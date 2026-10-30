"""Provenance-preserving conversion of manual impact timestamps to G1 targets."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotate(q, v):
    """Unit quaternion wxyz, broadcastable vectors."""
    return v + 2*np.cross(q[..., 1:], np.cross(q[..., 1:], v)+q[..., :1]*v)


def racket_series(data, mount):
    index = list(data['body_names']).index(mount['parent_body'])
    q = data['body_quat_w'][:, index]
    offset = rotate(q, np.asarray(mount['center_in_wrist_frame_m']))
    proxy = mount['collision_proxy']
    normal_local = rotate(np.asarray(proxy['quaternion_wxyz']), np.array([0., 0., 1.]))
    return dict(position=data['body_pos_w'][:, index]+offset,
        velocity=data['body_lin_vel_w'][:, index]+np.cross(data['body_ang_vel_w'][:, index], offset),
        normal=rotate(q, np.broadcast_to(normal_local, offset.shape)))


def prepare(manifest, annotations, output, mount_path, allow_estimated=False, max_uncertainty_s=.04,
            skill_group=None, allow_boundary_windows=False):
    manifest, annotations, output, mount_path = map(Path, (manifest, annotations, output, mount_path))
    output.mkdir(parents=True, exist_ok=False)
    mount = json.loads(mount_path.read_text(encoding='utf-8'))
    rows = json.loads(manifest.read_text(encoding='utf-8'))['motions']
    accepted, excluded = [], []
    for row in rows:
        key = row['id']
        revisions = sorted((annotations/key).glob('revision_*.json'))
        try:
            if not revisions:
                raise ValueError('not_annotated')
            evidence_path = revisions[-1]
            a = json.loads(evidence_path.read_text(encoding='utf-8'))
            if a['status'] != 'confirmed' or not a['events']:
                raise ValueError('not_confirmed_or_no_events')
            if skill_group and (a['stroke_type'] == 'serve') != (skill_group == 'serve'):
                excluded.append(dict(id=key, reason='other_skill_group'))
                continue
            motion = (manifest.parent/row['motion_file']).resolve()
            if a['manifest_sha256'] != sha(manifest) or a['g1_motion_sha256'] != sha(motion):
                raise ValueError('provenance_changed')
            quality = json.loads(motion.with_name('reference_quality.json').read_text(encoding='utf-8'))
            if quality['target_model_sha256'] != mount['target_model_sha256']:
                raise ValueError('racket_mount_model_mismatch')
            with np.load(motion, allow_pickle=False) as d:
                fps = float(np.asarray(d['fps']).flat[0])
                if fps != 50:
                    raise ValueError('expected_50Hz_reference')
                series = racket_series(d, mount)
                frames = len(d['joint_pos'])
            events = []
            for e in a['events']:
                if e['evidence'] == 'estimated_swing' and not allow_estimated:
                    raise ValueError('estimated_swing_requires_explicit_opt_in')
                if e['uncertainty_source_frames']/30 > max_uncertainty_s:
                    raise ValueError('timing_uncertainty_too_large')
                f = int(e['g1_frame'])
                mapped = int(np.floor(e['clip_time_s']*fps+.5))
                if f != mapped or not 0 <= f < frames:
                    raise ValueError('invalid_frame_mapping')
                boundary = f < 2 or f+2 >= frames
                if boundary and not allow_boundary_windows:
                    raise ValueError('incomplete_hit_window_or_frame_mapping')
                events.append(dict(frame=f, time_s=f/fps,
                    available_hit_window_frames=[max(0, f-2), min(frames-1, f+2)],
                    truncated_hit_window=boundary,
                    position=series['position'][f].tolist(), velocity=series['velocity'][f].tolist(),
                    normal=series['normal'][f].tolist(), annotation=e))
            events.sort(key=lambda e: e['frame'])
            if any(y['frame']-x['frame'] <= 4 for x, y in zip(events, events[1:])):
                raise ValueError('overlapping_hit_windows')
            accepted.append(dict(id=key, motion_file=str(motion), motion_sha256=sha(motion),
                frames=frames, fps=fps, stroke_type=a['stroke_type'], events=events,
                annotation_file=str(evidence_path.resolve()), annotation_sha256=sha(evidence_path),
                racket_state_source='retargeted_G1_wrist_rigid_attachment_not_measured',
                ball_state_source='not_available', physics_verified=False))
        except (ValueError, KeyError, OSError) as error:
            excluded.append(dict(id=key, reason=str(error)))
    result = dict(format='tennis_impact_stage1_v1', source_manifest=str(manifest.resolve()),
        source_manifest_sha256=sha(manifest), mount=mount, mount_sha256=sha(mount_path),
        motions=accepted, excluded=excluded, annotated_motion_count=len(accepted),
        hit_window_half_frames=2, max_uncertainty_s=max_uncertainty_s,
        skill_group=skill_group, boundary_windows_opt_in=allow_boundary_windows,
        estimated_swing_opt_in=allow_estimated, source_files_modified=False)
    (output/'impact_manifest.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mount', type=Path, default=Path(__file__).resolve().parents[2]/'configs/racket_mount.json')
    p.add_argument('--allow-estimated', action='store_true')
    p.add_argument('--max-uncertainty-s', type=float, default=.04)
    p.add_argument('--skill-group', choices=('rally', 'serve'))
    p.add_argument('--allow-boundary-windows', action='store_true')
    a = p.parse_args()
    r = prepare(a.manifest, a.annotations, a.output, a.mount, a.allow_estimated,
                a.max_uncertainty_s, a.skill_group, a.allow_boundary_windows)
    print(json.dumps(dict(ready=len(r['motions']), excluded=len(r['excluded']), output=str(a.output))))


if __name__ == '__main__':
    main()

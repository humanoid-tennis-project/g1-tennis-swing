"""Lossless input adapter from reviewed SMPL exports to the existing GMR driver."""
import hashlib
import json
from pathlib import Path
import numpy as np

BODY_NAMES = ('pelvis','left_hip','right_hip','spine1','left_knee','right_knee',
              'spine2','left_ankle','right_ankle','spine3','left_foot','right_foot',
              'neck','left_collar','right_collar','head','left_shoulder',
              'right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist')


def adapt(row, output):
    source = Path(row['human_motion_file'])
    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_hash != row['human_motion_sha256']:
        raise ValueError('Frozen motion hash mismatch')
    evidence = Path(row['review_evidence'])
    if hashlib.sha256(evidence.read_bytes()).hexdigest() != row['review_evidence_sha256']:
        raise ValueError('Frozen user decision hash mismatch')
    with np.load(source, allow_pickle=False) as data:
        n = len(data['global__body_pose'])
        if n < 3 or float(data['fps']) != 30:
            raise ValueError('Expected at least three source frames at 30Hz')
        if data['joints_global'].shape != (n,22,3):
            raise ValueError('Expected the 22 SMPL body joints')
        frames = data['parent_30Hz_frames'].copy()
        np.testing.assert_array_equal(np.diff(frames),np.ones(n-1,dtype=np.int64))
        np.testing.assert_allclose(data['parent_elapsed_seconds'],frames/30,rtol=0,atol=1e-12)
        # The source interval is half-open. Preserve its clock, starting at zero
        # locally; do not add an extra frame at the exclusive right boundary.
        times = (frames-frames[0]).astype(np.float64)/30
        joints = dict(joints_global=data['joints_global'].copy(),joints_incam=data['joints_incam'].copy(),
                      time_seconds=times,fps=np.array(30.),joint_names=np.array(BODY_NAMES),
                      K_fullimg=data['K_fullimg'].copy())
        params = {k:data[k].copy() for k in data.files if k.startswith(('global__','incam__'))}
        if not all(np.isfinite(v).all() for v in [*params.values(),joints['joints_global'],times]):
            raise ValueError('Nonfinite source motion')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'human_joints_local.npz',**joints)
    np.savez_compressed(output/'human_motion_parameters.npz',**params)
    record=dict(**row,adapter='adapt_selected_human.py',source_arrays_changed=False,
                source_coordinate_system='Y-up',source_fps=30,
                source_interval_frames=[int(frames[0]),int(frames[-1])+1],
                local_time_last_s=float(times[-1]),source_interval_duration_s=n/30,
                subject_style_selection='User approved; NG motions remain in the frozen exclusion list')
    (output/'provenance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    return record

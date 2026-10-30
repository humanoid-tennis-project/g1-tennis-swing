"""Compare all saved fixed evaluations; render recorded states without new rollouts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


from project_paths import REPO as ROOT, G1_XML_DIR
METRICS = ['joint_rmse_rad', 'body_aligned_rmse_m', 'racket_world_error_m',
           'racket_velocity_error_m_s', 'racket_normal_world_error_rad']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recover_numeric(run):
    """Read only explicit numeric files; never extract arbitrary archive paths."""
    for archive in sorted(run.glob('*.tar')):
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                parts = Path(member.name).parts
                if (not member.isfile() or len(parts) != 3 or parts[0] != run.name
                        or not parts[1].startswith('monitor_eval_')
                        or parts[2] not in ('summary.json', 'tracking.csv', 'simulated_states.npz')):
                    continue
                target = run / parts[1] / parts[2]
                if target.exists():
                    continue
                target.parent.mkdir(exist_ok=True)
                with tar.extractfile(member) as source:
                    target.write_bytes(source.read())


def load_model(xml):
    tree = ET.parse(xml)
    root = tree.getroot()
    compiler = root.find('compiler')
    meshdir = xml.parent / compiler.get('meshdir', '')
    assets = {}
    for i, mesh in enumerate(root.findall('.//asset/mesh')):
        source = meshdir / mesh.get('file')
        virtual = f'mesh_{i}{source.suffix}'
        assets[virtual] = source.read_bytes()
        mesh.set('file', virtual)
    compiler.attrib.pop('meshdir', None)
    world = root.find('worldbody')
    ET.SubElement(world, 'geom', name='comparison_floor', type='plane', size='30 30 .1',
                  rgba='.86 .88 .89 1', contype='0', conaffinity='0')
    ET.SubElement(world, 'light', pos='0 0 4', dir='0 0 -1', diffuse='.8 .8 .8')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'), assets=assets)
    model.vis.global_.offwidth = 480
    model.vis.global_.offheight = 360
    return model


def read_evaluation(run, folder, snapshot):
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    if summary['evaluated'] != summary['requested']:
        raise ValueError(f'Partial fixed evaluation: {folder}')
    with (folder / 'tracking.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    with np.load(folder / 'simulated_states.npz', allow_pickle=False) as z:
        states = {k: z[k] for k in z.files if k.endswith(('_qpos', '_qvel'))}
    episodes = {}
    for episode in summary['episodes']:
        key = f"clip{episode['clip']:04d}_start{round(episode['start_time_s']*1000):04d}"
        selected = [r for r in rows if int(r['clip']) == episode['clip']
                    and abs(float(r['start_time_s']) - episode['start_time_s']) < 1e-8]
        q = states[key + '_qpos']
        v = states[key + '_qvel']
        if len(selected) != len(q) or len(q) != episode['samples']:
            raise ValueError(f'State/metric alignment mismatch: {key}')
        times = np.array([float(r['sim_time_s']) for r in selected])
        if not np.allclose(times, .02 * np.arange(1, len(q)+1)):
            raise ValueError('Unexpected sample grid')
        episodes[key] = dict(episode=episode, q=q, v=v, times=times,
            metrics={k: np.array([float(r[k]) for r in selected]) for k in METRICS})
    return dict(folder=folder, summary=summary, episodes=episodes, snapshot=snapshot,
                source_hash=digest(folder / 'simulated_states.npz'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    evaluations = []
    for run in sorted((a.campaign / 'recovered').iterdir()):
        if not run.is_dir() or not (run / 'snapshots.json').exists():
            continue
        recover_numeric(run)
        snapshots = json.loads((run / 'snapshots.json').read_text())
        for folder in sorted(run.glob('monitor_eval_*')):
            if not all((folder / n).exists() for n in ('summary.json', 'tracking.csv', 'simulated_states.npz')):
                continue
            seconds = int(folder.name.split('_')[-1])
            snap = next(s for s in snapshots if Path(s['path']).name == f'wall_{seconds:05d}.pt')
            evaluations.append(read_evaluation(run, folder, snap))
    if not evaluations:
        raise ValueError('No complete saved evaluations')
    evaluations.sort(key=lambda e: e['snapshot']['env_steps'])
    keys = sorted(set.intersection(*(set(e['episodes']) for e in evaluations)))
    if len(keys) != 20:
        raise ValueError(f'Expected all 20 fixed conditions, found {len(keys)}')
    model_path = G1_XML_DIR / 'g1_rightracket.xml'
    model = load_model(model_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=360, width=480)
    result = dict(training_performed=False, new_rollouts=False,
        description='Recorded post-action physical states, CPU FK rendering only. No policy or physics integration.',
        model_sha256=digest(model_path), camera='Fixed per condition across checkpoints; no per-policy following',
        metrics='Per-step metrics averaged over the common simulation-time prefix of all checkpoints for this condition.',
        stages=[dict(label=f"この実行の約{e['snapshot']['wall_s']/3600:.0f}時間 / {e['snapshot']['env_steps']/1e6:.1f}M遷移", update=e['snapshot']['update'],
                     env_steps=e['snapshot']['env_steps'], source=e['folder'].relative_to(a.campaign).as_posix(),
                     checkpoint_sha256=e['summary']['checkpoint_sha256'], states_sha256=e['source_hash'],
                     evaluated=e['summary']['evaluated'], completed=e['summary']['completed']) for e in evaluations],
        conditions=[])
    montage_rows=[]
    try:
        for key in keys:
            entries = [e['episodes'][key] for e in evaluations]
            first = entries[0]['episode']
            common_n = min(len(e['q']) for e in entries)
            all_xy = np.concatenate([e['q'][:, :2] for e in entries])
            low, high = np.min(all_xy, axis=0), np.max(all_xy, axis=0)
            camera = mujoco.MjvCamera()
            camera.azimuth = 125
            camera.elevation = -15
            camera.lookat[:] = [*(low+high)/2, .95]
            camera.distance = max(3.4, float(np.max(high-low))*1.4+2.7)
            condition = dict(key=key, clip=first['clip'], clip_id=first['clip_id'],
                stroke_type=first['stroke_type'], start_time_s=first['start_time_s'],
                reference_duration_s=first['reference_duration_s'], common_seconds=common_n*.02, stages=[])
            posters=[]
            for i, entry in enumerate(entries):
                if entry['q'].shape[1] != model.nq or entry['v'].shape[1] != model.nv:
                    raise ValueError('Model and recorded states dimensions disagree')
                video = f'{key}_stage{i:02d}.mp4'
                poster = f'{key}_stage{i:02d}.jpg'
                # 50Hz states -> 25fps. First rendered frame is t=.04s,
                # then .08s, matching the evaluator's original writer.
                indices = list(range(1, len(entry['q']), 2))
                if not indices or indices[-1] != len(entry['q'])-1:
                    indices.append(len(entry['q'])-1)
                with imageio.get_writer(a.output/video, fps=25, codec='libx264', quality=7,
                                       macro_block_size=1, ffmpeg_params=['-movflags', '+faststart']) as writer:
                    for n, idx in enumerate(indices):
                        data.qpos[:] = entry['q'][idx]
                        data.qvel[:] = entry['v'][idx]
                        mujoco.mj_forward(model, data)
                        renderer.update_scene(data, camera=camera)
                        image = Image.fromarray(renderer.render())
                        draw = ImageDraw.Draw(image)
                        draw.rectangle((0,0,480,31), fill='#17252d')
                        draw.text((10,9), f"Recorded PPO | update {evaluations[i]['snapshot']['update']} | t={(idx+1)*.02:.2f}s", fill='white')
                        if n == len(indices)-1:
                            draw.rectangle((0,327,480,360), fill='#176445' if entry['episode']['completed'] else '#923c38')
                            draw.text((10,337), 'COMPLETE' if entry['episode']['completed'] else 'TERMINATED - NO FURTHER STATE', fill='white')
                        writer.append_data(np.asarray(image))
                        if n == 0:
                            image.save(a.output/poster)
                            posters.append(image.copy())
                with imageio.get_reader(a.output/video) as reader:
                    decoded = sum(1 for _ in reader)
                if decoded != len(indices):
                    raise ValueError('Video decode frame count mismatch')
                curves = dict(t=[round(float(t),3) for t in entry['times'][::5]])
                curves.update({k:[round(float(v),6) for v in entry['metrics'][k][::5]] for k in METRICS})
                condition['stages'].append(dict(video=video, poster=poster, frames=decoded,
                    video_first_sim_time_s=(indices[0]+1)*.02,
                    completed=entry['episode']['completed'], reached_s=entry['episode']['reached_s'],
                    termination=entry['episode']['termination_reason'],
                    common_prefix={k:float(np.mean(v[:common_n])) for k,v in entry['metrics'].items()}, curves=curves))
            result['conditions'].append(condition)
            row=Image.new('RGB',(480*len(posters),390),'#f2f5f7')
            for i,img in enumerate(posters):row.paste(img,(480*i,30))
            ImageDraw.Draw(row).text((10,9),f"{key} | {first['stroke_type']} | recorded rollout, first frame",fill='black')
            montage_rows.append(row)
            (a.output/'progress.json').write_text(json.dumps({'conditions':len(result['conditions']),'total':len(keys)}))
            print(key, 'done', flush=True)
    finally:
        renderer.close()
    sheet=Image.new('RGB',(480*len(evaluations),390*len(keys)),'white')
    for n,row in enumerate(montage_rows):sheet.paste(row,(0,390*n))
    sheet.save(a.output/'all_conditions_first_frame.jpg',quality=85)
    (a.output/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    fields=['clip','start_time_s','stroke_type','update','env_steps','completed','reached_s','common_seconds',*METRICS]
    with (a.output/'common_prefix_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for condition in result['conditions']:
            for stage,values in zip(result['stages'],condition['stages']):
                writer.writerow(dict(clip=condition['clip'],start_time_s=condition['start_time_s'],
                    stroke_type=condition['stroke_type'],update=stage['update'],env_steps=stage['env_steps'],
                    completed=values['completed'],reached_s=values['reached_s'],
                    common_seconds=condition['common_seconds'],**values['common_prefix']))
    (a.output/'index.html').write_text((Path(__file__).parent/'templates/ppo_motion_comparison.html').read_text(encoding='utf-8'),encoding='utf-8')
    print(json.dumps({'conditions':len(keys),'checkpoints':len(evaluations),'videos':len(keys)*len(evaluations)}),flush=True)


if __name__ == '__main__':
    main()

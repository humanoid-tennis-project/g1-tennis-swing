"""Read-only FK diagnosis of saved PPO states; no training or new rollout."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R

from build_ppo_motion_comparison import ROOT, load_model, digest
from project_paths import G1_XML_DIR

RUNS = ['impact474_recovery1_00_evaluate_20261029_v1',
        'impact474_fastfollowup_01_evaluate_20261029_v1']
EE = ['left_ankle_roll_link', 'right_ankle_roll_link',
      'left_wrist_yaw_link', 'right_wrist_yaw_link']


def yaw(q):
    w, x, y, z = np.asarray(q).T
    return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def rounded(a):
    return np.asarray(a).round(6).tolist()


def marker(scene, p, color, radius=.045):
    assert scene.ngeom < scene.maxgeom
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius]*3), np.asarray(p, dtype=float), np.eye(3).ravel(), np.array(color))
    scene.ngeom += 1


def sole_clearance(model, data):
    """Lowest point of the seven collision capsules of each foot, ground z=0."""
    result=[]
    for side in ('left','right'):
        ids=[i for i in range(model.ngeom)
             if model.geom(i).name.split('/')[-1].startswith(side+'_foot')
             and model.geom(i).name.endswith('_collision')]
        assert len(ids)==7
        bottom=[]
        for i in ids:
            assert model.geom_type[i]==mujoco.mjtGeom.mjGEOM_CAPSULE
            axis=data.geom_xmat[i].reshape(3,3)[:,2]
            bottom.append(data.geom_xpos[i,2]-abs(axis[2])*model.geom_size[i,1]-model.geom_size[i,0])
        result.append(min(bottom))
    return result


def main(root, before=None, after=None, bundle=None, output=None, updates=(3202,6348), labels=None,only_clips=None):
    out = output or root/'diagnosis'; out.mkdir(parents=True,exist_ok=True)
    bundle = bundle or root.parent/'cloud_bundle_v3'
    folders=[before or root/'recovered'/RUNS[0],after or root/'recovered'/RUNS[1]]
    labels=labels or [f'update {u}' for u in updates]
    manifest = json.loads((bundle/'impact_manifest.json').read_text(encoding='utf-8'))
    xml = G1_XML_DIR/'g1_rightracket.xml'
    model = load_model(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=320, width=400)
    bodies = {model.body(i).name.split('/')[-1]: i for i in range(model.nbody)}
    joints = {model.joint(i).name.split('/')[-1]: i for i in range(model.njnt)}
    site = next(i for i in range(model.nsite) if model.site(i).name.split('/')[-1] == 'racket_point')
    geom = next(i for i in range(model.ngeom) if model.geom(i).name.split('/')[-1] == 'tennis_racket_collision')
    evaluations=[]
    for folder in folders:
        summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        with (folder/'tracking.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
        arrays=np.load(folder/'simulated_states.npz',allow_pickle=False)
        evaluations.append((summary, rows, arrays))
    result=dict(new_rollouts=False, training=False, physics_stepping=False,
        source='Saved post-action physical states; reference reconstructed from frozen NPZ using runtime heading alignment.',
        model_sha256=digest(xml), conditions=[],
        caveats=['FK velocity/height are diagnostics, not recomputed GPU termination truth.',
                 'Foot horizontal speed is not a measured slip/contact classification.',
                 'Stored actuator_force is a saved simulator signal; no saturation claim without actuator limits.',
                 'Markers and G1 racket targets are model-derived, not measured ball contact.'])
    result['recorded_sources']=[dict(run=str(folder),
        checkpoint_sha256=evaluations[i][0]['checkpoint_sha256'],
        states_sha256=digest(folder/'simulated_states.npz'),
        metrics_sha256=digest(folder/'tracking.csv')) for i,folder in enumerate(folders)]
    eligible=[ep for ep in evaluations[1][0]['episodes'] if only_clips is None or ep['clip'] in only_clips]
    result['rendered_subset_clips']=sorted(only_clips) if only_clips is not None else None
    result['rendered_conditions_expected']=len(eligible)
    try:
        for ep in eligible:
            clip=ep['clip'];start=ep['start_time_s'];key=f'clip{clip:04d}_start{round(start*1000):04d}'
            src=bundle/f'inputs/clip{clip:04d}/motion.npz'
            assert digest(src)==manifest['motions'][clip]['motion_sha256']
            with np.load(src,allow_pickle=False) as z: ref={k:z[k].copy() for k in z.files}
            names=list(ref['body_names']);root_i=names.index('pelvis')
            rotation=R.from_euler('z',-float(yaw(ref['body_quat_w'][-1,root_i])))
            pivot=ref['body_pos_w'][0,root_i]
            ref_pos=rotation.apply((ref['body_pos_w']-pivot).reshape(-1,3)).reshape(ref['body_pos_w'].shape)+pivot
            ref_rot=rotation*R.from_quat(ref['body_quat_w'].reshape(-1,4),scalar_first=True)
            ref_quat=ref_rot.as_quat(scalar_first=True).reshape(ref['body_quat_w'].shape)
            qadr=np.array([model.jnt_qposadr[joints[n]] for n in ref['joint_names']])
            assert np.array_equal(qadr,np.arange(7,34))
            refq=np.zeros((len(ref_pos),model.nq));refq[:,:3]=ref_pos[:,root_i];refq[:,3:7]=ref_quat[:,root_i];refq[:,qadr]=ref['joint_pos']
            full_clearance=[]
            for qr in refq:
                data.qpos[:]=qr;data.qvel[:]=0;mujoco.mj_forward(model,data)
                full_clearance.append(sole_clearance(model,data))
            series=[]
            for stage,(summary,rows,arr) in enumerate(evaluations):
                episode=next(e for e in summary['episodes'] if e['clip']==clip and e['start_time_s']==start)
                selected=[r for r in rows if int(r['clip'])==clip and abs(float(r['start_time_s'])-start)<1e-8]
                q=arr[key+'_qpos'];v=arr[key+'_qvel'];times=arr[key+'_reference_time_s']
                goal=arr[key+'_applied_impact_goal_pos_vel_normal_time_hit']
                frames=np.clip(np.rint(times*50).astype(int),0,len(refq)-1)
                assert len(q)==len(selected)==episode['samples']
                assert np.max(np.abs(ref['joint_pos'][frames]-arr[key+'_reference_joint_pos']))<2e-4
                rp=ref_pos[frames];rq=ref_quat[frames]
                heights=[];feet_speed=[];rackets=[];normals=[];joint_err=[];fk_errors=[];goal_errors=[];clearances=[];actual_z=[]
                for i in range(len(q)):
                    data.qpos[:]=q[i];data.qvel[:]=v[i];mujoco.mj_forward(model,data)
                    actual=data.xpos[[bodies[n] for n in names]].copy()
                    clearances.append(sole_clearance(model,data));actual_z.append(actual[[names.index(n) for n in EE],2])
                    heights.append(actual[[names.index(n) for n in EE],2]-rp[i,[names.index(n) for n in EE],2])
                    rackets.append(data.site_xpos[site].copy());normals.append(data.geom_xmat[geom].reshape(3,3)[:,2].copy())
                    fs=[]
                    for name in EE[:2]:
                        vel=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,bodies[name],vel,0);fs.append(float(np.linalg.norm(vel[3:5])))
                    feet_speed.append(fs)
                    joint_err.append(q[i,qadr]-ref['joint_pos'][frames[i]])
                    rw=R.from_quat(rq[i,names.index('right_wrist_yaw_link')],scalar_first=True)
                    reference_racket=rp[i,names.index('right_wrist_yaw_link')]+rw.apply(model.site_pos[site])
                    fk_errors.append(abs(np.linalg.norm(data.site_xpos[site]-reference_racket)-float(selected[i]['racket_world_error_m'])))
                    goal_errors.append(abs(np.linalg.norm(data.site_xpos[site]-goal[i,:3])-float(selected[i]['target_position_error_m'])))
                # Verify reconstruction against the original evaluator before drawing references.
                assert max(fk_errors)<3e-4, (key,max(fk_errors))
                assert max(goal_errors)<3e-4
                heights=np.asarray(heights);joint_err=np.asarray(joint_err)
                terminal_abs=np.abs(heights[-1]);rank=np.argsort(-np.sqrt(np.mean(joint_err**2,axis=0)))[:5]
                impactframes=[e['frame'] for e in manifest['motions'][clip]['events']]
                focus_impact=impactframes[0]/50-start
                force=arr[key+'_actuator_force'] if key+'_actuator_force' in arr else None
                force_max=np.max(np.abs(force),axis=1) if force is not None else None
                worst=int(np.argmax(terminal_abs))
                chart={name:rounded([float(r[name]) for r in selected]) for name in ['sim_time_s','motion_time_s','base_tilt_rad','base_height_error_m','racket_world_error_m','racket_velocity_error_m_s','racket_plane_error_deg','target_position_error_m','target_velocity_error_m_s','target_plane_error_deg','racket_actual_speed_m_s','racket_reference_speed_m_s']}
                chart.update(ee_height_signed_m=rounded(heights),feet_horizontal_speed_m_s=rounded(feet_speed),
                    actual_ee_z_m=rounded(actual_z),reference_ee_z_m=rounded(rp[:,[names.index(n) for n in EE],2]),
                    actual_sole_clearance_m=rounded(clearances),reference_sole_clearance_m=rounded(np.asarray(full_clearance)[frames]),
                    force_max_abs=rounded(force_max) if force_max is not None else None,
                    hit=[bool(float(g[-1])) for g in goal],racket_position=rounded(rackets),goal_position=rounded(goal[:,:3]))
                series.append(dict(update=updates[stage],stage_index=stage,label=labels[stage],completed=episode['completed'],termination=episode['termination_reason'],
                    reached_s=episode['reached_s'],hit_samples=episode['hit_samples'],
                    terminal_height_error_m=dict(zip(EE,rounded(heights[-1]))),
                    terminal_largest_height_body=EE[worst],terminal_max_abs_height_error_m=float(terminal_abs[worst]),
                    first_height_cross_25cm_s=next((round((i+1)*.02,4) for i,h in enumerate(heights) if max(abs(h))>.25),None),
                    top_joint_rmse=[dict(joint=str(ref['joint_names'][i]),rmse_rad=float(np.sqrt(np.mean(joint_err[:,i]**2)))) for i in rank],
                    max_saved_actuator_force_abs=float(force_max.max()) if force_max is not None else None,
                    reference_reset_sole_clearance_m=rounded(full_clearance[round(start*50)]),
                    fk_reconstruction_max_error_m=max(fk_errors),charts=chart,q=q,v=v,goal=goal))
            frame_count=max(max(len(s['q']) for s in series),len(refq)-round(start*50))
            all_xy=np.concatenate([refq[:,:2],*[s['q'][:,:2] for s in series]])
            low,high=all_xy.min(0),all_xy.max(0)
            camera=mujoco.MjvCamera();camera.lookat[:]=[*(low+high)/2,.95];camera.distance=max(3.5,float(max(high-low))*1.3+2.7);camera.azimuth=125;camera.elevation=-14
            video=out/(key+'.mp4');posters=[]
            selected_steps=sorted(set([*range(0,frame_count,2),frame_count-1]))
            with imageio.get_writer(video,fps=25,codec='libx264',quality=7,macro_block_size=1,ffmpeg_params=['-movflags','+faststart']) as writer:
                for step in selected_steps:
                    panels=[]
                    for panel in range(3):
                        if panel==1:
                            frame=min(len(refq)-1,round(start*50)+step+1);data.qpos[:]=refq[frame];data.qvel[:]=0
                            label='G1 REFERENCE (kinematic)';ended=False;g=series[1]['goal'][min(step,len(series[1]['q'])-1)]
                        else:
                            entry=series[0 if panel==0 else 1];idx=min(step,len(entry['q'])-1)
                            data.qpos[:]=entry['q'][idx];data.qvel[:]=entry['v'][idx];g=entry['goal'][idx];ended=step>=len(entry['q'])
                            label=f"PHYSICAL | {entry['label']}"
                        mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera)
                        marker(renderer.scene,g[:3],[.05,.75,.25,.95]);marker(renderer.scene,data.site_xpos[site],[.95,.25,.1,.9],.025)
                        img=Image.fromarray(renderer.render());draw=ImageDraw.Draw(img)
                        draw.rectangle((0,0,400,42),fill='#1d2a35');draw.text((6,6),label,fill='white');draw.text((6,24),f't={(step+1)*.02:.2f}s | green = target; red = racket',fill='white')
                        if ended:
                            draw.rectangle((0,278,400,320),fill='#9b3333');draw.text((8,287),'ENDED - final recorded state held',fill='white')
                        panels.append(img)
                    picture=Image.new('RGB',(1200,320));[picture.paste(img,(400*i,0)) for i,img in enumerate(panels)];writer.append_data(np.asarray(picture))
                    if abs((step+1)*.02-focus_impact)<.025:posters.append(picture.copy())
                    if step==min(selected_steps,key=lambda i:abs(i-(len(series[1]['q'])-1))):
                        picture.save(out/(key+'_failure.jpg'),quality=85)
                picture.save(out/(key+'_terminal.jpg'),quality=85)
            if posters:posters[0].save(out/(key+'_impact.jpg'),quality=85)
            with imageio.get_reader(video) as reader: assert reader.count_frames()==len(selected_steps)
            clean=[]
            for entry in series: clean.append({k:v for k,v in entry.items() if k not in ['q','v','goal']})
            result['conditions'].append(dict(key=key,clip=clip,clip_id=ep['clip_id'],stroke_type=ep['stroke_type'],start_time_s=start,
                reference_duration_s=(len(refq)-1)/50,impact_sim_times_s=[f/50-start for f in impactframes],
                reference_sole_clearance_full_m=rounded(full_clearance),
                video=video.name,poster=key+'_terminal.jpg',video_first_sim_time_s=.02,
                stages=clean,source_motion_sha256=digest(src)))
            (out/'progress.json').write_text(json.dumps(dict(done=len(result['conditions']),total=len(eligible))),encoding='utf-8')
            print(key,'done',flush=True)
    finally:
        renderer.close()
        for _,_,arrays in evaluations:arrays.close()
    result['failure_body_counts']={labels[i]:dict(Counter(c['stages'][i]['terminal_largest_height_body'] for c in result['conditions'] if not c['stages'][i]['completed'])) for i in range(2)}
    (out/'diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    html=(Path(__file__).parent/'templates/impact_diagnosis.html').read_text(encoding='utf-8')
    html=html.replace('追加前・3202更新',labels[0]).replace('追加後・6348更新',labels[1])
    old='追加後の14失敗は、終了時の高さ誤差が最大の部位で集計すると手首8条件・足首6条件です。'
    html=html.replace(old,'今回の比較条件と失敗部位は下の各条件の数値に記録しています。')
    if only_clips is not None:html=html.replace('全20条件','可視化した条件')
    (out/'index.html').write_text(html,encoding='utf-8')
    print(json.dumps(result['failure_body_counts']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--before',type=Path);p.add_argument('--after',type=Path);p.add_argument('--bundle',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--before-update',type=int,default=3202);p.add_argument('--after-update',type=int,default=6348)
    p.add_argument('--before-label');p.add_argument('--after-label')
    p.add_argument('--only-clips',default='')
    a=p.parse_args()
    labels=[a.before_label or f'update {a.before_update}',a.after_label or f'update {a.after_update}']
    main(a.root.resolve(),a.before,a.after,a.bundle,a.output,(a.before_update,a.after_update),labels,
         {int(x) for x in a.only_clips.split(',') if x} if a.only_clips else None)

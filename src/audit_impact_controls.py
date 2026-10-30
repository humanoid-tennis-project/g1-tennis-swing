"""Audit saved position commands, joint response and actuator effort, without rollouts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from build_ppo_motion_comparison import digest

RUNS=['impact474_recovery1_00_evaluate_20261029_v1',
      'impact474_fastfollowup_01_evaluate_20261029_v1']


def build_metadata(root):
    """Build the same actuator-equipped entity on CPU to recover exact ctrl ordering."""
    from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g
    from mjlab.entity import Entity
    import os
    from project_paths import G1_XML_DIR
    xmlbase=G1_XML_DIR
    g.G1_RIGHT_XML=Path(os.path.relpath(xmlbase/'g1_rightracket.xml'))
    g.G1_LEFT_XML=Path(os.path.relpath(xmlbase/'g1_leftracket.xml'))
    model=Entity(g.get_g1_robot_cfg(racket_hand='right')).spec.compile()
    assert model.nq==34 and model.nu==27
    m=dict(actuator_joints=[model.joint(int(i)).name for i in model.actuator_trnid[:,0]],
        actuator_joint_qpos_indices=model.jnt_qposadr[model.actuator_trnid[:,0]].tolist(),
        actuator_joint_qvel_indices=model.jnt_dofadr[model.actuator_trnid[:,0]].tolist(),
        kp=model.actuator_gainprm[:,0].tolist(),kd=(-model.actuator_biasprm[:,2]).tolist(),
        force_limits=model.actuator_forcerange.tolist(),force_limited=model.actuator_forcelimited.tolist(),
        gear=model.actuator_gear[:,0].tolist(),keyframe_qpos=model.key_qpos.tolist())
    (root/'diagnosis/actuator_metadata.json').write_text(json.dumps(m,indent=2),encoding='utf-8')


def main(root,before=None,after=None,output=None,updates=(3202,6348)):
    out=output or root/'diagnosis';meta=json.loads((out/'actuator_metadata.json').read_text(encoding='utf-8'))
    bundle=root.parent/'cloud_bundle_v3'
    kp=np.asarray(meta['kp']);kd=np.asarray(meta['kd']);limits=np.max(np.abs(meta['force_limits']),axis=1)
    aq=np.asarray(meta['actuator_joint_qpos_indices']);av=np.asarray(meta['actuator_joint_qvel_indices'])
    # Saved actions are joint order, but MuJoCo ctrl/force arrays are actuator order.
    jidx=aq-7;assert sorted(jidx)==list(range(27))
    nominal=np.asarray(meta['keyframe_qpos'][0])[aq]
    scale=.25*limits/kp
    result=dict(new_training=False,new_rollouts=False,actuator_metadata_sha256=digest(out/'actuator_metadata.json'),
        actuator_order=meta['actuator_joints'],conditions=[],
        interpretation='Position setpoints need not equal reference joint angles: PD compensation and dynamics may require offsets. RMS terms are not an additive causal decomposition. Saturation is sampled at 50Hz, not every physics substep.')
    max_force_mapping_error=0.;max_action_mapping_error=0.
    for stage,folder in enumerate([before or root/'recovered'/RUNS[0],after or root/'recovered'/RUNS[1]]):
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        with (folder/'tracking.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        with np.load(folder/'simulated_states.npz',allow_pickle=False) as z:
            for ep in summary['episodes']:
                key=f"clip{ep['clip']:04d}_start{round(ep['start_time_s']*1000):04d}"
                q=z[key+'_qpos'][:,aq];v=z[key+'_qvel'][:,av];ctrl=z[key+'_ctrl'];force=z[key+'_actuator_force']
                qref=z[key+'_reference_joint_pos'][:,jidx]
                actions=z[key+'_action'][:,jidx]
                mapped=z[key+'_qfrc_actuator'][:,av]
                force_error=float(np.max(np.abs(mapped-force)));action_error=float(np.max(np.abs(ctrl-(nominal+scale*actions))))
                max_force_mapping_error=max(max_force_mapping_error,force_error);max_action_mapping_error=max(max_action_mapping_error,action_error)
                assert force_error<1e-4,(key,'force mapping',force_error)
                assert action_error<2e-5,(key,'action mapping',action_error)
                selected=[r for r in rows if int(r['clip'])==ep['clip'] and abs(float(r['start_time_s'])-ep['start_time_s'])<1e-8]
                assert len(selected)==len(q)
                hit=np.array([r['hit_window']=='True' for r in selected])
                relative_force=np.abs(force)/limits
                stats=[]
                for j,name in enumerate(meta['actuator_joints']):
                    stats.append(dict(joint=name,
                        actual_reference_rmse_rad=float(np.sqrt(np.mean((q[:,j]-qref[:,j])**2))),
                        command_reference_rmse_rad=float(np.sqrt(np.mean((ctrl[:,j]-qref[:,j])**2))),
                        actual_command_rmse_rad=float(np.sqrt(np.mean((q[:,j]-ctrl[:,j])**2))),
                        actual_reference_mean_rad=float(np.mean(q[:,j]-qref[:,j])),
                        command_reference_mean_rad=float(np.mean(ctrl[:,j]-qref[:,j])),
                        saturation_fraction_ge99pct=float(np.mean(relative_force[:,j]>=.99)),
                        peak_force_fraction=float(relative_force[:,j].max()),
                        force_limit_nm=float(limits[j]),
                        final_reference_rad=float(qref[-1,j]),final_command_rad=float(ctrl[-1,j]),final_actual_rad=float(q[-1,j])))
                entry=dict(key=key,update=updates[stage],stage_index=stage,clip=ep['clip'],stroke_type=ep['stroke_type'],
                    start_time_s=ep['start_time_s'],completed=ep['completed'],reached_s=ep['reached_s'],hit_samples=ep['hit_samples'],
                    stats=stats,force_mapping_max_error_nm=force_error,action_mapping_max_error_rad=action_error,
                    times_s=[float(r['sim_time_s']) for r in selected],hit=hit.tolist(),
                    reference=qref.round(6).tolist(),command=ctrl.round(6).tolist(),actual=q.round(6).tolist(),
                    force_fraction=relative_force.round(6).tolist())
                result['conditions'].append(entry)
    result['max_force_mapping_error_nm']=max_force_mapping_error;result['max_action_mapping_error_rad']=max_action_mapping_error
    result['checkpoint_sources']=[dict(run=str(folder),states_sha256=digest(folder/'simulated_states.npz')) for folder in [before or root/'recovered'/RUNS[0],after or root/'recovered'/RUNS[1]]]
    result['arm_summary']={}
    for side in ('left','right'):
        ss=[s for c in result['conditions'] if c['stage_index']==1 for s in c['stats']
            if s['joint'].startswith(side+'_') and any(x in s['joint'] for x in ('shoulder','elbow'))]
        result['arm_summary'][side]=dict(peak_force_fraction=max(s['peak_force_fraction'] for s in ss),
            sampled_saturation_fraction_mean=float(np.mean([s['saturation_fraction_ge99pct'] for s in ss])),
            command_reference_rmse_mean_rad=float(np.mean([s['command_reference_rmse_rad'] for s in ss])),
            actual_command_rmse_mean_rad=float(np.mean([s['actual_command_rmse_rad'] for s in ss])))
    (out/'controls.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    # Recompute only the known analytic kernels, not rewards for a new policy.
    hit=[r for r in rows if r['hit_window']=='True'];reward_audit={}
    for name,field,std in [('position','target_position_error_m',.15),
                           ('velocity','target_velocity_error_m_s',2.),('plane','target_plane_error_deg',.35)]:
        error=np.array([float(r[field])*(np.pi/180 if name=='plane' else 1) for r in hit])
        reward=np.exp(-(error/std)**2)
        reward_audit[name]=dict(hit_samples=len(hit),mean_reward=float(reward.mean()),
            median_reward=float(np.median(reward)),fraction_reward_below_001=float(np.mean(reward<.01)),
            mean_error=float(error.mean()))
    (out/'hit_reward_audit.json').write_text(json.dumps(dict(
        scope='nominal fixed20 evaluation hit samples; not training reward distribution',
        components=reward_audit),indent=2),encoding='utf-8')
    for c in result['conditions']:
        if c['stage_index']==1 and c['key'] in ['clip0361_start0000','clip0050_start0000','clip0004_start0000','clip0184_start0040']:
            print(c['key'])
            for s in c['stats']:
                if s['joint'].startswith('left_') and any(x in s['joint'] for x in ['shoulder','elbow']):print(json.dumps(s))
    print('MAPPING CHECK',max_force_mapping_error,max_action_mapping_error)


def arm_geometry_probe(root):
    """Hold recorded base/legs fixed and substitute reference left-arm angles for FK only."""
    import mujoco
    from build_ppo_motion_comparison import load_model
    from project_paths import G1_XML_DIR
    diagnosis=json.loads((root/'diagnosis/diagnosis.json').read_text(encoding='utf-8'))
    model=load_model(G1_XML_DIR/'g1_rightracket.xml')
    data=mujoco.MjData(model)
    ids=[model.jnt_qposadr[i] for i in range(model.njnt) if model.joint(i).name.startswith('left_')
         and any(t in model.joint(i).name for t in ('shoulder','elbow','wrist'))]
    wrist=next(i for i in range(model.nbody) if model.body(i).name=='left_wrist_yaw_link');checks=[]
    with np.load(root/'recovered'/RUNS[1]/'simulated_states.npz',allow_pickle=False) as states:
        for c in diagnosis['conditions']:
            stage=c['stages'][1]
            if stage['completed'] or stage['terminal_largest_height_body']!='left_wrist_yaw_link':continue
            key=c['key'];q=states[key+'_qpos'][-1].copy();ref=states[key+'_reference_joint_pos'][-1]
            with np.load(root.parent/f'cloud_bundle_v3/inputs/clip{c["clip"]:04d}/motion.npz',allow_pickle=False) as motion:
                frame=min(len(motion['joint_pos'])-1,round(states[key+'_reference_time_s'][-1]*50))
                rz=motion['body_pos_w'][frame,list(motion['body_names']).index('left_wrist_yaw_link'),2]
            data.qpos[:]=q;mujoco.mj_forward(model,data);original=data.xpos[wrist,2]-rz
            q[ids]=ref[np.asarray(ids)-7];data.qpos[:]=q;mujoco.mj_forward(model,data)
            checks.append(dict(key=key,original_height_error_m=float(original),
                left_arm_reference_only_height_error_m=float(data.xpos[wrist,2]-rz),physical_execution=False))
    (root/'diagnosis/left_arm_fk_counterfactual.json').write_text(json.dumps(dict(
        description='Geometric attribution only: same recorded root/legs/right arm; reference left-arm joint positions. No physics step or executable-control claim.',
        conditions=checks),indent=2),encoding='utf-8')


def plot_controls(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=json.loads((root/'diagnosis/controls.json').read_text(encoding='utf-8'))
    entries=[next(c for c in d['conditions'] if c['key']=='clip0361_start0000' and c['update']==u) for u in (3202,6348)]
    fig,axes=plt.subplots(3,2,figsize=(12,9))
    for row,joint in enumerate(('left_elbow_joint','left_shoulder_yaw_joint','left_shoulder_pitch_joint')):
        j=d['actuator_order'].index(joint);ax,force_ax=axes[row]
        ax.plot(entries[1]['times_s'],np.array(entries[1]['reference'])[:,j],color='black',label='Reference')
        for i,e in enumerate(entries):
            color=['#c47916','#087ba4'][i];alpha=.6 if i==0 else 1
            ax.plot(e['times_s'],np.array(e['command'])[:,j],ls='--',color=color,alpha=alpha,label=f"Command {e['update']}")
            ax.plot(e['times_s'],np.array(e['actual'])[:,j],color=color,alpha=alpha,label=f"Actual {e['update']}")
            force_ax.plot(e['times_s'],np.array(e['force_fraction'])[:,j],color=color,alpha=alpha,label=str(e['update']))
        force_ax.axhline(1,color='black',ls='--',label='Configured limit')
        ax.set(title=joint,ylabel='Joint angle (rad)',xlabel='Simulation time (s)');force_ax.set(title='Saved effort / configured effort limit',ylabel='Absolute ratio',xlabel='Simulation time (s)',ylim=(0,1.05))
        ax.grid(alpha=.2);force_ax.grid(alpha=.2)
        if row==0:ax.legend(fontsize=8,ncol=2);force_ax.legend(fontsize=8)
    fig.suptitle('clip361, smash: left-arm command differs from reference before hit\nNominal hit at 1.80 s; saved executions end at 0.52 / 0.78 s. No new rollout.')
    fig.tight_layout();fig.savefig(root/'diagnosis/left_arm_command_response.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--rebuild-metadata',action='store_true');p.add_argument('--plot',action='store_true');args=p.parse_args();root=args.root.resolve()
    if args.rebuild_metadata:build_metadata(root)
    main(root)
    arm_geometry_probe(root)
    if args.plot:plot_controls(root)

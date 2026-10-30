"""Evaluate many references with one loaded policy/environment; fresh CPU FK metrics."""
import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import mujoco
from .task import configure, install_reset_hook, GoalConfig
from tracking_experiment import seed_everything,write_json,snapshot,measurements,make_terminal_recorder,sha256
from windows_mujoco_paths import patch_adapt_paths
from racket_tracking_rewards import capture_reference_velocity,cpu_racket_metrics


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--indices',default='all');p.add_argument('--start-times',default='0,0.04')
    p.add_argument('--max-wall-seconds',type=int,default=3600)
    p.add_argument('--video-indices',default='')
    p.add_argument('--randomized-goals',action='store_true')
    from .reward_variants import VARIANTS
    p.add_argument('--reward-variant',choices=VARIANTS,default='baseline')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    seed_everything(100);patch_adapt_paths()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.recorder_manager import RecorderTermCfg
    from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
    metadata=json.loads(a.manifest.read_text(encoding='utf-8'))['motions']
    paths=[Path(r['motion_file']) for r in metadata]
    indices=list(range(len(paths))) if a.indices=='all' else [int(i) for i in a.indices.split(',')]
    videos={int(i) for i in a.video_indices.split(',') if i}
    starts=[float(t) for t in a.start_times.split(',')]
    cfg,agent,paths=configure(a.manifest,1,100,GoalConfig(randomized=a.randomized_goals))
    from .reward_variants import apply_reward_variant
    apply_reward_variant(cfg,a.manifest,GoalConfig(randomized=a.randomized_goals),a.reward_variant)
    cfg.commands['motion'].sampling_mode='start'
    cfg.recorders['terminal_capture']=RecorderTermCfg(func=make_terminal_recorder())
    started=time.monotonic();env=ManagerBasedRlEnv(cfg,device='cuda:0')
    from tracking_experiment import check_motion_layout
    write_json(a.output/'layouts.json',[check_motion_layout(env,path,i) for i,path in enumerate(paths)])
    install_reset_hook(env)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
    runner=MjlabOnPolicyRunner(wrapped,asdict(agent),device='cuda:0')
    checkpoint_meta=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    checkpoint_variant=checkpoint_meta.get('infos',{}).get('impact_scope',{}).get('reward_variant','baseline')
    if checkpoint_variant!=a.reward_variant:
        raise ValueError('Evaluation reward variant does not match checkpoint experiment')
    del checkpoint_meta
    runner.load(str(a.checkpoint.resolve()),load_cfg={'actor':True},map_location='cuda:0')
    policy=runner.get_inference_policy('cuda:0')
    motion=env.command_manager.get_term('motion')
    original_reset=motion._resample_command
    selected={'index':0,'frame':0}
    def deterministic_reset(ids):
        original_reset(ids)
        motion.motion_ids[ids]=selected['index']
        motion.reset_to_frame(ids,selected['frame'])
    motion._resample_command=deterministic_reset
    model=env.sim.mj_model;data=mujoco.MjData(model)
    episodes=[];all_rows=[];states={};renderer=None
    if videos:renderer=mujoco.Renderer(model,height=360,width=480)
    try:
        for index in indices:
            for start_time in starts:
                if time.monotonic()-started>a.max_wall_seconds-30:break
                selected.update(index=index,frame=round(start_time*50))
                obs,_=wrapped.reset();env._evaluation_terminal=None
                target=capture_reference_velocity(env,snapshot(env))
                expected=motion.motion.motions[index].joint_pos[selected['frame']].detach().cpu().numpy()
                reset_error=float(np.max(np.abs(target['joint_pos']-expected)))
                # Upstream reset computes the next action's reference before
                # returning the observation; compare the physical state with
                # the requested reset frame, not that one-step-ahead command.
                if reset_error>1e-4:
                    raise ValueError('Evaluation reset does not match selected reference')
                with (a.output/'reset_audit.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(dict(clip=index,requested_frame=selected['frame'],
                        reset_joint_max_abs_error_rad=reset_error,first_action_reference_time_s=target['motion_time_s']))+'\n')
                key=f'clip{index:04d}_start{round(start_time*1000):04d}'
                rows=[];qpos=[];qvel=[];actions=[];refq=[];frame_times=[];signals={};goals=[]
                writer=None
                if index in videos and start_time==starts[0]:
                    import imageio.v2 as imageio
                    writer=imageio.get_writer(a.output/(key+'.mp4'),fps=25,codec='libx264')
                horizon=(motion.motion.time_step_totals[index].item()-1)/50+1.1
                termination='evaluation_horizon';failure=False
                with torch.inference_mode():
                    for step in range(1,int(horizon/env.step_dt)+2):
                        goal=tuple(v[0].detach().cpu().numpy().copy() for v in env._impact_targets.current())
                        goals.append(np.concatenate((goal[0],goal[1],goal[2],[goal[3],goal[4]])))
                        action=policy(obs);obs,_,done,_=wrapped.step(action)
                        terminal=bool(done[0])
                        captured=env._evaluation_terminal if terminal else snapshot(env)
                        for field in ('ref_joint_pos','ref_body_pos','ref_body_quat','motion_time_s'):
                            captured[field]=target[field]
                        row=dict(clip=index,start_time_s=start_time,sim_time_s=step*env.step_dt,
                            motion_time_s=target['motion_time_s'],terminal=terminal,
                            **measurements(model,data,captured,motion.cfg.body_names),
                            **cpu_racket_metrics(model,data,target,motion.cfg.body_names))
                        site=next(i for i in range(model.nsite) if model.site(i).name.split('/')[-1]=='racket_point')
                        geom=next(i for i in range(model.ngeom) if model.geom(i).name.split('/')[-1]=='tennis_racket_collision')
                        velocity=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_SITE,site,velocity,0)
                        row.update(hit_window=bool(goal[4]),time_to_impact_s=float(goal[3]),
                            target_position_error_m=float(np.linalg.norm(data.site_xpos[site]-goal[0])),
                            target_velocity_error_m_s=float(np.linalg.norm(velocity[3:]-goal[1])),
                            target_plane_error_deg=float(np.rad2deg(np.arccos(np.clip(abs(np.dot(data.geom_xmat[geom].reshape(3,3)[:,2],goal[2])),0,1)))))
                        rows.append(row);qpos.append(captured['qpos']);qvel.append(captured['qvel'])
                        actions.append(action[0].cpu().numpy());refq.append(target['ref_joint_pos']);frame_times.append(target['motion_time_s'])
                        for name in ('ctrl','actuator_force','qfrc_actuator'):
                            if name in captured:signals.setdefault(name,[]).append(captured[name])
                        if writer is not None and (step%2==0 or terminal):
                            camera=mujoco.MjvCamera();camera.distance=3;camera.azimuth=125;camera.elevation=-15
                            camera.lookat[:]=data.qpos[:3];renderer.update_scene(data,camera=camera)
                            writer.append_data(renderer.render())
                        if terminal:
                            termination=captured['termination_reason'];failure=captured['failure'];break
                        target=capture_reference_velocity(env,snapshot(env))
                if writer is not None:writer.close()
                numerical=[k for k,v in rows[0].items() if isinstance(v,(float,np.floating)) and k not in ('sim_time_s','motion_time_s','start_time_s')]
                episode=dict(clip=index,clip_id=metadata[index]['id'],stroke_type=metadata[index].get('stroke_type','unknown'),
                    start_time_s=start_time,completed=('time_out' in termination and not failure),failure=failure,
                    termination_reason=termination,reached_s=len(rows)*env.step_dt,reference_duration_s=horizon-1.1,
                    metrics_mean={k:float(np.mean([r[k] for r in rows])) for k in numerical},samples=len(rows))
                hit_rows=[r for r in rows if r['hit_window']]
                episode['hit_samples']=len(hit_rows)
                episode['hit_target_metrics_mean']={k:float(np.mean([r[k] for r in hit_rows])) for k in ('target_position_error_m','target_velocity_error_m_s','target_plane_error_deg')} if hit_rows else None
                episode['randomized_goals']=a.randomized_goals
                episodes.append(episode);all_rows.extend(rows)
                # Store actual physical states separately from the reference;
                # successful complete trajectories can later be selected for MVAE.
                states[key+'_qpos']=np.asarray(qpos);states[key+'_qvel']=np.asarray(qvel)
                states[key+'_action']=np.asarray(actions);states[key+'_reference_joint_pos']=np.asarray(refq)
                states[key+'_reference_time_s']=np.asarray(frame_times)
                states[key+'_applied_impact_goal_pos_vel_normal_time_hit']=np.asarray(goals)
                for name,value in signals.items():states[key+'_'+name]=np.asarray(value)
                with (a.output/'episodes.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(episode)+'\n')
                write_json(a.output/'progress.json',dict(evaluated=len(episodes),requested=len(indices)*len(starts),last=episode))
                print(json.dumps(episode),flush=True)
    finally:
        if renderer is not None:renderer.close()
        wrapped.close()
    np.savez_compressed(a.output/'simulated_states.npz',**states)
    if all_rows:
        with (a.output/'tracking.csv').open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    write_json(a.output/'summary.json',dict(evaluated=len(episodes),requested=len(indices)*len(starts),
        completed=sum(e['completed'] for e in episodes),episodes=episodes,
        checkpoint_sha256=sha256(a.checkpoint),wall_s=time.monotonic()-started,
        metrics='Post-action fresh CPU FK, including terminal pre-reset. Mean errors are not full-prefix matched comparisons.',
        starts=starts,seed=100,training_performed=False))


if __name__=='__main__':main()

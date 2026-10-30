"""Evaluation-only independent worlds; validate against serial before full use.

No training, altered terminations, or simulated-state substitutions. Nominal
goals and disabled randomization are required so vectorization preserves the
per-condition experiment. Terminal states are captured before automatic reset.
"""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import time
import types

import numpy as np
import torch
import mujoco


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','checkpoint','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--indices',default='all');p.add_argument('--start-times',default='0,0.04')
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--max-wall-seconds',type=int,default=1800)
    p.add_argument('--reward-variant',required=True);a=p.parse_args()
    if not 1<=a.batch_size<=128:raise ValueError('Explicit evaluation batch1..128 required')
    from windows_mujoco_paths import patch_adapt_paths
    patch_adapt_paths()
    from impact_stage1.task import configure,install_reset_hook,GoalConfig
    from impact_stage1.reward_variants import apply_reward_variant
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.recorder_manager import RecorderTerm,RecorderTermCfg
    from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
    from tracking_experiment import seed_everything,write_json,measurements,sha256,check_motion_layout
    from racket_tracking_rewards import cpu_racket_metrics
    seed_everything(100)
    metadata=json.loads(a.manifest.read_text(encoding='utf-8'))['motions']
    indices=list(range(len(metadata))) if a.indices=='all' else [int(x) for x in a.indices.split(',')]
    starts=[float(x) for x in a.start_times.split(',')]
    cases=[(i,t) for i in indices for t in starts]
    # Nominal independent worlds: duration buckets avoid simulating many
    # already-ended short motions while one long motion remains active.
    cases.sort(key=lambda c:(metadata[c[0]]['frames']-round(c[1]*50),c[0],c[1]))
    cfg,agent,paths=configure(a.manifest,a.batch_size,100,GoalConfig(randomized=False))
    apply_reward_variant(cfg,a.manifest,GoalConfig(randomized=False),a.reward_variant)
    if cfg.events or any(t.noise is not None for t in cfg.observations['actor'].terms.values()):
        raise ValueError('Parallel protocol requires deterministic nominal observations and physics')
    cfg.commands['motion'].sampling_mode='start'
    a.output.mkdir(parents=True,exist_ok=False)

    def physical(env):
        c=env.command_manager.get_term('motion')
        result={name:getattr(env.sim.data,name)[:].detach().cpu().numpy().copy()
                for name in ['qpos','qvel','ctrl','actuator_force','qfrc_actuator']}
        result['joint_pos']=c.robot_joint_pos.detach().cpu().numpy().copy()
        return result

    class Capture(RecorderTerm):
        def record_pre_reset(self,ids):
            env=self._env
            if not getattr(env,'_parallel_eval_ready',False):return
            x=physical(env)
            failure=env.termination_manager.terminated.detach().cpu().numpy().copy()
            reasons={name:env.termination_manager.get_term(name).detach().cpu().numpy().copy()
                     for name in env.termination_manager.active_terms}
            for i in ids.tolist():
                env._parallel_terminals[i]=dict(**{k:v[i] for k,v in x.items()},
                    failure=bool(failure[i]),termination_reason=','.join(k for k,v in reasons.items() if v[i]))

    cfg.recorders['parallel_terminal']=RecorderTermCfg(func=Capture)
    begun=time.monotonic();env=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        write_json(a.output/'layouts.json',[check_motion_layout(env,path,i) for i,path in enumerate(paths)])
        install_reset_hook(env);wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
        runner=MjlabOnPolicyRunner(wrapped,asdict(agent),device='cuda:0')
        saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        if saved['infos']['impact_scope'].get('reward_variant','baseline')!=a.reward_variant:
            raise ValueError('Checkpoint variant mismatch')
        del saved
        runner.load(str(a.checkpoint.resolve()),load_cfg={'actor':True},map_location='cuda:0')
        policy=runner.get_inference_policy('cuda:0');motion=env.command_manager.get_term('motion')
        original=motion._resample_command;assignment=[]
        def reset(self,ids):
            original(ids)
            mid=torch.tensor([c[0] for c in assignment],device=env.device)
            frames=torch.tensor([round(c[1]*50) for c in assignment],device=env.device)
            self.motion_ids[ids]=mid[ids]
            for frame in frames[ids].unique().tolist():
                group=ids[frames[ids]==frame];self.reset_to_frame(group,int(frame))
        motion._resample_command=types.MethodType(reset,motion)
        model=env.sim.mj_model;data=mujoco.MjData(model)
        origins=env.scene.env_origins.detach().cpu().numpy().copy()
        site=next(i for i in range(model.nsite) if model.site(i).name.split('/')[-1]=='racket_point')
        geom=next(i for i in range(model.ngeom) if model.geom(i).name.split('/')[-1]=='tennis_racket_collision')
        episodes=[];all_rows=[];states={}
        for offset in range(0,len(cases),a.batch_size):
            if time.monotonic()-begun>a.max_wall_seconds-60:break
            batch=cases[offset:offset+a.batch_size];assignment[:]=batch+[batch[0]]*(a.batch_size-len(batch))
            env._parallel_eval_ready=False
            obs,_=wrapped.reset();env._parallel_terminals={};env._parallel_eval_ready=True
            physical_reset=physical(env)
            for j,(index,start) in enumerate(batch):
                expected=motion.motion.motions[index].joint_pos[round(start*50)].detach().cpu().numpy()
                error=float(np.abs(physical_reset['joint_pos'][j]-expected).max())
                if error>1e-4:raise ValueError('Parallel reset mismatch')
            alive=np.zeros(a.batch_size,dtype=bool);alive[:len(batch)]=True
            samples=[[] for _ in batch];reasons={}
            horizon=max((motion.motion.time_step_totals[i].item()-1)/50+1.1 for i,_ in batch)
            for step in range(1,int(horizon/env.step_dt)+2):
                refs={name:value.detach().cpu().numpy().copy() for name,value in {
                    'ref_joint_pos':motion.joint_pos,'ref_body_pos':motion.body_pos_w,
                    'ref_body_quat':motion.body_quat_w,'ref_body_lin_vel':motion.body_lin_vel_w,
                    'ref_body_ang_vel':motion.body_ang_vel_w,'motion_time_s':motion.current_time_seconds()}.items()}
                goal=[v.detach().cpu().numpy().copy() for v in env._impact_targets.current()]
                action=policy(obs);applied=action.detach().cpu().numpy().copy()
                obs,_,done,_=wrapped.step(action);post=physical(env);ended=done.detach().cpu().numpy().astype(bool)
                for j in np.flatnonzero(alive):
                    term=env._parallel_terminals.get(j) if ended[j] else None
                    actual=dict(term or {k:v[j].copy() for k,v in post.items()})
                    actual.update({k:v[j].copy() for k,v in refs.items()})
                    # Canonical single-world coordinates for CPU FK and comparison.
                    actual['qpos'][:3]-=origins[j];actual['ref_body_pos']-=origins[j]
                    g=np.concatenate((goal[0][j]-origins[j],goal[1][j],goal[2][j],[goal[3][j],goal[4][j]]))
                    samples[j].append((actual,applied[j],g))
                    if ended[j]:reasons[j]=(actual['termination_reason'],actual['failure']);alive[j]=False
                if not alive.any():break
            env._parallel_eval_ready=False
            for j,(index,start) in enumerate(batch):
                key=f'clip{index:04d}_start{round(start*1000):04d}';rows=[];buffers={}
                for n,(s,action,goal) in enumerate(samples[j],1):
                    terminal=(j in reasons and n==len(samples[j]))
                    row=dict(clip=index,start_time_s=start,sim_time_s=n*env.step_dt,
                        motion_time_s=float(s['motion_time_s']),terminal=terminal,
                        **measurements(model,data,s,motion.cfg.body_names),
                        **cpu_racket_metrics(model,data,s,motion.cfg.body_names))
                    velocity=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_SITE,site,velocity,0)
                    row.update(hit_window=bool(goal[10]),time_to_impact_s=float(goal[9]),
                        target_position_error_m=float(np.linalg.norm(data.site_xpos[site]-goal[:3])),
                        target_velocity_error_m_s=float(np.linalg.norm(velocity[3:]-goal[3:6])),
                        target_plane_error_deg=float(np.rad2deg(np.arccos(np.clip(abs(np.dot(data.geom_xmat[geom].reshape(3,3)[:,2],goal[6:9])),0,1)))))
                    rows.append(row)
                    for name,val in dict(qpos=s['qpos'],qvel=s['qvel'],action=action,reference_joint_pos=s['ref_joint_pos'],
                        reference_time_s=s['motion_time_s'],applied_impact_goal_pos_vel_normal_time_hit=goal,
                        ctrl=s['ctrl'],actuator_force=s['actuator_force'],qfrc_actuator=s['qfrc_actuator']).items():
                        buffers.setdefault(name,[]).append(val)
                termination,failure=reasons.get(j,('evaluation_horizon',False));hits=[r for r in rows if r['hit_window']]
                numerical=[k for k,v in rows[0].items() if isinstance(v,(float,np.floating)) and k not in ('sim_time_s','motion_time_s','start_time_s')]
                ep=dict(clip=index,clip_id=metadata[index]['id'],stroke_type=metadata[index].get('stroke_type','unknown'),
                    start_time_s=start,completed=('time_out' in termination and not failure),failure=failure,
                    termination_reason=termination,reached_s=len(rows)*env.step_dt,
                    reference_duration_s=(motion.motion.time_step_totals[index].item()-1)/50,
                    metrics_mean={k:float(np.mean([r[k] for r in rows])) for k in numerical},samples=len(rows),hit_samples=len(hits),
                    hit_target_metrics_mean={k:float(np.mean([r[k] for r in hits])) for k in ('target_position_error_m','target_velocity_error_m_s','target_plane_error_deg')} if hits else None,
                    randomized_goals=False)
                episodes.append(ep);all_rows.extend(rows)
                states.update({key+'_'+name:np.asarray(vals) for name,vals in buffers.items()})
                with (a.output/'episodes.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(ep)+'\n')
            write_json(a.output/'progress.json',dict(evaluated=len(episodes),requested=len(cases),batch_size=a.batch_size))
            print(json.dumps(dict(evaluated=len(episodes),requested=len(cases),wall_s=time.monotonic()-begun)),flush=True)
    finally:env.close()
    np.savez_compressed(a.output/'simulated_states.npz',**states)
    with (a.output/'tracking.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    write_json(a.output/'summary.json',dict(evaluated=len(episodes),requested=len(cases),
        completed=sum(e['completed'] for e in episodes),episodes=episodes,checkpoint_sha256=sha256(a.checkpoint),
        wall_s=time.monotonic()-begun,starts=starts,seed=100,training_performed=False,batch_size=a.batch_size,
        runtime=mujoco.__version__,metrics='Fresh post-action CPU FK and pre-reset terminal capture; vectorized nominal protocol; requires serial comparison before scientific use.'))

if __name__=='__main__':main()

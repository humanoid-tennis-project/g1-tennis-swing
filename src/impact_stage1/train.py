"""Bounded Stage 1 training using pinned RSL-RL; explicit execution only."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import statistics
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--impact-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--updates', type=int, default=2)
    p.add_argument('--seconds', type=float, default=120)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--randomized-goals', action='store_true')
    p.add_argument('--resume', type=Path)
    p.add_argument('--snapshot-seconds', type=float, default=600)
    p.add_argument('--monitor-eval-indices', default='')
    from .reward_variants import VARIANTS
    p.add_argument('--reward-variant',choices=VARIANTS,default='baseline')
    p.add_argument('--fork-parent-scope',action='store_true',
                   help='Explicitly branch the validated original task into a reward experiment.')
    p.add_argument('--runtime-bridge-evidence',type=Path,
                   help='Explicit same-checkpoint cloud/local physical comparison; local-only transition.')
    p.add_argument('--cross-runtime-continuation-proof',type=Path,
                   help='Separate tested runtime-transfer experiment; final cloud validation required.')
    p.add_argument('--foot-lock-data-fork',action='store_true',
                   help='Branch onto stance-foot-locked copies of exactly the parent references.')
    p.add_argument('--skill-group', choices=('rally', 'serve'), default='rally',
                   help='Use a separately prepared manifest; mixed serve/rally training is forbidden.')
    a = p.parse_args()
    if a.runtime_bridge_evidence and a.cross_runtime_continuation_proof:
        raise ValueError('Choose strict comparison or cross-runtime continuation, not both')
    if not (1 <= a.num_envs <= 4096 and 1 <= a.updates <= 300000 and 0 < a.seconds <= 18000):
        raise ValueError('Invalid explicit execution budget')
    data = json.loads(a.impact_manifest.read_text(encoding='utf-8'))
    motions = data.get('motions', [])
    if not motions or any((row['stroke_type'] == 'serve') != (a.skill_group == 'serve')
                          for row in motions):
        raise ValueError('Manifest must contain only the selected skill group; prepare separate rally/serve manifests')
    from windows_mujoco_paths import patch_adapt_paths
    patch_adapt_paths()
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from .task import configure, install_reset_hook, GoalConfig
    from .data import sha
    from tracking_experiment import seed_everything
    seed_everything(7)
    cfg, agent, paths = configure(a.impact_manifest, a.num_envs, 7, GoalConfig(randomized=a.randomized_goals))
    from .reward_variants import apply_reward_variant
    apply_reward_variant(cfg,a.impact_manifest,GoalConfig(randomized=a.randomized_goals),a.reward_variant)
    a.output.mkdir(parents=True, exist_ok=False)
    from tracking_experiment import write_json
    write_json(a.output/'configuration.json', dict(env=asdict(cfg), agent=asdict(agent),
        manifest=str(a.impact_manifest.resolve()), manifest_sha256=sha(a.impact_manifest),
        source='Independent paper-inspired implementation, not released author code',
        includes_ball=False, includes_upper_policy=False, skill_group=a.skill_group,
        randomized_goals=a.randomized_goals, resume=str(a.resume) if a.resume else None,
        reward_variant=a.reward_variant,explicit_parent_fork=a.fork_parent_scope))
    env = ManagerBasedRlEnv(cfg, device=a.device)
    try:
        # Match the established tracking entry point before the first reset:
        # newer mujoco-warp exposes shared joint limits with one world row.
        from tracking_experiment import check_motion_layout
        write_json(a.output/'layouts.json',
                   [check_motion_layout(env, path, i) for i, path in enumerate(paths)])
        install_reset_hook(env)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        from standard_joint_ppo import install_exposure, rollout_diagnostics
        install_exposure(wrapped)
        obs, _ = wrapped.reset()
        if obs['actor'].shape[-1] != 160 or obs['critic'].shape[-1] != 286:
            raise ValueError('Unexpected actual observation dimensions')
        runner = MjlabOnPolicyRunner(wrapped, asdict(agent), str(a.output/'tensorboard'), a.device)
        from standard_joint_ppo import verify_and_accelerate
        verify_and_accelerate(runner.alg.get_policy(), obs)
        verify_and_accelerate(runner.alg.critic, obs)
        # Paths differ across accounts; identity is the immutable data, goals,
        # observation/task architecture, and simulator version, not local paths.
        import mujoco
        scope = dict(format='impact_stage1_resume_v1',
            motions=[(r['id'], r['motion_sha256'], r['annotation_sha256']) for r in motions],
            goal_config=asdict(GoalConfig(randomized=a.randomized_goals)),
            skill_group=a.skill_group, num_envs=a.num_envs, mujoco=mujoco.__version__,
            implementation={name:sha(Path(__file__).with_name(name+'.py'))
                            for name in ('networks','policy','task')},
            actor_dims=[160,8,95,27], critic_dim=286)
        # JSON normalization makes tuples/lists invariant across serialization.
        scope = json.loads(json.dumps(scope))
        parent_scope = scope
        scope = dict(parent_scope,reward_variant=a.reward_variant,
                     reward_variant_sha256=sha(Path(__file__).with_name('reward_variants.py')))
        previous = 0
        if a.resume:
            source = torch.load(a.resume, map_location=a.device, weights_only=False)
            source_scope=source.get('infos', {}).get('impact_scope')
            from .scope_guard import check_reward_fork,check_local_runtime_bridge,check_cross_runtime_continuation
            allow_path=Path(__file__).with_name('approved_reward_forks.json')
            approved=json.loads(allow_path.read_text(encoding='utf-8')) if allow_path.exists() else {}
            if a.cross_runtime_continuation_proof:
                proof=json.loads(a.cross_runtime_continuation_proof.read_text(encoding='utf-8'))
                kind=check_cross_runtime_continuation(source_scope,parent_scope,scope,sha(a.resume),proof,a.fork_parent_scope,approved)
            elif a.runtime_bridge_evidence:
                proof=json.loads(a.runtime_bridge_evidence.read_text(encoding='utf-8'))
                kind=check_local_runtime_bridge(source_scope,parent_scope,scope,sha(a.resume),proof,a.fork_parent_scope,approved)
            elif a.foot_lock_data_fork:
                from .scope_guard import check_foot_lock_data_fork
                kind=check_foot_lock_data_fork(source_scope,parent_scope,scope,data,a.fork_parent_scope,approved)
            else:
                kind=check_reward_fork(source_scope,parent_scope,scope,a.fork_parent_scope,approved)
            exact_resume=kind=='exact'
            allowed_fork=kind in ('fork','runtime_fork','runtime_continuation','data_fork')
            if allowed_fork:
                write_json(a.output/'parent_fork.json',dict(parent_sha256=sha(a.resume),
                    parent_scope=source_scope,new_scope=scope,
                    reason={'runtime_continuation':'Reviewed cross-runtime continuation',
                            'data_fork':'Foot-locked references derived from the parent motions'}.get(kind,'Controlled reward-only experiment'),
                    parent_checkpoint_modified=False,resume_kind=kind,
                    runtime_bridge_evidence_sha256=sha(a.runtime_bridge_evidence or a.cross_runtime_continuation_proof) if (a.runtime_bridge_evidence or a.cross_runtime_continuation_proof) else None))
            runner.load(str(a.resume.resolve()), map_location=a.device)
            if 'adaptive_v4' in a.reward_variant:
                from .adaptive_kernel import ensure_kernel
                kernel=ensure_kernel(env)
                if 'adaptive_v4' in source_scope.get('reward_variant',''):
                    kernel.load_state_dict(source['infos']['impact_adaptive_kernel'])
            previous = runner.current_learning_iteration+1
            runner.current_learning_iteration = previous
            for model, key in ((runner.alg.get_policy(), 'actor_state_dict'),
                               (runner.alg.critic, 'critic_state_dict')):
                if not all(torch.equal(v, source[key][k]) for k,v in model.state_dict().items()):
                    raise AssertionError('Resume changed learned model/normalizer')
            from resume_learning_rate import restore_algorithm_learning_rate
            lr = restore_algorithm_learning_rate(runner.alg)
            write_json(a.output/'resume.json', dict(source=str(a.resume.resolve()), sha256=sha(a.resume),
                previous_updates=previous, exact_model_restore=True, learning_rate=lr,
                physics_and_rollout_restored=False,reward_variant=a.reward_variant,
                parent_scope_fork=bool(allowed_fork)))
            del source
        from shared_ppo_monitoring import attach_monitoring
        attach_monitoring(runner, wrapped, a.output, a.resume)
        monitored_save = runner.save
        def save(path, infos=None):
            extra=dict(infos or {},impact_scope=scope)
            if 'adaptive_v4' in a.reward_variant:
                from .adaptive_kernel import ensure_kernel
                extra['impact_adaptive_kernel']=ensure_kernel(env).state_dict()
            monitored_save(path,extra)
        runner.save = save
        runner.save(str(a.output/'initial.pt'))
        start = time.monotonic()
        rows = []
        diagnostics = {}
        original_update = runner.alg.update
        def update():
            record_diagnostic=len(rows)%50 == 0
            if record_diagnostic:
                diagnostics.update(rollout_diagnostics(runner.alg))
            losses = original_update()
            if record_diagnostic:
                post=rollout_diagnostics(runner.alg)
                diagnostics.update({f'post_update_{k}':post[k] for k in ('kl','clip_fraction')})
                diagnostics['diagnostic_at_additional_update']=len(rows)+1
                if hasattr(env,'_impact_adaptive_kernel'):
                    widths=env._impact_adaptive_kernel.sigma.sqrt().detach().cpu().tolist()
                    diagnostics.update(dict(zip(('kernel_position_width_m','kernel_velocity_width_m_s','kernel_plane_width_rad'),widths)))
            if not all(torch.isfinite(torch.as_tensor(v)).all() for v in losses.values()):
                raise FloatingPointError('Nonfinite PPO loss')
            return losses
        runner.alg.update = update
        original_log = runner.logger.log
        next_snapshot = a.snapshot_seconds
        class BudgetStop(Exception):
            pass
        def log(**kw):
            nonlocal next_snapshot
            original_log(**kw, print_minimal=True)
            iteration = len(rows)
            row = dict(update=kw['it']+1, additional_updates=iteration+1, wall_s=time.monotonic()-start,
                reward_variant=a.reward_variant,
                transitions=(kw['it']+1)*a.num_envs*agent.num_steps_per_env,
                collection_s=kw['collect_time'], learning_s=kw['learn_time'],
                learning_rate=kw['learning_rate'], action_std=float(kw['action_std'].mean()),
                episode_reward_mean=statistics.mean(runner.logger.rewbuffer) if runner.logger.rewbuffer else None,
                episode_length_mean=statistics.mean(runner.logger.lenbuffer) if runner.logger.lenbuffer else None,
                **kw['loss_dict'], **diagnostics)
            rows.append(row)
            with (a.output/'progress.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(row)+'\n')
            if row['wall_s'] >= next_snapshot:
                target = a.output/f'wall_{int(next_snapshot):05d}.pt'
                runner.save(str(target))
                if a.monitor_eval_indices and int(next_snapshot)%3600 == 0 and a.seconds-row['wall_s'] > 360:
                    evaluation = a.output/f'monitor_eval_{int(next_snapshot):05d}'
                    command = [sys.executable,'-m','impact_stage1.evaluate','--manifest',str(a.impact_manifest),
                        '--checkpoint',str(target),'--output',str(evaluation),'--indices',a.monitor_eval_indices,
                        '--video-indices',a.monitor_eval_indices,'--max-wall-seconds','300']
                    with (a.output/(evaluation.name+'.log')).open('x',encoding='utf-8') as f:
                        try:
                            code = subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=330).returncode
                            write_json(a.output/(evaluation.name+'_status.json'),dict(returncode=code))
                        except subprocess.TimeoutExpired:
                            write_json(a.output/(evaluation.name+'_status.json'),dict(status='timeout'))
                next_snapshot += a.snapshot_seconds
            if time.monotonic()-start >= a.seconds:
                raise BudgetStop()
        runner.logger.log = log
        try:
            runner.learn(a.updates, init_at_random_ep_len=False)
        except BudgetStop:
            pass
        finally:
            runner.save(str(a.output/'final.pt'))
            runner.logger.stop_logging_writer()
        write_json(a.output/'summary.json', dict(updates=len(rows), total_updates=previous+len(rows),
            reward_variant=a.reward_variant,
            wall_s=time.monotonic()-start, motion_count=len(paths),
            median_hot_update_s=statistics.median([r['collection_s']+r['learning_s'] for r in rows[2:]]) if len(rows)>2 else None,
            clip_steps=wrapped.clip_steps.tolist(), phase_steps=wrapped.phase_steps.tolist(),
            trained_not_validated=True, tracking_success_not_claimed=True,
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None))
    finally:
        env.close()


if __name__ == '__main__':
    main()

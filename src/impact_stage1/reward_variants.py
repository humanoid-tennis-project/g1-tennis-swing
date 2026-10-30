"""Explicit, auditable reward-only experiments; the original task is unchanged."""
from .task import outside_hit_reward
import math

VARIANTS = ('baseline', 'joint_pose_v1', 'swept_v2', 'joint_pose_swept_v2', 'joint_pose_ee_v3', 'joint_pose_swept_ee_v3', 'joint_pose_adaptive_v4', 'joint_pose_ee_adaptive_v4', 'joint_pose_hitbalanced_v5', 'joint_pose_ee_hitbalanced_v5', 'joint_pose_ee_hitbalanced_adaptive_v4', 'joint_pose_ee_hitbalanced_nonracket_adaptive_v4', 'joint_pose_ee_hitbalanced_nonracket_rootvel_adaptive_v4')


def apply_reward_variant(cfg, manifest, settings, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown reward experiment: '+variant)
    if variant == 'baseline':
        return
    # H8 inherits H6B exactly, then adds the root linear velocity tracking term
    # listed in badminton Stage1 App. A.2. The existing 14-body velocity term
    # (std1.0) gives the pelvis 1/14 share; std.5 m/s is our engineering choice.
    if variant == 'joint_pose_ee_hitbalanced_nonracket_rootvel_adaptive_v4':
        apply_reward_variant(cfg,manifest,settings,'joint_pose_ee_hitbalanced_nonracket_adaptive_v4')
        from mjlab.managers.reward_manager import RewardTermCfg
        from mjlab.tasks.adapt_tennis import mdp
        cfg.rewards['motion_root_lin_vel']=RewardTermCfg(func=outside_hit_reward,weight=1.,
            params=dict(original_func=mdp.motion_global_body_linear_velocity_error_exp,
                original_params=dict(command_name='motion',std=.5,body_names=('pelvis',)),
                manifest=str(manifest),settings=settings))
        return
    # H6 inherits H5 exactly, then adds one shared non-racket-arm pose term.
    if variant == 'joint_pose_ee_hitbalanced_nonracket_adaptive_v4':
        apply_reward_variant(cfg,manifest,settings,'joint_pose_ee_hitbalanced_adaptive_v4')
        from mjlab.managers.reward_manager import RewardTermCfg
        from mjlab.tasks.adapt_tennis import mdp
        cfg.rewards['motion_nonracket_arm_pose']=RewardTermCfg(func=outside_hit_reward,weight=1.,
            params=dict(original_func=mdp.motion_joint_position_error_exp,
                original_params=dict(command_name='motion',std=.4*math.sqrt(4),joint_names=(
                    'left_shoulder_pitch_joint','left_shoulder_roll_joint',
                    'left_shoulder_yaw_joint','left_elbow_joint')),
                manifest=str(manifest),settings=settings))
        return
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.tasks.adapt_tennis import mdp
    from mjlab.tasks.adapt_tennis.replay_joint_names import G1_REPLAY_JOINT_NAMES_27
    # Restore the joint-position tracking component explicitly described in
    # badminton Appendix A.2, using the existing pinned OSS implementation.
    # Weight/std are our controlled trial settings, not claimed author values.
    if variant in ('joint_pose_v1', 'joint_pose_swept_v2', 'joint_pose_ee_v3', 'joint_pose_swept_ee_v3', 'joint_pose_adaptive_v4', 'joint_pose_ee_adaptive_v4', 'joint_pose_hitbalanced_v5', 'joint_pose_ee_hitbalanced_v5', 'joint_pose_ee_hitbalanced_adaptive_v4'):
        cfg.rewards['motion_joint_pose'] = RewardTermCfg(
            func=outside_hit_reward, weight=1., params=dict(
                original_func=mdp.motion_joint_position_error_exp,
                # Upstream sums errors; sqrt(N) converts the per-joint RMS scale.
                original_params=dict(command_name='motion', std=.4*math.sqrt(27),
                                     joint_names=G1_REPLAY_JOINT_NAMES_27),
                manifest=str(manifest), settings=settings))
    if variant in ('swept_v2', 'joint_pose_swept_v2', 'joint_pose_swept_ee_v3'):
        cfg.rewards['impact_position']=RewardTermCfg(func=swept_position_reward,weight=1.,
            params=dict(manifest=str(manifest),settings=settings))
    if variant in ('joint_pose_ee_v3', 'joint_pose_swept_ee_v3', 'joint_pose_ee_adaptive_v4', 'joint_pose_ee_hitbalanced_v5', 'joint_pose_ee_hitbalanced_adaptive_v4'):
        # End-effector tracking is explicit in DeepMimic; use the existing
        # pinned mjlab position metric. Our scale/weight are engineering choices.
        cfg.rewards['motion_end_effector_position'] = RewardTermCfg(
            func=outside_hit_reward, weight=1., params=dict(
                original_func=mdp.motion_relative_body_position_error_exp,
                original_params=dict(command_name='motion', std=.15,
                    body_names=('left_ankle_roll_link','right_ankle_roll_link',
                                'left_wrist_yaw_link','right_wrist_yaw_link')),
                manifest=str(manifest), settings=settings))
    if variant in ('joint_pose_adaptive_v4','joint_pose_ee_adaptive_v4'):
        from .adaptive_kernel import adaptive_target_reward
        for component,name in enumerate(('position','velocity','plane')):
            cfg.rewards['impact_'+name]=RewardTermCfg(func=adaptive_target_reward,weight=1.,
                params=dict(component=component,manifest=str(manifest),settings=settings))
    if variant in ('joint_pose_hitbalanced_v5','joint_pose_ee_hitbalanced_v5'):
        from .hit_balance import balanced_hit_reward
        for component,name in enumerate(('position','velocity','plane')):
            cfg.rewards['impact_'+name]=RewardTermCfg(func=balanced_hit_reward,weight=1.,
                params=dict(component=component,manifest=str(manifest),settings=settings))
    if variant == 'joint_pose_ee_hitbalanced_adaptive_v4':
        from .hit_balance import balanced_adaptive_hit_reward
        for component,name in enumerate(('position','velocity','plane')):
            cfg.rewards['impact_'+name]=RewardTermCfg(func=balanced_adaptive_hit_reward,weight=1.,
                params=dict(component=component,manifest=str(manifest),settings=settings))


def phase_consistent_position_score(actual, target, velocity, remaining, hit, std=.15):
    import torch
    moving_target=target-velocity*remaining[:,None]
    return torch.exp(-((actual-moving_target)**2).sum(-1)/std**2)*hit.float()


def swept_position_reward(env, manifest, settings):
    from .task import state
    from racket_tracking_rewards import racket_quantities
    p,v,n,remaining,hit=state(env,manifest,settings).current()
    actual=racket_quantities(env)[0]
    return phase_consistent_position_score(actual,p,v,remaining,hit,settings.position_std_m)

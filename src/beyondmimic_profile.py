"""BeyondMimic III-B/D observation and reward adaptation, MIT wrapper.

Reuses pinned AdaPT/mjlab functions; not a clean upstream reproduction.
Robot, motor gains, reference loader and reset sampler remain pinned.
"""
import torch


def reference_phase(env, command_name):
    motion = env.command_manager.get_term(command_name)
    return torch.cat((motion.joint_pos, motion.joint_vel), dim=-1)


def masked_anchor_position(env, command_name, enabled):
    from mjlab.tasks.adapt_tennis import mdp
    value = mdp.motion_anchor_pos_b(env, command_name)
    return value if enabled else torch.zeros_like(value)


def masked_linear_velocity(env, sensor_name, enabled):
    from mjlab.tasks.adapt_tennis import mdp
    value = mdp.builtin_sensor(env, sensor_name=sensor_name)
    return value if enabled else torch.zeros_like(value)


def configure_beyondmimic(cfg, actor_inputs='both'):
    if actor_inputs not in ('both', 'position', 'velocity', 'neither'):
        raise ValueError(actor_inputs)
    from mjlab.managers.observation_manager import ObservationTermCfg, ObservationGroupCfg
    from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
    from mjlab.tasks.adapt_tennis import mdp

    def term(func, **params):
        return ObservationTermCfg(func=func, params=params)

    actor = {
        'command': term(reference_phase, command_name='motion'),
        'motion_anchor_pos_b': term(mdp.motion_anchor_pos_b, command_name='motion'),
        'motion_anchor_ori_b': term(mdp.motion_anchor_ori_b, command_name='motion'),
        'base_lin_vel': term(mdp.builtin_sensor, sensor_name='robot/imu_lin_vel'),
        'base_ang_vel': term(mdp.builtin_sensor, sensor_name='robot/imu_ang_vel'),
        'joint_pos': term(mdp.joint_pos_rel),
        'joint_vel': term(mdp.joint_vel_rel),
        'actions': term(mdp.last_action),
    }
    critic = dict(actor)
    critic['body_pos'] = term(mdp.robot_body_pos_b, command_name='motion')
    critic['body_ori'] = term(mdp.robot_body_ori_b, command_name='motion')
    # Replace Actor terms after copying: Critic retains both physical signals.
    actor['motion_anchor_pos_b'] = term(masked_anchor_position, command_name='motion',
                                      enabled=actor_inputs in ('both', 'position'))
    actor['base_lin_vel'] = term(masked_linear_velocity, sensor_name='robot/imu_lin_vel',
                               enabled=actor_inputs in ('both', 'velocity'))
    cfg.observations = {
        'actor': ObservationGroupCfg(terms=actor, concatenate_terms=True, enable_corruption=False),
        'critic': ObservationGroupCfg(terms=critic, concatenate_terms=True, enable_corruption=False),
    }
    # The inherited task already uses nominal-centered, per-motor scaled PD.
    # Remove tennis-only wrist and keyframe rewards. Use pinned general tracker.
    cfg.rewards = make_tracking_env_cfg().rewards
    cfg.commands['motion'].dynamic_dt_enabled = False
    cfg.commands['motion'].random_dt_training_enabled = False
    cfg.scale_rewards_by_motion_dt = False


def verify_profile(env, actor_inputs='both'):
    """Check actual observations, mapping and finite values before learning."""
    motion = env.command_manager.get_term('motion')
    phase = reference_phase(env, 'motion')
    obs = env.observation_manager.compute()
    action = env.action_manager.get_term('joint_pos')
    n = motion.joint_pos.shape[-1]
    assert obs['actor'].shape == (env.num_envs, 5*n+15)
    assert torch.equal(obs['actor'][:, :2*n], phase)
    assert all(torch.isfinite(value).all() for value in obs.values())
    for name, sl, enabled in [('position', slice(54,57), actor_inputs in ('both','position')),
                              ('velocity', slice(63,66), actor_inputs in ('both','velocity'))]:
        expected = obs['critic'][:,sl] if enabled else torch.zeros_like(obs['critic'][:,sl])
        assert torch.equal(obs['actor'][:,sl], expected), name
    untouched = torch.ones(150, dtype=torch.bool, device=env.device)
    untouched[54:57] = False
    untouched[63:66] = False
    assert torch.equal(obs['actor'][:,untouched], obs['critic'][:,:150][:,untouched])
    action.process_actions(torch.zeros((env.num_envs, n), device=env.device))
    error = float((action._processed_actions-action.offset).abs().max())
    assert error < 1e-7
    return dict(actor_input=obs['actor'].shape[-1], critic_input=obs['critic'].shape[-1],
                action_output=n, reference_phase_exact=True, all_finite=True,
                actor_inputs=actor_inputs, masked_signals_verified=True, critic_full_information=True,
                zero_action_nominal_pose_max_error=error,
                reward_names=list(env.cfg.rewards),
                note='Verification does not step physics or update policy')

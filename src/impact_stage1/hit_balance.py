"""Normalize event reward opportunity using the same rule for every motion.

T/H makes H ideal hit-frame scores integrate to T frame-scores. This is a
relative reward-weight choice, not an evaluation success relaxation. It does
not move the hit window or change targets, kernels, or reference trajectories.
"""
import torch


def hit_window_frames(row):
    n=int(row['frames'])
    return {f for e in row['events'] for f in range(max(0,e['frame']-2),min(n,e['frame']+3))}


def hit_duty_gain(env,manifest,settings):
    from .task import state
    if not hasattr(env,'_impact_hit_duty_gain'):
        rows=state(env,manifest,settings).rows
        values=[]
        for row in rows:
            h=len(hit_window_frames(row))
            if h<=0:raise ValueError('No valid annotated hit-window samples')
            values.append(int(row['frames'])/h)
        env._impact_hit_duty_gain=torch.tensor(values,device=env.device,dtype=torch.float32)
    ids=env.command_manager.get_term('motion').motion_ids
    return env._impact_hit_duty_gain[ids]


def balanced_hit_reward(env,component,manifest,settings):
    from .task import hit_target_reward
    return hit_target_reward(env,component,manifest,settings)*hit_duty_gain(env,manifest,settings)


def balanced_adaptive_hit_reward(env,component,manifest,settings):
    """Keep the event allocation while adapting only its error-kernel width."""
    from .adaptive_kernel import adaptive_target_reward
    return adaptive_target_reward(env,component,manifest,settings)*hit_duty_gain(env,manifest,settings)

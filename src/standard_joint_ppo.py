"""Fixed-task scalar PPO recipe, using the upstream RSL-RL runner (MIT).

The lecture's four-line learning entry is retained. Robot/reference/rewards
remain tennis-specific; this does not replace them with walking rewards.
"""
from pathlib import Path
import hashlib
import json
import types
import torch
from tracking_experiment import build_config, apply_tracking_profile, check_motion_layout
from beyondmimic_profile import configure_beyondmimic
from racket_tracking_rewards import configure_racket_rewards



def configure(paths,num_envs,seed):
    cfg,agent=build_config(paths[0],2.52,num_envs,seed)
    apply_tracking_profile(cfg,'video-nominal')
    configure_beyondmimic(cfg)
    configure_racket_rewards(cfg)
    cfg.commands['motion'].motion_files=tuple(str(p.resolve()) for p in paths)
    cfg.sim.nconmax,cfg.sim.njmax=128,512
    cfg.adadt_reward_groups=None
    agent.seed=seed;agent.algorithm.class_name='PPO'
    agent.logger='tensorboard';agent.upload_model=False
    agent.save_interval=1000
    return cfg,agent


def reference_audit(paths):
    import numpy as np
    rows=[]
    for i,path in enumerate(paths):
        with np.load(path,allow_pickle=False) as data:
            if not all(np.isfinite(data[k]).all() for k in
                       ('joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w')):
                raise ValueError(f'Nonfinite reference {path}')
            frames=len(data['joint_pos'])
        quality=json.loads(path.with_name('reference_quality.json').read_text(encoding='utf-8'))
        if quality['max_joint_limit_violation_rad']>1e-3: raise ValueError('Joint limit QA failed')
        penetration=quality.get('largest_self_penetration')
        rows.append(dict(clip=i,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            frames=frames,recorded_self_penetration=penetration,
            self_penetration_over_1cm=bool(penetration and penetration['penetration_m']>.01)))
    return rows


def install_exposure(wrapper):
    """Read-only bookkeeping at the applied clip/frame; leave PPO I/O intact."""
    env=wrapper.unwrapped;command=env.command_manager.get_term('motion')
    count=command.motion.num_motions
    wrapper.clip_steps=torch.zeros(count,dtype=torch.long,device=env.device)
    wrapper.clip_ends=torch.zeros_like(wrapper.clip_steps)
    wrapper.clip_timeouts=torch.zeros_like(wrapper.clip_steps)
    wrapper.phase_steps=torch.zeros((count,6),dtype=torch.long,device=env.device)
    original=wrapper.step
    def step(self,action):
        mids=command.motion_ids.clone()
        phase=(command.time_steps.float()/(command.motion.time_step_totals[mids]-1)*6).long().clamp(0,5)
        self.clip_steps+=torch.bincount(mids,minlength=count)
        self.phase_steps.view(-1).add_(torch.bincount(mids*6+phase,minlength=count*6))
        obs,reward,done,extras=original(action)
        self.clip_ends+=torch.bincount(mids[done.bool()],minlength=count)
        self.clip_timeouts+=torch.bincount(mids[env.reset_time_outs.bool()],minlength=count)
        return obs,reward,done,extras
    wrapper.step=types.MethodType(step,wrapper)


@torch.no_grad()
def rollout_diagnostics(algorithm,max_samples=4096):
    st=algorithm.storage
    obs=st.observations.flatten(0,1);action=st.actions.flatten(0,1)
    ids=torch.linspace(0,len(action)-1,min(max_samples,len(action)),device=algorithm.device).long()
    old=tuple(p.flatten(0,1)[ids] for p in st.distribution_params)
    actor=algorithm.get_policy()
    actor.distribution.update(actor(obs[ids],stochastic_output=False))
    kl=actor.distribution.kl_divergence(old,actor.distribution.params)
    ratio=(actor.distribution.log_prob(action[ids])-st.actions_log_prob.flatten(0,1)[ids].squeeze(-1)).exp()
    values,returns=st.values.flatten(),st.returns.flatten()
    variance=returns.var(unbiased=False)
    return dict(kl=float(kl.mean()),clip_fraction=float(((ratio-1).abs()>algorithm.clip_param).float().mean()),
                explained_variance=float(1-(returns-values).var(unbiased=False)/variance.clamp_min(1e-9)),
                sample_count=len(ids))


def stock_latent_fast(self,obs,masks=None,hidden_state=None):
    """Exact MLP latent math, omitting unused debug argmax/min GPU syncs."""
    return self.obs_normalizer(torch.cat([obs[key] for key in self.obs_groups],dim=-1))


def verify_and_accelerate(model,obs):
    with torch.no_grad():
        expected=model.get_latent(obs)
        actual=stock_latent_fast(model,obs)
        error=float((expected-actual).abs().max())
    if error!=0.: raise AssertionError('Latent acceleration changes numerical output')
    model.get_latent=types.MethodType(stock_latent_fast,model)
    return error

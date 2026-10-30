"""Draft: error-scale curriculum for hit targets; not attached to a live run.

The non-increasing error-EMA idea follows KungfuBot section3.2. This is our
own implementation, with explicit lower bounds equal to the current kernel.
It changes reward sensitivity, never the sampled goals or evaluation errors.
"""
import math
import torch


class AdaptiveTargetKernel:
    def __init__(self,device='cpu',step_dt=.02,tau_seconds=20.,
                 initial=(.45,4.,.7),minimum=(.15,2.,.35)):
        if step_dt<=0 or tau_seconds<=0 or any(a<b or b<=0 for a,b in zip(initial,minimum)):
            raise ValueError('Positive ordered widths and time constants required')
        self.initial=tuple(initial);self.minimum=tuple(minimum)
        self.step_dt=step_dt;self.tau_seconds=tau_seconds
        self.alpha=1-math.exp(-step_dt/tau_seconds)
        self.sigma=torch.tensor(initial,device=device,dtype=torch.float32).square()
        self.floor=torch.tensor(minimum,device=device,dtype=torch.float32).square()
        self.ema=self.sigma.clone();self.last_step=None;self.calls=0

    def scores(self,squared_errors,hit,step):
        if squared_errors.ndim!=2 or squared_errors.shape[1]!=3:
            raise ValueError('Expected [environments,position/velocity/plane] squared errors')
        if tuple(hit.shape)!=(len(squared_errors),):raise ValueError('Hit mask shape mismatch')
        mask=hit.to(dtype=squared_errors.dtype);count=mask.sum()
        if step!=self.last_step:
            error=(squared_errors.detach()*mask[:,None]).sum(0)/count.clamp_min(1)
            candidate=(1-self.alpha)*self.ema+self.alpha*error
            self.ema=torch.where(count>0,candidate,self.ema)
            self.sigma=torch.minimum(self.sigma,torch.maximum(self.floor,self.ema))
            self.last_step=step;self.calls+=1
        return torch.exp(-squared_errors/self.sigma)*mask[:,None]

    def state_dict(self):
        return dict(format='adaptive_hit_kernel_v1',initial=self.initial,minimum=self.minimum,
            step_dt=self.step_dt,tau_seconds=self.tau_seconds,
            sigma=self.sigma.detach().cpu().clone(),ema=self.ema.detach().cpu().clone(),calls=self.calls)

    def load_state_dict(self,value):
        expected=self.state_dict()
        for key in ['format','initial','minimum','step_dt','tau_seconds']:
            if value[key]!=expected[key]:raise ValueError('Curriculum definition changed: '+key)
        for key in ['sigma','ema']:
            t=value[key].to(device=self.sigma.device,dtype=self.sigma.dtype)
            if t.shape!=(3,) or not torch.isfinite(t).all():raise ValueError('Invalid saved kernel')
            setattr(self,key,t.clone())
        if (self.sigma<self.floor).any() or (self.sigma>torch.tensor(self.initial,device=self.sigma.device).square()+1e-7).any():
            raise ValueError('Saved kernel outside the declared bounds')
        self.calls=value['calls'];self.last_step=None


def ensure_kernel(env):
    if not hasattr(env,'_impact_adaptive_kernel'):
        env._impact_adaptive_kernel=AdaptiveTargetKernel(env.device,env.step_dt)
    return env._impact_adaptive_kernel


def squared_target_errors(position,velocity,normal,target_p,target_v,target_n):
    from .networks import unit_vector
    pe=((position-target_p)**2).sum(-1)
    ve=((velocity-target_v)**2).sum(-1)
    cosine=(unit_vector(normal)*unit_vector(target_n)).sum(-1).abs().clamp(0,1)
    ae=torch.acos(cosine).square()
    return torch.stack((pe,ve,ae),dim=-1)


def adaptive_target_reward(env,component,manifest,settings):
    from .task import state
    from racket_tracking_rewards import racket_quantities
    p,v,n,_,hit=state(env,manifest,settings).current()
    actual_p,_,actual_v,_,actual_n,_=racket_quantities(env)
    errors=squared_target_errors(actual_p,actual_v,actual_n,p,v,n)
    return ensure_kernel(env).scores(errors,hit,env.common_step_counter)[:,component]

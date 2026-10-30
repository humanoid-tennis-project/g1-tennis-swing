"""Read-only PPO monitoring and atomic resumable checkpoints; no reward changes."""
import json
import os
import random
from pathlib import Path
import numpy as np
import torch


def attach_monitoring(runner, wrapper, output, resume=None):
    output=Path(output)
    motion=wrapper.unwrapped.command_manager.get_term('motion')
    env=wrapper.unwrapped
    sampler_names=('bin_failed_count','_current_bin_failed')
    if resume:
        saved=torch.load(resume,map_location='cpu',weights_only=False).get('infos',{}).get('shared_monitoring',{})
        for name,value in saved.get('sampler',{}).items():
            current=getattr(motion,name)
            if current.shape!=value.shape:raise ValueError('Sampler shape changed')
            current.copy_(value.to(current.device))
        for name,value in saved.get('exposure',{}).items():
            getattr(wrapper,name).copy_(value.to(env.device))
        rng=saved.get('rng')
        if rng:
            random.setstate(rng['python']);np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch'])
            torch.cuda.set_rng_state_all(rng['cuda'])
    original_save=runner.save
    def save(path,infos=None):
        info=dict(infos or {})
        info['shared_monitoring']={
            'sampler':{n:getattr(motion,n).detach().cpu() for n in sampler_names if hasattr(motion,n)},
            'exposure':{n:getattr(wrapper,n).detach().cpu() for n in ('clip_steps','phase_steps','clip_ends','clip_timeouts')},
            'rng':{'python':random.getstate(),'numpy':np.random.get_state(),
                   'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},
            'exact_physics_resume':False,
        }
        temporary=Path(str(path)+'.writing')
        original_save(str(temporary),info)
        os.replace(temporary,path)
    runner.save=save
    original_log=runner.logger.log
    def log(**kw):
        extras={}
        for d in runner.logger.ep_extras:
            for key,value in d.items():
                try:extras.setdefault(key,[]).append(float(torch.as_tensor(value).float().mean()))
                except (ValueError,TypeError):pass
        original_log(**kw)
        row={'update':kw['it']+1,'scalars':{k:sum(v)/len(v) for k,v in extras.items()}}
        # Scalar episode metrics use the upstream logger's episodic aggregation;
        # these values do not establish full-motion completion.
        with (output/'episode_metrics.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n')
        if kw['it']%50==0:
            exposure={n:getattr(wrapper,n).detach().cpu().tolist() for n in ('clip_steps','phase_steps','clip_ends','clip_timeouts')}
            exposure.update(update=kw['it']+1,metrics_definition='Applied clip/phase counts; RSI timeouts are not full-clip success')
            temp=output/'exposure.json.writing'
            temp.write_text(json.dumps(exposure),encoding='utf-8');os.replace(temp,output/'exposure.json')
    runner.logger.log=log

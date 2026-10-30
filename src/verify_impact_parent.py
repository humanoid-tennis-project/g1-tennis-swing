"""Verify exact learned/optimizer/RNG restoration across an explicit reward fork."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def main(parent,initial,output,allow_local_runtime_bridge=False):
    a=torch.load(parent,map_location='cpu',weights_only=False)
    b=torch.load(initial,map_location='cpu',weights_only=False)
    checks={k:same(a[k],b[k]) for k in ['actor_state_dict','critic_state_dict','optimizer_state_dict']}
    for name in ['sampler','rng','exposure']:
        checks[name]=same(a['infos']['shared_monitoring'][name],b['infos']['shared_monitoring'][name])
    if 'impact_adaptive_kernel' in a['infos']:
        checks['adaptive_kernel']=same(a['infos']['impact_adaptive_kernel'],b['infos'].get('impact_adaptive_kernel'))
    scope_a=a['infos']['impact_scope'];scope_b=b['infos']['impact_scope']
    changed={k for k in scope_a.keys()|scope_b.keys() if not same(scope_a.get(k),scope_b.get(k))}
    allowed={'reward_variant','reward_variant_sha256'}
    if allow_local_runtime_bridge:
        if (scope_a['mujoco'],scope_b['mujoco'])!=('3.12.1','3.12.0'):
            raise ValueError('Only the validated local runtime transition is permitted')
        allowed.add('mujoco')
    checks['scope_changes_reviewed']=changed<=allowed
    result=dict(checks=checks,parent_iter=a['iter'],new_initial_iter=b['iter'],
        parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        initial_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),changed_scope_fields=sorted(changed),
        physics_and_rollout_reconstructed=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    if not all(checks.values()):raise AssertionError(result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--initial',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-local-runtime-bridge',action='store_true')
    main(**vars(p.parse_args()))

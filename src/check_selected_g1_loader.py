"""Read the entire generated corpus with the same MultiMotionLoader as PPO (CPU)."""
import argparse
import json
from pathlib import Path
import torch
from reference_manifest import load_motion_manifest
from retarget_tennis_bvh import TRACKED_BODIES
from mjlab.tasks.adapt_tennis.mdp.commands import MultiMotionLoader
from mjlab.tasks.adapt_tennis.replay_joint_names import G1_REPLAY_JOINT_NAMES_27


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    paths=load_motion_manifest(a.root/'reference_manifest.json')
    loader=MultiMotionLoader(tuple(str(p) for p in paths),torch.arange(14),device='cpu',
        target_body_names=TRACKED_BODIES,target_joint_names=G1_REPLAY_JOINT_NAMES_27)
    mids=torch.arange(len(paths));frames=(loader.lengths-1)//2
    q=loader.gather_joint_pos(mids,frames)
    qd=loader.gather_joint_vel(mids,frames)
    pos=loader.gather_body_pos_w(mids,frames)
    assert q.shape==(len(paths),27) and qd.shape==q.shape and pos.shape==(len(paths),14,3)
    assert all(torch.isfinite(x).all() for x in [q,qd,pos])
    result=dict(clips=len(paths),frames=int(loader.lengths.sum()),device='cpu',
        actual_loader='mjlab.tasks.adapt_tennis.mdp.commands.MultiMotionLoader',
        name_based_joint_mapping=True,name_based_body_mapping=True,
        joint_shape=list(q.shape),body_shape=list(pos.shape),all_finite=True,
        physics_stepped=False,policy_trained=False)
    with (a.root/'rsl_loader_check.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()

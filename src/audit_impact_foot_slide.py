"""Test whether pelvis drift at the annotated center equals the reference's stance-foot sliding.

A robot that keeps its stance foot planted and tracks body poses relative to the
pelvis can only reproduce the part of the reference pelvis travel that is carried
by stepping. The remainder is the integral of the stance-foot horizontal velocity
(foot skate from video-based retargeting). If the measured drift vector matches
minus that slide integral, the drift comes from the reference, not the optimizer.
Saved states and reference NPZ only; no new rollouts.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from audit_impact_root_drift import yaw

ANKLES=(3,6)  # left/right_ankle_roll_link in the 14-body reference layout


def audit(evaluation,manifest):
    motions=json.loads(Path(manifest).read_text(encoding='utf-8'))['motions']
    states=np.load(evaluation/'simulated_states.npz')
    with (evaluation/'tracking.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    centers={}
    for r in rows:
        if r['time_to_impact_s'] and abs(float(r['time_to_impact_s']))<1e-6:
            centers.setdefault((int(r['clip']),round(float(r['start_time_s']),3)),[]).append(r)
    keys=sorted({k.rsplit('_qpos',1)[0] for k in states.files if k.endswith('_qpos')})
    hit=[]
    for key in keys:
        clip=int(key[4:8]);start=int(key[-4:])/1000
        q=states[key+'_qpos'];rt=states[key+'_reference_time_s']
        with np.load(motions[clip]['motion_file']) as ref:B=ref['body_pos_w'];Q=ref['body_quat_w'][:,0];V=ref['body_lin_vel_w']
        f=np.minimum(np.round(rt*50).astype(int),len(B)-1)
        dpsi=yaw(q[0,3:7])-yaw(Q[f[0]]);c,s=np.cos(dpsi),np.sin(dpsi);R=np.array([[c,-s],[s,c]])
        ref_xy=B[f,0,:2]@R.T;ref_xy=ref_xy+(q[0,:2]-ref_xy[0])
        stance=np.where(B[:,ANKLES[0],2]<=B[:,ANKLES[1],2],ANKLES[0],ANKLES[1])
        foot_v=V[np.arange(len(V)),stance,:2]@R.T
        for r in centers.get((clip,round(start,3)),[]):
            i=min(max(int(round(float(r['sim_time_s'])/.02))-1,0),len(q)-1)
            frames=f[:i+1]
            slide=foot_v[frames].sum(0)*.02;travel=ref_xy[i]-ref_xy[0];drift=q[i,:2]-ref_xy[i]
            hit.append(dict(clip=clip,start_time_s=start,drift=drift.tolist(),slide=slide.tolist(),travel=travel.tolist(),
                target_position_error_m=float(r['target_position_error_m'])))
    D=np.array([h['drift'] for h in hit]);S=np.array([h['slide'] for h in hit]);T=np.array([h['travel'] for h in hit])
    E=np.array([h['target_position_error_m'] for h in hit])
    norm=lambda x:np.linalg.norm(x,axis=1)
    def explained(pred):
        # Fraction of drift vector energy removed by subtracting the prediction.
        return float(1-(norm(D-pred)**2).sum()/(norm(D)**2).sum())
    k=float((D*-S).sum()/(S*S).sum())
    stat=lambda x:dict(mean=float(x.mean()),median=float(np.median(x)))
    return dict(evaluation=str(evaluation),centers=len(hit),
        drift_m=stat(norm(D)),reference_travel_m=stat(norm(T)),reference_stance_slide_m=stat(norm(S)),
        drift_energy_explained_by_minus_slide=explained(-S),best_scale_on_minus_slide=k,
        drift_energy_explained_by_scaled_minus_slide=explained(-k*S),
        drift_energy_explained_by_minus_travel=explained(-T),
        correlation_slide_norm_vs_position_error=float(np.corrcoef(norm(S),E)[0,1]),
        correlation_travel_norm_vs_position_error=float(np.corrcoef(norm(T),E)[0,1]),
        position_error_by_slide_quartile=[float(E[(norm(S)>=lo)&(norm(S)<=hi)].mean()) for lo,hi in
            zip(np.percentile(norm(S),[0,25,50,75]),np.percentile(norm(S),[25,50,75,100]))],
        note='Stance foot = lower ankle per reference frame. Slide integrates its horizontal velocity from the episode start to the center.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.evaluation,a.manifest);a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='evaluation'},ensure_ascii=False))

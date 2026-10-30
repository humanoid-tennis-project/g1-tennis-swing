"""Decompose annotated-center racket error into pelvis drift and body-relative error.

Uses only saved evaluation states (no new rollouts). The reference pelvis is placed
by the rigid xy/yaw transform that matches the robot at the first saved sample, so
drift is measured from where the episode actually started.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def yaw(q):
    w,x,y,z=q[...,0],q[...,1],q[...,2],q[...,3]
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))


def audit(evaluation,manifest):
    motions=json.loads(Path(manifest).read_text(encoding='utf-8'))['motions']
    states=np.load(evaluation/'simulated_states.npz')
    with (evaluation/'tracking.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    centers={}
    for r in rows:
        if r['time_to_impact_s'] and abs(float(r['time_to_impact_s']))<1e-6:
            centers.setdefault((int(r['clip']),round(float(r['start_time_s']),3)),[]).append(r)
    keys=sorted({k.rsplit('_qpos',1)[0] for k in states.files if k.endswith('_qpos')})
    drift_by_time={};speed=dict(reference=[],robot=[]);velocity_error=[];hit=[]
    for key in keys:
        clip=int(key[4:8]);start=int(key[-4:])/1000
        q=states[key+'_qpos'];v=states[key+'_qvel'];rt=states[key+'_reference_time_s']
        with np.load(motions[clip]['motion_file']) as ref:P=ref['body_pos_w'][:,0];Q=ref['body_quat_w'][:,0];V=ref['body_lin_vel_w'][:,0]
        f=np.minimum(np.round(rt*50).astype(int),len(P)-1)
        dpsi=yaw(q[0,3:7])-yaw(Q[f[0]]);c,s=np.cos(dpsi),np.sin(dpsi);R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        ref_xy=P[f]@R.T;ref_xy=ref_xy[:,:2]+(q[0,:2]-ref_xy[0,:2]);ref_v=V[f]@R.T
        drift=np.linalg.norm(q[:,:2]-ref_xy,axis=1)
        yaw_error=np.abs(np.angle(np.exp(1j*(yaw(q[:,3:7])-yaw(Q[f])-dpsi))))
        for i,d in enumerate(drift):drift_by_time.setdefault(min(int(i*.02/.25),8),[]).append(float(d))
        speed['reference'].append(float(np.linalg.norm(ref_v[:,:2],axis=1).mean()));speed['robot'].append(float(np.linalg.norm(v[:,:2],axis=1).mean()))
        velocity_error+=list(np.linalg.norm(v[:,:3]-ref_v,axis=1))
        for r in centers.get((clip,round(start,3)),[]):
            i=min(max(int(round(float(r['sim_time_s'])/.02))-1,0),len(q)-1)
            hit.append(dict(clip=clip,start_time_s=start,sim_time_s=float(r['sim_time_s']),pelvis_xy_drift_m=float(drift[i]),
                pelvis_yaw_error_deg=float(np.degrees(yaw_error[i])),target_position_error_m=float(r['target_position_error_m']),
                racket_aligned_error_m=float(r['racket_aligned_error_m'])))
    d=np.array([h['pelvis_xy_drift_m'] for h in hit]);e=np.array([h['target_position_error_m'] for h in hit])
    a=np.array([h['racket_aligned_error_m'] for h in hit]);y=np.array([h['pelvis_yaw_error_deg'] for h in hit])
    stat=lambda x:dict(mean=float(x.mean()),median=float(np.median(x)),p75=float(np.percentile(x,75)))
    return dict(evaluation=str(evaluation),conditions=len(keys),centers=len(hit),
        at_center=dict(pelvis_xy_drift_m=stat(d),pelvis_yaw_error_deg=stat(y),target_position_error_m=stat(e),racket_yaw_xy_aligned_error_m=stat(a),
            correlation_drift_vs_position_error=float(np.corrcoef(d,e)[0,1])),
        drift_by_elapsed_s={f'{k*.25:.2f}':float(np.mean(v)) for k,v in sorted(drift_by_time.items())},
        horizontal_speed_m_s=dict(reference=float(np.mean(speed['reference'])),robot=float(np.mean(speed['robot']))),
        root_velocity_error_m_s=stat(np.array(velocity_error)),
        note='Saved-state analysis only. Drift is relative to the actual start pose; no ball contact is simulated.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.evaluation,a.manifest);a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='evaluation'},ensure_ascii=False))

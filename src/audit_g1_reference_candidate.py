"""Kinematic screening only; no physical-feasibility or exact-mesh guarantee."""
import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from sole_grounding import capsule_bottom, foot_geometry


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    ref=a.reference.resolve();out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    previous=Path.cwd()
    try:
        from load_reference_model import load_reference_model
        os.chdir(ref);model=load_reference_model(ref)
    finally:os.chdir(previous)
    with np.load(ref/'motion.npz',allow_pickle=False) as d:motion={k:d[k].copy() for k in d.files}
    with np.load(ref/'qpos_reference_local.npz',allow_pickle=False) as d:qpos=d['qpos'].copy()
    assert len(qpos)==len(motion['joint_pos'])
    racket=model.geom('racket_geom').id;racket_body=int(model.geom_bodyid[racket])
    pairs=[]
    for i in range(model.ngeom):
        name=model.body(int(model.geom_bodyid[i])).name
        if (i!=racket and model.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH
            and int(model.geom_bodyid[i])!=racket_body and 'right_wrist' not in name):
            pairs.append(i)
    data=mujoco.MjData(model);feet=foot_geometry(model);worst=None;foot=[];all_min=[]
    for frame,q in enumerate(qpos):
        data.qpos[:]=q;mujoco.mj_forward(model,data)
        foot.append([capsule_bottom(model,data,f[2]) for f in feet])
        minimum=2.
        for g in pairs:
            distance=float(mujoco.mj_geomDistance(model,data,racket,g,2.,None))
            minimum=min(minimum,distance)
            if worst is None or distance<worst['signed_distance_m']:
                worst=dict(frame=frame,time_s=frame/50,body=model.body(int(model.geom_bodyid[g])).name,
                           geom=model.geom(g).name,signed_distance_m=distance)
        all_min.append(minimum)
    foot=np.asarray(foot);q=motion['joint_pos'];fps=float(motion['fps'][0])
    speed=np.abs(motion['joint_vel']);step=np.abs(np.diff(q,axis=0))
    known=json.loads((ref/'reference_quality.json').read_text(encoding='utf-8'))
    flags=[]
    if min(all_min)<-.005:flags.append('racket_mesh_convex_hull_overlap_requires_review')
    if max(speed.max(),step.max()*fps)>20:flags.append('joint_speed_above_20rad_s_requires_review')
    if step.max()>.5:flags.append('joint_step_above_0p5rad_requires_review')
    if (known.get('largest_self_penetration') or {}).get('penetration_m',0)>.005:
        flags.append('body_proxy_overlap_above_5mm_requires_review')
    if np.mean(foot.min(1)>.05)>.2:
        flags.append('extended_both_feet_clearance_requires_phase_grounding_review')
    if np.any(foot.min(1)<-.005):
        flags.append('ground_penetration_below_minus5mm_requires_review')
    if known.get('height_fit',{}).get('supported_frame_fraction',1)==0:
        flags.append('no_predicted_support_anchors_for_height_fit_requires_review')
    if known['max_joint_limit_violation_rad']>1e-3:flags.append('joint_limit_violation')
    if not all(np.isfinite(x).all() for x in [qpos,q,speed,foot]):flags.append('nonfinite')
    result=dict(frames=len(q),fps=fps,screening_status='held_for_review' if flags else 'kinematic_screen_only',
                flags=flags,max_joint_speed_rad_s=float(speed.max()),max_joint_step_rad=float(step.max()),
                max_adjacent_frame_joint_rate_rad_s=float(step.max()*fps),
                full_racket_vs_body_mesh_hull_worst=worst,
                hull_overlap_frames_below_minus5mm=int(np.sum(np.array(all_min)<-.005)),
                left_right_foot_min_m=foot.min(0).tolist(),
                both_feet_above_5cm_fraction=float(np.mean(foot.min(1)>.05)),
                both_feet_above_10cm_fraction=float(np.mean(foot.min(1)>.1)),
                ground_penetration_below_minus5mm_fraction=float(np.mean(foot.min(1)<-.005)),
                physics_validated=False,stroke_complete='unconfirmed',human_skill_level='unknown',
                production_accepted=False,
                notes=['Mesh distances use convex hulls, not exact triangle intersection.',
                       'Holding wrist meshes excluded; other pairs require visual review.',
                       'Foot clearance requires phase/contact context; flight is not automatically invalid.',
                       'Thresholds are review triggers, not established universal feasibility limits.',
                       'Sampling at 50 Hz does not guarantee collision-free continuous interpolation.'])
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

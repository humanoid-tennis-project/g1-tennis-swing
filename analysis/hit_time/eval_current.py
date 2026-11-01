"""Step 1 (no training): the current final policy given annotated vs auto hit times (same 948 conditions, nominal goals).

Scored against the annotated contact on the 326 visible-contact references: racket point vs the annotated target
(the racket target of the annotated run, same runtime frame) at the annotated motion time, and the closest approach within +-0.3 s.
Also reports when the racket is closest to its own input target, relative to the input hit time.
"""
import json
import sys
from pathlib import Path
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; EXP = HERE.parent/'impact_improvement_20261030_v1'
sys.path.insert(0, str(ROOT/'Prototypes/Humanoid_Tennis_OSS'))
from build_ppo_motion_comparison import load_model

model = load_model(ROOT/'Prototypes/Humanoid_Tennis_OSS/upstream/AdaPT/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1_rightracket.xml'); data = mujoco.MjData(model)
site = next(i for i in range(model.nsite) if model.site(i).name.split('/')[-1] == 'racket_point')
ann = json.loads((EXP/'impact_manifest_v12_footlock20_local.json').read_text(encoding='utf-8'))['motions']
RUNS = dict(annotated=EXP/'std_floor_cont_v1/final_eval948', auto=HERE/'current_policy_autohit_eval948')
HALF = .02/2; WIN = .04+1e-6; NEAR = .3


def racket(qpos):
    out = []
    for q in qpos: data.qpos[:] = q; mujoco.mj_kinematics(model, data); out.append(data.site_xpos[site].copy())
    return np.array(out)


def run(name, z, targets):
    r = dict(conditions=0, reached_annotated_time=0, error_at_annotated_time=[], error_min_in_window=[], closest_rel_annotated_s=[], closest_rel_input_s=[])
    for k in z.files:
        if not k.endswith('_qpos'): continue
        key = k[:-5]; e = ann[int(key[4:8])]['events'][0]
        if e['annotation']['evidence'] != 'visible_contact': continue
        r['conditions'] += 1; p = racket(z[k]); t = z[key+'_reference_time_s']; g = z[key+'_applied_impact_goal_pos_vel_normal_time_hit']
        d = np.linalg.norm(p-targets[key], axis=1); rel = t-e['time_s']
        hit = np.flatnonzero(np.abs(rel) < HALF)
        if len(hit):
            r['reached_annotated_time'] += 1; r['error_at_annotated_time'].append(float(d[hit[0]]))
            r['error_min_in_window'].append(float(d[np.abs(rel) <= WIN].min()))
            w = np.flatnonzero(np.abs(rel) <= NEAR); r['closest_rel_annotated_s'].append(float(rel[w[np.argmin(d[w])]]))
        own = np.flatnonzero(np.abs(g[:, 9]) < 1e-6)
        if len(own):
            di = np.linalg.norm(p-g[:, :3], axis=1); w = np.flatnonzero(np.abs(g[:, 9]) <= NEAR); r['closest_rel_input_s'].append(float(-g[w[np.argmin(di[w])], 9]))
    return r


def summary(r):
    s = lambda v, f: round(float(f(np.array(v))), 3) if v else None
    return dict(conditions=r['conditions'], reached_annotated_time=r['reached_annotated_time'],
                error_at_annotated_time_median_cm=s(r['error_at_annotated_time'], np.median)*100, error_at_annotated_time_mean_cm=round(s(r['error_at_annotated_time'], np.mean)*100, 1),
                within_15cm_at_annotated_time=int((np.array(r['error_at_annotated_time']) <= .15).sum()),
                error_min_in_window_median_cm=round(s(r['error_min_in_window'], np.median)*100, 1),
                closest_rel_annotated_median_s=s(r['closest_rel_annotated_s'], np.median), closest_within_window_of_annotated=round(float((np.abs(r['closest_rel_annotated_s']) <= WIN).mean()), 3),
                closest_rel_input_median_s=s(r['closest_rel_input_s'], np.median), closest_within_window_of_input=round(float((np.abs(r['closest_rel_input_s']) <= WIN).mean()), 3))


if __name__ == '__main__':
    zs = {k: np.load(v/'simulated_states.npz') for k, v in RUNS.items()}
    za = zs['annotated']; targets = {k[:-len('_applied_impact_goal_pos_vel_normal_time_hit')]: za[k][0, :3] for k in za.files if k.endswith('_applied_impact_goal_pos_vel_normal_time_hit')}
    res = {k: summary(run(k, z, targets)) for k, z in zs.items()}
    res['note'] = 'Current final policy, trained with annotated hit times; only the hit-time input and its target change in the auto run.'
    (HERE/'eval_current.json').write_text(json.dumps(res, indent=2), encoding='utf-8'); print(json.dumps(res, indent=1))

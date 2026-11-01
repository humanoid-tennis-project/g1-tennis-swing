"""Does the racket follow the shifted target? Paired nominal vs randomized runs of the same checkpoint and conditions.

At the hit step (time to hit = 0) of each condition that reached it in both runs, CPU forward kinematics gives the racket point.
Slope = sum(d.o)/sum(o.o), d = racket shift between runs, o = target shift. 1 = follows the target, 0 = ignores it.
"""
import json
import sys
from pathlib import Path
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'Prototypes/Humanoid_Tennis_OSS'))
from build_ppo_motion_comparison import load_model

model = load_model(ROOT/'Prototypes/Humanoid_Tennis_OSS/upstream/AdaPT/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1_rightracket.xml'); data = mujoco.MjData(model)
site = next(i for i in range(model.nsite) if model.site(i).name.split('/')[-1] == 'racket_point')
RUNS = {'nominal': HERE.parent/'impact_improvement_20261030_v1/std_floor_cont_v1/final_eval948', 'randomized': HERE/'eval948_seed100'}


def hits(folder):
    z = np.load(folder/'simulated_states.npz'); out = {}
    for k in z.files:
        if not k.endswith('_qpos'): continue
        key = k[:-5]; g = z[key+'_applied_impact_goal_pos_vel_normal_time_hit']; idx = np.flatnonzero(np.abs(g[:, 9]) < 1e-6)
        if not len(idx): continue
        i = idx[0]; data.qpos[:] = z[k][i]; mujoco.mj_kinematics(model, data)
        vel = np.zeros(6); mujoco.mj_forward(model, data)
        out[key] = dict(racket=data.site_xpos[site].copy(), goal=g[i, :3].copy(), goal_vel=g[i, 3:6].copy())
    return out


A, B = (hits(RUNS[k]) for k in ('nominal', 'randomized')); keys = sorted(set(A) & set(B))
fk_check = float(np.median([np.linalg.norm(A[k]['racket']-A[k]['goal']) for k in keys])*100)
d = np.array([B[k]['racket']-A[k]['racket'] for k in keys]); o = np.array([B[k]['goal']-A[k]['goal'] for k in keys])
slope = float((d*o).sum()/(o*o).sum()); per_axis = [float((d[:, j]*o[:, j]).sum()/(o[:, j]**2).sum()) for j in range(3)]
corr = [float(np.corrcoef(d[:, j], o[:, j])[0, 1]) for j in range(3)]
res = dict(paired_conditions=len(keys), fk_median_error_nominal_cm=round(fk_check, 1), target_shift_mean_cm=round(float(np.linalg.norm(o, axis=1).mean()*100), 2),
           racket_shift_mean_cm=round(float(np.linalg.norm(d, axis=1).mean()*100), 2), slope_overall=round(slope, 3),
           slope_xyz=[round(x, 3) for x in per_axis], correlation_xyz=[round(x, 3) for x in corr],
           note='slope 1: racket moves with the target; 0: racket ignores it. Position only.')
np.savez(HERE/'adaptation_pairs.npz', keys=np.array(keys), racket_shift_m=d, target_shift_m=o)
(HERE/'adaptation.json').write_text(json.dumps(res, indent=2), encoding='utf-8'); print(json.dumps(res, indent=1))

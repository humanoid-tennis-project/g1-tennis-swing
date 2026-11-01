"""Evaluation gaps of the H11 final policy, from the saved simulated states (no new simulation).

1. Velocity and face-direction following (adaptation.py covered position only).
2. A contact criterion from the racket model: the target point must lie on the face disc (collision cylinder radius 0.15 m),
   measured in the actual face plane at the hit step and, leniently, at any step of the hit window (|t| <= 0.04 s).
3. Ignore-the-target baseline: the racket stays where it was under the nominal target, while the target moves.
4. Failure breakdown: termination term, fall (pelvis below 0.45 m), and whether the episode ended before the hit.
"""
import collections
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
geom = next(i for i in range(model.ngeom) if model.geom(i).name.split('/')[-1] == 'tennis_racket_collision')
FACE_R = float(model.geom_size[geom][0]); WINDOW = .04+1e-6; FALL_Z = .45
RUNS = {'nominal': HERE.parent/'impact_improvement_20261030_v1/std_floor_cont_v1/final_eval948', 'randomized_x1': HERE/'eval948_seed100',
        'randomized_x3': HERE/'eval948_seed100_range3'}


def racket(qpos, qvel):
    data.qpos[:] = qpos; data.qvel[:] = qvel; mujoco.mj_forward(model, data)
    v = np.zeros(6); mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site, v, 0)
    return data.site_xpos[site].copy(), v[3:].copy(), data.geom_xmat[geom].reshape(3, 3)[:, 2].copy()


def in_plane(goal, pos, n):
    o = goal-pos; return float(np.linalg.norm(o-np.dot(o, n)*n)), float(abs(np.dot(o, n)))


def load(folder):
    z = np.load(folder/'simulated_states.npz'); eps = {f"clip{e['clip']:04d}_start{round(e['start_time_s']*1000):04d}": e for e in map(json.loads, open(folder/'episodes.jsonl', encoding='utf-8'))}
    out = {}
    for key, e in eps.items():
        qp, qv, g = z[key+'_qpos'], z[key+'_qvel'], z[key+'_applied_impact_goal_pos_vel_normal_time_hit']
        hit = np.flatnonzero(np.abs(g[:, 9]) < 1e-6); win = np.flatnonzero(np.abs(g[:, 9]) <= WINDOW)
        r = dict(stroke=e['stroke_type'], termination=e['termination_reason'], completed=e['completed'], fell=bool(qp[-1, 2] < FALL_Z),
                 before_hit=bool(g[-1, 9] > 1e-6), hit=None, window_on_face=False)
        for i in win:
            p, _, n = racket(qp[i], qv[i])
            if in_plane(g[i, :3], p, n)[0] <= FACE_R: r['window_on_face'] = True
        if len(hit):
            i = hit[0]; p, v, n = racket(qp[i], qv[i]); t = g[i]
            r['hit'] = dict(pos=p, vel=v, normal=n*np.sign(np.dot(n, t[6:9]) or 1), goal=t[:3], goal_vel=t[3:6], goal_normal=t[6:9],
                            in_plane=in_plane(t[:3], p, n)[0], off_plane=in_plane(t[:3], p, n)[1], error=float(np.linalg.norm(t[:3]-p)))
        out[key] = r
    return out


def slope(d, o): return round(float((d*o).sum()/(o*o).sum()), 3)


def criteria(runs):
    res = {}
    for name, R in runs.items():
        hits = [r['hit'] for r in R.values() if r['hit']]; n = len(R)
        by = collections.defaultdict(lambda: [0, 0, 0])
        for r in R.values():
            b = by[r['stroke']]; b[0] += 1; b[1] += bool(r['hit'] and r['hit']['in_plane'] <= FACE_R); b[2] += r['window_on_face']
        res[name] = dict(conditions=n, reached_hit=len(hits),
                         on_face_at_hit=sum(h['in_plane'] <= FACE_R for h in hits), on_face_in_window=sum(r['window_on_face'] for r in R.values()),
                         within_15cm_3d=sum(h['error'] <= FACE_R for h in hits),
                         in_plane_median_cm=round(float(np.median([h['in_plane'] for h in hits]))*100, 1),
                         off_plane_median_cm=round(float(np.median([h['off_plane'] for h in hits]))*100, 1),
                         by_stroke={k: dict(conditions=v[0], on_face_at_hit=v[1], on_face_in_window=v[2]) for k, v in sorted(by.items())})
    return res


def following(A, B):
    keys = sorted(k for k in A if A[k]['hit'] and B[k]['hit']); a = [A[k]['hit'] for k in keys]; b = [B[k]['hit'] for k in keys]
    arr = lambda hs, f: np.array([h[f] for h in hs])
    o_p, d_p = arr(b, 'goal')-arr(a, 'goal'), arr(b, 'pos')-arr(a, 'pos')
    o_v, d_v = arr(b, 'goal_vel')-arr(a, 'goal_vel'), arr(b, 'vel')-arr(a, 'vel')
    o_n, d_n = arr(b, 'goal_normal')-arr(a, 'goal_normal'), arr(b, 'normal')-arr(a, 'normal')
    speed_ratio = np.log(np.linalg.norm(arr(b, 'vel'), axis=1)/np.linalg.norm(arr(a, 'vel'), axis=1))
    goal_ratio = np.log(np.linalg.norm(arr(b, 'goal_vel'), axis=1)/np.linalg.norm(arr(a, 'goal_vel'), axis=1))
    angle = lambda x, y: np.degrees(np.arccos(np.clip(np.abs((x*y).sum(1)), 0, 1)))
    ignore = np.linalg.norm(arr(b, 'goal')-arr(a, 'pos'), axis=1); actual = arr(b, 'error')
    ignore_in_plane = [in_plane(g, p, n)[0] for g, p, n in zip(arr(b, 'goal'), arr(a, 'pos'), arr(a, 'normal'))]
    return dict(paired_conditions=len(keys),
                target_shift_mean=dict(position_cm=round(float(np.linalg.norm(o_p, axis=1).mean()*100), 2), velocity_m_s=round(float(np.linalg.norm(o_v, axis=1).mean()), 2),
                                       face_deg=round(float(angle(arr(a, 'goal_normal'), arr(b, 'goal_normal')).mean()), 1)),
                slope=dict(position=slope(d_p, o_p), velocity_vector=slope(d_v, o_v), speed_log_ratio=slope(speed_ratio, goal_ratio), face_normal=slope(d_n, o_n)),
                note='slope 1: follows the target change; 0: ignores it.',
                ignore_target_baseline=dict(error_median_cm=round(float(np.median(ignore))*100, 1), actual_error_median_cm=round(float(np.median(actual))*100, 1),
                                            on_face_at_hit=int(sum(x <= FACE_R for x in ignore_in_plane)), actual_on_face_at_hit=int(sum(h['in_plane'] <= FACE_R for h in b))))


def failures(R):
    c = collections.Counter(); by = collections.defaultdict(collections.Counter)
    for r in R.values():
        if r['completed']: k = 'completed'
        else: k = ('fell' if r['fell'] else 'deviated_no_fall')+('_before_hit' if r['before_hit'] else '_after_hit')
        c[k] += 1; by[r['stroke']][k] += 1
    return dict(total=dict(c), termination_terms=dict(collections.Counter(r['termination'] for r in R.values() if not r['completed'])),
                by_stroke={k: dict(v) for k, v in sorted(by.items())}, fall_definition=f'pelvis height below {FALL_Z} m at the last step')


runs = {k: load(v) for k, v in RUNS.items() if (v/'simulated_states.npz').exists()}
res = dict(face_radius_m=FACE_R, contact_criterion=criteria(runs), failures={k: failures(v) for k, v in runs.items()},
           following={k: following(runs['nominal'], v) for k, v in runs.items() if k != 'nominal'})
(HERE/'extended_eval.json').write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8'); print(json.dumps(res, indent=1, ensure_ascii=False))

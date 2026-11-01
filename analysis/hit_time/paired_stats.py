"""Paired comparison of the two arms (both given annotated hit times) on the 326 visible-contact references; bootstrap over references."""
import json
import numpy as np
from eval_current import HERE, ann, racket, HALF, WIN, NEAR

def per_condition(folder, targets):
    z = np.load(HERE/folder/'simulated_states.npz'); out = {}
    for k in z.files:
        if not k.endswith('_qpos'): continue
        key = k[:-5]; e = ann[int(key[4:8])]['events'][0]
        if e['annotation']['evidence'] != 'visible_contact': continue
        rel = z[key+'_reference_time_s']-e['time_s']; hit = np.flatnonzero(np.abs(rel) < HALF)
        if not len(hit): continue
        d = np.linalg.norm(racket(z[k])-targets[key], axis=1); w = np.flatnonzero(np.abs(rel) <= NEAR)
        out[key] = (d[hit[0]], d[np.abs(rel) <= WIN].min(), abs(rel[w[np.argmin(d[w])]]) <= WIN)
    return out

za = np.load(HERE/'eval_annotatedarm_annotatedtime_948/simulated_states.npz')
targets = {k[:-len('_applied_impact_goal_pos_vel_normal_time_hit')]: za[k][0, :3] for k in za.files if k.endswith('_applied_impact_goal_pos_vel_normal_time_hit')}
A = per_condition('eval_annotatedarm_annotatedtime_948', targets); B = per_condition('eval_autoarm_annotatedtime_948', targets)
keys = sorted(set(A) & set(B)); clips = np.array([int(k[4:8]) for k in keys]); uc = np.unique(clips)
a = np.array([A[k] for k in keys], float); b = np.array([B[k] for k in keys], float)
rng = np.random.default_rng(0); idx = {c: np.flatnonzero(clips == c) for c in uc}
def stat(sel):
    return [float(np.median(b[sel, 0]-a[sel, 0])*100), float(np.median(b[sel, 1]-a[sel, 1])*100), float((a[sel, 2].mean()-b[sel, 2].mean())*100),
            float(((b[sel, 0] <= .15).mean()-(a[sel, 0] <= .15).mean())*100)]
boot = np.array([stat(np.concatenate([idx[c] for c in rng.choice(uc, len(uc))])) for _ in range(2000)])
names = ['median paired error diff at contact (auto-annotated, cm)', 'median paired window-min error diff (auto-annotated, cm)',
         'closest-approach within +-0.04 s of contact (annotated-auto, points)', 'within 15 cm at contact (auto-annotated, points)']
arms = {n: dict(error_at_contact_median_cm=round(float(np.median(x[:, 0]))*100, 1), closest_within_window_pct=round(float(x[:, 2].mean())*100, 1))
        for n, x in [('annotated_arm', a), ('auto_arm', b)]}  # each arm on the same paired set, so the columns subtract to the difference
res = dict(paired_conditions=len(keys), arms_on_paired_set=arms, references=len(uc), seed_note='one training seed per arm; seed-to-seed variation not measured',
           results={n: dict(value=round(v, 2), ci95=[round(x, 2) for x in np.percentile(boot[:, i], [2.5, 97.5])]) for i, (n, v) in enumerate(zip(names, stat(np.arange(len(keys)))))})
(HERE/'paired_stats.json').write_text(json.dumps(res, indent=2), encoding='utf-8'); print(json.dumps(res, indent=1))

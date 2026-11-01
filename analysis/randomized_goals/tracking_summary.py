"""Whole-body tracking of the final policy (nominal 948 evaluation) and per-reference counts, from the saved episode metrics."""
import collections
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; RUN = HERE.parent/'impact_improvement_20261030_v1/std_floor_cont_v1/final_eval948'
eps = [json.loads(l) for l in open(RUN/'episodes.jsonl', encoding='utf-8')]
ext = json.loads((HERE/'extended_eval.json').read_text(encoding='utf-8'))
m = lambda k: np.array([e['metrics_mean'][k] for e in eps])
res = dict(trials=len(eps), metrics_definition='mean over each trial until it ended, from the evaluator (metrics_mean)',
           joint_rmse_deg=dict(median=round(float(np.degrees(np.median(m('joint_rmse_rad')))), 1), mean=round(float(np.degrees(m('joint_rmse_rad').mean())), 1)),
           body_aligned_rmse_cm=dict(median=round(float(np.median(m('body_aligned_rmse_m')))*100, 1), mean=round(float(m('body_aligned_rmse_m').mean())*100, 1)),
           racket_world_error_cm=dict(median=round(float(np.median(m('racket_world_error_m')))*100, 1)),
           within_15cm_3d_at_hit=ext['contact_criterion']['nominal']['within_15cm_3d'],
           off_plane_median_cm=ext['contact_criterion']['nominal']['off_plane_median_cm'])
z = np.load(RUN/'simulated_states.npz'); hit = collections.defaultdict(list)
for k in z.files:
    if k.endswith('_applied_impact_goal_pos_vel_normal_time_hit'): hit[int(k[4:8])].append(bool(z[k][-1, 9] <= 1e-6))
res['references'] = dict(total=len(hit), reached_hit_both_starts=sum(all(v) for v in hit.values()), reached_hit_one_start=sum(sum(v) == 1 for v in hit.values()),
                         reached_hit_neither=sum(not any(v) for v in hit.values()))
(HERE/'tracking_summary.json').write_text(json.dumps(res, indent=2), encoding='utf-8'); print(json.dumps(res, indent=1))

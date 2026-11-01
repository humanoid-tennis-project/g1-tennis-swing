"""Compare the nominal, randomized-goal (training ranges) and x3-range evaluations of the H11 final checkpoint (same 948 conditions)."""
import csv
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = {'nominal': HERE.parent/'impact_improvement_20261030_v1/std_floor_cont_v1/final_eval948', 'randomized': HERE/'eval948_seed100', 'randomized_x3': HERE/'eval948_seed100_range3'}


def stats(d):
    s = json.loads((d/'summary.json').read_text(encoding='utf-8'))
    hit = [r for r in csv.DictReader(open(d/'tracking.csv', encoding='utf-8')) if r['time_to_impact_s'] and abs(float(r['time_to_impact_s'])) < 1e-6]
    p = np.array([float(r['target_position_error_m'])*100 for r in hit]); v = np.array([float(r['target_velocity_error_m_s']) for r in hit])
    n = np.array([float(r['target_plane_error_deg']) for r in hit])
    out = dict(evaluated=s['evaluated'], completed=s['completed'], reached_hit=len(hit), position_mean_cm=round(p.mean(), 1), position_median_cm=round(float(np.median(p)), 1),
               within_10cm=int((p <= 10).sum()), within_20cm=int((p <= 20).sum()), velocity_within_1_5=round(float((v <= 1.5).mean()*100), 1),
               plane_within_15deg=round(float((n <= 15).mean()*100), 1), all_three=int(((p <= 10) & (v <= 1.5) & (n <= 15)).sum()),
               randomized_goals=s['episodes'][0].get('randomized_goals'), checkpoint_sha256=s['checkpoint_sha256'])
    eps = s['episodes']
    if 'goal_offset_m' in eps[0]:
        o = np.array([e['goal_offset_m'] for e in eps])*100
        out['offset_abs_mean_cm'] = [round(x, 2) for x in np.abs(o).mean(0)]; out['offset_norm_mean_cm'] = round(float(np.linalg.norm(o, axis=1).mean()), 2)
        out['velocity_scale_range'] = [round(min(e['goal_velocity_scale'] for e in eps), 3), round(max(e['goal_velocity_scale'] for e in eps), 3)]
        out['normal_angle_abs_max_deg'] = round(max(abs(e['goal_normal_angle_deg']) for e in eps), 2)
    return out


res = {k: stats(d) for k, d in RUNS.items() if (d/'summary.json').exists()}
(HERE/'comparison.json').write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(res, indent=1, ensure_ascii=False))

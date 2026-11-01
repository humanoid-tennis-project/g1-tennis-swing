"""Score the two from-scratch arms against the annotated contact on the 326 visible-contact references (see eval_current.py)."""
import json
import numpy as np
from eval_current import HERE, run, summary

RUNS = dict(annotated_arm_annotated_time='eval_annotatedarm_annotatedtime_948', auto_arm_annotated_time='eval_autoarm_annotatedtime_948',
            auto_arm_auto_time='eval_autoarm_autotime_948')
zs = {k: np.load(HERE/v/'simulated_states.npz') for k, v in RUNS.items()}
za = zs['annotated_arm_annotated_time']
targets = {k[:-len('_applied_impact_goal_pos_vel_normal_time_hit')]: za[k][0, :3] for k in za.files if k.endswith('_applied_impact_goal_pos_vel_normal_time_hit')}
res = {k: summary(run(k, z, targets)) for k, z in zs.items()}
for k, v in RUNS.items():
    s = json.loads((HERE/v/'summary.json').read_text(encoding='utf-8')); res[k]['completed_of_948'] = s['completed']
(HERE/'score_arms.json').write_text(json.dumps(res, indent=2), encoding='utf-8'); print(json.dumps(res, indent=1))

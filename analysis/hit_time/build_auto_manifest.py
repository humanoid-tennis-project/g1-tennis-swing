"""Auto hit-time manifest: same references and racket mount, hit frame = peak reference racket speed instead of the hand annotation.

Targets are recomputed with impact_stage1.data.racket_series on the same foot-locked references; the script first checks that this
reproduces the annotated manifest's targets at the annotated frames.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; EXP = HERE.parent/'impact_improvement_20261030_v1'
sys.path.insert(0, str(EXP/'bundle_v12/Prototypes/Humanoid_Tennis_OSS'))
from impact_stage1.data import racket_series

SRC = EXP/'impact_manifest_v12_footlock20_local.json'; m = json.loads(SRC.read_text(encoding='utf-8'))
worst = 0.; shifts = []
for x in m['motions']:
    d = np.load(x['motion_file']); s = racket_series(d, m['mount']); e = x['events'][0]; f = e['frame']
    worst = max(worst, *(float(np.abs(s[k][f]-np.asarray(e[k])).max()) for k in ('position', 'velocity', 'normal')))
    a = int(np.argmax(np.linalg.norm(s['velocity'], axis=1))); n = x['frames']; shifts.append(a-f)
    e2 = dict(e, frame=a, time_s=a/x['fps'], available_hit_window_frames=[max(0, a-2), min(n-1, a+2)], truncated_hit_window=a-2 < 0 or a+2 > n-1,
              position=s['position'][a].tolist(), velocity=s['velocity'][a].tolist(), normal=s['normal'][a].tolist(),
              annotation=dict(e['annotation'], evidence='auto_peak_racket_speed', annotated_frame=f))
    for k in ('original_position', 'original_velocity', 'foot_lock_offset_m'): e2.pop(k, None)
    x['events'] = [e2]
assert worst < 1e-4, f'racket_series does not reproduce the annotated targets ({worst})'
m['hit_time_source'] = dict(rule='argmax |reference racket velocity|', annotated_manifest=str(SRC.name),
                            annotated_manifest_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(), reproduction_max_abs_error=worst)
(HERE/'impact_manifest_v12_footlock20_autohit.json').write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding='utf-8')
sh = np.array(shifts); print(dict(reproduction_max_abs_error=worst, shift_median_abs=float(np.median(np.abs(sh))), within_2=float((np.abs(sh) <= 2).mean()),
                               truncated=sum(x['events'][0]['truncated_hit_window'] for x in m['motions']), at_edges=int(sum(x['events'][0]['frame'] in (0, x['frames']-1) for x in m['motions']))))

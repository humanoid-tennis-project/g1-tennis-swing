"""Compare paired fixed evaluations over matched simulation-time prefixes."""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

METRICS = ['joint_rmse_rad', 'body_aligned_rmse_m', 'racket_world_error_m',
           'racket_velocity_error_m_s', 'racket_plane_error_deg']
HIT_METRICS = ['target_position_error_m', 'target_velocity_error_m_s',
               'target_plane_error_deg']


def load(folder,expected_count=20):
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    assert summary['evaluated'] == summary['requested'] == expected_count
    rows = {}
    with (folder / 'tracking.csv').open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            key = (int(row['clip']), round(float(row['start_time_s']), 6))
            rows.setdefault(key, []).append(row)
    episodes = {(e['clip'], round(e['start_time_s'], 6)): e for e in summary['episodes']}
    assert set(rows) == set(episodes) and len(episodes) == expected_count
    for key, data in rows.items():
        assert len(data) == episodes[key]['samples']
        assert all(abs(float(r['sim_time_s']) - .02 * (i+1)) < 1e-6
                   for i, r in enumerate(data))
    return summary, episodes, rows


def main(root, before=None, after=None, before_update=3202, after_update=6348, output_name='fastfollowup_comparison.json',expected_count=20):
    before = before or root / 'recovered/impact474_recovery1_00_evaluate_20261029_v1'
    after = after or root / 'recovered/impact474_fastfollowup_01_evaluate_20261029_v1'
    bs, be, br = load(before,expected_count)
    ass, ae, ar = load(after,expected_count)
    assert bs['seed'] == ass['seed'] == 100 and set(be) == set(ae)
    assert bs['starts'] == ass['starts']
    conditions = []
    for key in sorted(be):
        b, a = be[key], ae[key]
        assert b['clip_id'] == a['clip_id'] and b['stroke_type'] == a['stroke_type']
        assert not b['randomized_goals'] and not a['randomized_goals']
        n = min(len(br[key]), len(ar[key]))
        paired = {}
        for metric in METRICS:
            values = [[float(r[metric]) for r in rows[key][:n]] for rows in (br, ar)]
            assert all(math.isfinite(v) for vs in values for v in vs)
            bm, am = map(mean, values)
            paired[metric] = dict(before=bm, after=am, change=am-bm)
        conditions.append(dict(clip=key[0], start_time_s=key[1], clip_id=b['clip_id'],
            stroke_type=b['stroke_type'], common_samples=n, common_seconds=n*.02,
            before_completed=b['completed'], after_completed=a['completed'],
            before_reached_s=b['reached_s'], after_reached_s=a['reached_s'],
            before_hit_samples=b['hit_samples'], after_hit_samples=a['hit_samples'],
            metrics=paired))
    cohorts = {}
    for name, entries in [('all', conditions)] + [
            (s, [c for c in conditions if c['stroke_type'] == s])
            for s in sorted({c['stroke_type'] for c in conditions})]:
        keys = [(c['clip'], c['start_time_s']) for c in entries]
        both_hit = [k for k in keys if be[k]['hit_samples'] and ae[k]['hit_samples']]
        result = dict(conditions=len(entries), before_completed=sum(c['before_completed'] for c in entries),
            after_completed=sum(c['after_completed'] for c in entries),
            before_hit_reached=sum(be[k]['hit_samples'] > 0 for k in keys),
            after_hit_reached=sum(ae[k]['hit_samples'] > 0 for k in keys),
            both_hit_conditions=len(both_hit), common_prefix_metrics={}, both_hit_metrics={},
            before_termination=dict(Counter(be[k]['termination_reason'] for k in keys)),
            after_termination=dict(Counter(ae[k]['termination_reason'] for k in keys)))
        for metric in METRICS:
            result['common_prefix_metrics'][metric] = {
                side: mean(c['metrics'][metric][side] for c in entries)
                for side in ['before', 'after', 'change']}
        for metric in HIT_METRICS:
            result['both_hit_metrics'][metric] = {
                'before': mean(be[k]['hit_target_metrics_mean'][metric] for k in both_hit) if both_hit else None,
                'after': mean(ae[k]['hit_target_metrics_mean'][metric] for k in both_hit) if both_hit else None}
        cohorts[name] = result
    result = dict(before_checkpoint_sha256=bs['checkpoint_sha256'],
        after_checkpoint_sha256=ass['checkpoint_sha256'], before_update=before_update, after_update=after_update,
        before_evaluation=str(before),after_evaluation=str(after),
        seed=100, randomized_goals=False, new_rollouts=False,
        averaging='Equal weight per condition, paired minimum reached-time prefix. Hit-window means use only conditions reaching the window in both evaluations; missing values are not zero.',
        expected_condition_count=expected_count,
        limitations=f'{expected_count} fixed trained conditions; no unseen generalization, randomized-goal, or ball test.',
        cohorts=cohorts, conditions=conditions)
    (root / output_name).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(cohorts, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--before',type=Path);parser.add_argument('--after',type=Path)
    parser.add_argument('--before-update',type=int,default=3202);parser.add_argument('--after-update',type=int,default=6348)
    parser.add_argument('--output-name',default='fastfollowup_comparison.json')
    parser.add_argument('--expected-count',type=int,default=20)
    args=parser.parse_args()
    main(args.root,args.before,args.after,args.before_update,args.after_update,args.output_name,args.expected_count)

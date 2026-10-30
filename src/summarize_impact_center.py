"""Evaluate the annotated impact center separately from the five-frame window.

Missing/unreached centers never become zero error or a successful hit. This is a
racket-state proxy, not a simulated ball-contact success criterion.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean,median


def summarize(folder):
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    with (folder/'tracking.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    results=[]
    for ep in summary['episodes']:
        selected=[r for r in rows if int(r['clip'])==ep['clip'] and abs(float(r['start_time_s'])-ep['start_time_s'])<1e-8]
        centers=[r for r in selected if abs(float(r['time_to_impact_s']))<1e-6]
        metrics=[]
        for r in centers:
            metrics.append(dict(sim_time_s=float(r['sim_time_s']),motion_time_s=float(r['motion_time_s']),
                position_m=float(r['target_position_error_m']),velocity_m_s=float(r['target_velocity_error_m_s']),
                plane_deg=float(r['target_plane_error_deg'])))
        passed=bool(metrics) and all(m['position_m']<=.10 and m['velocity_m_s']<=1.5 and m['plane_deg']<=15 for m in metrics)
        results.append(dict(clip=ep['clip'],clip_id=ep['clip_id'],stroke_type=ep['stroke_type'],start_time_s=ep['start_time_s'],
            completed=ep['completed'],hit_window_samples=ep['hit_samples'],center_observed=bool(metrics),
            centers=metrics,center_proxy_passed=passed,complete_and_center_proxy_passed=bool(ep['completed'] and passed)))
    observed=[c for r in results for c in r['centers']]
    result=dict(checkpoint_sha256=summary['checkpoint_sha256'],evaluated=summary['evaluated'],requested=summary['requested'],
        completed=sum(r['completed'] for r in results),center_observed_conditions=sum(r['center_observed'] for r in results),
        center_proxy_passed=sum(r['center_proxy_passed'] for r in results),
        complete_and_center_proxy_passed=sum(r['complete_and_center_proxy_passed'] for r in results),
        thresholds=dict(position_m=.10,velocity_m_s=1.5,plane_deg=15),
        metrics_at_observed_centers={k:dict(mean=mean(c[k] for c in observed),median=median(c[k] for c in observed)) if observed else None
                                    for k in ('position_m','velocity_m_s','plane_deg')},
        note='Annotated center only; preserved window metrics remain in original summary. No ball-contact success or full474 claim.',conditions=results)
    (folder/'impact_center_statistics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True)
    result=summarize(p.parse_args().evaluation)
    print(json.dumps({k:v for k,v in result.items() if k!='conditions'},ensure_ascii=False))

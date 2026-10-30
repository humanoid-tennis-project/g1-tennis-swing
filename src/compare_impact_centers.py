"""Paired annotated centers only; never fill missing center errors with zero."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

FIELDS=['target_position_error_m','target_velocity_error_m_s','target_plane_error_deg']

def load(folder):
    s=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    if s['evaluated']!=s['requested']:raise ValueError('Complete evaluation required')
    episodes={(e['clip'],e['start_time_s']):e for e in s['episodes']}
    centers={k:{} for k in episodes}
    with (folder/'tracking.csv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            if abs(float(r['time_to_impact_s']))<1e-6:
                key=(int(r['clip']),float(r['start_time_s']));t=round(float(r['motion_time_s']),6)
                centers[key][t]={k:float(r[k]) for k in FIELDS}
    return s,episodes,centers

def main(before,after,output):
    bs,be,bc=load(before);ass,ae,ac=load(after)
    if set(be)!=set(ae) or bs['seed']!=ass['seed']:raise ValueError('Condition/seed mismatch')
    conditions=[]
    for k in sorted(be):
        common=sorted(set(bc[k])&set(ac[k]))
        conditions.append(dict(clip=k[0],start_time_s=k[1],stroke_type=be[k]['stroke_type'],
            before_center_count=len(bc[k]),after_center_count=len(ac[k]),shared_center_times_s=common,
            metrics={m:dict(before=mean(bc[k][t][m] for t in common),after=mean(ac[k][t][m] for t in common)) for m in FIELDS} if common else None))
    cohorts={}
    for name,entries in [('all',conditions)]+[(st,[c for c in conditions if c['stroke_type']==st]) for st in sorted({c['stroke_type'] for c in conditions})]:
        observed=[c for c in entries if c['metrics'] is not None]
        cohorts[name]=dict(total_conditions=len(entries),paired_center_conditions=len(observed),
            before_observed=sum(c['before_center_count']>0 for c in entries),after_observed=sum(c['after_center_count']>0 for c in entries),
            metrics={m:{side:mean(c['metrics'][m][side] for c in observed) for side in ['before','after']} for m in FIELDS} if observed else None)
    d=dict(before_checkpoint_sha256=bs['checkpoint_sha256'],after_checkpoint_sha256=ass['checkpoint_sha256'],
        averaging='Equal weight per paired condition; multiple common centers within a condition averaged first. Missing centers excluded and counts reported, never zero.',
        before=str(before),after=str(after),cohorts=cohorts,conditions=conditions)
    output.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(cohorts['all']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(**vars(p.parse_args()))

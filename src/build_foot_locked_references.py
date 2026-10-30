"""Remove stance-foot horizontal sliding from the retargeted references (new files only).

Video-based GVHMR->GMR references slide the supporting foot (median 0.30 m/s). A
robot that plants its foot cannot reproduce the pelvis travel carried by that
slide. Contact-aware correction is a standard step in video-to-humanoid pipelines
(PBHC/KungfuBot contact-aware motion correction; VideoMimic foot-skating penalty).
This is the simplest rigid variant: whenever the lower ankle is near the ground,
its horizontal velocity is subtracted from every body, so the whole pose moves
with a planted foot. Joint angles, heights, orientations and relative poses are
untouched; hit targets are shifted by the same per-frame offset. Originals are
not modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ANKLES=(3,6)
CONTACT_Z=.12  # default; lower ankle_roll_link height median .093, p75 .125, p95 .198


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lock(data,contact_z=CONTACT_Z):
    B=data['body_pos_w'].copy();V=data['body_lin_vel_w'].copy();fps=float(np.asarray(data['fps']).flat[0])
    left_lower=B[:,ANKLES[0],2]<=B[:,ANKLES[1],2];stance=np.where(left_lower,ANKLES[0],ANKLES[1])
    contact=B[np.arange(len(B)),stance,2]<contact_z
    slide_v=np.where(contact[:,None],V[np.arange(len(V)),stance,:2],0.)
    # Positions shift by the integrated slide; frame 0 is unchanged.
    offset=np.concatenate((np.zeros((1,2)),np.cumsum(slide_v[:-1],0)/fps))
    B[:,:,:2]-=offset[:,None];V[:,:,:2]-=slide_v[:,None]
    return B,V,offset,slide_v,contact


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest-output',type=Path,required=True);p.add_argument('--contact-z',type=float,default=CONTACT_Z);a=p.parse_args()
    source=json.loads(a.manifest.read_text(encoding='utf-8'));a.output.mkdir(parents=True,exist_ok=False)
    rows=[];stats=[]
    for row in source['motions']:
        with np.load(row['motion_file'],allow_pickle=False) as d:
            data={k:d[k] for k in d.files}
        B,V,offset,slide_v,contact=lock(data,a.contact_z)
        out=a.output/Path(row['motion_file']).parent.name/'motion.npz';out.parent.mkdir()
        np.savez(out,**dict(data,body_pos_w=B,body_lin_vel_w=V))
        events=[]
        for e in row['events']:
            f=e['frame'];pos=list(e['position']);vel=list(e['velocity'])
            pos[0]-=float(offset[f,0]);pos[1]-=float(offset[f,1]);vel[0]-=float(slide_v[f,0]);vel[1]-=float(slide_v[f,1])
            events.append(dict(e,position=pos,velocity=vel,foot_lock_offset_m=offset[f].tolist(),original_position=e['position'],original_velocity=e['velocity']))
        rows.append(dict(row,motion_file=str(out.resolve()),motion_sha256=sha(out),events=events,
            original_motion_file=row['motion_file'],original_motion_sha256=row['motion_sha256']))
        stats.append(dict(id=row['id'],contact_fraction=float(contact.mean()),
            removed_travel_at_last_event_m=float(np.linalg.norm(offset[row['events'][-1]['frame']]))))
    result=dict(source,motions=rows,foot_lock=dict(method='rigid stance-foot horizontal slide removal',contact_ankle_z_m=a.contact_z,
        source_manifest=str(a.manifest.resolve()),source_manifest_sha256=sha(a.manifest),
        mean_contact_fraction=float(np.mean([s['contact_fraction'] for s in stats])),
        removed_travel_at_last_event_m=dict(mean=float(np.mean([s['removed_travel_at_last_event_m'] for s in stats])),
            median=float(np.median([s['removed_travel_at_last_event_m'] for s in stats]))),
        joint_pos_modified=False,originals_modified=False))
    a.manifest_output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    (a.output/'foot_lock_stats.json').write_text(json.dumps(stats,indent=1),encoding='utf-8')
    print(json.dumps(result['foot_lock'],ensure_ascii=False))


if __name__=='__main__':main()

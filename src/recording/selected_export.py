"""Extract user intervals from completed segment inference, preserving clocks."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def export_segment(root, segment):
    root=Path(root);folder=root/segment['segment_id']
    windows=json.loads((folder/'motion_windows.json').read_text())['windows']
    tracking=json.loads((folder/'tracking_observations.json').read_text())
    track=next(t for t in tracking['tracks'] if t['track_id']==tracking['selected_track_id'])
    detected=set(track['frame_ids']);output=[]
    for annotation in segment['intervals']:
        lo=annotation['start_frame']-segment['start_frame'];hi=annotation['end_frame']-segment['start_frame']
        owner=[w for w in windows if lo>=w['core_start'] and hi<=w['core_end']]
        if len(owner)!=1:raise ValueError('User interval spans independent world windows; do not silently join')
        w=owner[0];a=lo-w['input_start'];b=hi-w['input_start'];name=f'window_{w["index"]:05d}'
        params=torch.load(folder/'motion_windows'/(name+'.pt'),map_location='cpu')
        arrays={f'{space}__{key}':value[a:b].numpy() for space in ('incam','global') for key,value in params[f'smpl_params_{space}'].items()}
        with np.load(folder/'motion_windows'/(name+'_joints.npz')) as joints:
            arrays.update(joints_global=joints['joints_global'][a:b],joints_incam=joints['joints_incam'][a:b],K_fullimg=joints['K'][a:b])
        if hi-lo<2 or any(len(v)!=hi-lo or not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Invalid extracted human motion')
        arrays.update(fps=np.array(30.),segment_frames=np.arange(lo,hi),
            parent_30Hz_frames=np.arange(annotation['start_frame'],annotation['end_frame']),
            parent_elapsed_seconds=np.arange(annotation['start_frame'],annotation['end_frame'])/30,
            target_detection_observed=np.array([i in detected for i in range(lo,hi)]))
        key=segment['segment_id']+'__'+annotation['id'];dest=root/'clips'/key;dest.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(dest/'human_motion.npz',**arrays)
        rot=Rotation.from_rotvec(arrays['global__global_orient'])
        angular=(rot[:-1].inv()*rot[1:]).magnitude()*180/np.pi
        root_step=np.linalg.norm(np.diff(arrays['global__transl'],axis=0),axis=-1)
        wrist_v=np.linalg.norm(np.diff(arrays['joints_global'][:,[20,21]],axis=0)*30,axis=-1)
        quality=dict(numerically_finite=True,frames=hi-lo,duration_s=(hi-lo)/30,
            direct_detection_fraction=float(arrays['target_detection_observed'].mean()),
            interpolated_tracking_frames=int((~arrays['target_detection_observed']).sum()),
            max_root_step_m=float(root_step.max()),max_root_rotation_step_deg=float(angular.max()),
            wrist_peak_estimated_world_speed_m_s=float(wrist_v.max()),
            full_clip_identity_verified=False,physical_validity_verified=False,
            note='Diagnostic metrics only; no automatic quality approval or human skill inference')
        metadata=dict(id=key,annotation=annotation,annotation_revision=segment['annotation_revision'],
            annotation_sha256=segment['annotation_sha256'],source=segment['source'],
            segment_id=segment['segment_id'],segment_parent_frames=[segment['start_frame'],segment['end_frame']],
            user_boundary_parent_frames=[annotation['start_frame'],annotation['end_frame']],
            annotation_time_basis=segment['annotation_time_basis'],selected_track_id=tracking['selected_track_id'],
            target_hint=segment['target_hint'],target_anchor=segment.get('target_anchor'),
            fps=30,joint_order='SMPL first22',coordinates='GVHMR y-up; independent world frame per computational segment',
            global_origin_stitched_across_segments=False,quality=quality,G1_retargeted=False,
            public_distribution=False,path=str(dest/'human_motion.npz'),review_video=str(folder/'review.mp4'),
            review_local_seconds=[lo/30,hi/30])
        with (dest/'metadata.json').open('x',encoding='utf-8') as f:json.dump(metadata,f,indent=2,ensure_ascii=False)
        output.append(metadata)
    return output


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--config',required=True)
    a=p.parse_args();root=Path(a.root)
    config=json.loads(Path(a.config).read_text(encoding='utf-8'));rows=[]
    for item in config['ready']:rows.extend(export_segment(root,item['segment']))
    with (root/'clip_catalog.json').open('x',encoding='utf-8') as f:json.dump(rows,f,indent=2,ensure_ascii=False)
    print(json.dumps(dict(clips=len(rows),frames=sum(r['quality']['frames'] for r in rows))))


if __name__=='__main__':main()

"""Bounded CPU batch of the existing GMR driver, with per-clip provenance and QA."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from adapt_selected_human import adapt


def save(path, data):
    tmp=path.with_suffix('.new');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--body-model',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--limit',type=int)
    p.add_argument('--preview',action='store_true')
    a=p.parse_args()
    if not 1<=a.workers<=4:raise ValueError('Use 1..4 CPU workers')
    selection=json.loads(a.selection.read_text(encoding='utf-8'))
    if selection['format']!='reviewed_human_motions_for_g1_v1':raise ValueError('Wrong selection format')
    rows=selection['motions'][:a.limit] if a.limit else selection['motions']
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'.gitignore').write_text('**/*.npz\n**/*.xml\n**/*.mp4\n**/*.png\n**/*.log\n',encoding='utf-8')
    code=Path(__file__).resolve().parent
    expected_parents=np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19])
    with np.load(a.body_model,allow_pickle=False) as bm:
        parents=bm['kintree_table'][0,:22].astype(np.int64);parents[0]=-1
    np.testing.assert_array_equal(parents,expected_parents)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1')
    started=time.monotonic();results=[]
    save(a.output/'run_manifest.json',dict(selection=str(a.selection.resolve()),
        selection_sha256=hashlib.sha256(a.selection.read_bytes()).hexdigest(),
        body_model_sha256=hashlib.sha256(a.body_model.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256((code/'retarget_gvhmr_local.py').read_bytes()).hexdigest(),
        count=len(rows),workers=a.workers,preview=a.preview,cloud_used=False,
        method='Existing GMR SMPLX mapping + AdaPT 27 joint right-racket model; unchanged driver',
        paid_compute=False,training_performed=False))

    def run(row):
        start=time.monotonic();folder=a.output/row['id'];folder.mkdir()
        result=dict(id=row['id'],status='running')
        try:
            adapt(row,folder/'human')
            commands=[
                [sys.executable,str(code/'retarget_gvhmr_local.py'),'--joints',str(folder/'human/human_joints_local.npz'),
                 '--parameters',str(folder/'human/human_motion_parameters.npz'),'--body-model',str(a.body_model.resolve()),
                 '--output',str(folder/'reference'),*(['--preview'] if a.preview else [])],
                [sys.executable,str(code/'audit_g1_reference_candidate.py'),'--reference',str(folder/'reference'),
                 '--output',str(folder/'audit.json')]]
            for index,command in enumerate(commands):
                with (folder/f'stage{index}.log').open('x',encoding='utf-8') as log:
                    subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240,check=True)
            ref=folder/'reference/motion.npz'
            with np.load(ref,allow_pickle=False) as d:
                n=len(d['joint_pos']);names=d['joint_names'].tolist();bodies=d['body_names'].tolist()
                shapes={'joint_pos':(n,27),'joint_vel':(n,27),'body_pos_w':(n,14,3),
                        'body_quat_w':(n,14,4),'body_lin_vel_w':(n,14,3),'body_ang_vel_w':(n,14,3)}
                if len(set(names))!=27 or len(set(bodies))!=14 or float(d['fps'][0])!=50:
                    raise ValueError('Invalid runtime motion layout')
                for k,shape in shapes.items():
                    if d[k].shape!=shape or not np.isfinite(d[k]).all():raise ValueError(f'Invalid {k}')
                np.testing.assert_allclose(np.linalg.norm(d['body_quat_w'],axis=-1),1,atol=1e-5)
            audit=json.loads((folder/'audit.json').read_text(encoding='utf-8'))
            quality=json.loads((folder/'reference/reference_quality.json').read_text(encoding='utf-8'))
            meta=dict(**row,format='adapt_mjlab_g1_tracking_npz_v1',fps=50,joints=names,tracked_bodies=bodies,
                world_axes='right-handed Z-up',quaternion_order='wxyz',racket_hand='right',
                racket_pose_source='Rigid model attachment inferred from wrist; not measured racket pose',
                motion_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),
                units=dict(joint_pos='rad',joint_vel='rad/s',position='m',linear_velocity='m/s',angular_velocity='rad/s'),
                endpoint_policy='No extrapolation past last human sample; original half-open bounds retained in annotation',
                human_style_approved=True,robot_kinematic_flags=audit['flags'],physics_verified=False,
                source_to_target_clock=dict(source_fps=30,target_fps=50,
                    source_start_frame=row['annotation']['start_frame'],target_local_time_start_s=0,
                    target_local_time_last_s=quality['last_timestamp_s']))
            save(folder/'reference/metadata.json',meta)
            result.update(status='converted',frames=n,motion_file=str(ref),audit_flags=audit['flags'],
                max_joint_limit_violation_rad=quality['max_joint_limit_violation_rad'],
                motion_sha256=meta['motion_sha256'],physics_verified=False)
        except Exception as error:
            result.update(status='failed',error=repr(error))
        result['wall_seconds']=time.monotonic()-start;save(folder/'result.json',result)
        return result

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,row) for row in rows]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            save(a.output/'progress.json',dict(total=len(rows),finished=len(results),
                converted=sum(r['status']=='converted' for r in results),failed=sum(r['status']=='failed' for r in results),
                wall_seconds=time.monotonic()-started,latest=result))
            print(json.dumps(result),flush=True)
    results.sort(key=lambda r:r['id'])
    indexed={r['id']:r for r in rows}
    save(a.output/'reference_manifest.json',dict(format='reviewed_g1_references_v1',
        motions=[dict(id=r['id'],motion_file=str(Path(r['motion_file']).relative_to(a.output)),
            identity_review=indexed[r['id']]['identity_review'],pose_review=indexed[r['id']]['pose_review'],
            review_evidence=indexed[r['id']]['review_evidence'],
            audit_file=f"{r['id']}/audit.json",audit_flags=r['audit_flags'],physics_verified=False)
            for r in results if r['status']=='converted'],
        note='User review applies to the human source; robot kinematic flags remain separate and must be reported before training'))
    save(a.output/'summary.json',dict(total=len(rows),converted=sum(r['status']=='converted' for r in results),
        failed=sum(r['status']=='failed' for r in results),wall_seconds=time.monotonic()-started,
        kinematic_flagged=sum(bool(r.get('audit_flags')) for r in results),results=results,
        note='Numerical/kinematic conversion, not proof of physical tracking; flags do not delete source data'))


if __name__=='__main__':main()

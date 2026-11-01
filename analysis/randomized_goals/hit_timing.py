"""Hit-time annotation check: annotated hit vs the reference racket speed peak, and when the policy racket is closest to the target. Run from impact_improvement_20261030_v1."""
import json,csv,collections,numpy as np
from scipy.spatial.transform import Rotation as R
m=json.load(open('impact_manifest_v12_footlock20_local.json',encoding='utf-8'))['motions']
OFF=np.array([0.4367835,-0.003,0.145]); dev=[];ev=[];peakspd=[];annspd=[]
for i,x in enumerate(m):
    z=np.load(f'foot_locked_refs_v2/clip{i:04d}/motion.npz'); names=[str(n) for n in z['body_names']]; w=names.index('right_wrist_yaw_link')
    q=z['body_quat_w'][:,w]; v=z['body_lin_vel_w'][:,w]+np.cross(z['body_ang_vel_w'][:,w],R.from_quat(q,scalar_first=True).apply(OFF))
    s=np.linalg.norm(v,axis=1); e=x['events'][0]; f=e['frame']; p=int(np.argmax(s))
    dev.append(p-f); ev.append(e['annotation']['evidence']); peakspd.append(s[p]); annspd.append(s[f])
dev=np.array(dev); ev=np.array(ev)
print('evidence',collections.Counter(ev))
for g in ['all']+sorted(set(ev)):
    d=dev if g=='all' else dev[ev==g]; a=np.abs(d)
    print(g,len(d),'|peak-annot| frames median',np.median(a),'<=2:',round((a<=2).mean()*100,1),'% >5:',round((a>5).mean()*100,1),'%')
print('speed at annot / peak median',round(float(np.median(np.array(annspd)/np.array(peakspd))),2))
# policy: when is the racket closest to the target (nominal eval)
rows=collections.defaultdict(list)
for r in csv.DictReader(open('std_floor_cont_v1/final_eval948/tracking.csv',encoding='utf-8')):
    if r['time_to_impact_s']: rows[(r['clip'],r['start_time_s'])].append((float(r['time_to_impact_s']),float(r['target_position_error_m'])))
tm=[];errs=collections.defaultdict(list)
for (c,s),L in rows.items():
    L=[t for t in L if abs(t[0])<=0.3]
    if not any(abs(t[0])<1e-6 for t in L) or len(L)<20: continue
    tm.append(-min(L,key=lambda t:t[1])[0]); errs[ev[int(c)]].append([t[1] for t in L if abs(t[0])<1e-6][0])
tm=np.array(tm)
print('policy closest-approach time rel. hit (s): n',len(tm),'median',np.median(tm),'|.|<=0.04:',round((np.abs(tm)<=0.04+1e-6).mean()*100,1),'%','<=0.1:',round((np.abs(tm)<=0.1+1e-6).mean()*100,1),'%')
for g,v in errs.items(): print('nominal error at hit',g,len(v),'median cm',round(np.median(v)*100,1))

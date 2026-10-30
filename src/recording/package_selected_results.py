"""Validate all locally collected clips and generate a private review gallery."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--plan',required=True)
    a=p.parse_args();root=Path(a.root);plan=json.loads(Path(a.plan).read_text(encoding='utf-8'))
    expected={(s['parent_recording_id'],r['id']) for s in plan['segments'] for r in s['intervals']}
    rows=[];seen=set();types=Counter();checks=[]
    for meta in sorted((root/'downloads').glob('*/clips/*/metadata.json')):
        row=json.loads(meta.read_text(encoding='utf-8'));source=row['source']['recording_id'];ann=row['annotation']
        key=(source,ann['id'])
        if key not in expected or key in seen:raise ValueError('Unexpected/duplicate source annotation')
        seen.add(key);motion=meta.with_name('human_motion.npz');n=ann['end_frame']-ann['start_frame']
        with np.load(motion,allow_pickle=False) as arrays:
            assert arrays['global__body_pose'].shape==(n,63)
            assert arrays['joints_global'].shape==(n,22,3)
            assert float(arrays['fps'])==30 and all(np.isfinite(arrays[k]).all() for k in arrays.files)
            np.testing.assert_array_equal(arrays['parent_30Hz_frames'],np.arange(ann['start_frame'],ann['end_frame']))
            np.testing.assert_allclose(arrays['parent_elapsed_seconds'],arrays['parent_30Hz_frames']/30,rtol=0,atol=1e-12)
        local_run=meta.parents[2]
        review=local_run/row['segment_id']/'review.mp4'
        if not review.is_file():raise FileNotFoundError(review)
        types[ann['stroke_type']]+=1
        row.update(local_motion_path=motion.relative_to(root).as_posix(),local_review_video=review.relative_to(root).as_posix(),
            local_metadata_path=meta.relative_to(root).as_posix())
        rows.append(row);checks.append(dict(path=row['local_motion_path'],sha256=hashlib.sha256(motion.read_bytes()).hexdigest(),frames=n))
    missing=sorted(expected-seen)
    summary=dict(expected_user_intervals=len(expected),exported_user_intervals=len(rows),missing=missing,
        seconds=sum(r['quality']['duration_s'] for r in rows),stroke_types=dict(types),
        numerical_and_frame_mapping_verified=True,all_identity_verified=False,physical_quality_verified=False,
        G1_retargeted=False,public_distribution=False,
        low_direct_detection_under_95pct=sum(r['quality']['direct_detection_fraction']<.95 for r in rows))
    for name,value in [('catalog.json',rows),('checksums.json',checks),('summary.json',summary)]:
        with (root/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False)
    fields=['id','recording_id','annotation_id','type','start_s','end_s','frames','direct_detection_fraction','max_root_step_m','max_root_rotation_step_deg','target_hint','local_motion_path']
    with (root/'clip_matrix.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:
            ann=r['annotation'];q=r['quality']
            w.writerow(dict(id=r['id'],recording_id=r['source']['recording_id'],annotation_id=ann['id'],
                type=ann['stroke_type'],start_s=ann['start_frame']/30,end_s=ann['end_frame']/30,
                frames=q['frames'],direct_detection_fraction=q['direct_detection_fraction'],
                max_root_step_m=q['max_root_step_m'],max_root_rotation_step_deg=q['max_root_rotation_step_deg'],
                target_hint=r['target_hint'],local_motion_path=r['local_motion_path']))
    # Embedded catalog permits offline opening without fetch/CORS dependencies.
    compact=[dict(id=r['id'],recording=r['source']['recording_id'],type=r['annotation']['stroke_type'],
        hint=r['target_hint'],parent_start=r['annotation']['start_frame']/30,parent_end=r['annotation']['end_frame']/30,
        video=r['local_review_video'],start=r['review_local_seconds'][0],end=r['review_local_seconds'][1],
        direct=r['quality']['direct_detection_fraction'],step=r['quality']['max_root_step_m'],
        rotation=r['quality']['max_root_rotation_step_deg']) for r in rows]
    page='''<!doctype html><html lang="ja"><meta charset="utf-8"><title>選択動作の3D確認</title>
<style>body{font:16px system-ui;background:#141a22;color:#eee;margin:24px}video{width:100%;max-height:55vh;background:black}select,button{font:inherit;padding:8px;background:#283749;color:white;border:1px solid #65758a;border-radius:5px}td,th{padding:8px;border-bottom:1px solid #526172;text-align:left}table{width:100%;font-size:14px}p{color:#b5c4d7}.layout{max-width:1500px;margin:auto}a{color:#9de}</style>
<div class="layout"><h1>選択動作の3D確認</h1><p id="summary"></p><p>左：元映像、中央：選択人物の推定骨格、右：身体中心表示。座標は計算区間ごとに独立。数値検査と人物・姿勢の目視合格は別です。</p>
<video id="video" controls></video><div><button id="previous">前</button> <button id="next">次</button> <button id="replay">区間を再生</button> <label><input id="loop" type="checkbox" checked>繰返し</label> <span id="title"></span></div><p><select id="source"><option value="">全録画</option></select> <select id="kind"><option value="">全打球種</option></select> <select id="order"><option value="original">元の順</option><option value="direct">直接検出率が低い順</option><option value="rotation">姿勢急変が大きい順</option></select> <a href="clip_matrix.csv">区間表CSV</a></p>
<table><thead><tr><th>区間</th><th>打球種</th><th>元時刻</th><th>直接検出</th><th>人物メモ</th><th></th></tr></thead><tbody id="rows"></tbody></table></div><script>const DATA=__DATA__,SUMMARY=__SUMMARY__;
const $=id=>document.getElementById(id),v=$('video');let visible=DATA,index=0,active;
const labels={forehand:'フォア',backhand:'バック',serve:'サーブ',volley:'ボレー',overhead:'オーバーヘッド',unknown:'未分類'};
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
$('summary').textContent=`${SUMMARY.exported_user_intervals} / ${SUMMARY.expected_user_intervals}区間を保存・数値検査。未出力 ${SUMMARY.missing.length}区間。G1変換前。`;
for(const x of [...new Set(DATA.map(r=>r.recording))])$('source').insertAdjacentHTML('beforeend',`<option>${esc(x)}</option>`);
for(const x of [...new Set(DATA.map(r=>r.type))])$('kind').insertAdjacentHTML('beforeend',`<option value="${esc(x)}">${esc(labels[x]||x)}</option>`);
function play(i){if(!visible.length)return;index=(i+visible.length)%visible.length;active=visible[index];$('title').textContent=`${active.recording} / ${labels[active.type]} / ${active.parent_start.toFixed(2)}〜${active.parent_end.toFixed(2)}秒`;v.src=active.video;v.onloadedmetadata=()=>{v.currentTime=active.start;v.play().catch(()=>{})}}
function refresh(){visible=DATA.filter(r=>(!$('source').value||r.recording===$('source').value)&&(!$('kind').value||r.type===$('kind').value));if($('order').value==='direct')visible.sort((a,b)=>a.direct-b.direct);if($('order').value==='rotation')visible.sort((a,b)=>b.rotation-a.rotation);$('rows').innerHTML=visible.map((r,i)=>`<tr><td>${esc(r.id)}</td><td>${esc(labels[r.type])}</td><td>${r.parent_start.toFixed(2)}〜${r.parent_end.toFixed(2)}秒</td><td>${(r.direct*100).toFixed(1)}%</td><td>${esc(r.hint)}</td><td><button data-i="${i}">再生</button></td></tr>`).join('')}
for(const id of ['source','kind','order'])$(id).onchange=refresh;$('rows').onclick=e=>{if(e.target.dataset.i!==undefined)play(Number(e.target.dataset.i))};$('previous').onclick=()=>play(index-1);$('next').onclick=()=>play(index+1);$('replay').onclick=()=>play(index);
function tick(){if(active&&v.currentTime>=active.end){if($('loop').checked)v.currentTime=active.start;else v.pause()}}v.ontimeupdate=tick;if(v.requestVideoFrameCallback){const t=()=>{tick();v.requestVideoFrameCallback(t)};v.requestVideoFrameCallback(t)}refresh();</script></html>'''
    page=page.replace('__DATA__',json.dumps(compact,ensure_ascii=True).replace('</','<\\/')).replace('__SUMMARY__',json.dumps(summary,ensure_ascii=True))
    with (root/'review.html').open('x',encoding='utf-8') as f:f.write(page)
    print(json.dumps(summary))


if __name__=='__main__':main()

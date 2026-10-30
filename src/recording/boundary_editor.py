"""Loopback-only video boundary editor with append-only revision snapshots."""
import argparse
import csv
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import re
import threading
from urllib.parse import urlsplit, parse_qs

TYPES={'unknown','forehand','backhand','serve','volley','overhead','nonstroke'}
STATES={'candidate','hold','exclude'}
KINDS={'single_swing','continuous_motion','nonstroke'}


class Store:
    def __init__(self,manifest,data):
        self.manifest=json.loads(Path(manifest).read_text(encoding='utf-8'))
        self.records={r['recording_id']:r for r in self.manifest['recordings']}
        self.data=Path(data);self.data.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock()
        for rid,r in self.records.items():
            if not re.fullmatch(r'rec_[0-9a-f]+',rid):raise ValueError('Invalid recording ID')
            if r.get('media_kind')=='original_remote':
                url=urlsplit(r['video'])
                expected='/datasets/demalenk/caltennis/resolve/'+r['source']['revision']+'/'
                if url.scheme!='https' or url.netloc!='huggingface.co' or not url.path.startswith(expected):
                    raise ValueError('Only the pinned public CalTennis source is allowed')
            elif not Path(r['video']).is_file():raise FileNotFoundError(r['video'])
    def load(self,rid):
        if rid not in self.records:raise KeyError(rid)
        files=sorted((self.data/rid).glob('revision_*.json'))
        if files:return json.loads(files[-1].read_text(encoding='utf-8'))
        return dict(recording_id=rid,revision=0,intervals=self.records[rid].get('suggestions',[]))
    def save(self,rid,value):
        with self.lock:
            current=self.load(rid)
            if value.get('expected_revision')!=current['revision']:raise RuntimeError('revision_conflict')
            intervals=value.get('intervals')
            if not isinstance(intervals,list) or len(intervals)>10000:raise ValueError('Invalid intervals')
            record=self.records[rid];bound=record['frames']
            observed=value.get('observed_duration_s')
            if record.get('media_kind')=='original_remote':
                if type(observed) not in (int,float) or not math.isfinite(observed) or not 0<observed<=86400:
                    raise ValueError('動画の長さが読めてから保存してください')
                bound=round(observed*30)
            clean=[];ids=set()
            for r in intervals:
                lo=r.get('start_frame');hi=r.get('end_frame')
                if type(lo) is not int or type(hi) is not int or not 0<=lo<hi<=bound:
                    raise ValueError('開始・終了の順序または範囲を確認してください')
                if r.get('stroke_type') not in TYPES or r.get('status') not in STATES or r.get('kind') not in KINDS:
                    raise ValueError('Invalid labels')
                key=r.get('id','')
                if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',key) or key in ids:raise ValueError('Invalid interval ID')
                ids.add(key)
                clean.append(dict(id=key,start_frame=lo,end_frame=hi,start_s=lo/30,end_s=hi/30,
                    stroke_type=r['stroke_type'],status=r['status'],kind=r['kind'],
                    note=str(r.get('note',''))[:2000],user_boundary_confirmed=r.get('user_boundary_confirmed') is True,
                    origin=str(r.get('origin','user'))[:200],all_frames_reviewed=False))
            folder=self.data/rid;folder.mkdir(exist_ok=True)
            result=dict(recording_id=rid,revision=current['revision']+1,
                saved_at=datetime.now(timezone.utc).isoformat(),fps=30,frame_interval='[start_frame,end_frame)',
                video=self.records[rid]['video'],source=self.records[rid]['source'],
                recording_folder=self.records[rid]['recording_folder'],intervals=clean)
            result.update(media_kind=record.get('media_kind','comparison_local'),
                time_basis=('original_video_elapsed_seconds_on_30Hz_grid' if record.get('media_kind')=='original_remote'
                            else 'normalized_30fps_input'),observed_duration_s=observed,
                duration_basis='browser_metadata' if record.get('media_kind')=='original_remote' else 'completed_3D_frames')
            # Exclusive creation: previous revisions remain available for recovery.
            with (folder/f'revision_{result["revision"]:08d}.json').open('x',encoding='utf-8') as f:
                json.dump(result,f,indent=2,ensure_ascii=False);f.flush()
                import os
                os.fsync(f.fileno())
            return result


def make_handler(store,port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,body,ctype='application/json; charset=utf-8',code=200):
            if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False).encode('utf-8')
            self.send_response(code);self.send_header('Content-Type',ctype)
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
            if self.command!='HEAD':self.wfile.write(body)
        def do_HEAD(self):self.do_GET()
        def do_GET(self):
            path=urlsplit(self.path);query=parse_qs(path.query)
            try:
                if path.path=='/':return self.send(Path(__file__).with_name('boundary_editor.html').read_bytes(),'text/html; charset=utf-8')
                if path.path=='/api/manifest':
                    records=[{k:v for k,v in r.items() if k not in ('video','suggestions')} for r in store.records.values()]
                    return self.send(dict(recordings=records,storage=str(store.data.resolve())))
                rid=query.get('id',[''])[0]
                if rid not in store.records:return self.send(dict(error='Unknown recording'),code=404)
                if path.path=='/api/annotations':return self.send(store.load(rid))
                if path.path=='/api/export':
                    rows=store.load(rid)
                    if query.get('format',['json'])[0]=='csv':
                        output=io.StringIO();keys=['id','start_frame','end_frame','start_s','end_s','stroke_type','status','kind','user_boundary_confirmed','note']
                        writer=csv.DictWriter(output,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows['intervals'])
                        return self.send(('\ufeff'+output.getvalue()).encode('utf-8'),'text/csv; charset=utf-8')
                    return self.send(rows)
                if path.path=='/video':
                    record=store.records[rid]
                    if record.get('media_kind')=='original_remote':
                        self.send_response(302);self.send_header('Location',record['video'])
                        self.send_header('Content-Length','0');self.end_headers();return
                    return self.video(Path(record['video']))
                return self.send(dict(error='Not found'),code=404)
            except (ConnectionResetError,BrokenPipeError):pass
            except Exception as e:self.send(dict(error=str(e)),code=400)
        def video(self,path):
            size=path.stat().st_size;start=0;end=size-1;partial=False
            header=self.headers.get('Range')
            if header:
                match=re.fullmatch(r'bytes=(\d*)-(\d*)',header)
                if not match or not any(match.groups()):return self.send(dict(error='Invalid range'),code=416)
                left,right=match.groups()
                if left:start=int(left);end=min(int(right),end) if right else end
                else:start=max(0,size-int(right))
                if start>=size or end<start:
                    self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.end_headers();return
                partial=True
            self.send_response(206 if partial else 200);self.send_header('Content-Type','video/mp4')
            self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
            if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            if self.command=='HEAD':return
            with path.open('rb') as f:
                f.seek(start);remaining=end-start+1
                while remaining:
                    block=f.read(min(256*1024,remaining))
                    if not block:break
                    self.wfile.write(block);remaining-=len(block)
        def do_POST(self):
            if self.headers.get('X-Boundary-Editor')!='local' or self.headers.get('Origin') not in (None,f'http://127.0.0.1:{port}',f'http://localhost:{port}'):
                return self.send(dict(error='Local editor request required'),code=403)
            if urlsplit(self.path).path!='/api/annotations':return self.send(dict(error='Not found'),code=404)
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4*1024*1024:raise ValueError('Invalid request size')
                value=json.loads(self.rfile.read(length));result=store.save(value['recording_id'],value)
                self.send(result)
            except RuntimeError:self.send(dict(error='別タブで更新されています。再読み込みしてください。'),code=409)
            except Exception as e:self.send(dict(error=str(e)),code=400)
    return Handler


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--data',required=True)
    p.add_argument('--port',type=int,default=8771);a=p.parse_args()
    store=Store(a.manifest,a.data)
    server=ThreadingHTTPServer(('127.0.0.1',a.port),make_handler(store,a.port))
    print(f'http://127.0.0.1:{a.port}',flush=True);server.serve_forever()


if __name__=='__main__':main()

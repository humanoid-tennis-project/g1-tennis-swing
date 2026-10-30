"""Loopback-only impact annotation. Source/G1 data and old reviews are read-only."""
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit
from boundary_editor import make_handler as media_handler


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def map_event(row, source_frame, uncertainty):
    """Source frame is absolute on the existing 30 Hz annotation grid."""
    start, end = row['annotation']['start_frame'], row['annotation']['end_frame']
    if type(source_frame) is not int or not start <= source_frame < end:
        raise ValueError('打撃フレームが選択区間の外です')
    if type(uncertainty) is not int or not 0 <= uncertainty <= 15:
        raise ValueError('時刻の不確かさは0〜15フレームです')
    time_s = (source_frame-start)/30.0
    g1_frame = math.floor(time_s*50+0.5)
    return dict(source_frame=source_frame, human_local_frame=source_frame-start,
        review_time_s=row['review_local_seconds'][0]+time_s,
        clip_time_s=time_s, g1_frame=g1_frame, g1_time_s=g1_frame/50,
        time_rounding_error_s=g1_frame/50-time_s, uncertainty_source_frames=uncertainty)


class ImpactStore:
    def __init__(self, root, manifest, data):
        self.root, self.manifest, self.data = map(lambda p: Path(p).resolve(), (root, manifest, data))
        self.data.mkdir(parents=True, exist_ok=True)
        self.catalog_hash = digest(self.root/'catalog.json')
        self.manifest_hash = digest(self.manifest)
        catalog = {r['id']: r for r in json.loads((self.root/'catalog.json').read_text(encoding='utf-8'))}
        motions = json.loads(self.manifest.read_text(encoding='utf-8'))['motions']
        self.records = {}
        for motion in motions:
            key = motion['id']
            if key in self.records or '/' in key or '\\' in key or key in ('.', '..'):
                raise ValueError('Invalid/duplicate ID')
            if motion['identity_review'] != 'approved' or motion['pose_review'] != 'approved':
                raise ValueError('Unapproved source')
            row = dict(catalog[key])
            row['g1_motion_path'] = str((self.manifest.parent/motion['motion_file']).resolve())
            self.media_path(row['local_review_video'])
            self.records[key] = row
        self.lock = threading.Lock()

    def media_path(self, relative):
        path = (self.root/relative).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError('Invalid media')
        return path

    def load(self, key):
        if key not in self.records:
            raise KeyError(key)
        files = sorted((self.data/key).glob('revision_*.json'))
        if not files:
            return dict(id=key, revision=0, status='pending', events=[], note='')
        value = json.loads(files[-1].read_text(encoding='utf-8'))
        if value['catalog_sha256'] != self.catalog_hash or value['manifest_sha256'] != self.manifest_hash:
            raise ValueError('Input changed; annotation reconciliation required')
        return value

    def save(self, value):
        key = value['id']
        with self.lock:
            current = self.load(key)
            if type(value.get('expected_revision')) is not int or value['expected_revision'] != current['revision']:
                raise RuntimeError('revision_conflict')
            status = value.get('status')
            if status not in ('confirmed', 'uncertain', 'no_hit', 'pending'):
                raise ValueError('Invalid status')
            reviewer = value.get('reviewer', 'user')
            if reviewer not in ('user', 'codex'):
                raise ValueError('Invalid reviewer')
            events = value.get('events')
            if not isinstance(events, list) or len(events) > 40:
                raise ValueError('Invalid events')
            if (status == 'confirmed' and not events) or (status == 'no_hit' and events):
                raise ValueError('打撃なし・確認済みとイベント数が一致しません')
            row = self.records[key]
            clean, seen = [], set()
            for event in events:
                frame = event.get('source_frame')
                if event.get('evidence') not in ('visible_contact', 'audio_timing', 'estimated_swing'):
                    raise ValueError('Invalid evidence')
                if frame in seen:
                    raise ValueError('同じフレームが重複しています')
                seen.add(frame)
                clean.append(dict(**map_event(row, frame, event.get('uncertainty_source_frames', 1)),
                    evidence=event['evidence']))
            clean.sort(key=lambda e: e['source_frame'])
            result = dict(format='tennis_impact_annotation_v1', id=key, revision=current['revision']+1,
                status=status, events=clean, note=str(value.get('note', ''))[:2000], reviewer=reviewer,
                saved_at=datetime.now(timezone.utc).isoformat(), catalog_sha256=self.catalog_hash,
                manifest_sha256=self.manifest_hash, source=row['source'],
                source_interval=[row['annotation']['start_frame'], row['annotation']['end_frame']],
                stroke_type=row['annotation']['stroke_type'], source_time_basis=row['annotation_time_basis'],
                g1_motion_sha256=digest(row['g1_motion_path']),
                racket_state_provenance='G1 rigid attachment, not measured racket pose',
                ball_state_provenance='not measured', frame_indexing='zero-based')
            folder = self.data/key
            folder.mkdir(exist_ok=True)
            with (folder/f"revision_{result['revision']:08d}.json").open('x', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            return result

    def index(self):
        clips = []
        for key, row in self.records.items():
            clips.append(dict(id=key, type=row['annotation']['stroke_type'], recording=row['source']['recording_id'],
                source_start=row['annotation']['start_frame'], source_end=row['annotation']['end_frame'],
                media_start=row['review_local_seconds'][0], media_end=row['review_local_seconds'][1],
                annotation=self.load(key)))
        return dict(clips=clips, counts={s: sum(c['annotation']['status']==s for c in clips)
            for s in ('pending', 'confirmed', 'uncertain', 'no_hit')})


def make_handler(store, port):
    class Handler(media_handler(store, port)):
        def do_GET(self):
            url = urlsplit(self.path)
            try:
                if url.path == '/':
                    return self.send(Path(__file__).with_suffix('.html').read_bytes(), 'text/html; charset=utf-8')
                if url.path == '/api/index':
                    return self.send(store.index())
                if url.path == '/api/export':
                    return self.send(dict(format='tennis_impact_export_v1',
                        annotations=[store.load(key) for key in store.records]))
                key = parse_qs(url.query).get('id', [''])[0]
                if url.path == '/video' and key in store.records:
                    return self.video(store.media_path(store.records[key]['local_review_video']))
                return self.send(dict(error='Not found'), code=404)
            except (ConnectionResetError, BrokenPipeError):
                pass
            except Exception as error:
                self.send(dict(error=str(error)), code=400)

        def do_POST(self):
            if self.headers.get('X-Impact-Editor') != 'local' or self.headers.get('Origin') not in (
                None, f'http://127.0.0.1:{port}', f'http://localhost:{port}'):
                return self.send(dict(error='Local request required'), code=403)
            if urlsplit(self.path).path != '/api/annotation':
                return self.send(dict(error='Not found'), code=404)
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 32768:
                    raise ValueError('Invalid body size')
                self.send(store.save(json.loads(self.rfile.read(length))))
            except RuntimeError:
                self.send(dict(error='別タブで変更されました。再読み込みしてください。'), code=409)
            except Exception as error:
                self.send(dict(error=str(error)), code=400)
    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--port', type=int, default=8783)
    a = p.parse_args()
    store = ImpactStore(a.root, a.manifest, a.data)
    server = ThreadingHTTPServer(('127.0.0.1', a.port), make_handler(store, a.port))
    server.daemon_threads = True
    print(f'http://127.0.0.1:{a.port}/ | {len(store.records)} clips', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()

"""Local job server: upload a video from the browser, run the pipeline, poll progress.

  .venv-ns/Scripts/python.exe pipeline/server.py      (start.bat does this for you)

  POST /api/jobs?filename=walk.mp4&ceiling=2.75&title=My%20Home   body = raw video bytes
  GET  /api/jobs/<id>                                              -> status, step, log tail
  GET  /api/scenes                                                 -> reconstructed scenes
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / 'work' / 'uploads'
SCENES = ROOT / 'web' / 'public' / 'scenes'
PORT = int(os.environ.get('WAYFARER_PORT', 8765))
STEP_RE = re.compile(r'^\[(\d)/5\] (\w+)')
jobs = {}  # ponytail: in-memory, one machine, one user; persist to disk if jobs must survive restarts
lock = threading.Lock()  # one GPU -> one training job at a time


def slug(s):
    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')[:40] or 'scene'


def pipeline_env():
    env = os.environ.copy()
    scripts = Path(sys.executable).parent  # venv Scripts/bin: ns-process-data, ns-train, ns-export
    extra = [str(scripts)]
    if os.environ.get('WAYFARER_COLMAP'):
        extra.append(os.environ['WAYFARER_COLMAP'])
    if sys.platform == 'win32':  # PATH edits made after this process started live only in the registry
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as k:
                extra.append(os.path.expandvars(winreg.QueryValueEx(k, 'Path')[0]))
        except OSError:
            pass
    env['PATH'] = os.pathsep.join(extra + [env.get('PATH', '')])
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    return env


def run_job(job):
    job['status'] = 'queued'
    with lock:
        job['status'] = 'running'
        cmd = [sys.executable, str(ROOT / 'pipeline' / 'wayfarer.py'), job['video'], '--name', job['scene'], '--title', job['title']]
        if job['ceiling']:
            cmd += ['--ceiling', str(job['ceiling'])]
        if job['quality']:
            cmd += ['--quality', job['quality']]
        log = open(job['log'], 'w', encoding='utf-8', errors='replace')
        p = subprocess.Popen(cmd, cwd=ROOT, env=pipeline_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding='utf-8', errors='replace', bufsize=1)
        for line in p.stdout:
            log.write(line)
            log.flush()
            if m := STEP_RE.match(line):
                job['step'], job['stepName'] = int(m[1]), m[2]
        p.wait()
        log.close()
        job['status'] = 'done' if p.returncode == 0 else 'failed'
        job['finished'] = time.time()


def tail(path, n=40):
    try:
        text = Path(path).read_text(encoding='utf-8', errors='replace')
    except FileNotFoundError:
        return []
    lines = [l.split('\r')[-1] for l in text.splitlines()]  # progress bars redraw with \r: keep the last frame
    return [l for l in lines if l.strip()][-n:]


class Handler(BaseHTTPRequestHandler):
    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # keep the console for pipeline output
        pass

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == '/api/scenes':
            scenes = []
            for d in sorted(SCENES.iterdir()) if SCENES.exists() else []:
                f = d / 'scene.json'
                if f.exists():
                    s = json.loads(f.read_text(encoding='utf-8'))
                    scenes.append({'id': d.name, 'name': s.get('name', d.name), 'splat': bool(s.get('splat')), 'pois': len(s.get('pois', [])),
                                   'updated': f.stat().st_mtime})
            return self.send_json(scenes)
        if url.path.startswith('/api/jobs/'):
            job = jobs.get(url.path.rsplit('/', 1)[-1])
            if not job:
                return self.send_json({'error': 'no such job'}, 404)
            return self.send_json({k: v for k, v in job.items() if k not in ('video', 'log')} | {'log': tail(job['log'])})
        if url.path == '/api/jobs':
            return self.send_json([{k: v for k, v in j.items() if k not in ('video', 'log')} for j in jobs.values()])
        self.send_json({'error': 'not found'}, 404)

    def reject(self, msg, code=400):
        # drain the upload first: replying mid-body makes proxies drop the response
        left = int(self.headers.get('Content-Length', 0))
        while left > 0 and (chunk := self.rfile.read(min(left, 1 << 20))):
            left -= len(chunk)
        self.send_json({'error': msg}, code)

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != '/api/jobs':
            return self.reject('not found', 404)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        filename = q.get('filename', 'capture.mp4')
        ext = Path(filename).suffix.lower()
        if ext not in {'.mp4', '.mov', '.m4v', '.mkv', '.avi', '.webm'}:
            return self.reject(f'unsupported file type {ext or "(none)"}')
        try:
            ceiling = float(q['ceiling']) if q.get('ceiling') else None
        except ValueError:
            return self.reject('ceiling must be a number in metres')
        if ceiling is not None and not 1.8 <= ceiling <= 12:
            return self.reject('ceiling should be between 1.8 and 12 m')
        quality = q.get('quality') if q.get('quality') in ('fast', 'default', 'high') else None
        size = int(self.headers.get('Content-Length', 0))
        if size <= 0:
            return self.send_json({'error': 'empty upload'}, 400)

        jid = uuid.uuid4().hex[:8]
        title = q.get('title') or Path(filename).stem
        scene = f'{slug(title)}-{jid[:4]}'
        UPLOADS.mkdir(parents=True, exist_ok=True)
        video = UPLOADS / f'{scene}{ext}'
        with open(video, 'wb') as f:  # stream to disk: phone videos are often > 1 GB
            left = size
            while left > 0:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        if left > 0:
            video.unlink(missing_ok=True)
            return self.send_json({'error': 'upload interrupted'}, 400)

        job = {'id': jid, 'scene': scene, 'title': title, 'ceiling': ceiling, 'quality': quality, 'video': str(video),
               'log': str(UPLOADS / f'{scene}.log'), 'status': 'queued', 'step': 0, 'stepName': 'upload',
               'created': time.time(), 'sizeMB': round(size / 1e6, 1)}
        jobs[jid] = job
        threading.Thread(target=run_job, args=(job,), daemon=True).start()
        self.send_json({'id': jid, 'scene': scene}, 201)


if __name__ == '__main__':
    if not shutil.which('colmap', path=pipeline_env()['PATH']):
        print('WARNING: colmap not found on PATH (set WAYFARER_COLMAP to its folder). Uploads will fail at the sfm step.')
    print(f'Wayfarer job server on http://127.0.0.1:{PORT}')
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()

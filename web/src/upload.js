// Wayfarer Studio: upload a capture, follow the pipeline, open the twin.
const $ = (id) => document.getElementById(id);
const form = $('new-job');
let file = null;

function pick(f) {
  if (!f) return;
  file = f;
  $('drop-title').textContent = f.name;
  $('drop-sub').textContent = `${(f.size / 1e6).toFixed(0)} MB · ready`;
  if (!form.title.value) form.title.value = f.name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').slice(0, 40);
  $('go').disabled = false;
}
$('file').addEventListener('change', (e) => pick(e.target.files[0]));
const drop = $('drop');
['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => pick([...e.dataTransfer.files].find((f) => f.type.startsWith('video/')) ?? e.dataTransfer.files[0]));

form.addEventListener('submit', (e) => {
  e.preventDefault();
  if (!file) return;
  const q = new URLSearchParams({ filename: file.name, title: form.title.value.trim(), ceiling: form.ceiling.value, quality: form.quality.value });
  // XHR (not fetch) for upload progress on multi-GB videos
  const xhr = new XMLHttpRequest();
  xhr.open('POST', `/api/jobs?${q}`);
  $('go').disabled = true;
  $('upload-bar').hidden = false;
  xhr.upload.onprogress = (ev) => { $('upload-bar').firstElementChild.style.width = `${(ev.loaded / ev.total) * 100}%`; };
  xhr.onload = () => {
    $('upload-bar').hidden = true;
    const res = JSON.parse(xhr.responseText || '{}');
    if (xhr.status !== 201) { alertBox(res.error || `upload failed (${xhr.status})`); $('go').disabled = false; return; }
    try { localStorage.setItem('wayfarer:job', res.id); } catch { /* storage blocked */ }
    follow(res.id);
  };
  xhr.onerror = () => { $('upload-bar').hidden = true; $('go').disabled = false; $('server-down').hidden = false; };
  xhr.send(file);
});

function alertBox(msg) {
  $('job').hidden = false;
  $('job-title').textContent = 'Upload rejected';
  $('job-status').className = 'pill failed';
  $('job-status').textContent = 'error';
  $('log').textContent = msg;
}

let timer = null;
async function follow(id) {
  clearTimeout(timer);
  let job;
  try {
    const r = await fetch(`/api/jobs/${id}`);
    if (r.status === 404) { try { localStorage.removeItem('wayfarer:job'); } catch { /* */ } return; }
    job = await r.json();
  } catch { $('server-down').hidden = false; timer = setTimeout(() => follow(id), 4000); return; }
  $('server-down').hidden = true;
  $('job').hidden = false;
  $('job-title').textContent = job.title;
  $('job-status').className = `pill ${job.status}`;
  $('job-status').textContent = job.status === 'running' ? `step ${job.step}/5 · ${elapsed(job.created)}` : job.status;
  document.querySelectorAll('#steps li').forEach((li) => {
    const s = +li.dataset.step;
    li.className = job.status === 'done' || s < job.step ? 'done' : s === job.step && job.status === 'running' ? 'active' : '';
  });
  const log = $('log'), atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
  log.textContent = job.log.join('\n');
  if (atBottom) log.scrollTop = log.scrollHeight;
  $('open').hidden = job.status !== 'done';
  $('open').href = `./?scene=${encodeURIComponent(job.scene)}`;
  if (job.status === 'done' || job.status === 'failed') { loadScenes(); return; }
  timer = setTimeout(() => follow(id), 2000);
}

const elapsed = (t) => { const s = Math.round(Date.now() / 1000 - t); return `${Math.floor(s / 60)}m ${s % 60}s`; };

async function loadScenes() {
  let scenes = [];
  try { scenes = await (await fetch('/api/scenes')).json(); } catch { $('server-down').hidden = false; }
  if (!scenes.length) scenes = [{ id: 'demo', name: 'Demo Home (procedural)', splat: false, pois: 10 }];
  $('scenes').innerHTML = '';
  for (const s of scenes.sort((a, b) => (b.updated ?? 0) - (a.updated ?? 0))) {
    const li = document.createElement('li');
    li.innerHTML = '<div><div class="n"></div><small></small></div><a class="button primary">Open</a>';
    li.querySelector('.n').textContent = s.name;
    li.querySelector('small').textContent = `${s.splat ? '3D capture' : 'blueprint'} · ${s.pois} POIs · ?scene=${s.id}`;
    li.querySelector('a').href = `./?scene=${encodeURIComponent(s.id)}`;
    $('scenes').appendChild(li);
  }
}

loadScenes();
try { const id = localStorage.getItem('wayfarer:job'); if (id) follow(id); } catch { /* storage blocked */ }

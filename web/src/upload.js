// Wayfarer Studio: upload a capture, follow the pipeline, open the twin.
const $ = (id) => document.getElementById(id);
const form = $('new-job');
let files = [];

function pick(list) {
  const vids = [...list].filter((f) => f.type.startsWith('video/') || /\.(mp4|mov|m4v|mkv|webm|avi)$/i.test(f.name));
  files.push(...vids.filter((f) => !files.some((g) => g.name === f.name && g.size === f.size)));
  renderClips();
}
function renderClips() {
  $('clips').innerHTML = '';
  files.forEach((f, i) => {
    const li = document.createElement('li');
    li.innerHTML = '<span></span><small></small><button type="button" aria-label="Remove clip">✕</button>';
    li.firstChild.textContent = `${i + 1}. ${f.name}`;
    li.querySelector('small').textContent = `${(f.size / 1e6).toFixed(0)} MB`;
    li.querySelector('button').onclick = () => { files.splice(i, 1); renderClips(); };
    $('clips').appendChild(li);
  });
  const total = files.reduce((s, f) => s + f.size, 0);
  $('drop-title').textContent = files.length ? `${files.length} clip${files.length > 1 ? 's' : ''} · ${(total / 1e6).toFixed(0)} MB` : 'Drop your capture video(s) here';
  $('drop-sub').textContent = files.length > 1 ? 'clips must overlap at doorways; they merge into one space' : 'or click to choose · MP4 / MOV · several room clips merge into one space';
  if (files.length && !form.title.value) form.title.value = files[0].name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').slice(0, 40);
  $('go').disabled = !files.length;
}
$('file').addEventListener('change', (e) => { pick(e.target.files); e.target.value = ''; });
const drop = $('drop');
['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => pick(e.dataTransfer.files));

// XHR (not fetch) for upload progress on multi-GB videos
function uploadOne(f, done, total) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `/api/uploads?${new URLSearchParams({ filename: f.name })}`);
    xhr.upload.onprogress = (ev) => { $('upload-bar').firstElementChild.style.width = `${((done + ev.loaded) / total) * 100}%`; };
    xhr.onload = () => {
      const res = JSON.parse(xhr.responseText || '{}');
      xhr.status === 201 ? resolve(res.upload) : reject(new Error(res.error || `upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(Object.assign(new Error('server unreachable'), { down: true }));
    xhr.send(f);
  });
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!files.length) return;
  $('go').disabled = true;
  $('upload-bar').hidden = false;
  try {
    const total = files.reduce((s, f) => s + f.size, 0), ids = [];
    let done = 0;
    for (const f of files) { ids.push(await uploadOne(f, done, total)); done += f.size; }
    const r = await fetch('/api/jobs', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ uploads: ids, title: form.title.value.trim(), ceiling: form.ceiling.value, quality: form.quality.value, aiRepair: form.airepair.checked, maskPeople: form.maskpeople.checked }),
    });
    const res = await r.json();
    if (r.status !== 201) throw new Error(res.error || `could not start (${r.status})`);
    try { localStorage.setItem('wayfarer:job', res.id); } catch { /* storage blocked */ }
    files = []; renderClips();
    follow(res.id);
  } catch (err) {
    if (err.down) $('server-down').hidden = false; else alertBox(err.message);
    $('go').disabled = false;
  } finally {
    $('upload-bar').hidden = true;
  }
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
    li.querySelector('small').textContent = [s.splat ? '3D capture' : 'blueprint', `${s.pois} POIs`, s.psnr && `PSNR ${s.psnr} dB`, `?scene=${s.id}`].filter(Boolean).join(' · ');
    li.querySelector('a').href = `./?scene=${encodeURIComponent(s.id)}`;
    $('scenes').appendChild(li);
  }
}

loadScenes();
try { const id = localStorage.getItem('wayfarer:job'); if (id) follow(id); } catch { /* storage blocked */ }

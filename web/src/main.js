import * as THREE from 'three';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { parseGrid, inflate, findRoute, nextInstruction, toCell, isBlocked, routeLength } from './nav.js';
import { Player, isTyping } from './player.js';
import { Minimap } from './minimap.js';

const $ = (id) => document.getElementById(id);
const sceneName = new URLSearchParams(location.search).get('scene') || 'demo';
const sceneDir = `${import.meta.env.BASE_URL}scenes/${sceneName}/`;
const WALK_SPEED = 1.3; // m/s, for ETA

// ---------- three.js setup ----------
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
$('app').appendChild(renderer.domElement);
const labels = new CSS2DRenderer();
labels.setSize(innerWidth, innerHeight);
Object.assign(labels.domElement.style, { position: 'fixed', inset: '0', pointerEvents: 'none' });
document.body.insertBefore(labels.domElement, $('topbar'));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e1a2b);
const camera = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, 0.05, 200);
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight); labels.setSize(innerWidth, innerHeight);
});

let data, grid, margin, player, minimap, blueprint, splat;
let pois = [];
const poiObjects = new Map();
let route = null; // { target: {name,x,z}, pts, arrivedAt }
let selected = null;

boot().catch((e) => { console.error(e); $('status').textContent = `Failed to load scene "${sceneName}": ${e.message}`; });

async function boot() {
  data = await (await fetch(`${sceneDir}scene.json`)).json();
  $('scene-name').textContent = data.name;
  grid = parseGrid(data);
  margin = { inflated: inflate(grid, 0.3) };
  const floorY = data.floorY ?? 0;

  scene.add(new THREE.HemisphereLight(0xdfefff, 0x1b2a3c, 1.4));
  const sun = new THREE.DirectionalLight(0xffffff, 1.2); sun.position.set(5, 10, 3); scene.add(sun);
  blueprint = buildBlueprint(data, grid, floorY);
  scene.add(blueprint);

  if (data.splat) {
    $('status').textContent = 'Loading 3D capture…';
    const { SparkRenderer, SplatMesh } = await import('@sparkjsdev/spark');
    // sqrt(5) std-devs instead of sqrt(8): visually identical indoors, noticeably cheaper fill rate on laptops
    scene.add(new SparkRenderer({ renderer, maxStdDev: Math.sqrt(5) }));
    splat = new SplatMesh({
      url: new URL(data.splat.url, location.origin + sceneDir).href,
      lod: true, // builds a level-of-detail tree in a worker: fixed splat budget per frame on any GPU
      onProgress: (e) => { if (e.total) $('status').textContent = `Loading 3D capture… ${Math.round((e.loaded / e.total) * 100)}%`; },
    });
    splat.position.fromArray(data.splat.position);
    splat.quaternion.fromArray(data.splat.rotation);
    splat.scale.fromArray(data.splat.scale);
    scene.add(splat);
    await splat.initialized;
    setBlueprint(false); // splat is the visual; the grid stays for collision + nav
  }

  player = new Player(camera, renderer.domElement, grid, { eyeHeight: data.eyeHeight ?? 1.6, floorY });
  const sp = data.spawn ?? { x: grid.ox + grid.w * grid.cell / 2, z: grid.oz + grid.h * grid.cell / 2, yaw: 0 };
  player.teleport(sp.x, sp.z, sp.yaw);
  player.onLockChange = (locked) => {
    if (!locked) return;
    $('overlay').hidden = true;
    hide('poi-card'); hide('tag-form'); $('results').innerHTML = '';
  };
  minimap = new Minimap($('minimap'), grid, data.rows);

  pois = loadPois();
  pois.forEach(addPoiObject);
  wireUi();

  $('status').textContent = `${pois.length} points of interest · ${(grid.w * grid.cell).toFixed(1)} × ${(grid.h * grid.cell).toFixed(1)} m`;
  $('start').disabled = false;
  renderer.setAnimationLoop(frame);
}

// ---------- blueprint (procedural geometry from the grid) ----------
function buildBlueprint(d, g, floorY) {
  const group = new THREE.Group();
  const W = g.w * g.cell, H = g.h * g.cell;
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(W, H), new THREE.MeshStandardMaterial({ color: 0x15263c, roughness: 0.95 }));
  floor.rotation.x = -Math.PI / 2; floor.position.set(g.ox + W / 2, floorY, g.oz + H / 2);
  group.add(floor);
  const lines = new THREE.GridHelper(Math.max(W, H), Math.round(Math.max(W, H)), 0x2a4a6b, 0x1d3550);
  lines.position.set(g.ox + W / 2, floorY + 0.002, g.oz + H / 2);
  group.add(lines);

  const wallH = d.wallHeight ?? 2.6;
  const kinds = { '#': { h: wallH, color: 0x2f5a85 }, o: { h: 0.8, color: 0x4cc3e6 } };
  for (const [ch, { h, color }] of Object.entries(kinds)) {
    const cells = [];
    d.rows.forEach((row, z) => [...row].forEach((c, x) => { if (c === ch) cells.push([x, z]); })); // ' ' = unknown, not drawn
    if (!cells.length) continue;
    const mesh = new THREE.InstancedMesh(
      new THREE.BoxGeometry(g.cell, h, g.cell),
      new THREE.MeshStandardMaterial({ color, roughness: 0.7, transparent: true, opacity: ch === 'o' ? 0.55 : 0.92 }),
      cells.length,
    );
    const m = new THREE.Matrix4();
    cells.forEach(([x, z], i) => mesh.setMatrixAt(i, m.makeTranslation(g.ox + (x + 0.5) * g.cell, floorY + h / 2, g.oz + (z + 0.5) * g.cell)));
    group.add(mesh);
  }
  return group;
}
function setBlueprint(on) { blueprint.visible = on; }

// ---------- POIs ----------
const storeKey = `wayfarer:pois:${sceneName}`;
function loadPois() {
  try { const s = localStorage.getItem(storeKey); if (s) return JSON.parse(s); } catch { /* storage blocked */ }
  return structuredClone(data.pois ?? []);
}
function savePois() { try { localStorage.setItem(storeKey, JSON.stringify(pois)); } catch { /* storage blocked */ } }

function addPoiObject(p) {
  const el = document.createElement('div');
  el.className = `poi-label ${p.category}`;
  el.innerHTML = '<div class="pin"></div><div class="name"></div>';
  el.querySelector('.name').textContent = p.name;
  el.style.pointerEvents = 'auto';
  el.addEventListener('click', (e) => { e.stopPropagation(); openCard(p); });
  const obj = new CSS2DObject(el);
  obj.position.set(p.x, p.y ?? (data.floorY ?? 0) + 1.3, p.z);
  scene.add(obj);
  poiObjects.set(p.id, obj);
}
function removePoi(p) {
  pois = pois.filter((q) => q !== p);
  const o = poiObjects.get(p.id); o.element.remove(); scene.remove(o); poiObjects.delete(p.id);
  if (route?.target === p) stopRoute();
  savePois();
}

function openCard(p) {
  selected = p;
  player.unlock();
  $('poi-name').textContent = p.name;
  $('poi-desc').textContent = [p.category, p.desc].filter(Boolean).join(' · ');
  show('poi-card'); hide('tag-form');
  poiObjects.forEach((o, id) => o.element.classList.toggle('show', id === p.id));
}

// Where is the crosshair pointing? March the view ray over the occupancy grid until floor or obstacle.
function crosshairTarget() {
  const dir = new THREE.Vector3(); camera.getWorldDirection(dir);
  const p = camera.position.clone(), floorY = data.floorY ?? 0, step = 0.04;
  let last = p.clone();
  for (let t = 0; t < 25; t += step) {
    p.addScaledVector(dir, step);
    const [cx, cz] = toCell(grid, p.x, p.z);
    if (p.y <= floorY) return { x: p.x, z: p.z, y: floorY + 1.2 };
    if (isBlocked(grid, cx, cz)) return { x: last.x, z: last.z, y: Math.max(floorY + 0.4, p.y) };
    last = p.clone();
  }
  return null;
}

let pendingTag = null;
function startTag() {
  const hit = player.locked ? crosshairTarget() : { x: player.x, z: player.z, y: player.y - 0.3 };
  if (!hit) return;
  pendingTag = hit;
  player.unlock(); hide('poi-card'); show('tag-form');
  $('tag-form').name.value = ''; $('tag-form').desc.value = '';
  setTimeout(() => $('tag-form').name.focus(), 0);
}

// ---------- search ----------
function score(p, q) {
  const n = p.name.toLowerCase(), hay = `${n} ${p.desc ?? ''} ${p.category ?? ''}`.toLowerCase();
  if (n === q) return 100;
  if (n.startsWith(q)) return 80;
  if (n.split(/\W+/).some((w) => w.startsWith(q))) return 60;
  if (n.includes(q)) return 40;
  if (hay.includes(q)) return 20;
  let i = 0; for (const c of n) if (c === q[i]) i++; // loose subsequence match ("mstr bed")
  return i === q.length ? 10 : 0;
}
function search(q) {
  q = q.trim().toLowerCase();
  if (!q) return [];
  return pois.map((p) => [score(p, q), p]).filter(([s]) => s > 0).sort((a, b) => b[0] - a[0]).map(([, p]) => p).slice(0, 8);
}
let hits = [], hitIdx = 0;
function renderResults() {
  $('results').innerHTML = '';
  hits.forEach((p, i) => {
    const li = document.createElement('li');
    li.setAttribute('role', 'option'); li.setAttribute('aria-selected', i === hitIdx);
    const d = Math.hypot(p.x - player.x, p.z - player.z);
    li.innerHTML = '<span></span><small></small>';
    li.firstChild.textContent = p.name;
    li.lastChild.textContent = `${p.category} · ${d.toFixed(1)} m`;
    li.addEventListener('mousedown', (e) => { e.preventDefault(); pick(p); });
    $('results').appendChild(li);
  });
}
function pick(p) {
  $('q').value = ''; hits = []; renderResults(); $('q').blur();
  openCard(p);
  startRoute(p);
}

// ---------- navigation ----------
const arrowMesh = (() => {
  const s = new THREE.Shape();
  s.moveTo(0, -0.18); s.lineTo(0.16, 0.06); s.lineTo(0.08, 0.06); s.lineTo(0, -0.06); s.lineTo(-0.08, 0.06); s.lineTo(-0.16, 0.06); s.closePath();
  const geo = new THREE.ShapeGeometry(s); geo.rotateX(Math.PI / 2); // lie flat, tip toward -z
  const m = new THREE.InstancedMesh(geo, new THREE.MeshBasicMaterial({ color: 0xff8a3d, side: THREE.DoubleSide, transparent: true, opacity: 0.95, depthTest: false }), 400);
  m.renderOrder = 10; m.count = 0; m.frustumCulled = false;
  scene.add(m);
  return m;
})();

function startRoute(target) {
  route = { target, pts: null, arrivedAt: 0 };
  reroute();
  $('nav-hud').hidden = false;
}
function stopRoute() { route = null; arrowMesh.count = 0; $('nav-hud').hidden = true; }
function reroute() {
  if (!route) return;
  route.pts = findRoute(grid, [player.x, player.z], [route.target.x, route.target.z], margin);
}

const tmpM = new THREE.Matrix4(), tmpQ = new THREE.Quaternion(), tmpS = new THREE.Vector3(1, 1, 1), up = new THREE.Vector3(0, 1, 0);
function drawArrows(t) {
  const pts = route?.pts;
  if (!pts) { arrowMesh.count = 0; return; }
  const gap = 0.5, phase = (t * 0.8) % gap, y = (data.floorY ?? 0) + 0.03;
  let n = 0, carry = phase;
  for (let i = 1; i < pts.length && n < 400; i++) {
    const [ax, az] = pts[i - 1], [bx, bz] = pts[i], len = Math.hypot(bx - ax, bz - az);
    tmpQ.setFromAxisAngle(up, Math.atan2(-(bx - ax), -(bz - az)));
    for (let d = carry; d < len && n < 400; d += gap) {
      const k = d / len;
      arrowMesh.setMatrixAt(n++, tmpM.compose(new THREE.Vector3(ax + (bx - ax) * k, y, az + (bz - az) * k), tmpQ, tmpS));
    }
    carry = (carry - len) % gap; if (carry < 0) carry += gap;
  }
  arrowMesh.count = n;
  arrowMesh.instanceMatrix.needsUpdate = true;
}

function updateHud() {
  if (!route) return;
  if (!route.pts) { $('nav-text').textContent = 'No walkable path found'; $('nav-sub').textContent = route.target.name; return; }
  const ins = nextInstruction(route.pts, [player.x, player.z], player.yaw);
  if (!ins) return;
  $('nav-arrow').style.transform = `rotate(${(-ins.turn * 180) / Math.PI}deg)`;
  $('nav-text').textContent = ins.remaining < 0.6 ? `Arrived at ${route.target.name}` : ins.text;
  $('nav-sub').textContent = `${route.target.name} · ${ins.remaining.toFixed(1)} m · ~${Math.max(1, Math.round(ins.remaining / WALK_SPEED))} s`;
  if (ins.remaining < 0.6 && !route.arrivedAt) route.arrivedAt = performance.now();
  if (route.arrivedAt && performance.now() - route.arrivedAt > 4000) stopRoute();
}

// ---------- UI wiring ----------
function show(id) { $(id).hidden = false; }
function hide(id) { $(id).hidden = true; }

function wireUi() {
  $('start').addEventListener('click', () => player.lock());
  renderer.domElement.addEventListener('click', () => { if (!player.locked) { hide('poi-card'); player.lock(); } });

  addEventListener('keydown', (e) => {
    if (isTyping(e)) {
      if (e.key === 'Escape') e.target.blur();
      return;
    }
    if (e.code === 'Slash') { e.preventDefault(); player.unlock(); $('q').focus(); }
    else if (e.code === 'KeyT') startTag();
    else if (e.code === 'KeyF') player.toggleFly();
    else if (e.code === 'KeyG') setBlueprint(!blueprint.visible);
    else if (e.code === 'KeyM') $('minimap').classList.toggle('big');
    else if (e.code === 'KeyN') stopRoute();
  });

  $('q').addEventListener('input', () => { hits = search($('q').value); hitIdx = 0; renderResults(); });
  $('q').addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      hitIdx = (hitIdx + (e.key === 'ArrowDown' ? 1 : hits.length - 1)) % Math.max(1, hits.length);
      renderResults();
    } else if (e.key === 'Enter' && hits[hitIdx]) pick(hits[hitIdx]);
  });
  $('q').addEventListener('blur', () => setTimeout(() => { hits = []; renderResults(); }, 100));

  $('poi-go').addEventListener('click', () => { startRoute(selected); hide('poi-card'); player.lock(); });
  $('poi-jump').addEventListener('click', () => {
    const p = selected, r = findRoute(grid, [player.x, player.z], [p.x, p.z], margin);
    // land on the route ~1.2 m before the POI, facing it
    const pts = r ?? [[p.x, p.z]];
    let [x, z] = pts[pts.length - 1];
    if (pts.length > 1) {
      const [ax, az] = pts[pts.length - 2], len = Math.hypot(x - ax, z - az), k = Math.max(0, len - 1.2) / (len || 1);
      x = ax + (x - ax) * k; z = az + (z - az) * k;
    }
    player.teleport(x, z, Math.atan2(-(p.x - x), -(p.z - z)));
    hide('poi-card'); player.lock();
  });
  $('poi-del').addEventListener('click', () => { removePoi(selected); hide('poi-card'); });
  $('poi-close').addEventListener('click', () => hide('poi-card'));

  $('btn-tag').addEventListener('click', startTag);
  $('tag-cancel').addEventListener('click', () => hide('tag-form'));
  $('tag-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const f = e.target, name = f.name.value.trim();
    if (!name || !pendingTag) return;
    const p = { id: `${name.toLowerCase().replace(/\W+/g, '-')}-${Date.now().toString(36)}`, name, category: f.category.value, desc: f.desc.value.trim(), ...pendingTag };
    p.x = +p.x.toFixed(2); p.z = +p.z.toFixed(2); p.y = +p.y.toFixed(2);
    pois.push(p); addPoiObject(p); savePois();
    hide('tag-form'); openCard(p);
  });
  $('btn-export').addEventListener('click', () => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(pois, null, 2)], { type: 'application/json' }));
    a.download = `${sceneName}-pois.json`; a.click(); URL.revokeObjectURL(a.href);
  });

  $('nav-stop').addEventListener('click', stopRoute);
  $('minimap').addEventListener('click', (e) => {
    if (player.locked) return;
    const [x, z] = minimap.toWorld(e.offsetX, e.offsetY);
    startRoute({ name: 'Dropped pin', x, z });
  });
}

// ---------- frame loop ----------
let prevT = performance.now() / 1000;
let lastReroute = 0;
const fwd = new THREE.Vector3(), toP = new THREE.Vector3();
function frame() {
  const t = performance.now() / 1000, dt = Math.min(t - prevT, 0.05);
  prevT = t;
  player.update(dt);
  if (route && t - lastReroute > 0.25) { reroute(); lastReroute = t; }
  drawArrows(t);
  updateHud();

  // look-at-to-reveal: name the POI under the crosshair
  camera.getWorldDirection(fwd);
  let best = null, bestAng = 0.09;
  for (const p of pois) {
    const o = poiObjects.get(p.id);
    toP.subVectors(o.position, camera.position);
    const d = toP.length(), ang = fwd.angleTo(toP);
    if (d < 20 && ang < bestAng) { best = { p, d }; bestAng = ang; }
  }
  $('look-label').hidden = !best || !player.locked;
  if (best) $('look-label').textContent = `${best.p.name} · ${best.d.toFixed(1)} m`;

  minimap.draw(player, pois, route?.pts);
  renderer.render(scene, camera);
  labels.render(scene, camera);
}

// exposed for debugging / demo recording from the console
Object.assign(window, { wayfarer: { get player() { return player; }, get route() { return route; }, get splat() { return splat; }, routeLength } });

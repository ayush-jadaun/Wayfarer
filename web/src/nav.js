// Occupancy grid + A* pathfinding. Pure functions, no three.js, so it runs in tests.
// Grid rows are strings: '.' walkable, anything else blocked. Row index = z, column = x.

export function parseGrid({ rows, cellSize, origin }) {
  const h = rows.length, w = rows[0].length;
  const blocked = new Uint8Array(w * h);
  for (let z = 0; z < h; z++)
    for (let x = 0; x < w; x++) blocked[z * w + x] = rows[z][x] === '.' ? 0 : 1;
  return { w, h, cell: cellSize, ox: origin[0], oz: origin[1], blocked };
}

export const toCell = (g, x, z) => [Math.floor((x - g.ox) / g.cell), Math.floor((z - g.oz) / g.cell)];
export const toWorld = (g, cx, cz) => [g.ox + (cx + 0.5) * g.cell, g.oz + (cz + 0.5) * g.cell];

export function isBlocked(g, cx, cz, blocked = g.blocked) {
  return cx < 0 || cz < 0 || cx >= g.w || cz >= g.h || blocked[cz * g.w + cx] === 1;
}

// Is the world point (x,z) with body radius r free of walls?
export function isFree(g, x, z, r) {
  const [x0, z0] = toCell(g, x - r, z - r), [x1, z1] = toCell(g, x + r, z + r);
  for (let cz = z0; cz <= z1; cz++)
    for (let cx = x0; cx <= x1; cx++) {
      if (!isBlocked(g, cx, cz)) continue;
      // closest point of the cell to (x,z)
      const nx = Math.max(g.ox + cx * g.cell, Math.min(x, g.ox + (cx + 1) * g.cell));
      const nz = Math.max(g.oz + cz * g.cell, Math.min(z, g.oz + (cz + 1) * g.cell));
      if ((nx - x) ** 2 + (nz - z) ** 2 < r * r) return false;
    }
  return true;
}

// Grow obstacles by `radius` metres so paths keep the walker's body off walls.
export function inflate(g, radius) {
  const rc = Math.ceil(radius / g.cell), out = new Uint8Array(g.blocked);
  const offs = [];
  for (let dz = -rc; dz <= rc; dz++)
    for (let dx = -rc; dx <= rc; dx++) if (dx * dx + dz * dz <= rc * rc) offs.push([dx, dz]);
  for (let z = 0; z < g.h; z++)
    for (let x = 0; x < g.w; x++) {
      if (!g.blocked[z * g.w + x]) continue;
      for (const [dx, dz] of offs) {
        const nx = x + dx, nz = z + dz;
        if (nx >= 0 && nz >= 0 && nx < g.w && nz < g.h) out[nz * g.w + nx] = 1;
      }
    }
  return out;
}

// Breadth-first search for the closest free cell (snaps POIs / player standing in an inflated margin).
export function nearestFree(g, [cx, cz], blocked = g.blocked) {
  if (!isBlocked(g, cx, cz, blocked)) return [cx, cz];
  const seen = new Uint8Array(g.w * g.h), q = [[cx, cz]];
  for (let i = 0; i < q.length; i++) {
    const [x, z] = q[i];
    for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const nx = x + dx, nz = z + dz;
      if (nx < 0 || nz < 0 || nx >= g.w || nz >= g.h || seen[nz * g.w + nx]) continue;
      if (!blocked[nz * g.w + nx]) return [nx, nz];
      seen[nz * g.w + nx] = 1;
      q.push([nx, nz]);
    }
  }
  return null;
}

const DIRS = [[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1],
  [1, 1, Math.SQRT2], [1, -1, Math.SQRT2], [-1, 1, Math.SQRT2], [-1, -1, Math.SQRT2]];

// 8-connected A* with octile heuristic; diagonals may not cut wall corners. Returns cells or null.
export function astar(g, start, goal, blocked = g.blocked) {
  const N = g.w * g.h, s = start[1] * g.w + start[0], t = goal[1] * g.w + goal[0];
  if (blocked[s] || blocked[t]) return null;
  const gs = new Float64Array(N).fill(Infinity), from = new Int32Array(N).fill(-1), closed = new Uint8Array(N);
  const hx = (i) => {
    const dx = Math.abs((i % g.w) - goal[0]), dz = Math.abs(((i / g.w) | 0) - goal[1]);
    return Math.max(dx, dz) + (Math.SQRT2 - 1) * Math.min(dx, dz);
  };
  const heap = new Heap();
  gs[s] = 0;
  heap.push(s, hx(s));
  while (heap.size) {
    const i = heap.pop();
    if (i === t) break;
    if (closed[i]) continue;
    closed[i] = 1;
    const x = i % g.w, z = (i / g.w) | 0;
    for (const [dx, dz, c] of DIRS) {
      const nx = x + dx, nz = z + dz;
      if (isBlocked(g, nx, nz, blocked)) continue;
      if (dx && dz && (isBlocked(g, x + dx, z, blocked) || isBlocked(g, x, z + dz, blocked))) continue;
      const j = nz * g.w + nx, ng = gs[i] + c;
      if (ng < gs[j]) { gs[j] = ng; from[j] = i; heap.push(j, ng + hx(j)); }
    }
  }
  if (s !== t && from[t] === -1) return null;
  const path = [];
  for (let i = t; i !== -1; i = from[i]) path.push([i % g.w, (i / g.w) | 0]);
  return path.reverse();
}

// Can a straight line between two cells be walked without touching a blocked cell? (supercover walk)
export function lineOfSight(g, [x0, z0], [x1, z1], blocked = g.blocked) {
  let dx = Math.abs(x1 - x0), dz = Math.abs(z1 - z0), x = x0, z = z0;
  const sx = x1 > x0 ? 1 : -1, sz = z1 > z0 ? 1 : -1;
  let err = dx - dz;
  for (let n = dx + dz; n > 0; n--) {
    const e2 = 2 * err;
    if (e2 > -dz && e2 < dx) { // passes exactly through a corner: both neighbours must be free
      if (isBlocked(g, x + sx, z, blocked) || isBlocked(g, x, z + sz, blocked)) return false;
      err += dx - dz; x += sx; z += sz; n--;
    } else if (e2 > -dz) { err -= dz; x += sx; }
    else { err += dx; z += sz; }
    if (isBlocked(g, x, z, blocked)) return false;
  }
  return true;
}

// Drop intermediate cells when a straight walk is possible -> natural-looking corners only.
export function smooth(g, path, blocked = g.blocked) {
  if (!path || path.length < 3) return path;
  const out = [path[0]];
  let anchor = 0;
  for (let i = 2; i < path.length; i++)
    if (!lineOfSight(g, path[anchor], path[i], blocked)) { anchor = i - 1; out.push(path[anchor]); }
  out.push(path[path.length - 1]);
  return out;
}

// World-space route from (x,z) to (x,z). Keeps a body-radius margin, falls back to raw grid for tight doors.
export function findRoute(g, from, to, margin) {
  for (const blocked of [margin.inflated, g.blocked]) {
    const a = nearestFree(g, toCell(g, ...from), blocked), b = nearestFree(g, toCell(g, ...to), blocked);
    if (!a || !b) continue;
    const cells = astar(g, a, b, blocked);
    if (!cells) continue;
    const pts = smooth(g, cells, blocked).map(([cx, cz]) => toWorld(g, cx, cz));
    pts[0] = from;
    pts[pts.length - 1] = to;
    return pts;
  }
  return null;
}

export const routeLength = (pts) => pts.reduce((d, p, i) => i ? d + Math.hypot(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1]) : 0, 0);

// Turn-by-turn hint for a walker at `pos` facing `yaw` (radians, three.js convention: forward = -z).
export function nextInstruction(pts, pos, yaw) {
  if (!pts || pts.length < 2) return null;
  const remaining = routeLength([pos, ...pts.slice(1)]);
  if (remaining < 0.6) return { text: 'You have arrived', remaining, turn: 0 };
  const [tx, tz] = pts[1];
  const want = Math.atan2(-(tx - pos[0]), -(tz - pos[1]));
  let turn = want - yaw;
  turn = Math.atan2(Math.sin(turn), Math.cos(turn)); // wrap to [-pi, pi]
  const dist = Math.hypot(tx - pos[0], tz - pos[1]);
  const deg = (turn * 180) / Math.PI;
  const text = Math.abs(deg) < 25 ? `Go straight ${dist.toFixed(1)} m`
    : Math.abs(deg) > 150 ? 'Turn around'
      : `${deg > 0 ? 'Turn left' : 'Turn right'}${Math.abs(deg) < 60 ? ' slightly' : ''}`;
  return { text, remaining, turn };
}

class Heap {
  constructor() { this.k = []; this.p = []; }
  get size() { return this.k.length; }
  push(k, p) {
    const { k: K, p: P } = this;
    let i = K.length;
    K.push(k); P.push(p);
    while (i > 0) {
      const u = (i - 1) >> 1;
      if (P[u] <= p) break;
      K[i] = K[u]; P[i] = P[u]; i = u;
    }
    K[i] = k; P[i] = p;
  }
  pop() {
    const { k: K, p: P } = this, top = K[0], k = K.pop(), p = P.pop();
    if (K.length) {
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= K.length) break;
        if (c + 1 < K.length && P[c + 1] < P[c]) c++;
        if (P[c] >= p) break;
        K[i] = K[c]; P[i] = P[c]; i = c;
      }
      K[i] = k; P[i] = p;
    }
    return top;
  }
}

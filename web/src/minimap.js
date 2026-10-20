// Top-down map: grid baked once into an offscreen canvas, player/POIs/route drawn per frame.
export class Minimap {
  constructor(canvas, grid, rows) {
    this.canvas = canvas; this.g = grid;
    this.base = document.createElement('canvas');
    this.base.width = grid.w; this.base.height = grid.h;
    const ctx = this.base.getContext('2d'), img = ctx.createImageData(grid.w, grid.h);
    const pal = { '.': [24, 44, 70, 230], o: [76, 195, 230, 255], '#': [235, 240, 245, 255], ' ': [0, 0, 0, 0] };
    rows.forEach((row, z) => [...row].forEach((c, x) => img.data.set(pal[c] ?? pal['#'], (z * grid.w + x) * 4)));
    ctx.putImageData(img, 0, 0);
  }

  // world (x,z) -> canvas px, fitted with padding
  fit() {
    const c = this.canvas, dpr = devicePixelRatio || 1;
    const W = c.clientWidth * dpr, H = c.clientHeight * dpr;
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    const s = Math.min((W - 16 * dpr) / this.g.w, (H - 16 * dpr) / this.g.h);
    return { s, ox: (W - s * this.g.w) / 2, oy: (H - s * this.g.h) / 2, dpr };
  }

  toPx(x, z, f) { return [f.ox + ((x - this.g.ox) / this.g.cell) * f.s, f.oy + ((z - this.g.oz) / this.g.cell) * f.s]; }

  toWorld(px, py) {
    const f = this.fit(), d = f.dpr;
    return [this.g.ox + ((px * d - f.ox) / f.s) * this.g.cell, this.g.oz + ((py * d - f.oy) / f.s) * this.g.cell];
  }

  draw(player, pois, route) {
    const f = this.fit(), ctx = this.canvas.getContext('2d');
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.base, f.ox, f.oy, this.g.w * f.s, this.g.h * f.s);

    if (route) {
      ctx.strokeStyle = '#ff8a3d'; ctx.lineWidth = 3 * f.dpr; ctx.lineJoin = 'round';
      ctx.beginPath();
      route.forEach(([x, z], i) => ctx[i ? 'lineTo' : 'moveTo'](...this.toPx(x, z, f)));
      ctx.stroke();
    }
    for (const p of pois) {
      const [px, py] = this.toPx(p.x, p.z, f);
      ctx.fillStyle = p.category === 'item' ? '#4cc3e6' : '#ff8a3d';
      ctx.beginPath(); ctx.arc(px, py, 3.5 * f.dpr, 0, 7); ctx.fill();
      if (this.canvas.classList.contains('big')) {
        ctx.fillStyle = '#f3efe6'; ctx.font = `${12 * f.dpr}px system-ui`;
        ctx.fillText(p.name, px + 6 * f.dpr, py + 4 * f.dpr);
      }
    }
    // player: arrow pointing along view direction
    const [px, py] = this.toPx(player.x, player.z, f), r = 7 * f.dpr;
    ctx.save(); ctx.translate(px, py); ctx.rotate(-player.yaw);
    ctx.fillStyle = '#fff'; ctx.strokeStyle = '#0e1a2b'; ctx.lineWidth = 1.5 * f.dpr;
    ctx.beginPath(); ctx.moveTo(0, -r); ctx.lineTo(r * 0.7, r * 0.7); ctx.lineTo(0, r * 0.3); ctx.lineTo(-r * 0.7, r * 0.7); ctx.closePath();
    ctx.fill(); ctx.stroke(); ctx.restore();
  }
}


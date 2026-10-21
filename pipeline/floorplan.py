"""Export a twin's occupancy grid as a clean vector floor plan (SVG): walls, furniture, floor, POIs, scale.

  python pipeline/floorplan.py web/public/scenes/home      -> web/public/scenes/home/floorplan.svg
"""
import html
import json
import sys
from pathlib import Path

PX = 40  # pixels per metre


def runs(rows, ch):
    """Horizontal runs of `ch` per row: [(z, x0, x1)] -> far fewer SVG rects than one per cell."""
    out = []
    for z, row in enumerate(rows):
        x = 0
        while x < len(row):
            if row[x] == ch:
                x0 = x
                while x < len(row) and row[x] == ch:
                    x += 1
                out.append((z, x0, x))
            else:
                x += 1
    return out


def floorplan(scene):
    rows, cell = scene['rows'], scene['cellSize']
    W, H = len(rows[0]), len(rows)
    k = cell * PX  # px per cell
    pad, head = 30, 60
    w, h = max(W * k + 2 * pad, 460), H * k + 2 * pad + head + 64
    ox, oz = scene['origin']
    to_px = lambda x, z: (pad + (x - ox) / cell * k, head + pad + (z - oz) / cell * k)
    rect = lambda z, x0, x1, fill: f'<rect x="{pad + x0 * k:.1f}" y="{head + pad + z * k:.1f}" width="{(x1 - x0) * k:.1f}" height="{k + 0.4:.1f}" fill="{fill}"/>'

    walk = sum(r.count('.') for r in rows) * cell * cell
    pois = scene.get('pois', [])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" viewBox="0 0 {w:.0f} {h:.0f}" '
        'font-family="Segoe UI, Arial, sans-serif">',
        f'<rect width="{w:.0f}" height="{h:.0f}" fill="#f7f5f0"/>',
        f'<text x="{pad}" y="34" font-size="22" font-weight="700" fill="#14202e">{html.escape(scene.get("name", "Floor plan"))}</text>',
        f'<text x="{pad}" y="54" font-size="12" fill="#5b6b7c">Wayfarer floor plan · {walk:.0f} m² walkable · '
        f'{len(pois)} points of interest · {"generated from the 3D capture" if scene.get("splat") else "procedural layout"}</text>',
        '<g>', *(rect(z, a, b, '#e3ecf3') for z, a, b in runs(rows, '.')), '</g>',
        '<g>', *(rect(z, a, b, '#8fb8d0') for z, a, b in runs(rows, 'o')), '</g>',
        '<g>', *(rect(z, a, b, '#14202e') for z, a, b in runs(rows, '#')), '</g>',
    ]
    for p in pois:
        x, y = to_px(p['x'], p['z'])
        color = '#e8702a' if p.get('category') != 'item' else '#1d8fb3'
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" stroke="#fff" stroke-width="1.5"/>')
        parts.append(f'<text x="{x + 8:.1f}" y="{y + 4:.1f}" font-size="11" font-weight="600" fill="#14202e" '
                     f'paint-order="stroke" stroke="#f7f5f0" stroke-width="3">{html.escape(p["name"])}</text>')
    # scale bar: 1 m ticks up to 5 m (or the plan width)
    n = max(1, min(5, int(W * cell)))
    sy = h - 50
    parts.append(f'<g stroke="#14202e" stroke-width="2"><line x1="{pad}" y1="{sy}" x2="{pad + n * PX}" y2="{sy}"/>'
                 + ''.join(f'<line x1="{pad + i * PX}" y1="{sy - 5}" x2="{pad + i * PX}" y2="{sy + 5}"/>' for i in range(n + 1)) + '</g>')
    parts.append(f'<text x="{pad + n * PX + 10}" y="{sy + 4}" font-size="12" fill="#14202e">{n} m</text>')
    legend = [('#14202e', 'wall'), ('#8fb8d0', 'furniture'), ('#e3ecf3', 'walkable'), ('#e8702a', 'room'), ('#1d8fb3', 'item')]
    for i, (c, label) in enumerate(legend):  # own row under the scale bar
        parts.append(f'<rect x="{pad + i * 80}" y="{sy + 17}" width="12" height="12" fill="{c}"/>'
                     f'<text x="{pad + i * 80 + 17}" y="{sy + 27}" font-size="11" fill="#14202e">{label}</text>')
    parts.append('</svg>')
    return '\n'.join(parts)


if __name__ == '__main__':
    d = Path(sys.argv[1])
    svg = floorplan(json.loads((d / 'scene.json').read_text(encoding='utf-8')))
    (d / 'floorplan.svg').write_text(svg, encoding='utf-8')
    print(f'  floor plan -> {d / "floorplan.svg"}')

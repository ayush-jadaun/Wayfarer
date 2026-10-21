# Wayfarer

**Google Maps walking directions, but indoors.** Wayfarer turns an ordinary smartphone video of an indoor space into a photorealistic, walkable 3D digital twin in the browser. You can explore it in first person, search for rooms and items, and follow obstacle-aware walking routes.

Built for **Namma Space, Techfest 2026-27, IIT Bombay**.

```
 phone video ──► sharp-frame picker ──► COLMAP SfM ──► Gaussian splatting ──► navgrid ──► web twin
  (SOP)            ffmpeg + Laplacian     camera poses    nerfstudio / gsplat     floor, metric    three.js
                   blur rejection         + sparse cloud  on the GPU              scale, walls,    walk · search
                                                                                   walkable grid    A* routes
```

## What you get

| | |
|---|---|
| **Photoreal twin** | 3D Gaussian Splatting, rendered in real time in any desktop browser with nothing to install |
| **First-person roaming** | WASD + mouse look, run, collision with real walls and furniture, fly mode for a dollhouse view |
| **Robust camera poses** | COLMAP run directly with GPU SIFT and exhaustive matching (sequential + loop closure for long captures); the largest model is kept and registration is gated |
| **Metric, level model** | Floor found with RANSAC; scale from a printed ArUco marker (about 1%), cross-checked against the tape-measured ceiling and camera height |
| **Walkable grid** | Obstacles sliced 0.12–1.9 m above the floor; height-aware free-space carving from rendered depth removes floaters; doorways the camera walked through always stay open |
| **Self-labelling twin** | A COCO object detector finds beds, sofas, TVs, fridges and more in the frames and places them in 3D as searchable POIs |
| **Measured quality** | Every twin is scored on held-out views (PSNR/SSIM/LPIPS); the numbers ship with the scene |
| **POIs + search** | Tag rooms and items at the crosshair; ranked fuzzy search; look at a POI to reveal its name |
| **Navigation** | A* on the grid with a body-radius safety margin, line-of-sight smoothing, animated floor arrows, live re-routing and turn-by-turn hints |
| **Fast web viewer** | [Spark](https://sparkjs.dev) renderer with level-of-detail (fixed splat budget per frame) and SOG compression: about 11x smaller than PLY, with full view-dependent colour |

## Quick start (viewer only)

```bash
cd web
npm install
npm run dev          # http://localhost:5173  -> procedural demo home
```

Open `?scene=<name>` to load a reconstructed scene from `web/public/scenes/<name>/`.

| Key | Action | Key | Action |
|---|---|---|---|
| `W A S D` | move | Mouse | look |
| `Shift` | run | `F` | fly / walk |
| `/` | search | `T` | tag a POI at the crosshair |
| `M` | big map (click it to route there) | `G` | blueprint overlay |
| `N` | stop route | `Esc` | free the mouse |
| `B` | edit map (paint/erase walls) | `P` | auto tour (record with Win+Alt+R) |

## Capture → twin

1. Film the space following the **[capture SOP](docs/SOP.md)**.
2. Install the pipeline once: **[docs/SETUP.md](docs/SETUP.md)**.
3. Double-click **`start.bat`**, drop the video (or several overlapping room clips) into **Wayfarer Studio**, and press *Build twin*. You'll see live progress through each step, then an *Open twin* button.

Or from the command line:

```bash
python pipeline/wayfarer.py walk.mp4 --name home --ceiling 2.75
```

This writes `web/public/scenes/home/` (`scene.sog`, `scene.json`, and a `grid.png` preview). Each step is resumable, and `--force <step>` re-runs from that step.

## Repository layout

```
pipeline/
  wayfarer.py       one-command orchestrator (frames → sfm → train → export → navgrid)
  navgrid.py        splat → floor-aligned metric occupancy grid + SOG + scene.json
  marker_scale.py   metric scale from a printed ArUco marker (triangulated from the COLMAP poses)
  depthcarve.py     height-aware free-space carving from rendered splat depth
  autopoi.py        object detection -> 3D POIs
  train_mcmc.py     gsplat MCMC trainer (memory-capped fallback: --trainer mcmc)
  server.py         local job server behind the upload studio (stdlib only)
  test_navgrid.py   synthetic-room self-check (python pipeline/test_navgrid.py)
  test_marker_scale.py  rendered-marker self-check
web/
  src/nav.js        occupancy grid, A*, smoothing, routing, turn instructions (pure, unit-tested)
  src/player.js     first-person controller with grid collision
  src/minimap.js    top-down map
  src/main.js       scene loading, splat rendering, POIs, search, navigation UI
  upload.html       Wayfarer Studio: drag-and-drop capture upload with live pipeline progress
  test/             node:test unit tests (npm test)
  public/scenes/    one folder per twin; demo/ is a procedural sample
docs/
  SOP.md            capture rulebook
  marker.svg        printable A4 scale marker
```

### `scene.json`

```jsonc
{
  "name": "Home", "cellSize": 0.1, "origin": [x0, z0],   // grid placement in metres
  "rows": ["  ###..", "..."],                            // '.' walkable, '#' wall, 'o' furniture, ' ' unknown
  "floorY": 0, "wallHeight": 2.75, "eyeHeight": 1.6,
  "spawn": { "x": 1.2, "z": 3.4, "yaw": 0 },
  "splat": { "url": "scene.sog", "position": [...], "rotation": [x, y, z, w], "scale": [s, s, s] },
  "pois": [{ "id": "kitchen", "name": "Kitchen", "category": "room", "x": 4.1, "z": 2.0 }]
}
```

The grid is plain text on purpose: you can fix a wall the scan missed in any text editor.

## Tests

```bash
cd web && npm test                  # pathfinding, collision, instructions
python pipeline/test_navgrid.py     # floor/scale/wall recovery on a synthetic splat
.venv-ns/Scripts/python pipeline/test_marker_scale.py   # ArUco scale recovery on rendered views
```

## Credits

Open-source software we build on:
- [three.js](https://threejs.org) (MIT)
- [Spark](https://github.com/sparkjsdev/spark) (MIT)
- [splat-transform](https://github.com/playcanvas/splat-transform) (MIT)
- [OpenCV](https://opencv.org) (Apache-2.0)
- [nerfstudio](https://nerf.studio) (Apache-2.0)
- [gsplat](https://github.com/nerfstudio-project/gsplat) (Apache-2.0)
- [COLMAP](https://colmap.github.io) (BSD)
- [FFmpeg](https://ffmpeg.org) (LGPL)
- [Vite](https://vitejs.dev) (MIT)

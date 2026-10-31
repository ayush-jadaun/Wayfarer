# Sample data

`sample-room/frames/`: 111 frames (960 px) of one room, taken from a 2 min 23 s phone video (1080p, 30 fps)
filmed following [docs/SOP.md](../docs/SOP.md). They are every second frame the pipeline's sharp-frame picker kept.

Rebuild a walkable twin from them (setup: [docs/SETUP.md](../docs/SETUP.md)):

```bash
python pipeline/wayfarer.py test/sample-room/frames --name sample-room --quality fast --trainer mcmc
cd web && npm run dev        # then open http://localhost:5173/?scene=sample-room
```

`--quality fast` (7k steps) takes about 15 minutes on an 8 GB laptop GPU. Drop it for the full-quality run
(about an hour). The twin built from the full video is committed as `web/public/scenes/vid1028/`.

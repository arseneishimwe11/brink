# Brink

Brink finds dangerous ground shapes from one handheld phone video of a pit,
quarry face, soil pile or excavation: undercut bases, overhangs, faces too
steep for their material, cracks along the top edge and water at the foot.

It reports what it measured, with a confidence level. It never says ground is
free of danger, and it is not an engineering assessment.
Background and roadmap: [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md).

**Status: Phase 0.** Single-capture hazard engine and benchmark. All accuracy
numbers so far come from synthetic scenes; see [docs/RESULTS.md](docs/RESULTS.md).

## Set up

```bash
bash scripts/setup_sandbox.sh
```

```bash
. .venv/bin/activate
```

Runs on a CPU. No GPU is needed.

## Check one video

```bash
brink run path/to/video.mp4 --material damp_soil --site "North pit"
```

Outputs go to `data/derived/runs/<video name>/out/`:

| File | What it is |
|---|---|
| `report.pdf` | One-page inspection record |
| `report.json` | The same content for software, with a SHA-256 of each other file |
| `hazards.jpg` | A frame from the video with the hazards drawn on it |
| `model.ply` | 3D model coloured by hazard |
| `viewer/index.html` | 3D viewer; open it in a browser straight from disk |

`--material` picks the slope limit from the rules file
(`src/brink/rules/default_rules.yaml`). **Those rules are provisional** until a
qualified engineer approves them.

If a file named `<video>.sensors.json` sits next to the video, Brink reads the
direction of gravity from it. Otherwise it estimates "up" from the footage,
which is less accurate.

## Benchmark

```bash
brink bench all
```

reruns every tier on the frozen test split and rewrites `docs/RESULTS.md`.
Individual tiers, for development:

```bash
brink bench geometry --split dev
```

```bash
brink bench full --split dev
```

- `geometry`: hazard code alone, on surfaces sampled from known scenes.
- `full`: rendered phone videos through the whole pipeline.
- `dev` scenes are for tuning. `test` scenes are frozen and only reported.

Every experiment is logged in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Real videos

Follow [docs/TEST_PROTOCOL.md](docs/TEST_PROTOCOL.md) and put the videos in
`data/raw/`. Nothing under `data/derived/` is committed; it is regenerated.

## Layout

```
src/brink/
  domain/     Site, Capture, Reconstruction, Face, Hazard, Report
  stages/     frames, sfm (cameras), dense, orient (up and scale)
  hazards/    faces, profiles, undercut and overhang, engine
  rules/      thresholds by material (provisional)
  pipeline/   runs the stages in order, with caching
  outputs/    PDF, JSON, hazard image, coloured model, viewer
  synth/      synthetic scenes and rendered videos with known truth
  bench/      benchmark tiers, metrics and the results writer
```

Licences of every dependency: [docs/LICENSES.md](docs/LICENSES.md).

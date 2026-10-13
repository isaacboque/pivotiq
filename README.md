# PivotIQ

Automated futsal **moment-intelligence** tool. PivotIQ takes a match video and
(in the finished product) produces an annotated video plus a report of shots,
goals and chances with build-up clips.

> **Status: Milestone 1 (skeleton).** The CLI, config, data models and pitch
> geometry exist and are tested. The analysis pipeline itself is stubbed:
> `pivotiq analyze` parses all flags and prints the run configuration, but does
> not process video yet. Detection, tracking, registration, events and reports
> land in milestones 2–5.

PivotIQ lives next to the legacy `futsal-analytics` tool (YOLO + ByteTrack + KPIs,
see the original README) in the same repo and package distribution (`iafs`).

## Planned pipeline

```
video ─► ingest ─► detect (players + ball) ─► track ─► teams
                        │                                │
                        └────► register (pitch keypoints → homography)
                                         │
                                         ▼
                                events (shots / goals / chances) ─► render + report
```

Package layout (`src/pivotiq/`):

| Module | State | Purpose |
|--------|-------|---------|
| `cli.py` | working (stub pipeline) | `pivotiq analyze`, `pivotiq version` (Typer) |
| `config.py` | working | Dataclass config, optional TOML override |
| `models/` | working | Pydantic models: `BBox`, `Detection`, `Frame`, `PlayerTrack`, `BallState`, `Homography`, `Event` |
| `geometry/` | working | Futsal pitch geometry, 29 canonical keypoints, point-in-polygon, drawing helpers |
| `ingest/`, `detect/`, `track/`, `teams/`, `register/`, `events/`, `render/`, `report/` | placeholder | Planned milestones 2–5 |

## Installation

```bash
git clone https://github.com/isaacboque/pivotiq.git
cd pivotiq
git checkout training          # PivotIQ code currently lives on this branch

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Usage

```bash
pivotiq version
pivotiq analyze match.mp4 \
    --ball-weights runs/ball_detect/train/weights/best.pt \
    --keypoint-weights runs/keypoint_detect/train/weights/best.pt \
    --out-video out/annotated.mp4 \
    --report-dir out/report \
    --device auto \
    --max-frames 3000
```

| Flag | Description | Default |
|------|-------------|---------|
| `--ball-weights` | Fine-tuned ball detector (`.pt`). Without it, a pretrained YOLO is used (weaker on small futsal balls) | fallback YOLO |
| `--keypoint-weights` | Pitch keypoint detector (`.pt`). Without it, an identity homography is used (pixels treated as pitch coordinates) | mock registration |
| `--out-video` | Annotated MP4 output | — |
| `--report-dir` | Directory for PNG/JSON/clip outputs | — |
| `--device` | `auto`, `cpu`, `cuda` or `mps` | `auto` |
| `--max-frames` | Process only N frames (smoke tests) | all |
| `--config` | TOML file overriding built-in defaults | — |
| `-v`, `--verbose` | DEBUG logging | off |

## Configuration

All thresholds live in `pivotiq.config.PivotIQConfig` and can be overridden from
a TOML file; unset keys keep their defaults.

```toml
[detection]
player_confidence = 0.40
ball_confidence = 0.30

[tracking]
track_buffer = 30

[registration]
min_visible_keypoints = 4

[events]
shot_speed_threshold_mps = 5.0
chance_xg_threshold = 0.20
```

Sections: `detection`, `tracking` (ByteTrack), `registration` (RANSAC homography,
temporal smoothing), `events` (shot / chance thresholds).

## Training the detectors

Two small YOLO models are fine-tuned on your own labelled frames.

| Model | Script | Dataset config | Task |
|-------|--------|----------------|------|
| Ball detector | `training/ball_detector/train.py` | `dataset.yaml` (1 class: `ball`) | Detection |
| Pitch keypoint detector | `training/keypoint_detector/train.py` | `dataset.yaml` (1 class `pitch`, 29 keypoints) | YOLO-Pose |

Workflow:

1. Extract frames from video (see `docs/YOUTUBE_WORKFLOW.md`).
2. Annotate in Roboflow or similar (see `docs/ANNOTATION_GUIDE.md`).
3. Unzip the export into `data/ball_detector/` or `data/keypoint_detector/`.
4. Train, e.g. `python training/ball_detector/train.py --epochs 100 --device auto`.
5. Pass the resulting `best.pt` to `pivotiq analyze`.

Dataset guidance: roughly 400 labelled source frames for the ball (about 1,200 after
augmentation, ~20 % frames with no ball visible) and 500+ frames for keypoints with
varied pan and zoom. More detail in `docs/TRAINING.md`.

## Tests

```bash
pytest tests/pivotiq -v
```

70 unit tests cover the data models, pitch geometry and CLI.

## License

MIT — see `pyproject.toml`.

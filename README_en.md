# ReID using OpenVINO

[简体中文](README.md) | English

[![Python](https://img.shields.io/badge/Python-3.7%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-Inference%20Engine-0071C5?logo=intel&logoColor=white)](https://www.intel.cn/content/www/cn/zh/developer/tools/openvino-toolkit/overview.html)
[![Stars](https://img.shields.io/github/stars/yiyiliu0415/ReID_using-openvino?style=flat&color=yellow)](https://github.com/yiyiliu0415/ReID_using-openvino/stargazers)

**Give it one photo of a person, and it finds every moment that person appears in hours of surveillance footage.**

Detection, multi-object tracking and person re-identification are chained into a single pipeline, and every
inference runs on Intel OpenVINO — so it works on an ordinary office PC **without a dedicated GPU**. The
output is a timestamped text report telling you which camera and which second the target showed up.

![pipeline preview](common/show_pic/image.png)

---

## Contents

- [Why this repo](#why-this-repo)
- [How it works](#how-it-works)
- [Setup](#setup)
- [Quick start](#quick-start)
- [Arguments](#arguments)
- [Output](#output)
- [Repository layout](#repository-layout)
- [FAQ](#faq)
- [Acknowledgements](#acknowledgements)

---

## Why this repo

| | |
|---|---|
| **Runs on CPU** | Detection and ReID feature extraction both use OpenVINO IR models — usable frame rates on Intel CPUs |
| **Track first, identify later** | The tracker links detections into trajectories, then features are matched per trajectory instead of per frame |
| **Frame skipping** | `--skip N` processes every N-th frame, so long videos scale linearly in time |
| **Reproducible** | Intermediate results are cached as `detections.npy` / `rm.npy`; tune the decision logic without re-running inference |
| **Multi-camera** | `-v` accepts several videos and merges them onto a single timeline |

## How it works

```
video ──► person detection (OpenVINO Detector) ──► multi-object tracking (MCT / SCT)
                                                          │
                                        per-frame bbox ──► ReID features (hr18)
                                                          │
target photo ──► gallery features ──────────────────────► cosine similarity
                                                          │
                                            top-k hits ──► timestamped report
```

- **Detection** — `Detector` in `utils/network_wrappers.py` wraps the OpenVINO Inference Engine and loads
  `model/detection.xml` by default. Wrappers for SSD, RetinaFace, YOLO, CenterNet and others live in
  `common/python/models/` if you want to swap the detector.
- **Tracking** — `mc_tracker/mct.py` (multi-camera, cross-camera association) and `mc_tracker/sct.py`
  (single camera), orchestrated by `Tracker` in `tracker.py`.
- **ReID** — `model/hr18.xml` (ResNet18 backbone) embeds representative crops of each trajectory and
  compares them against the target photo.
- **Decision** — `--top_k` keeps the k most similar trajectories; `path_use.py` converts frame indices to
  `HH:MM:SS` and writes the report.

## Setup

- Python 3.7+
- Intel OpenVINO ([install guide](https://www.intel.cn/content/www/cn/zh/developer/tools/openvino-toolkit/overview.html))
- Dependencies:

```bash
pip install -r requirements.txt
```

> `requirements.txt` pins `numpy<1.20` and `scipy~=1.5.4`. Use a fresh virtualenv to avoid conflicts with
> an existing scientific stack.

### Models and sample data

Model weights are not included. Models, a sample video and a sample target image are packaged on
[Baidu Netdisk](https://pan.baidu.com/s/1nkM6b2jDnjJHJTUZV-qGDw?pwd=oj8k) (code `oj8k`). Unzip over the
project root:

```
model/
├── detection.xml / detection.bin   # person detector
└── hr18.xml / hr18.bin             # ReID feature extractor
```

## Quick start

### 1. Create `configs/configs.json`

This file maps each video to a location and a wall-clock start time, so frame indices can be turned into
real timestamps. A supermarket camera `0.mp4` that starts at 8:00 becomes:

```json
{
    "place": {
        "supermarket": {
            "0": 28800
        }
    }
}
```

`"0"` is the file name without extension, `28800` is the start time in seconds (8 × 60 × 60).

### 2. Add videos and the target photo

```bash
mkdir -p video_input target_input
# video:        video_input/0.mp4
# target photo: target_input/xxx.jpg  (one person, clear face or full body works best)
```

### 3. Run

```bash
python ReID_main.py -v ./video_input -t ./target_input \
    --data_output ./data_output --top_k 50 --skip 10
```

### 4. Reuse cached inference

```bash
python ReID_main.py -v ./video_input -t ./target_input \
    --is_only_txt True --npy_path ./data_output/2 \
    --top_k 50 --skip 10
```

## Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `-v, --video_input` | required | Input video(s), accepts multiple |
| `-t, --target_input` | required | Target photo(s), accepts multiple |
| `--data_output` | `./data_output/2` | Output directory |
| `--npy_path` | `./data_output/2` | Cached `.npy` directory, used with `--is_only_txt` |
| `--json_path` | `./configs/configs.json` | Location / start-time config |
| `-m, --m_detector` | `./model/detection.xml` | Detection model (OpenVINO IR) |
| `--m_reid_xml` / `--m_reid_bin` | `./model/hr18.xml` / `.bin` | ReID model |
| `--config` | `./configs/person.py` | Scenario config; use `configs/vehicle.py` for vehicles |
| `--t_detector` | `0.95` | Detection confidence threshold — lower it if you miss people |
| `--top_k` | `-1` | Keep top-k trajectories; **both too large and too small hurt accuracy**, 30–50 is a good range |
| `--skip` | `10` | Process every N-th frame; use `1` for no skipping |
| `-d, --device` | `CPU` | Inference device (`CPU` / `GPU` / `MYRIAD`) |
| `-l, --cpu_extension` | — | Custom layer CPU extension (legacy OpenVINO only) |
| `-u, --utilization_monitors` | — | Bound hardware monitors, e.g. `GPU,0` |
| `--is_save` | `True` | Save the annotated video |
| `--is_tracker` | `True` | Enable tracking (disabling it compares every frame and is much slower) |
| `--is_txt` | `True` | Write the text report |
| `--is_only_txt` | `False` | Skip inference, build the report from cached `.npy` |
| `--loop` | `False` | Loop the input video |

## Output

```
data_output/
├── message/            # report: camera / time range / similarity
├── *.npy               # detections.npy, rm.npy, ...
└── *.mp4               # annotated video (when --is_save True)
```

A line such as `[supermarket][0] 08:13:24 - 08:14:02  score: 0.87` means the target appeared in camera 0 of
the supermarket between 08:13:24 and 08:14:02.

## Repository layout

```
ReID_using-openvino/
├── ReID_main.py           # entry point: argument parsing + pipeline
├── tracker.py             # Tracker: tracking + ReID matching
├── path_use.py            # time conversion, report writing
├── visualize.py           # result visualisation
├── configs/
│   ├── person.py          # person scenario (tracker hyper-parameters)
│   └── vehicle.py         # vehicle scenario
├── mc_tracker/
│   ├── mct.py             # multi-camera tracker
│   └── sct.py             # single-camera tracker
├── utils/
│   ├── network_wrappers.py  # OpenVINO detector wrapper
│   ├── video.py             # multi-source capture, CLAHE normalisation
│   ├── visualization.py     # drawing, frame stacking
│   └── ...
└── common/python/         # OpenVINO demo utilities and model wrappers
```

## FAQ

**`configs/configs.json` is missing.**
It is not committed — create it as shown in step 1. The pipeline still runs without it, but reported times
stay relative to the video.

**One person is split into several trajectories.**
Tune `mct_config.global_match_thresh` (cross-camera) and `sct_config.match_threshold` (within a camera) in
`configs/person.py`. Lower values link trajectories more aggressively.

**The target is never found.**
Check, in order: `--t_detector` too high (missed detections), `--top_k` too small, a large viewpoint or
lighting gap between the photo and the footage, and whether `--skip` jumps over short appearances.

**High memory usage.**
Split long videos, or raise `--skip`. Model wrappers in `common/python` are loaded on demand.

## Acknowledgements

The detection and tracking scaffolding follows Intel OpenVINO's multi-camera-multi-person-tracking example;
the ReID retrieval and timestamped reporting are built on top of it.

Issues and PRs are welcome — a star is appreciated if this saved you some time.

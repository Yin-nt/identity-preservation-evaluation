# Evaluating Identity Preservation in Face Restoration

Code accompanying the paper *"Degradation-Dependent Robustness of Identity
Verification from Blind Face Restoration and
Image-to-Video Generation Pipelines"*.

The study benchmarks general-purpose video-generation models (Wan 2.1 / Wan 2.2)
against dedicated face-restoration models (GFPGAN, OSDFace) for **identity
preservation** under five degradation types at three severity levels, using
ArcFace (InsightFace) and FaceNet512 (DeepFace) as recognition backbones.

## Pipeline

The scripts are organised in the order they are run:

| Stage | Folder | What it does |
|-------|--------|--------------|
| 1. Data preparation | `01_data_preparation/` | Select frontal/side identities from LFW and build genuine/impostor verification pairs. |
| 2. Degradations | `02_degradations/` | Apply the five paper degradations to the clean inputs. |
| 3. Generation | `03_generation_wan/` | Pack degraded images into Wan2GP image-to-video queues. |
| 4. Evaluation | `04_evaluation/` | Extract identity cosine similarity, then compute bootstrap metrics and EER analysis. |

> Restoration baselines (GFPGAN, OSDFace) were run from each model's own
> official repository and are not bundled here.

### 1. Data preparation
- `filter_faces_by_pose.py` — estimate head pose with InsightFace and select up
  to 500 shared identities, each contributing four distinct photos: an
  independent source + reference pair for the frontal protocol, and a
  separate independent source + reference pair for the side protocol (no
  photo shared between the two protocols). Writes `lfw_frontal_test.csv` and
  `lfw_side_test.csv`.
- `build_verification_pairs.py` — turn a per-identity CSV into balanced genuine
  (label=1) and impostor (label=0) pairs.

**Pose protocol** (as implemented in `filter_faces_by_pose.py`; copy these
values into the manuscript's Methods section):
- Estimator: InsightFace `buffalo_l`, yaw/pitch from `face.pose`.
- Side candidate: `|yaw| > 25°` or `|pitch| > 20°`.
- Frontal candidate: `|yaw| < 10°` and `|pitch| < 10°`.
- An identity qualifies once it has at least one side candidate and at least
  three frontal candidates; it then appears in both protocols with four
  distinct photos, ranked by Laplacian sharpness among the frontal
  candidates: the sharpest becomes the frontal protocol's `test_path`
  (frontal source), the second-sharpest becomes the frontal protocol's
  `ref_path` (frontal reference), the third-sharpest becomes the side
  protocol's `ref_path` (side reference), and the strongest off-axis
  candidate becomes the side protocol's `test_path` (side source). Photos between the frontal and side thresholds (e.g. `15° < |yaw| < 25°`).


### 2. Degradations (paper severity levels L1/L2/L3)
| Script | Degradation | Levels |
|--------|-------------|--------|
| `gaussian_blur.py` | Gaussian blur | sigma ∈ {10, 12, 15} |
| `motion_blur.py` | Motion blur | L ∈ {12, 22, 35} |
| `down_up_scaling.py` | Down-up scaling | r ∈ {5, 8, 12} |
| `salt_pepper.py` | Salt-and-pepper (impulse) noise — pure, no Gaussian/blur | p ∈ {0.005, 0.012, 0.025} |
| `low_light.py` | Low light + Poisson noise | (0.25, 90), (0.10, 30), (0.05, 15) |

Each script reads an index CSV and writes degraded images plus an output CSV
containing a `Source_Path` column.

### 3. Generation
- `build_queue.py` — pack degraded images into chunked `.zip` queues with a
  `queue.json` manifest consumed by [Wan2GP](https://github.com/deepbeepmeep/Wan2GP)
  for image-to-video generation. The shared Wan i2v job configuration is loaded
  from an external `queue_template.json` (kept out of source control); copy
  `queue_template.example.json` and fill in your settings.

### 4. Evaluation
- `extract_similarity_arcface.py` — InsightFace/ArcFace cosine extraction
  (with video representative-frame selection) **and** bootstrap in one script
  (two modes: `extract` and `scores`). Uses the `buffalo_l` model pack by
  default.
- `extract_similarity_deepface.py` — the same extraction with FaceNet512
  (DeepFace), detector backend `retinaface`.
- `bootstrap_metrics.py` — Accuracy/Precision/Recall/F1/AUC + 95% bootstrap CIs
  from existing score CSVs.
- `eer_analysis.py` — per-experiment and unified EER, the 0.15 threshold
  sensitivity sweep (Section 4.3), and ROC/sensitivity plots.

**Frame selection (reference-free).** For Wan videos, the first/last 25%
of frames are dropped, every remaining frame in the candidate set is
checked (`--frame-step 1` / `FRAME_STRIDE = 1` by default), and the *single*
frame whose best face has the highest RetinaFace/detector confidence is kept
as the representative frame.
**Threshold.** A unified cosine threshold of `0.15` is used across architectures
(see `eer_analysis.py`).

## Setup

```bash
pip install -r requirements.txt
```

### Data layout

All scripts read and write under a single dataset root, resolved from the
`IPE_DATA_ROOT` environment variable (default: `data/`):

```bash
export IPE_DATA_ROOT=/path/to/your/data
```

The evaluation covers two facial poses (`frontal` and `side`). The degradation
scripts are pose-parametrised via the `IPE_POSE` environment variable, so the
same code runs both poses without edits:

```bash
IPE_POSE=frontal python 02_degradations/gaussian_blur.py
IPE_POSE=side    python 02_degradations/gaussian_blur.py
```

Each pose reads `lfw_<pose>_test.csv` and writes under `image/<pose>/...` and
`csv/<pose>/...`.

Expected sub-structure under the root (created as needed):

```
$IPE_DATA_ROOT/
├── input/lfw/lfw-deepfunneled/   # raw LFW images
├── lfw_*_test.csv                # identity/pair index CSVs
├── csv/<pose>/...                # per-degradation index CSVs
├── image/                        # degraded images
├── video/<pose>/<experiment>/    # generated Wan videos
└── reports/<backbone>/<pose>/    # extracted scores + metrics
```

This repo's own top-level `reports_legacy/` folder holds a consolidated raw
results snapshot (all conditions merged into a handful of CSVs) from an this pipeline run

Per-script defaults (input/output CSV names, prefixes) sit in a configuration
block near the top of each file, or are exposed via `argparse` where noted.

## Data

LFW is publicly available: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset

## License

Released under the MIT License (see `LICENSE`).

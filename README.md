# Evaluating Identity Preservation in Face Restoration

The study benchmarks general-purpose image-to-video generation models (Wan 2.1 /
Wan 2.2) against dedicated blind face-restoration models (GFPGAN, OSDFace,
CodeFormer) for **identity preservation** under five degradation types at three
severity levels, using ArcFace (InsightFace) and FaceNet512 (DeepFace) as
recognition backbones.

## Pipeline

The scripts are organised in the order they are run:

| Stage | Folder | What it does |
|-------|--------|--------------|
| 1. Data preparation | `01_data_preparation/` | Select frontal/side identities from LFW and build genuine/impostor verification pairs. |
| 2. Degradations | `02_degradations/` | Apply the five paper degradations to the clean inputs. |
| 3. Generation | `03_generation_gfpgan/`, `03_generation_osdface/`, `03_generation_codeformer/`, `03_generation_wan/` | Run the three restoration models on the degraded images; pack degraded images into Wan2GP image-to-video queues. |
| 4. Evaluation | `04_evaluation/`, `04_evaluation_model_i2i/` | Extract identity embeddings / cosine similarity (Wan and restoration outputs), then compute threshold-independent verification and identification metrics with bootstrap intervals. |
| Results | `results/main/` | Per-cell metrics for all pipelines under both backbones. |

> Each restoration runner wraps the model's official code and released weights
> (installed from the model's own repository).

### 1. Data preparation
- `filter_faces_by_pose.py` — estimate head pose with InsightFace and select up
  to 500 shared identities, each contributing four distinct photos: an
  independent source + reference pair for the frontal protocol, and a
  separate independent source + reference pair for the side protocol (no
  photo shared between the two protocols). Writes `lfw_frontal_test.csv` and
  `lfw_side_test.csv`.
- `build_verification_pairs.py` — turn a per-identity CSV into balanced genuine
  (label=1) and impostor (label=0) pairs.

**Pose protocol** (as implemented in `filter_faces_by_pose.py`):
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
  candidate becomes the side protocol's `test_path` (side source). Photos
  between the frontal and side thresholds (e.g. `15° < |yaw| < 25°`) are
  neither frontal nor side candidates and are not used.


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
- `03_generation_gfpgan/run_gfpgan.py` — GFPGAN v1.4 (`GFPGANv1.4.pth`) on every
  degraded image listed in the degradation CSVs.
- `03_generation_osdface/run_osdface.py` — OSDFace one-step restoration on a
  Stable Diffusion 2.1 base (`--pretrained-model-path`, `--ckpt-path`,
  `--mixed-precision fp16`).
- `03_generation_codeformer/run_codeformer.py` — CodeFormer, fidelity weight
  0.5, no upscaling. Two passes per folder: detect-and-align mode first; images
  with no detected face are resized to 512×512 and re-run with `--has_aligned`.
  The mode used for each image (`detect` / `aligned` / `failed`) is logged in
  `codeformer_pairs.csv`.
- `03_generation_wan/build_queue.py` — pack degraded images into chunked `.zip` queues with a
  `queue.json` manifest consumed by [Wan2GP](https://github.com/deepbeepmeep/Wan2GP)
  for image-to-video generation. The shared Wan i2v job configuration is loaded
  from an external `queue_template.json` (kept out of source control); copy
  `queue_template.example.json` and fill in your settings.

### 4. Evaluation
- `04_evaluation/compute_verification_metrics.py` — computes the metrics
  reported in the paper from per-pipeline embedding files. For every cell
  (backbone × pose × degradation × level) the 250 genuine pairs are scored
  against **all 62,250 cross-identity impostor pairs** (250 × 249), giving AUC,
  EER, TAR at FAR = 1% and 0.1%, closed-set Rank-1 against the 250 references,
  the genuine-to-nearest-impostor margin, and the number of identities closer
  to another identity. Candidates without an embedding are kept and always
  rejected (`--fill -1`). Writes `metrics.csv`, `per_identity.csv`,
  `ci_per_condition.csv` and `bootstrap.csv` (identity-resampled bootstrap).
- `04_evaluation/extract_similarity_arcface.py` — InsightFace/ArcFace cosine
  extraction for Wan videos (with representative-frame selection) **and**
  bootstrap in one script (two modes: `extract` and `scores`). Uses the
  `buffalo_l` model pack by default.
- `04_evaluation/extract_similarity_deepface.py` — the same extraction with
  FaceNet512 (DeepFace), detector backend `retinaface`.
- `04_evaluation_model_i2i/` — the same ArcFace / FaceNet512 extraction and
  bootstrap for the restoration outputs (GFPGAN, OSDFace).
- `bootstrap_metrics.py`, `eer_analysis.py` — fixed-threshold metrics
  (Accuracy/F1 at a cosine threshold of 0.15) and EER analysis on score CSVs.
  The paper makes no claim from fixed-threshold metrics; they are kept for the
  threshold diagnostics only.

**Frame selection (reference-free).** Each Wan sequence has 18 frames. The
primary Wan condition is a **fixed temporal position, frame 9 of 18**
(`wan21_fixed`, `wan22_fixed`), so every pipeline contributes exactly one
candidate per input. Two secondary conditions measure what selection
contributes: a frame drawn uniformly at random from the central region
(`wan21_random`, `wan22_random`), and detector-based selection, where the
first/last 25% of frames are dropped and the frame whose best face has the
highest RetinaFace confidence is kept.

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

### Results

`results/main/all_pipelines_metrics.csv` holds one row per cell × condition
(2 backbones × 2 poses × 5 degradations × 3 levels). `source` is the
evaluation run (`wan` or `i2i`); `condition` is the candidate image
(`GT` = clean source, `unprocessed` = degraded input, `gfpgan`, `osdface`,
`codeformer`, `wan21_fixed`, `wan22_fixed`, `wan21_random`, `wan22_random`).
Columns follow `compute_verification_metrics.py`.

Per-script defaults (input/output CSV names, prefixes) sit in a configuration
block near the top of each file, or are exposed via `argparse` where noted.

## Data

LFW is publicly available: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset

## License

Released under the MIT License (see `LICENSE`).

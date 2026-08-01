# -*- coding: utf-8 -*-
"""Cosine similarity + bootstrap with InsightFace / ArcFace.

Two modes:

MODE scores - score CSVs already exist:
    CSV columns: Person, Label, Baseline, Wan2.1, GroundTruth
    -> Only reads the scores and bootstraps; no model is run.

    python extract_similarity_arcface.py --mode scores --input reports/insightface/side --recursive

MODE extract - recompute embeddings with InsightFace from source CSV + videos:
    Source CSV needs: label, person, ref_path, Source_Path, test_path
    Video path: video/<pose>/<experiment>/<basename(Source_Path)>.mp4

    python extract_similarity_arcface.py --mode extract --csv-dir csv/frontal/negativeSample \\
        --video-dir video/frontal --reports-dir reports/insightface/frontal --groups motion

Group filtering examples:
    python extract_similarity_arcface.py --mode extract --groups motion
    python extract_similarity_arcface.py --mode extract --groups motion_L2,motion_L3
    python extract_similarity_arcface.py --mode scores --input reports/insightface/side --recursive --groups motion

Frame selection (reference-free): the first/last quarters of each video are
dropped, then the single frame with the highest RetinaFace detection
confidence (`det_score`) is kept as the representative frame. 
Notes:
- Uses the InsightFace FaceAnalysis buffalo_l pack by default.
- Embeddings are cast via np.asarray(..., dtype=np.float32).copy() to avoid an
  "invalid __array_struct__" error.
- Bootstrap defaults to 1000 iterations.
"""

import os
import glob
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from insightface.app import FaceAnalysis


# Dataset root. Override with the IPE_DATA_ROOT environment variable.
DATA_ROOT = os.environ.get("IPE_DATA_ROOT", "data")


# ==============================================================================
# 1. ARGUMENTS
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="InsightFace cosine scoring + bootstrap.")

    parser.add_argument(
        "--mode",
        default="scores",
        choices=["scores", "extract"],
        help="scores = read existing score CSVs; extract = compute embeddings with InsightFace then bootstrap.",
    )

    # MODE scores
    parser.add_argument(
        "--input",
        default=os.path.join(DATA_ROOT, "reports/insightface/side"),
        help="Score CSV file or folder (used with --mode scores).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search CSVs recursively under --input (--mode scores).",
    )

    # MODE extract
    parser.add_argument(
        "--csv-dir",
        default=os.path.join(DATA_ROOT, "csv/side/negativeSample"),
        help="Source CSV folder (--mode extract).",
    )
    parser.add_argument(
        "--video-dir",
        default=os.path.join(DATA_ROOT, "video/side"),
        help="Per-experiment video folder (--mode extract).",
    )
    parser.add_argument(
        "--reports-dir",
        default=os.path.join(DATA_ROOT, "reports/insightface/side"),
        help="Output folder (--mode extract).",
    )

    # Common
    parser.add_argument(
        "--groups",
        default="",
        help="Filter groups, e.g. motion or motion_L3,lowlight_L3. Empty = run all.",
    )
    parser.add_argument("--threshold", type=float, default=0.15, help="Cosine threshold.")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap iterations.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0, help="Row limit per CSV. 0 = all.")

    # InsightFace
    parser.add_argument("--model-name", default="buffalo_l", help="InsightFace model pack.")
    parser.add_argument("--det-size", type=int, default=640, help="Detector input size, e.g. 640.")
    parser.add_argument(
        "--providers",
        default="CUDAExecutionProvider,CPUExecutionProvider",
        help="ONNXRuntime providers, e.g. CUDAExecutionProvider,CPUExecutionProvider or CPUExecutionProvider.",
    )

    # Video sampling (reference-free frame selection)
    parser.add_argument("--skip-start", type=float, default=0.25, help="Fraction of video to skip at the start.")
    parser.add_argument("--skip-end", type=float, default=0.25, help="Fraction of video to skip at the end.")
    parser.add_argument("--frame-step", type=int, default=1, help="Frame stride when reading the video.")

    # Path clean
    parser.add_argument(
        "--path-prefix",
        default="",
        help="Optional prefix to strip from paths stored in the CSV (empty = keep as-is).",
    )

    return parser.parse_args()


# ==============================================================================
# 2. COMMON UTILS
# ==============================================================================
def parse_groups(groups_text):
    if not groups_text.strip():
        return []
    return [g.strip().lower() for g in groups_text.split(",") if g.strip()]


def infer_experiment_name(csv_path):
    p = Path(csv_path)

    if p.name == "identity_comparison_report.csv":
        return p.parent.name

    name = p.stem
    for suffix in [
        "_negative",
        "_positive",
        "_raw_scores",
        "_raw",
        "_metrics_summary",
        "_bootstrap_ci",
        "_bootstrap_delta_vs_baseline",
        "_identity_comparison_report",
    ]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def group_match(exp_name, groups):
    if not groups:
        return True

    name = exp_name.lower()
    for g in groups:
        if name == g:
            return True
        if name.startswith(g + "_"):
            return True
        if name.startswith(g):
            return True
        if g in name:
            return True

    return False


def clean_path(path, prefix):
    if not isinstance(path, str):
        return ""
    path = path.strip()
    if prefix and path.startswith(prefix):
        path = path.replace(prefix, "", 1)
    return path


def read_image(path):
    if not path:
        return None
    if not os.path.exists(path):
        return None
    return cv2.imread(path)


def cosine_similarity(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 0.0

    # Cast to avoid an "invalid __array_struct__" error.
    v1 = np.asarray(emb1, dtype=np.float32).copy().reshape(-1)
    v2 = np.asarray(emb2, dtype=np.float32).copy().reshape(-1)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 == 0 or n2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (n1 * n2))


def safe_auc(y_true, scores):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return np.nan


def compute_metrics(y_true, scores, threshold):
    scores = np.asarray(scores, dtype=np.float32)
    y_pred = (scores >= threshold).astype(int)

    return {
        "MeanCos": float(np.mean(scores)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "AUC": safe_auc(y_true, scores),
    }


def ci95(values):
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    return (
        float(np.mean(arr)),
        float(np.percentile(arr, 2.5)),
        float(np.percentile(arr, 97.5)),
    )


# ==============================================================================
# 3. INSIGHTFACE FEATURE EXTRACTION
# ==============================================================================
def build_insightface_app(args):
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]

    print("Loading InsightFace...")
    print("Model     :", args.model_name)
    print("Providers :", providers)

    app = FaceAnalysis(name=args.model_name, providers=providers)
    app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1, det_size=(args.det_size, args.det_size))
    return app


def det_score_of(face):
    """RetinaFace detection confidence, falling back to bbox area if absent."""
    score = getattr(face, "det_score", None)
    if score is not None:
        return float(score)
    return float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))


def get_embedding_robust(img, app):
    """Embedding of the highest-confidence detected face. Reference-free."""
    if img is None:
        return None

    try:
        faces = app.get(img)

        if faces is None or len(faces) == 0:
            return None

        face = max(faces, key=det_score_of)

        emb = np.asarray(face.embedding, dtype=np.float32).copy()

        norm = np.linalg.norm(emb)
        if norm == 0:
            return None

        return emb / norm

    except Exception as e:
        print(f"[InsightFace ERROR] {type(e).__name__}: {e}")
        return None


def get_best_frame_embedding(video_path, app, args):
    """Reference-free representative-frame selection.

    Drops the first/last `skip_start`/`skip_end` fraction of the video, then
    keeps the single frame whose best face has the highest RetinaFace
    detection confidence. The reference embedding never enters this search;
    it is only used afterwards by the caller to score the selected frame.
    """
    if not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None

    start_f = int(total_frames * args.skip_start)
    end_f = int(total_frames * (1 - args.skip_end))

    best_score = -1.0
    best_emb = None

    for i in range(start_f, end_f, args.frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if not ret:
            break

        try:
            faces = frame is not None and app.get(frame)
        except Exception as e:
            print(f"[InsightFace ERROR] {type(e).__name__}: {e}")
            continue

        if not faces:
            continue

        face = max(faces, key=det_score_of)
        score = det_score_of(face)

        if score > best_score:
            emb = np.asarray(face.embedding, dtype=np.float32).copy()
            norm = np.linalg.norm(emb)
            if norm > 0:
                best_score = score
                best_emb = emb / norm

    cap.release()
    return best_emb


# ==============================================================================
# 4. BOOTSTRAP
# ==============================================================================
def bootstrap_metrics(df, methods, threshold, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    y_true = df["Label"].astype(int).values
    n = len(df)

    metric_names = ["MeanCos", "Accuracy", "Precision", "Recall", "F1", "AUC"]

    boot = {
        method: {metric: [] for metric in metric_names}
        for method in methods
    }

    deltas = {}
    if "Baseline" in methods:
        for method in methods:
            if method != "Baseline":
                deltas[method] = {f"Delta{metric}": [] for metric in metric_names}

    for _ in tqdm(range(n_bootstrap), desc="Bootstrapping", leave=False):
        idx = rng.integers(0, n, n)
        y_s = y_true[idx]

        cache = {}

        for method in methods:
            scores = df[method].values[idx]
            met = compute_metrics(y_s, scores, threshold)
            cache[method] = met

            for metric in metric_names:
                boot[method][metric].append(met[metric])

        if "Baseline" in methods:
            base = cache["Baseline"]
            for method in methods:
                if method == "Baseline":
                    continue

                for metric in metric_names:
                    deltas[method][f"Delta{metric}"].append(cache[method][metric] - base[metric])

    return boot, deltas


def save_bootstrap_outputs(df, exp_name, output_dir, args):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [m for m in ["Baseline", "Wan2.1", "GroundTruth"] if m in df.columns]

    y_true = df["Label"].astype(int).values

    # Basic summary
    summary_rows = []
    for method in methods:
        met = compute_metrics(y_true, df[method].values, args.threshold)
        summary_rows.append({"Experiment": exp_name, "Method": method, **met})

    df_summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / f"{exp_name}_metrics_summary.csv"
    df_summary.to_csv(summary_path, index=False)

    print("\n[BASIC METRICS]")
    print(df_summary.to_string(index=False))

    # Bootstrap
    boot, deltas = bootstrap_metrics(
        df=df,
        methods=methods,
        threshold=args.threshold,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    ci_rows = []
    for method, metric_dict in boot.items():
        for metric, values in metric_dict.items():
            mean, low, high = ci95(values)
            ci_rows.append({
                "Experiment": exp_name,
                "Method": method,
                "Metric": metric,
                "Mean": mean,
                "CI95_Low": low,
                "CI95_High": high,
            })

    df_ci = pd.DataFrame(ci_rows)
    ci_path = output_dir / f"{exp_name}_bootstrap_ci.csv"
    df_ci.to_csv(ci_path, index=False)

    delta_rows = []
    for method, metric_dict in deltas.items():
        for metric, values in metric_dict.items():
            mean, low, high = ci95(values)
            delta_rows.append({
                "Experiment": exp_name,
                "Comparison": f"{method} - Baseline",
                "Metric": metric,
                "MeanDelta": mean,
                "CI95_Low": low,
                "CI95_High": high,
                "SignificantPositive": bool(low > 0) if not np.isnan(low) else False,
                "SignificantNegative": bool(high < 0) if not np.isnan(high) else False,
            })

    df_delta = pd.DataFrame(delta_rows)
    delta_path = output_dir / f"{exp_name}_bootstrap_delta_vs_baseline.csv"
    df_delta.to_csv(delta_path, index=False)

    # Text report
    report_lines = [
        f"=== INSIGHTFACE BOOTSTRAP REPORT: {exp_name.upper()} ===",
        f"Rows       : {len(df)}",
        f"Threshold  : {args.threshold:.4f}",
        f"Iterations : {args.n_bootstrap}",
        "",
        "1. Basic metrics",
    ]

    for _, row in df_summary.iterrows():
        report_lines.append(
            f"- {row['Method']}: "
            f"MeanCos={row['MeanCos']:.4f}, "
            f"Acc={row['Accuracy']:.4f}, "
            f"Precision={row['Precision']:.4f}, "
            f"Recall={row['Recall']:.4f}, "
            f"F1={row['F1']:.4f}, "
            f"AUC={row['AUC']:.4f}"
        )

    report_lines.append("")
    report_lines.append("2. Bootstrap 95% CI")
    for _, row in df_ci.iterrows():
        report_lines.append(
            f"- {row['Method']} {row['Metric']}: "
            f"{row['Mean']:.4f} [{row['CI95_Low']:.4f}, {row['CI95_High']:.4f}]"
        )

    if not df_delta.empty:
        report_lines.append("")
        report_lines.append("3. Delta vs Baseline")
        for _, row in df_delta.iterrows():
            note = ""
            if row["SignificantPositive"]:
                note = " (positive)"
            elif row["SignificantNegative"]:
                note = " (negative)"

            report_lines.append(
                f"- {row['Comparison']} {row['Metric']}: "
                f"{row['MeanDelta']:+.4f} "
                f"[{row['CI95_Low']:.4f}, {row['CI95_High']:.4f}]{note}"
            )

    report_path = output_dir / f"{exp_name}_bootstrap_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nSaved summary : {summary_path}")
    print(f"Saved CI      : {ci_path}")
    print(f"Saved delta   : {delta_path}")
    print(f"Saved report  : {report_path}")

    return {
        "summary": str(summary_path),
        "ci": str(ci_path),
        "delta": str(delta_path),
        "report": str(report_path),
    }


# ==============================================================================
# 5. MODE scores
# ==============================================================================
def discover_score_csvs(args):
    input_path = Path(args.input)
    groups = parse_groups(args.groups)

    if input_path.is_file():
        files = [input_path]
    else:
        pattern = "**/*.csv" if args.recursive else "*.csv"
        files = list(input_path.glob(pattern))

    keep = []

    for f in files:
        # skip generated bootstrap outputs
        if any(s in f.name for s in ["bootstrap_ci", "bootstrap_delta", "metrics_summary"]):
            continue

        # likely score files
        if f.name not in ["identity_comparison_report.csv"] and not f.name.endswith("_raw_scores.csv") and not f.name.endswith("_raw.csv"):
            continue

        exp = infer_experiment_name(f)
        if group_match(exp, groups):
            keep.append(f)

    return sorted(set(keep))


def process_score_csv(csv_path, args):
    exp_name = infer_experiment_name(csv_path)

    print("\n" + "#" * 90)
    print(f"Processing score CSV: {exp_name}")
    print(f"CSV: {csv_path}")
    print("#" * 90)

    if Path(csv_path).stat().st_size == 0:
        print("Skip empty CSV")
        return None

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print("Skip empty CSV")
        return None

    df.columns = [c.strip() for c in df.columns]

    required = ["Label", "Baseline", "Wan2.1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Skip (missing columns): {missing}")
        print("Current columns:", list(df.columns))
        return None

    df = df.dropna(subset=["Label"]).copy()
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df = df.dropna(subset=["Label"]).copy()
    df["Label"] = df["Label"].astype(int)

    for col in ["Baseline", "Wan2.1", "GroundTruth"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    if len(df) == 0:
        print("Skip (empty dataframe after cleaning)")
        return None

    output_dir = Path(csv_path).parent
    save_bootstrap_outputs(df, exp_name, output_dir, args)
    return exp_name


# ==============================================================================
# 6. MODE extract
# ==============================================================================
def discover_extract_csvs(args):
    groups = parse_groups(args.groups)
    files = sorted(glob.glob(os.path.join(args.csv_dir, "*.csv")))

    keep = []
    for f in files:
        exp = infer_experiment_name(f)
        if group_match(exp, groups):
            keep.append(f)

    return keep


def process_extract_csv(csv_path, app, args):
    exp_name = infer_experiment_name(csv_path)

    print("\n" + "#" * 90)
    print(f"Extracting with InsightFace: {exp_name}")
    print(f"CSV: {csv_path}")
    print("#" * 90)

    current_video_dir = os.path.join(args.video_dir, exp_name)
    current_output_dir = os.path.join(args.reports_dir, exp_name)

    if not os.path.exists(current_video_dir):
        print(f"Skip (video dir not found): {current_video_dir}")
        return None

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    required_cols = ["label", "person", "ref_path", "Source_Path", "test_path"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Skip (missing source columns): {missing}")
        print("Current columns:", list(df.columns))
        return None

    for col in ["ref_path", "Source_Path", "test_path"]:
        df[col] = df[col].astype(str).apply(lambda x: clean_path(x, args.path_prefix))

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    results = []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {exp_name}"):
        label = int(row["label"])
        person = row["person"]
        ref_p = row["ref_path"]
        deg_p = row["Source_Path"]
        orig_p = row["test_path"]

        video_name = f"{os.path.splitext(os.path.basename(deg_p))[0]}.mp4"
        video_path = os.path.join(current_video_dir, video_name)

        img_ref = read_image(ref_p)
        img_deg = read_image(deg_p)
        img_gt = read_image(orig_p)

        if img_ref is None:
            skipped += 1
            continue

        emb_ref = get_embedding_robust(img_ref, app)
        if emb_ref is None:
            skipped += 1
            continue

        emb_base = get_embedding_robust(img_deg, app)
        emb_wan = get_best_frame_embedding(video_path, app, args)
        emb_gt = get_embedding_robust(img_gt, app)

        s_base = cosine_similarity(emb_base, emb_ref)
        s_wan = cosine_similarity(emb_wan, emb_ref)
        s_gt = cosine_similarity(emb_gt, emb_ref)

        results.append({
            "Person": person,
            "Label": label,
            "GroundTruth": s_gt,
            "Baseline": s_base,
            "Wan2.1": s_wan,
        })

    if not results:
        print(f"No valid rows. skipped={skipped}")
        return None

    out_dir = Path(current_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_scores = pd.DataFrame(results)
    raw_path = out_dir / f"{exp_name}_raw_scores.csv"
    identity_path = out_dir / "identity_comparison_report.csv"

    df_scores.to_csv(raw_path, index=False)
    df_scores.to_csv(identity_path, index=False)

    print(f"Rows valid : {len(df_scores)}")
    print(f"Rows skip  : {skipped}")
    print(f"Saved raw scores: {raw_path}")
    print(f"Saved identity report: {identity_path}")

    save_bootstrap_outputs(df_scores, exp_name, out_dir, args)
    return exp_name


# ==============================================================================
# 7. MAIN
# ==============================================================================
def main():
    args = parse_args()

    print("=" * 100)
    print("INSIGHTFACE BOOTSTRAP")
    print("=" * 100)
    print(f"Mode       : {args.mode}")
    print(f"Threshold  : {args.threshold}")
    print(f"Bootstrap  : {args.n_bootstrap}")
    print(f"Groups     : {args.groups if args.groups else 'ALL'}")
    print("=" * 100)

    completed = []

    if args.mode == "scores":
        files = discover_score_csvs(args)

        print(f"Found {len(files)} score CSV file(s):")
        for f in files:
            print(f"  - {f}")

        for f in files:
            exp = process_score_csv(f, args)
            if exp:
                completed.append(exp)

    else:
        files = discover_extract_csvs(args)

        print(f"Found {len(files)} source CSV file(s):")
        for f in files:
            print(f"  - {f}")

        app = build_insightface_app(args)

        for f in files:
            exp = process_extract_csv(f, app, args)
            if exp:
                completed.append(exp)

    print("\n" + "*" * 100)
    print("DONE")
    print("Completed:", completed)
    print("*" * 100)


if __name__ == "__main__":
    main()

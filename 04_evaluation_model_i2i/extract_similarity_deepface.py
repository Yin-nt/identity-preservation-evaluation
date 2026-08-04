
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from deepface import DeepFace

# Dataset root / pose. Override with the IPE_DATA_ROOT / IPE_POSE env vars.
DATA_ROOT = os.environ.get("IPE_DATA_ROOT", "data")
POSE = os.environ.get("IPE_POSE", "frontal")

METRIC_NAMES = ["MeanCos", "Accuracy", "Precision", "Recall", "F1", "AUC"]
MODEL_COLUMNS = {"gfpgan_path": "GFPGAN", "OSDFace_path": "OSDFace"}


# ==============================================================================
# 1. ARGUMENTS
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="DeepFace i2i cosine scoring + bootstrap (GFPGAN/OSDFace).")

    parser.add_argument(
        "--csv-dir",
        default=os.path.join(DATA_ROOT, f"csv/{POSE}/negativeSample"),
        help="Folder of source CSVs, one per experiment.",
    )
    parser.add_argument(
        "--reports-dir",
        default=os.path.join(DATA_ROOT, "reports/i2i/deepface", POSE),
        help="Output folder (a <model>/<experiment> subfolder is created per CSV).",
    )
    parser.add_argument("--groups", default="", help="Filter groups, e.g. motion or motion_L3,lowlight_L3.")
    parser.add_argument("--threshold", type=float, default=0.15, help="Cosine threshold.")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap iterations.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0, help="Row limit per CSV. 0 = all.")

    parser.add_argument("--model-name", default="Facenet512", help="DeepFace model name.")
    parser.add_argument("--detector-backend", default="retinaface", help="DeepFace detector backend.")

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
    return Path(csv_path).stem


def group_match(exp_name, groups):
    if not groups:
        return True
    name = exp_name.lower()
    return any(name == g or name.startswith(g + "_") or name.startswith(g) or g in name for g in groups)


def discover_csvs(args):
    groups = parse_groups(args.groups)
    files = sorted(glob.glob(os.path.join(args.csv_dir, "*.csv")))
    return [f for f in files if group_match(infer_experiment_name(f), groups)]


def detect_model_column(df):
    for col, name in MODEL_COLUMNS.items():
        if col in df.columns:
            return col, name
    return None, None


def clean_path(path, prefix):
    if not isinstance(path, str):
        return ""
    path = path.strip()
    if prefix and path.startswith(prefix):
        path = path.replace(prefix, "", 1)
    return path


def cosine_similarity(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 0.0
    v1 = np.asarray(emb1, dtype=np.float32).copy().reshape(-1)
    v2 = np.asarray(emb2, dtype=np.float32).copy().reshape(-1)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
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
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


# ==============================================================================
# 3. DEEPFACE FEATURE EXTRACTION (best-of-all-faces matching)
# ==============================================================================
def get_all_embeddings_deepface(img_path, args):
    """All (normalized) face embeddings detected in an image. Empty list on failure."""
    if not isinstance(img_path, str) or not os.path.exists(img_path):
        return []
    try:
        results = DeepFace.represent(
            img_path=img_path,
            model_name=args.model_name,
            detector_backend=args.detector_backend,
            enforce_detection=False,
            align=True,
        )
        if isinstance(results, dict):
            results = [results]
        embeddings = []
        for face in results or []:
            emb = np.asarray(face["embedding"], dtype=np.float32)
            norm = np.linalg.norm(emb)
            embeddings.append(emb / norm if norm > 0 else emb)
        return embeddings
    except Exception as e:
        print(f"[DeepFace ERROR] {img_path}: {type(e).__name__}: {e}")
        return []


def best_cosine_deepface(emb_list1, emb_list2):
    """Highest cosine similarity across every face-embedding pair between two images."""
    if not emb_list1 or not emb_list2:
        return 0.0
    best = -1.0
    for e1 in emb_list1:
        for e2 in emb_list2:
            sim = cosine_similarity(e1, e2)
            if sim > best:
                best = sim
    return best


def read_path(path):
    if not isinstance(path, str) or not path or not os.path.exists(path):
        return None
    return path


# ==============================================================================
# 4. BOOTSTRAP
# ==============================================================================
def bootstrap_metrics(df, pair_names, threshold, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    y_true = df["label"].astype(int).values
    n = len(df)

    boot = {p: {m: [] for m in METRIC_NAMES} for p in pair_names}
    deltas = {p: {f"Delta{m}": [] for m in METRIC_NAMES} for p in pair_names if p != "ref_vs_degra"}

    for _ in tqdm(range(n_bootstrap), desc="Bootstrapping", leave=False):
        idx = rng.integers(0, n, n)
        y_s = y_true[idx]
        cache = {}

        for p in pair_names:
            met = compute_metrics(y_s, df[p].values[idx], threshold)
            cache[p] = met
            for m in METRIC_NAMES:
                boot[p][m].append(met[m])

        if "ref_vs_degra" in cache:
            base = cache["ref_vs_degra"]
            for p in pair_names:
                if p == "ref_vs_degra":
                    continue
                for m in METRIC_NAMES:
                    deltas[p][f"Delta{m}"].append(cache[p][m] - base[m])

    return boot, deltas


def save_outputs(df, pair_names, exp_name, model_name, output_dir, args):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = df["label"].astype(int).values

    summary_rows = [
        {"Experiment": exp_name, "Pair": p, **compute_metrics(y_true, df[p].values, args.threshold)}
        for p in pair_names
    ]
    df_summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / f"{exp_name}_metrics_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print("\n[BASIC METRICS]")
    print(df_summary.to_string(index=False))

    boot, deltas = bootstrap_metrics(df, pair_names, args.threshold, args.n_bootstrap, args.seed)

    ci_rows = [
        {"Experiment": exp_name, "Pair": p, "Metric": m, "Mean": mean, "CI95_Low": low, "CI95_High": high}
        for p, metric_dict in boot.items()
        for m, values in metric_dict.items()
        for mean, low, high in [ci95(values)]
    ]
    df_ci = pd.DataFrame(ci_rows)
    ci_path = output_dir / f"{exp_name}_bootstrap_ci.csv"
    df_ci.to_csv(ci_path, index=False)

    delta_rows = []
    for p, metric_dict in deltas.items():
        for m, values in metric_dict.items():
            mean, low, high = ci95(values)
            delta_rows.append({
                "Experiment": exp_name, "Comparison": f"{p} - ref_vs_degra", "Metric": m,
                "MeanDelta": mean, "CI95_Low": low, "CI95_High": high,
                "SignificantPositive": bool(low > 0) if not np.isnan(low) else False,
                "SignificantNegative": bool(high < 0) if not np.isnan(high) else False,
            })
    df_delta = pd.DataFrame(delta_rows)
    delta_path = output_dir / f"{exp_name}_bootstrap_delta_vs_baseline.csv"
    df_delta.to_csv(delta_path, index=False)

    report_lines = [
        f"=== DEEPFACE I2I BOOTSTRAP REPORT: {exp_name.upper()} ({model_name}) ===",
        f"Rows       : {len(df)}",
        f"Threshold  : {args.threshold:.4f}",
        f"Iterations : {args.n_bootstrap}",
        "",
        "1. Basic metrics",
    ]
    for _, row in df_summary.iterrows():
        report_lines.append(
            f"- {row['Pair']}: MeanCos={row['MeanCos']:.4f}, Acc={row['Accuracy']:.4f}, "
            f"Precision={row['Precision']:.4f}, Recall={row['Recall']:.4f}, F1={row['F1']:.4f}, AUC={row['AUC']:.4f}"
        )
    report_lines += ["", "2. Bootstrap 95% CI"]
    for _, row in df_ci.iterrows():
        report_lines.append(f"- {row['Pair']} {row['Metric']}: {row['Mean']:.4f} [{row['CI95_Low']:.4f}, {row['CI95_High']:.4f}]")
    if not df_delta.empty:
        report_lines += ["", "3. Delta vs ref_vs_degra (restoration effect)"]
        for _, row in df_delta.iterrows():
            note = " (positive)" if row["SignificantPositive"] else (" (negative)" if row["SignificantNegative"] else "")
            report_lines.append(f"- {row['Comparison']} {row['Metric']}: {row['MeanDelta']:+.4f} [{row['CI95_Low']:.4f}, {row['CI95_High']:.4f}]{note}")

    report_path = output_dir / f"{exp_name}_bootstrap_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nSaved summary : {summary_path}")
    print(f"Saved CI      : {ci_path}")
    print(f"Saved delta   : {delta_path}")
    print(f"Saved report  : {report_path}")


# ==============================================================================
# 5. EXTRACT
# ==============================================================================
def process_csv(csv_path, args):
    exp_name = infer_experiment_name(csv_path)
    print("\n" + "#" * 90)
    print(f"Extracting with DeepFace: {exp_name}")
    print(f"CSV: {csv_path}")
    print("#" * 90)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    model_col, model_name = detect_model_column(df)
    if model_col is None:
        print("Skip (no gfpgan_path/OSDFace_path column — run 03_generation_gfpgan or 03_generation_osdface first).")
        return None

    required_cols = ["label", "ref_path", "Source_Path", "test_path", model_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Skip (missing columns): {missing}")
        return None

    for col in ["ref_path", "Source_Path", "test_path", model_col]:
        df[col] = df[col].astype(str).apply(lambda x: clean_path(x, args.path_prefix))

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    pairs = [
        ("ref_vs_degra", "ref_path", "Source_Path"),
        (f"ref_vs_{model_name}", "ref_path", model_col),
        ("ref_vs_GT", "ref_path", "test_path"),
        (f"GT_vs_{model_name}", "test_path", model_col),
        ("GT_vs_degra", "test_path", "Source_Path"),
    ]
    pair_names = [p[0] for p in pairs]
    score_cols = {name: [] for name in pair_names}

    embedding_cache = {}

    def cached_embeddings(path):
        if path not in embedding_cache:
            embedding_cache[path] = get_all_embeddings_deepface(path, args)
        return embedding_cache[path]

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {exp_name}"):
        for pair_name, path_a, path_b in pairs:
            emb_a = cached_embeddings(read_path(row[path_a]))
            emb_b = cached_embeddings(read_path(row[path_b]))
            score_cols[pair_name].append(best_cosine_deepface(emb_a, emb_b))

    for name, values in score_cols.items():
        df[name] = values

    out_dir = os.path.join(args.reports_dir, model_name.lower(), exp_name)
    os.makedirs(out_dir, exist_ok=True)
    scores_path = os.path.join(out_dir, "data_with_scores.csv")
    df.to_csv(scores_path, index=False)

    print(f"Rows total : {len(df)}")
    print(f"Unique images embedded: {len(embedding_cache)}")
    print(f"Saved scores: {scores_path}")

    save_outputs(df, pair_names, exp_name, model_name, out_dir, args)
    return exp_name


# ==============================================================================
# 6. MAIN
# ==============================================================================
def main():
    args = parse_args()

    print("=" * 100)
    print("DEEPFACE I2I BOOTSTRAP (GFPGAN / OSDFace)")
    print("=" * 100)
    print(f"Threshold  : {args.threshold}")
    print(f"Bootstrap  : {args.n_bootstrap}")
    print(f"Groups     : {args.groups if args.groups else 'ALL'}")
    print("=" * 100)

    csv_files = discover_csvs(args)
    print(f"Found {len(csv_files)} source CSV file(s):")
    for f in csv_files:
        print(f"  - {f}")

    completed = []
    for f in csv_files:
        exp = process_csv(f, args)
        if exp:
            completed.append(exp)

    print("\n" + "*" * 100)
    print("DONE")
    print("Completed:", completed)
    print("*" * 100)


if __name__ == "__main__":
    main()

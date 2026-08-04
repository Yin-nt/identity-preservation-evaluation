import argparse
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

METRIC_NAMES = ["MeanCos", "Accuracy", "Precision", "Recall", "F1", "AUC"]
BASELINE_PAIR = "ref_vs_degra"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV file or folder of CSVs.")
    parser.add_argument("--recursive", action="store_true", help="Search CSVs recursively.")
    parser.add_argument("--groups", default="", help="Filter groups, e.g. motion or motion_L1,lowlight_L3.")
    parser.add_argument("--threshold", type=float, default=0.15, help="Cosine threshold.")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap iterations.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="", help="Output folder. Empty = beside the CSV.")
    return parser.parse_args()


def parse_groups(groups_text):
    if not groups_text.strip():
        return []
    return [g.strip().lower() for g in groups_text.split(",") if g.strip()]


def infer_experiment_name(csv_path):
    p = Path(csv_path)
    if p.name == "data_with_scores.csv":
        return p.parent.name

    name = p.stem
    for suffix in ["_raw", "_raw_scores", "_metrics_summary", "_bootstrap_ci", "_bootstrap_delta_vs_baseline", "_with_scores"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def group_match(exp_name, groups):
    if not groups:
        return True
    name = exp_name.lower()
    return any(name == g or name.startswith(g + "_") or name.startswith(g) or g in name for g in groups)


def discover_csv_files(input_path, recursive=False, groups=None):
    p = Path(input_path)
    groups = groups or []

    if p.is_file():
        files = [p]
    else:
        pattern = "**/*.csv" if recursive else "*.csv"
        files = list(p.glob(pattern))

    keep = []
    for f in files:
        if any(x in f.name for x in ["bootstrap_ci", "bootstrap_delta", "metrics_summary"]):
            continue
        if group_match(infer_experiment_name(f), groups):
            keep.append(f)

    return sorted(set(keep))


def safe_read_csv(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"Skip (missing): {csv_path}")
        return None
    if csv_path.stat().st_size == 0:
        print(f"Skip (0 bytes): {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Skip (no rows): {csv_path}")
            return None
        return df
    except pd.errors.EmptyDataError:
        print(f"Skip (empty CSV): {csv_path}")
        return None
    except Exception as e:
        print(f"Skip (read error): {csv_path}\n   {type(e).__name__}: {e}")
        return None


def detect_pair_columns(df):
    return [c for c in df.columns if "_vs_" in c]


def safe_auc(y_true, scores):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, scores)
    except Exception:
        return np.nan


def compute_metrics(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)
    return {
        "MeanCos": float(np.mean(scores)),
        "Accuracy": float(accuracy_score(y_true, pred)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "AUC": float(safe_auc(y_true, scores)),
    }


def ci95(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def bootstrap_one(df, pairs, threshold, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    y = df["Label"].astype(int).values
    n = len(df)

    boot = {p: {k: [] for k in METRIC_NAMES} for p in pairs}
    deltas = {}
    if BASELINE_PAIR in pairs:
        for p in pairs:
            if p != BASELINE_PAIR:
                deltas[p] = {f"Delta{k}": [] for k in METRIC_NAMES}

    for _ in tqdm(range(n_bootstrap), desc="Bootstrapping", leave=False):
        idx = rng.integers(0, n, n)
        y_s = y[idx]
        cache = {}

        for p in pairs:
            met = compute_metrics(y_s, df[p].values[idx], threshold)
            cache[p] = met
            for k in METRIC_NAMES:
                boot[p][k].append(met[k])

        if BASELINE_PAIR in pairs:
            base = cache[BASELINE_PAIR]
            for p in pairs:
                if p == BASELINE_PAIR:
                    continue
                for k in METRIC_NAMES:
                    deltas[p][f"Delta{k}"].append(cache[p][k] - base[k])

    return boot, deltas


def process_csv(csv_path, args):
    exp = infer_experiment_name(csv_path)
    print("\n" + "#" * 90)
    print(f"Processing: {exp}")
    print(f"CSV: {csv_path}")
    print("#" * 90)

    df = safe_read_csv(csv_path)
    if df is None:
        return None

    df.columns = [c.strip() for c in df.columns]
    if "Label" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "Label"})

    pairs = detect_pair_columns(df)

    if "Label" not in df.columns or not pairs:
        print("Skip (need a Label column and at least one *_vs_* pair column).")
        print(f"   Available columns: {list(df.columns)}")
        return None

    df = df.dropna(subset=["Label"]).copy()
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df = df.dropna(subset=["Label"]).copy()
    df["Label"] = df["Label"].astype(int)
    for p in pairs:
        df[p] = pd.to_numeric(df[p], errors="coerce").fillna(0.0)

    if len(df) == 0:
        print("Skip (empty after cleaning).")
        return None
    if len(np.unique(df["Label"].values)) < 2:
        print("Warning: only one label class present, AUC may be NaN.")

    out_dir = Path(args.output_root) / exp if args.output_root else Path(csv_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = [
        {"Experiment": exp, "Pair": p, **compute_metrics(df["Label"].values, df[p].values, args.threshold)}
        for p in pairs
    ]
    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"{exp}_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Rows      : {len(df)}")
    print(f"Threshold : {args.threshold}")
    print("\n[METRICS]")
    print(summary.to_string(index=False))

    boot, deltas = bootstrap_one(df, pairs, args.threshold, args.n_bootstrap, args.seed)

    ci_rows = [
        {"Experiment": exp, "Pair": p, "Metric": metric, "Mean": mean, "CI95_Low": low, "CI95_High": high}
        for p, metric_dict in boot.items()
        for metric, vals in metric_dict.items()
        for mean, low, high in [ci95(vals)]
    ]
    ci_df = pd.DataFrame(ci_rows)
    ci_path = out_dir / f"{exp}_bootstrap_ci.csv"
    ci_df.to_csv(ci_path, index=False)

    delta_rows = []
    for p, metric_dict in deltas.items():
        for metric, vals in metric_dict.items():
            mean, low, high = ci95(vals)
            delta_rows.append({
                "Experiment": exp, "Comparison": f"{p} - {BASELINE_PAIR}", "Metric": metric,
                "MeanDelta": mean, "CI95_Low": low, "CI95_High": high,
                "SignificantPositive": bool(low > 0) if not np.isnan(low) else False,
                "SignificantNegative": bool(high < 0) if not np.isnan(high) else False,
            })
    delta_df = pd.DataFrame(delta_rows)
    delta_path = out_dir / f"{exp}_bootstrap_delta_vs_baseline.csv"
    delta_df.to_csv(delta_path, index=False)

    report_lines = [
        f"=== I2I BOOTSTRAP REPORT: {exp.upper()} ===",
        f"Input CSV  : {csv_path}",
        f"Rows       : {len(df)}",
        f"Threshold  : {args.threshold}",
        f"Iterations : {args.n_bootstrap}",
        "",
        "1. Basic metrics",
    ]
    for _, r in summary.iterrows():
        report_lines.append(
            f"- {r['Pair']}: MeanCos={r['MeanCos']:.4f}, "
            f"Acc={r['Accuracy']:.4f}, Precision={r['Precision']:.4f}, "
            f"Recall={r['Recall']:.4f}, F1={r['F1']:.4f}, AUC={r['AUC']:.4f}"
        )

    report_lines += ["", "2. Bootstrap 95% CI"]
    for _, r in ci_df.iterrows():
        report_lines.append(f"- {r['Pair']} {r['Metric']}: {r['Mean']:.4f} [{r['CI95_Low']:.4f}, {r['CI95_High']:.4f}]")

    if not delta_df.empty:
        report_lines += ["", f"3. Delta vs {BASELINE_PAIR} (restoration effect)"]
        for _, r in delta_df.iterrows():
            note = " (positive)" if r["SignificantPositive"] else (" (negative)" if r["SignificantNegative"] else "")
            report_lines.append(f"- {r['Comparison']} {r['Metric']}: {r['MeanDelta']:+.4f} [{r['CI95_Low']:.4f}, {r['CI95_High']:.4f}]{note}")

    report_path = out_dir / f"{exp}_bootstrap_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {ci_path}")
    print(f"Saved: {delta_path}")
    print(f"Saved: {report_path}")
    return exp


def main():
    args = parse_args()
    groups = parse_groups(args.groups)

    print("=" * 100)
    print("I2I BOOTSTRAP FROM EXISTING SCORE CSVS (GFPGAN / OSDFace)")
    print("=" * 100)
    print(f"Input       : {args.input}")
    print(f"Recursive   : {args.recursive}")
    print(f"Groups      : {args.groups if args.groups else 'ALL'}")
    print(f"Threshold   : {args.threshold}")
    print(f"Bootstrap   : {args.n_bootstrap}")
    print("=" * 100)

    csv_files = discover_csv_files(args.input, args.recursive, groups)
    if not csv_files:
        print("No matching CSV found.")
        return

    print(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  - {f}")

    completed, skipped = [], []
    for f in csv_files:
        exp = process_csv(f, args)
        (completed if exp else skipped).append(exp or str(f))

    print("\n" + "*" * 100)
    print("DONE")
    print(f"Completed experiments: {completed}")
    print(f"Skipped files: {len(skipped)}")
    for f in skipped:
        print(f"  - {f}")
    print("*" * 100)


if __name__ == "__main__":
    main()

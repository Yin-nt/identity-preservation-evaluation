# -*- coding: utf-8 -*-
"""EER + threshold sensitivity analysis (batch).

Reuses the discovery / skip-empty logic of bootstrap_metrics.py and adds:
  1. EER + AUC for each CSV (per-experiment).
  2. A single "unified EER" per method, pooling every discovered CSV into one
     dataset (supports the unified 0.15 lower-bound threshold in Section 4.3).
  3. EER pooled by severity level (L1/L2/L3) to check threshold stability
     (appendix table).
  4. A sensitivity table (Accuracy/Precision/Recall/F1) sweeping the threshold
     0.05 -> 0.35 on the unified dataset.
  5. Unified ROC curve + sensitivity curve plots.

Input CSV needs columns: Person, Label, GroundTruth, Baseline, Wan2.1
(any extra numeric method column such as Wan2.2 / GFPGAN / OSDFace is picked up
automatically).

Examples:
    python eer_analysis.py --input reports/deepface --recursive
    python eer_analysis.py --input reports/deepface --recursive --groups motion
    python eer_analysis.py --input reports/arcface --recursive --output-root eer_results/arcface
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Discovery / skip-empty logic shared with bootstrap_metrics.py so that both
# scripts scan files consistently.
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV file or folder of CSVs.")
    parser.add_argument("--recursive", action="store_true", help="Search CSVs recursively.")
    parser.add_argument("--groups", default="", help="Filter groups, e.g. motion or motion_L1,lowlight_L3.")
    parser.add_argument("--fixed-threshold", type=float, default=0.15,
                         help="Fixed threshold used in the paper (default 0.15).")
    parser.add_argument("--thr-min", type=float, default=0.05)
    parser.add_argument("--thr-max", type=float, default=0.35)
    parser.add_argument("--thr-step", type=float, default=0.01)
    parser.add_argument("--output-root", default="eer_batch_output",
                         help="Output folder (per-experiment + unified results).")
    return parser.parse_args()


def parse_groups(groups_text):
    if not groups_text.strip():
        return []
    return [g.strip().lower() for g in groups_text.split(",") if g.strip()]


def infer_experiment_name(csv_path):
    p = Path(csv_path)
    if p.name == "identity_comparison_report.csv":
        return p.parent.name

    name = p.stem
    for suffix in ["_raw", "_raw_scores", "_metrics_summary", "_bootstrap_ci", "_bootstrap_delta_vs_baseline"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def group_match(exp_name, groups):
    if not groups:
        return True
    name = exp_name.lower()
    for g in groups:
        if name == g or name.startswith(g + "_") or name.startswith(g) or g in name:
            return True
    return False


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
        if any(x in f.name for x in ["bootstrap_ci", "bootstrap_delta", "metrics_summary",
                                       "eer_summary", "sensitivity_"]):
            continue
        exp_name = infer_experiment_name(f)
        if group_match(exp_name, groups):
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
        print(f"Skip (read error): {csv_path}")
        print(f"   Error: {type(e).__name__}: {e}")
        return None


def clean_df(df, methods_wanted):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "Label" not in df.columns:
        return None, []

    methods = [m for m in methods_wanted if m in df.columns]
    # Also pick up any other numeric column besides Person/Label.
    extra = [c for c in df.columns if c not in ["Person", "Label"] + methods]
    for c in extra:
        if pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce")):
            methods.append(c)

    if not methods:
        return None, []

    df = df.dropna(subset=["Label"]).copy()
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df = df.dropna(subset=["Label"]).copy()
    if df.empty:
        return None, []
    df["Label"] = df["Label"].astype(int)

    for m in methods:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    return df, methods


def infer_severity_level(exp_name):
    """Extract 'L1'/'L2'/'L3' from an experiment name, e.g. 'motion_L3' -> 'L3'."""
    match = re.search(r"[Ll]([123])\b", exp_name)
    if match:
        return f"L{match.group(1)}"
    return "unknown"


def infer_degradation_type(exp_name):
    """Extract the degradation type, e.g. 'saltpepper_L3' -> 'saltpepper'."""
    name = exp_name.lower()
    name = re.sub(r"_l[123]$", "", name)
    name = re.sub(r"_l[123]_", "_", name)
    return name


# ---------------------------------------------------------------------------
# EER + sensitivity analysis
# ---------------------------------------------------------------------------

def compute_eer(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    valid_mask = ~np.isnan(y_score)
    y_true, y_score = y_true[valid_mask], y_score[valid_mask]

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, None, None, None

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    finite = np.isfinite(thresholds)
    fpr_v, fnr_v, thr_v = fpr[finite], fnr[finite], thresholds[finite]
    if len(thr_v) == 0:
        return np.nan, np.nan, fpr, tpr, thresholds
    eer_idx = np.nanargmin(np.abs(fpr_v - fnr_v))
    eer = (fpr_v[eer_idx] + fnr_v[eer_idx]) / 2
    eer_threshold = thr_v[eer_idx]
    return eer, eer_threshold, fpr, tpr, thresholds


def sensitivity_table(y_true, y_score, thresholds_to_test):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    valid_mask = ~np.isnan(y_score)
    y_true, y_score = y_true[valid_mask], y_score[valid_mask]

    rows = []
    for t in thresholds_to_test:
        pred = (y_score >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
        rows.append({"threshold": round(float(t), 3), "accuracy": acc,
                      "precision": precision, "recall": recall, "f1": f1,
                      "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return pd.DataFrame(rows)


def plot_roc(curves, out_path, title):
    plt.figure(figsize=(7, 6))
    for label, (fpr, tpr, roc_auc, eer) in curves.items():
        if fpr is None:
            continue
        plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f}, EER={eer:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_sensitivity(sens_df, fixed_threshold, out_path, title):
    plt.figure(figsize=(7, 5))
    plt.plot(sens_df["threshold"], sens_df["precision"], label="Precision")
    plt.plot(sens_df["threshold"], sens_df["recall"], label="Recall")
    plt.plot(sens_df["threshold"], sens_df["f1"], label="F1")
    plt.axvline(fixed_threshold, color="red", linestyle="--",
                label=f"Fixed threshold ({fixed_threshold})")
    plt.xlabel("Cosine similarity threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    groups = parse_groups(args.groups)
    thresholds_to_test = np.arange(args.thr_min, args.thr_max + args.thr_step / 2, args.thr_step)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("EER + SENSITIVITY ANALYSIS - BATCH MODE")
    print("=" * 100)
    print(f"Input          : {args.input}")
    print(f"Recursive      : {args.recursive}")
    print(f"Groups         : {args.groups if args.groups else 'ALL'}")
    print(f"Fixed threshold: {args.fixed_threshold}")
    print(f"Sweep range    : {args.thr_min} - {args.thr_max} (step {args.thr_step})")
    print("=" * 100)

    csv_files = discover_csv_files(args.input, args.recursive, groups)
    if not csv_files:
        print("No matching CSV found.")
        return

    print(f"Found {len(csv_files)} CSV file(s).")

    methods_wanted = ["GroundTruth", "Baseline", "Wan2.1", "Wan2.2", "GFPGAN", "OSDFace"]

    # per_experiment_rows: per-file EER/AUC (to inspect variance across settings)
    per_experiment_rows = []
    # all_data[method] = list of (y_true, y_score) pooled from ALL files -> unified
    all_data = {}
    # by_severity[method][level] = list of (y_true, y_score)
    by_severity = {}

    skipped = []

    for f in csv_files:
        exp = infer_experiment_name(f)
        level = infer_severity_level(exp)
        degradation = infer_degradation_type(exp)

        df_raw = safe_read_csv(f)
        if df_raw is None:
            skipped.append(str(f))
            continue

        df, methods = clean_df(df_raw, methods_wanted)
        if df is None or df.empty:
            print(f"Skip (no valid columns or empty after cleaning): {f}")
            skipped.append(str(f))
            continue

        y_true = df["Label"].values

        for m in methods:
            y_score = df[m].values
            eer, eer_thr, fpr, tpr, _ = compute_eer(y_true, y_score)
            roc_auc = auc(fpr, tpr) if fpr is not None else np.nan

            per_experiment_rows.append({
                "experiment": exp, "degradation": degradation, "severity": level,
                "method": m, "n": len(df),
                "n_genuine": int((y_true == 1).sum()), "n_impostor": int((y_true == 0).sum()),
                "AUC": roc_auc, "EER": eer, "EER_threshold": eer_thr,
            })

            all_data.setdefault(m, []).append((y_true, y_score))
            by_severity.setdefault(m, {}).setdefault(level, []).append((y_true, y_score))

    # ---- Save per-experiment table ----
    per_exp_df = pd.DataFrame(per_experiment_rows)
    per_exp_path = out_root / "eer_per_experiment.csv"
    per_exp_df.to_csv(per_exp_path, index=False)
    print(f"\nSaved per-experiment EER table: {per_exp_path}")

    # ---- Unified (whole pooled dataset) ----
    print("\n" + "=" * 100)
    print("UNIFIED EER (all scanned files pooled)")
    print("=" * 100)

    unified_rows = []
    roc_curves_for_plot = {}

    for m, pairs in all_data.items():
        y_true_all = np.concatenate([p[0] for p in pairs])
        y_score_all = np.concatenate([p[1] for p in pairs])

        eer, eer_thr, fpr, tpr, _ = compute_eer(y_true_all, y_score_all)
        roc_auc = auc(fpr, tpr) if fpr is not None else np.nan
        roc_curves_for_plot[m] = (fpr, tpr, roc_auc, eer)

        pred_fixed = (y_score_all >= args.fixed_threshold).astype(int)
        tp = int(((pred_fixed == 1) & (y_true_all == 1)).sum())
        fp = int(((pred_fixed == 1) & (y_true_all == 0)).sum())
        fn = int(((pred_fixed == 0) & (y_true_all == 1)).sum())
        tn = int(((pred_fixed == 0) & (y_true_all == 0)).sum())
        precision_fixed = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_fixed = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_fixed = 2 * precision_fixed * recall_fixed / (precision_fixed + recall_fixed) \
            if (precision_fixed + recall_fixed) > 0 else 0.0

        unified_rows.append({
            "method": m, "n_total": len(y_true_all),
            "n_genuine": int((y_true_all == 1).sum()), "n_impostor": int((y_true_all == 0).sum()),
            "AUC": roc_auc, "EER": eer, "EER_threshold": eer_thr,
            "fixed_threshold": args.fixed_threshold,
            "F1_at_fixed": f1_fixed, "Precision_at_fixed": precision_fixed,
            "Recall_at_fixed": recall_fixed,
        })

        # Unified sensitivity table for this method.
        sens_df = sensitivity_table(y_true_all, y_score_all, thresholds_to_test)
        sens_path = out_root / f"sensitivity_unified_{m}.csv"
        sens_df.to_csv(sens_path, index=False)

        print(f"[{m}] n={len(y_true_all)} | AUC={roc_auc:.4f} | EER={eer:.4f} "
              f"at thr={eer_thr:.4f} | F1@{args.fixed_threshold}={f1_fixed:.4f}")

    unified_df = pd.DataFrame(unified_rows)
    unified_path = out_root / "eer_unified_summary.csv"
    unified_df.to_csv(unified_path, index=False)
    print(f"\nSaved unified EER summary: {unified_path}")

    # Unified ROC plot.
    roc_plot_path = out_root / "roc_curves_unified.png"
    plot_roc(roc_curves_for_plot, roc_plot_path, "Unified ROC Curves (all settings pooled)")
    print(f"Saved: {roc_plot_path}")

    # Sensitivity plot for Wan2.1 (or the last method if absent).
    example_model = "Wan2.1" if "Wan2.1" in all_data else list(all_data.keys())[-1]
    example_sens = pd.read_csv(out_root / f"sensitivity_unified_{example_model}.csv")
    sens_plot_path = out_root / f"sensitivity_plot_unified_{example_model}.png"
    plot_sensitivity(example_sens, args.fixed_threshold, sens_plot_path,
                      f"Unified Sensitivity Analysis - {example_model}")
    print(f"Saved: {sens_plot_path}")

    # ---- Pool by severity level (L1/L2/L3) ----
    print("\n" + "=" * 100)
    print("EER BY SEVERITY LEVEL (threshold stability check)")
    print("=" * 100)

    severity_rows = []
    for m, level_dict in by_severity.items():
        for level, pairs in level_dict.items():
            y_true_all = np.concatenate([p[0] for p in pairs])
            y_score_all = np.concatenate([p[1] for p in pairs])
            eer, eer_thr, fpr, tpr, _ = compute_eer(y_true_all, y_score_all)
            roc_auc = auc(fpr, tpr) if fpr is not None else np.nan
            severity_rows.append({
                "method": m, "severity": level, "n": len(y_true_all),
                "AUC": roc_auc, "EER": eer, "EER_threshold": eer_thr,
            })
            print(f"[{m} | {level}] n={len(y_true_all)} | AUC={roc_auc:.4f} | "
                  f"EER={eer:.4f} at thr={eer_thr:.4f}")

    severity_df = pd.DataFrame(severity_rows).sort_values(["method", "severity"])
    severity_path = out_root / "eer_by_severity.csv"
    severity_df.to_csv(severity_path, index=False)
    print(f"\nSaved severity-level EER table: {severity_path}")

    # ---- Summary ----
    print("\n" + "*" * 100)
    print("DONE")
    print(f"Total CSV scanned : {len(csv_files)}")
    print(f"Skipped           : {len(skipped)}")
    for f in skipped:
        print(f"  - {f}")
    print(f"\nOutput folder: {out_root}/")
    print("  - eer_per_experiment.csv          : per-setting EER/AUC (motion_L1, saltpepper_L3, ...)")
    print("  - eer_unified_summary.csv         : pooled EER/AUC -> used for Section 4.3")
    print("  - eer_by_severity.csv             : EER by L1/L2/L3 -> appendix stability check")
    print("  - sensitivity_unified_<model>.csv : threshold sweep 0.05-0.35 per model")
    print("  - roc_curves_unified.png          : pooled ROC plot")
    print("  - sensitivity_plot_unified_*.png  : sensitivity plot")
    print("*" * 100)


if __name__ == "__main__":
    main()
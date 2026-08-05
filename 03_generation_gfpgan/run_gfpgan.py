import argparse
import glob
import os
from pathlib import Path

import cv2
import pandas as pd
from basicsr.utils import imwrite
from gfpgan import GFPGANer
from tqdm import tqdm

# Dataset root / pose. Override with the IPE_DATA_ROOT / IPE_POSE env vars.
DATA_ROOT = os.environ.get("IPE_DATA_ROOT", "data")
POSE = os.environ.get("IPE_POSE", "frontal")

MODEL_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"


def build_restorer(args):
    return GFPGANer(
        model_path=MODEL_URL,
        upscale=args.upscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )


# ==============================================================================
# CSV DISCOVERY (same conventions as 03_generation_osdface/run_osdface.py)
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


# ==============================================================================
# INFERENCE
# ==============================================================================
def process_csv(csv_path, restorer, args):
    exp_name = infer_experiment_name(csv_path)
    print("\n" + "#" * 90)
    print(f"GFPGAN: {exp_name}")
    print(f"CSV: {csv_path}")
    print("#" * 90)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "Source_Path" not in df.columns:
        print("Skip (missing Source_Path column).")
        return

    if "gfpgan_path" not in df.columns:
        df["gfpgan_path"] = ""

    out_dir = os.path.join(DATA_ROOT, "image", POSE, "gfpgan", exp_name)
    os.makedirs(out_dir, exist_ok=True)

    updated = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=exp_name):
        source_path = str(row["Source_Path"])
        existing = row.get("gfpgan_path", "")
        if isinstance(existing, str) and existing and os.path.exists(existing):
            continue
        if not os.path.exists(source_path):
            print(f"  missing source image: {source_path}")
            continue

        input_img = cv2.imread(source_path)
        if input_img is None:
            print(f"  unreadable image: {source_path}")
            continue

        _, _, restored_img = restorer.enhance(
            input_img,
            has_aligned=args.aligned,
            only_center_face=args.only_center_face,
            paste_back=True,
            weight=args.weight,
        )
        if restored_img is None:
            print(f"  no face restored, skipping: {source_path}")
            continue

        out_path = os.path.join(out_dir, os.path.basename(source_path))
        imwrite(restored_img, out_path)
        df.at[idx, "gfpgan_path"] = os.path.abspath(out_path)
        updated += 1

    df.to_csv(csv_path, index=False)
    print(f"Updated gfpgan_path for {updated} row(s) -> {csv_path}")


# ==============================================================================
# MAIN
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run GFPGAN restoration over a degradation index CSV.")

    parser.add_argument(
        "--csv-dir",
        default=os.path.join(DATA_ROOT, f"csv/{POSE}/negativeSample"),
        help="Folder of index CSVs, one per experiment (each needs a Source_Path column).",
    )
    parser.add_argument("--groups", default="", help="Filter experiments, e.g. motion or motion_L2,lowlight_L3.")

    parser.add_argument("-s", "--upscale", type=int, default=4, help="Final upsampling scale.")
    parser.add_argument("-w", "--weight", type=float, default=0.5, help="Adjustable weight for v1.4.")
    parser.add_argument("--aligned", action="store_true", help="Input images are already cropped/aligned faces.")
    parser.add_argument("--only-center-face", dest="only_center_face", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    restorer = build_restorer(args)

    csv_files = discover_csvs(args)
    print(f"Found {len(csv_files)} experiment CSV(s):")
    for f in csv_files:
        print(f"  - {f}")

    for csv_path in csv_files:
        process_csv(csv_path, restorer, args)

    print("\nDone.")


if __name__ == "__main__":
    main()

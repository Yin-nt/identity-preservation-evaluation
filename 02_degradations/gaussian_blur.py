"""Gaussian blur degradation.

Convolves each source image with a 2D Gaussian kernel and resizes it to the
common target size. Paper severity levels use sigma in {10, 12, 15} (L1/L2/L3).

Input CSV columns : person, test_path, ref_path (paths relative to BASE_DIR)
Output            : degraded images on disk + an index CSV with a Source_Path
                    column pointing at each degraded image.
"""

import os

import pandas as pd
from PIL import Image, ImageFilter
from tqdm import tqdm

# --- Configuration ---
# Dataset root and pose. Override with the IPE_DATA_ROOT / IPE_POSE env vars.
BASE_DIR = os.environ.get("IPE_DATA_ROOT", "data")
POSE = os.environ.get("IPE_POSE", "side")  # "frontal" or "side"

TARGET_SIZE = (512, 512)
BLUR_SIGMA = 10  # Paper levels: 10 (L1), 12 (L2), 15 (L3)

INPUT_CSV = os.path.join(BASE_DIR, f"lfw_{POSE}_test.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, f"image/{POSE}/gaussian_{BLUR_SIGMA}")
OUTPUT_CSV = os.path.join(BASE_DIR, f"csv/{POSE}/gaussian_{BLUR_SIGMA}.csv")


def resize_and_blur_dataset():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: input CSV not found: {INPUT_CSV}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    source_paths = []

    print(f"Processing {len(df)} images | size={TARGET_SIZE} | sigma={BLUR_SIGMA}")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Gaussian blur"):
        original_path = os.path.join(BASE_DIR, row["test_path"])
        name_part, ext = os.path.splitext(os.path.basename(original_path))
        save_path = os.path.join(OUTPUT_DIR, f"{name_part}_blurred_{BLUR_SIGMA}{ext}")

        try:
            if not os.path.exists(original_path):
                raise FileNotFoundError(original_path)
            with Image.open(original_path) as img:
                img = img.convert("RGB")
                blurred = img.filter(ImageFilter.GaussianBlur(radius=BLUR_SIGMA))
                resized = blurred.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                resized.save(save_path, quality=95, subsampling=0)
                source_paths.append(os.path.abspath(save_path))
        except Exception as e:
            print(f"\nError at row {index} ({name_part}): {e}")
            source_paths.append(None)

    df["Source_Path"] = source_paths
    df_clean = df.dropna(subset=["Source_Path"])
    df_clean.to_csv(OUTPUT_CSV, index=False)

    print(f"Done. Images: {OUTPUT_DIR} | index CSV: {OUTPUT_CSV}")
    print(f"Succeeded: {len(df_clean)}/{len(df)}")


if __name__ == "__main__":
    resize_and_blur_dataset()

"""Down-up scaling degradation.

Downsamples each image by an integer factor with bicubic interpolation, then
upsamples it back to the target size. Paper severity levels use factors
r in {5, 8, 12} (L1/L2/L3).

Input CSV columns : person, test_path, ref_path (paths relative to BASE_DIR)
Output            : degraded images on disk + one index CSV per level.
"""

import os

import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
# Dataset root and pose. Override with the IPE_DATA_ROOT / IPE_POSE env vars.
BASE_DIR = os.environ.get("IPE_DATA_ROOT", "data")
POSE = os.environ.get("IPE_POSE", "frontal")  # "frontal" or "side"
INPUT_CSV = os.path.join(BASE_DIR, f"lfw_{POSE}_test.csv")
OUTPUT_ROOT = os.path.join(BASE_DIR, f"image/{POSE}")
CSV_OUTPUT_DIR = os.path.join(BASE_DIR, f"csv/{POSE}")

TARGET_SIZE = (512, 512)
LEVEL_FACTORS = {1: 5, 2: 8, 3: 12}  # Paper down-up scaling factors


def apply_down_up_bicubic(img_cv, factor):
    h, w = img_cv.shape[:2]
    low = cv2.resize(
        img_cv,
        (max(1, int(w / factor)), max(1, int(h / factor))),
        interpolation=cv2.INTER_CUBIC,
    )
    final = cv2.resize(low, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: input CSV not found: {INPUT_CSV}")
        return

    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
    for level in LEVEL_FACTORS:
        os.makedirs(os.path.join(OUTPUT_ROOT, "downup", f"L{level}"), exist_ok=True)

    df_input = pd.read_csv(INPUT_CSV)
    results = {level: [] for level in LEVEL_FACTORS}

    for _, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Down-up scaling"):
        orig_path = os.path.join(BASE_DIR, row["test_path"])
        if not os.path.exists(orig_path):
            continue

        img_cv = cv2.imread(orig_path)
        if img_cv is None:
            continue

        base_name = os.path.splitext(os.path.basename(orig_path))[0]
        for level, factor in LEVEL_FACTORS.items():
            try:
                img_res = apply_down_up_bicubic(img_cv, factor)
                save_name = f"{base_name}_downup_L{level}.png"
                save_path = os.path.join(OUTPUT_ROOT, "downup", f"L{level}", save_name)
                img_res.save(save_path)
                results[level].append({
                    "person": row["person"],
                    "test_path": row["test_path"],
                    "ref_path": row["ref_path"],
                    "yaw": row.get("yaw", 0),
                    "Source_Path": os.path.abspath(save_path),
                })
            except Exception as e:
                print(f"Error downup L{level} at {base_name}: {e}")

    print("\nSaving CSV files...")
    for level, data in results.items():
        if data:
            csv_path = os.path.join(CSV_OUTPUT_DIR, f"downup_L{level}.csv")
            pd.DataFrame(data).to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()

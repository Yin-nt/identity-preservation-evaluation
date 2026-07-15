"""Motion blur degradation.

Convolves each image with a linear motion kernel of length L at a fixed
orientation, then resizes to the target size. Paper severity levels use kernel
lengths L in {12, 22, 35} (L1/L2/L3).

Input CSV columns : person, test_path, ref_path (paths relative to BASE_DIR)
Output            : degraded images on disk + one index CSV per level.
"""

import os

import cv2
import numpy as np
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
KERNEL_ANGLE = 35            # Fixed kernel orientation in degrees
LEVEL_LENGTHS = {1: 12, 2: 22, 3: 35}  # Paper motion-blur kernel lengths


def apply_motion_blur(img_cv, length):
    kernel = np.zeros((length, length))
    kernel[int((length - 1) / 2), :] = np.ones(length)
    matrix = cv2.getRotationMatrix2D((length / 2, length / 2), KERNEL_ANGLE, 1)
    kernel = cv2.warpAffine(kernel, matrix, (length, length))
    kernel /= kernel.sum()
    res = cv2.filter2D(img_cv, -1, kernel)
    res = cv2.resize(res, TARGET_SIZE)
    return Image.fromarray(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: input CSV not found: {INPUT_CSV}")
        return

    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
    for level in LEVEL_LENGTHS:
        os.makedirs(os.path.join(OUTPUT_ROOT, "motion", f"L{level}"), exist_ok=True)

    df_input = pd.read_csv(INPUT_CSV)
    results = {level: [] for level in LEVEL_LENGTHS}

    for _, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Motion blur"):
        orig_path = os.path.join(BASE_DIR, row["test_path"])
        if not os.path.exists(orig_path):
            continue

        img_cv = cv2.imread(orig_path)
        if img_cv is None:
            continue

        base_name = os.path.splitext(os.path.basename(orig_path))[0]
        for level, length in LEVEL_LENGTHS.items():
            try:
                img_res = apply_motion_blur(img_cv, length)
                save_name = f"{base_name}_motion_L{level}.png"
                save_path = os.path.join(OUTPUT_ROOT, "motion", f"L{level}", save_name)
                img_res.save(save_path)
                results[level].append({
                    "person": row["person"],
                    "test_path": row["test_path"],
                    "ref_path": row["ref_path"],
                    "yaw": row.get("yaw", 0),
                    "Source_Path": os.path.abspath(save_path),
                })
            except Exception as e:
                print(f"Error motion L{level} at {base_name}: {e}")

    print("\nSaving CSV files...")
    for level, data in results.items():
        if data:
            csv_path = os.path.join(CSV_OUTPUT_DIR, f"motion_L{level}.csv")
            pd.DataFrame(data).to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()

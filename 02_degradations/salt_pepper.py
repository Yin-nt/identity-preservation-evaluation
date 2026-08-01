"""Pure salt-and-pepper (impulse) noise degradation.

Each pixel is independently flipped to pure black or pure white with
probability p; no Gaussian noise and no blur are applied. This matches the
manuscript's description of the salt-and-pepper condition. Paper severity
levels use p in {0.005, 0.012, 0.025} (L1/L2/L3).

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
POSE = os.environ.get("IPE_POSE", "side")  # "frontal" or "side"
INPUT_CSV = os.path.join(BASE_DIR, f"lfw_{POSE}_test.csv")
OUTPUT_ROOT = os.path.join(BASE_DIR, f"image/{POSE}")
CSV_OUTPUT_DIR = os.path.join(BASE_DIR, f"csv/{POSE}")

TARGET_SIZE = (512, 512)
LEVEL_PROBS = {1: 0.005, 2: 0.012, 3: 0.025}  # Paper salt-and-pepper probabilities


def apply_salt_and_pepper(img_cv, p_sp):
    img_float = img_cv.astype(np.float32)
    h, w, _c = img_float.shape

    if p_sp > 0:
        rand = np.random.rand(h, w)
        img_float[rand < (p_sp / 2)] = 255.0
        img_float[(rand >= (p_sp / 2)) & (rand < p_sp)] = 0.0

    result = np.clip(img_float, 0, 255).astype(np.uint8)
    result = cv2.resize(result, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: input CSV not found: {INPUT_CSV}")
        return

    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
    for level in LEVEL_PROBS:
        os.makedirs(os.path.join(OUTPUT_ROOT, "saltpepper", f"L{level}"), exist_ok=True)

    df_input = pd.read_csv(INPUT_CSV)
    results = {level: [] for level in LEVEL_PROBS}

    for _, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Salt & pepper"):
        orig_path = os.path.join(BASE_DIR, row["test_path"])
        if not os.path.exists(orig_path):
            continue

        img_cv = cv2.imread(orig_path)
        if img_cv is None:
            continue

        base_name = os.path.splitext(os.path.basename(orig_path))[0]
        for level, p_sp in LEVEL_PROBS.items():
            try:
                img_res = apply_salt_and_pepper(img_cv, p_sp)
                save_name = f"{base_name}_saltpepper_L{level}.png"
                save_path = os.path.join(OUTPUT_ROOT, "saltpepper", f"L{level}", save_name)
                img_res.save(save_path)
                results[level].append({
                    "person": row["person"],
                    "test_path": row["test_path"],
                    "ref_path": row["ref_path"],
                    "yaw": row.get("yaw", 0),
                    "Source_Path": os.path.abspath(save_path),
                })
            except Exception as e:
                print(f"Error saltpepper L{level} at {base_name}: {e}")

    print("\nSaving CSV files...")
    for level, data in results.items():
        if data:
            csv_path = os.path.join(CSV_OUTPUT_DIR, f"saltpepper_L{level}.csv")
            pd.DataFrame(data).to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()

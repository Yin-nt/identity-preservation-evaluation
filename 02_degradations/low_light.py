"""Low-light degradation.

Combines an illumination-reduction transform with Poisson (shot) noise, then
applies a gain + gamma correction to emulate a camera ISO boost. Paper severity
levels use (light_factor, peak) of (0.25, 90), (0.10, 30), (0.05, 15) for
L1/L2/L3 respectively.

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
OUTPUT_ROOT = os.path.join(BASE_DIR, f"image/{POSE}/low-light")
CSV_OUTPUT_DIR = os.path.join(BASE_DIR, f"csv/{POSE}")

TARGET_SIZE = (512, 512)

# (light_factor, peak_photon):
#   light_factor - source intensity scale (lower = darker)
#   peak_photon  - simulated sensor photon count (lower = noisier Poisson spread)
LOW_LIGHT_CONFIGS = {
    1: (0.25, 90.0),   # L1: mildly dark, very light noise
    2: (0.10, 30.0),   # L2: darker, visible noise, face still recognisable
    3: (0.05, 15.0),   # L3: very dark, strong grain but face not fully destroyed
}


def apply_low_light_poisson(img_cv, level):
    if level not in LOW_LIGHT_CONFIGS:
        raise ValueError(f"Invalid level: {level}")

    light_factor, peak = LOW_LIGHT_CONFIGS[level]

    # Move to linear float space and reduce intensity (low-light environment).
    img_float = img_cv.astype(np.float32) / 255.0
    signal = img_float * light_factor

    # Poisson noise depends on per-pixel intensity of the original signal.
    noisy = np.random.poisson(signal * peak) / peak

    # Brighten back for display / evaluation (equivalent to raising ISO).
    gain = 1.0 / (light_factor * 0.85)
    enhanced = np.clip(noisy * gain, 1e-7, 1.0)

    gamma = 0.85 if level == 1 else 0.75
    enhanced = np.power(enhanced, gamma)

    result = (enhanced * 255).astype(np.uint8)
    result = cv2.resize(result, TARGET_SIZE)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: input CSV not found: {INPUT_CSV}")
        return

    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
    for level in LOW_LIGHT_CONFIGS:
        os.makedirs(os.path.join(OUTPUT_ROOT, f"L{level}"), exist_ok=True)

    df_input = pd.read_csv(INPUT_CSV)
    results = {level: [] for level in LOW_LIGHT_CONFIGS}

    for _, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Low-light"):
        orig_path = os.path.join(BASE_DIR, row["test_path"])
        if not os.path.exists(orig_path):
            continue

        img_cv = cv2.imread(orig_path)
        if img_cv is None:
            continue

        base_name = os.path.splitext(os.path.basename(orig_path))[0]
        for level in LOW_LIGHT_CONFIGS:
            try:
                img_res = apply_low_light_poisson(img_cv, level)
                save_name = f"{base_name}_lowlight_L{level}.png"
                save_path = os.path.join(OUTPUT_ROOT, f"L{level}", save_name)
                img_res.save(save_path)
                results[level].append({
                    "person": row["person"],
                    "test_path": row["test_path"],
                    "ref_path": row["ref_path"],
                    "yaw": row.get("yaw", 0),
                    "Source_Path": os.path.abspath(save_path),
                })
            except Exception as e:
                print(f"Error L{level} at {base_name}: {e}")

    print("\nSaving CSV files...")
    for level, data in results.items():
        if data:
            csv_path = os.path.join(CSV_OUTPUT_DIR, f"poisson_low_light_L{level}.csv")
            pd.DataFrame(data).to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()

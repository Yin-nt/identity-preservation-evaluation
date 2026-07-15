"""Salt-and-pepper noise degradation.

Each pixel is flipped to pure black or pure white with probability p, optionally
combined with additive Gaussian noise and a light final blur. Paper severity
levels use p in {0.005, 0.012, 0.025} (L1/L2/L3).

Input CSV columns : person, test_path, ref_path (paths relative to BASE_DIR)
Output            : degraded images on disk + an index CSV with a Source_Path.
"""

import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
# Dataset root and pose. Override with the IPE_DATA_ROOT / IPE_POSE env vars.
BASE_DIR = os.environ.get("IPE_DATA_ROOT", "data")
POSE = os.environ.get("IPE_POSE", "side")  # "frontal" or "side"

TARGET_SIZE = (512, 512)
GAUSSIAN_SIGMA = 22.0    # Additive Gaussian noise std (0 to disable)
P_SALT_PEPPER = 0.008    # Paper levels: 0.005 (L1), 0.012 (L2), 0.025 (L3)
FINAL_BLUR_SIGMA = 0.7   # Light blur applied after noise (0 to disable)

INPUT_CSV = os.path.join(BASE_DIR, f"lfw_{POSE}_test.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, f"image/{POSE}/sp{P_SALT_PEPPER}")
OUTPUT_CSV = os.path.join(BASE_DIR, f"csv/{POSE}/sp{P_SALT_PEPPER}.csv")


def apply_combined_noise(image_path, sigma_gauss, p_sp, final_blur_sigma, target_size):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_float = img.astype(np.float32)
    h, w, c = img_float.shape

    if sigma_gauss > 0:
        img_float += np.random.normal(0, sigma_gauss, (h, w, c))
    img_float = np.clip(img_float, 0, 255)

    if p_sp > 0:
        rand = np.random.rand(h, w)
        img_float[rand < (p_sp / 2)] = 255.0
        img_float[(rand >= (p_sp / 2)) & (rand < p_sp)] = 0.0

    if final_blur_sigma > 0:
        img_float = cv2.GaussianBlur(
            img_float, (0, 0), sigmaX=final_blur_sigma, sigmaY=final_blur_sigma
        )

    img_uint8 = np.clip(img_float, 0, 255).astype(np.uint8)
    return cv2.resize(img_uint8, target_size, interpolation=cv2.INTER_CUBIC)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    source_paths = []

    print(
        f"Gaussian (sigma={GAUSSIAN_SIGMA}) + Salt&Pepper (p={P_SALT_PEPPER}) "
        f"+ light blur (sigma={FINAL_BLUR_SIGMA})..."
    )

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Salt & pepper"):
        original_path = os.path.join(BASE_DIR, row["test_path"])
        name_part = os.path.splitext(os.path.basename(original_path))[0]
        save_name = (
            f"{name_part}_combined_g{GAUSSIAN_SIGMA}_sp{P_SALT_PEPPER}"
            f"_b{FINAL_BLUR_SIGMA:.1f}.png"
        )
        save_path = os.path.join(OUTPUT_DIR, save_name)

        try:
            degraded = apply_combined_noise(
                original_path,
                sigma_gauss=GAUSSIAN_SIGMA,
                p_sp=P_SALT_PEPPER,
                final_blur_sigma=FINAL_BLUR_SIGMA,
                target_size=TARGET_SIZE,
            )
            if degraded is not None:
                cv2.imwrite(save_path, degraded)
                source_paths.append(os.path.abspath(save_path))
            else:
                source_paths.append(None)
        except Exception as e:
            print(f"Error at {original_path}: {e}")
            source_paths.append(None)

    df["Source_Path"] = source_paths
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done. Images: {OUTPUT_DIR} | index CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

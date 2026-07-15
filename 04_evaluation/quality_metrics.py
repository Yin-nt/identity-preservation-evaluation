"""Perceptual image-quality metrics (PSNR / SSIM / LPIPS).

For each genuine (positive) pair, compares the restored best-frame against the
clean test image and reports PSNR, SSIM and LPIPS. Restored frames are read from
`<RESTORED_BASE_DIR>/<experiment>/best_frames_debug/<person>/` (as written by the
similarity-extraction scripts). Only rows with label == 1 are used.
"""

import glob
import os

import cv2
import lpips
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from tqdm import tqdm

# --- Configuration ---
# Dataset root. Override with the IPE_DATA_ROOT environment variable.
DATA_ROOT = os.environ.get("IPE_DATA_ROOT", "data")
# Point CSV_INPUT_DIR at the folder holding the (positive) pair CSVs.
CSV_INPUT_DIR = os.path.join(DATA_ROOT, "csv/side/negativeSample")
RESTORED_BASE_DIR = os.path.join(DATA_ROOT, "reports/insightface/side")
OUTPUT_CSV = os.path.join(RESTORED_BASE_DIR, "all_quality_metrics_combined.csv")

print("Loading LPIPS (VGG) model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)
loss_fn_vgg.eval()


def calculate_metrics(img_gt, img_restored):
    if img_gt.shape != img_restored.shape:
        img_restored = cv2.resize(img_restored, (img_gt.shape[1], img_gt.shape[0]))

    img_gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
    img_res_rgb = cv2.cvtColor(img_restored, cv2.COLOR_BGR2RGB)

    psnr_val = compute_psnr(img_gt_rgb, img_res_rgb, data_range=255)
    ssim_val = compute_ssim(img_gt_rgb, img_res_rgb, channel_axis=-1, data_range=255)

    # LPIPS expects tensors in [-1, 1].
    gt_tensor = torch.from_numpy(img_gt_rgb.transpose(2, 0, 1)).float() / 255.0 * 2.0 - 1.0
    res_tensor = torch.from_numpy(img_res_rgb.transpose(2, 0, 1)).float() / 255.0 * 2.0 - 1.0
    gt_tensor = gt_tensor.unsqueeze(0).to(device)
    res_tensor = res_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        lpips_val = loss_fn_vgg(gt_tensor, res_tensor).item()

    return psnr_val, ssim_val, lpips_val


def main():
    if not os.path.exists(CSV_INPUT_DIR):
        print(f"Error: folder {CSV_INPUT_DIR} does not exist.")
        return

    csv_files = glob.glob(os.path.join(CSV_INPUT_DIR, "*.csv"))
    if not csv_files:
        print(f"No .csv files found in {CSV_INPUT_DIR}.")
        return

    print(f"Found {len(csv_files)} CSV files. Running on {device}...\n")
    all_results = []

    for csv_path in csv_files:
        csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
        experiment = csv_basename.replace("_negative", "").replace("_positive", "")
        print(f"\nProcessing: {experiment} (from {csv_basename}.csv)")

        df = pd.read_csv(csv_path)

        # Image quality is only defined for genuine (positive) pairs.
        if "label" in df.columns:
            df = df[df["label"] == 1]
        elif "Label" in df.columns:
            df = df[df["Label"] == 1]
        if df.empty:
            print("   Skip: no positive (label==1) samples.")
            continue

        restored_dir = os.path.join(RESTORED_BASE_DIR, experiment, "best_frames_debug")
        if not os.path.exists(restored_dir):
            print(f"   Skip: restored frames not found at {restored_dir}")
            continue

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Metrics {experiment}"):
            person = row["person"]
            target_dir = os.path.join(restored_dir, person.replace(" ", "_"))
            if not os.path.exists(target_dir):
                continue

            restored_files = [f for f in os.listdir(target_dir) if f.endswith((".jpg", ".png"))]
            if not restored_files:
                continue

            img_gt = cv2.imread(row["test_path"])
            img_restored = cv2.imread(os.path.join(target_dir, restored_files[0]))
            if img_gt is None or img_restored is None:
                continue

            try:
                psnr_score, ssim_score, lpips_score = calculate_metrics(img_gt, img_restored)
                all_results.append({
                    "Experiment": experiment, "Person": person,
                    "PSNR": psnr_score, "SSIM": ssim_score, "LPIPS": lpips_score,
                })
            except Exception:
                pass

    if not all_results:
        print("\nNo results produced. Check that CSV_INPUT_DIR holds positive-labelled pairs.")
        return

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 75)
    print(f"{'MEAN METRICS PER EXPERIMENT':^75}")
    print("=" * 75)
    summary = df_results.groupby("Experiment")[["PSNR", "SSIM", "LPIPS"]].mean().reset_index()
    print(f"{'Experiment':<25} | {'PSNR (up)':<12} | {'SSIM (up)':<12} | {'LPIPS (down)':<12}")
    print("-" * 75)
    for _, row in summary.iterrows():
        print(f"{row['Experiment']:<25} | {row['PSNR']:<12.2f} | {row['SSIM']:<12.4f} | {row['LPIPS']:<12.4f}")
    print("=" * 75)
    print(f"Raw results saved to: {os.path.abspath(OUTPUT_CSV)}")


if __name__ == "__main__":
    main()

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import cv2
from tqdm.auto import tqdm

EXTS = (".png", ".jpg", ".jpeg", ".bmp")
ALIGNED_SIZE = 512


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="/content/data", help="root folder of degraded images")
    p.add_argument("--out-dir", default="/content/result_codeformer")
    p.add_argument("--log-dir", default="/content/result_codeformer_logs")
    p.add_argument("--tmp-dir", default="/content/cf_tmp")
    p.add_argument("--codeformer-dir", default="/content/CodeFormer", help="clone of sczhou/CodeFormer")
    p.add_argument("-w", "--fidelity", type=float, default=0.5, help="CodeFormer fidelity weight (paper: 0.5)")
    p.add_argument("--setup", action="store_true", help="patch and prepare the CodeFormer clone, then run")
    return p.parse_args()


# ==============================================================================
# SETUP
# ==============================================================================
def setup_codeformer(cf_dir):
    """Same fixes as the notebook's install cells."""
    # torchvision >= 0.17 removed transforms.functional_tensor
    for root, _, files in os.walk(cf_dir):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            with open(path, encoding="utf-8", errors="ignore") as fh:
                src = fh.read()
            if "torchvision.transforms.functional_tensor" in src:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(src.replace("torchvision.transforms.functional_tensor",
                                         "torchvision.transforms.functional"))
    r = subprocess.run([sys.executable, "basicsr/setup.py", "develop"], cwd=cf_dir,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0:
        print("basicsr/setup.py develop failed, falling back to version.py only")
    version_file = os.path.join(cf_dir, "basicsr", "version.py")
    if not os.path.exists(version_file):
        with open(version_file, "w") as fh:
            fh.write("__version__ = '1.3.2'\n__gitsha__ = 'unknown'\nversion_info = (1, 3, 2)\n")
        print("Created basicsr/version.py")


def codeformer_commit(cf_dir):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cf_dir, text=True).strip()
    except Exception:
        return "unknown"


# ==============================================================================
# CODEFORMER CALLS
# ==============================================================================
def run_cf(args, input_dir, output_dir, n_imgs, pbar, aligned=False):
    """Run inference_codeformer.py on a folder; advance pbar as outputs appear."""
    cmd = [sys.executable, "inference_codeformer.py", "-w", str(args.fidelity), "-s", "1",
           "--input_path", input_dir, "--output_path", output_dir]
    if aligned:
        cmd.append("--has_aligned")
    # detect mode writes final_results/, aligned mode writes restored_faces/
    watch = os.path.join(output_dir, "restored_faces" if aligned else "final_results", "*.png")
    err_file = output_dir.rstrip("/\\") + "_stderr.txt"
    os.makedirs(os.path.dirname(err_file), exist_ok=True)
    done = 0
    with open(err_file, "w") as ef:
        p = subprocess.Popen(cmd, cwd=args.codeformer_dir, stdout=subprocess.DEVNULL, stderr=ef)
        while p.poll() is None:
            time.sleep(1)
            n = min(len(glob.glob(watch)), n_imgs)
            pbar.update(n - done)
            done = n
    n = min(len(glob.glob(watch)), n_imgs)
    pbar.update(n - done)
    if p.returncode != 0:
        with open(err_file) as ef:
            raise RuntimeError(ef.read()[-1500:])
    return n


def save_like(img, dst_path):
    """Save with the input's extension; JPEG at quality 100."""
    q = [cv2.IMWRITE_JPEG_QUALITY, 100] if dst_path.lower().endswith((".jpg", ".jpeg")) else []
    cv2.imwrite(dst_path, img, q)


def list_imgs(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(EXTS))


# ==============================================================================
# MAIN
# ==============================================================================
def process_folder(args, d, pbar):
    """Two-pass restoration of one folder. Returns (log rows, n_failed_detect, pass-1 time)."""
    rel = os.path.relpath(d, args.data_dir)
    out_d = os.path.join(args.out_dir, rel)
    imgs = list_imgs(d)
    os.makedirs(out_d, exist_ok=True)

    tmp_detect = os.path.join(args.tmp_dir, rel, "detect")
    tmp_aligned = os.path.join(args.tmp_dir, rel, "aligned")
    tmp_aligned_in = os.path.join(args.tmp_dir, rel, "aligned_input")
    for p in (tmp_detect, tmp_aligned, tmp_aligned_in):
        shutil.rmtree(p, ignore_errors=True)

    # ---- Pass 1: detect mode ----
    t0 = time.time()
    run_cf(args, d, tmp_detect, len(imgs), pbar, aligned=False)
    t_detect = time.time() - t0

    rows, failed = [], []
    for f in imgs:
        stem = os.path.splitext(f)[0]
        n_faces = len(glob.glob(os.path.join(tmp_detect, "restored_faces", stem + "_*.png")))
        if n_faces > 0:
            save_like(cv2.imread(os.path.join(tmp_detect, "final_results", stem + ".png")),
                      os.path.join(out_d, f))
            rows.append([os.path.join(d, f), os.path.join(out_d, f), "detect", n_faces])
        else:
            failed.append(f)

    # ---- Pass 2: re-run undetected images with --has_aligned ----
    if failed:
        pbar.set_postfix_str(f"{rel} (aligned: {len(failed)})")
        os.makedirs(tmp_aligned_in, exist_ok=True)
        orig_size = {}
        for f in failed:
            im = cv2.imread(os.path.join(d, f))
            orig_size[f] = (im.shape[1], im.shape[0])
            cv2.imwrite(os.path.join(tmp_aligned_in, os.path.splitext(f)[0] + ".png"),
                        cv2.resize(im, (ALIGNED_SIZE, ALIGNED_SIZE), interpolation=cv2.INTER_CUBIC))
        # these images were already counted in the main bar during pass 1
        sub = tqdm(total=len(failed), leave=False, desc="  aligned")
        run_cf(args, tmp_aligned_in, tmp_aligned, len(failed), sub, aligned=True)
        sub.close()
        for f in failed:
            stem = os.path.splitext(f)[0]
            cands = glob.glob(os.path.join(tmp_aligned, "restored_faces", stem + "*.png"))
            if cands:
                out = cv2.resize(cv2.imread(cands[0]), orig_size[f], interpolation=cv2.INTER_CUBIC)
                save_like(out, os.path.join(out_d, f))
                rows.append([os.path.join(d, f), os.path.join(out_d, f), "aligned", 1])
            else:
                rows.append([os.path.join(d, f), "", "failed", 0])
    return rows, len(failed), t_detect


def main():
    args = parse_args()
    if args.setup:
        setup_codeformer(args.codeformer_dir)

    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, "run_info.json"), "w") as fh:
        json.dump({"codeformer_commit": codeformer_commit(args.codeformer_dir),
                   "fidelity_weight": args.fidelity, "upscale": 1,
                   "aligned_fallback_size": ALIGNED_SIZE}, fh, indent=1)

    img_dirs = sorted({root for root, _, fs in os.walk(args.data_dir)
                       if any(f.lower().endswith(EXTS) for f in fs)})

    def log_path_of(d):
        return os.path.join(args.log_dir, os.path.relpath(d, args.data_dir).replace(os.sep, "__") + ".csv")

    # folders that already have a log are skipped (resume after a Colab disconnect)
    todo = [d for d in img_dirs if not os.path.exists(log_path_of(d))]
    total = sum(len(list_imgs(d)) for d in todo)
    print(f"{len(img_dirs)} folders, {len(todo)} still to process ({total} images)")

    pbar = tqdm(total=total, desc="CodeFormer", unit="img")
    for d in todo:
        rel = os.path.relpath(d, args.data_dir)
        pbar.set_postfix_str(rel)
        try:
            rows, n_failed, t_detect = process_folder(args, d, pbar)
        except Exception as e:
            tqdm.write(f"[ERROR] {rel}: {e}")
            continue
        with open(log_path_of(d), "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
        n = len(rows)
        tqdm.write(f"[done] {rel}: {n} images | detect: {n - n_failed} | aligned: {n_failed} | "
                   f"{t_detect / max(n, 1):.3f}s/img (pass 1)")
    pbar.close()

    # ---- Merge per-folder logs ----
    pairs_csv = os.path.join(args.log_dir, "codeformer_pairs.csv")
    with open(pairs_csv, "w", newline="") as out:
        w = csv.writer(out)
        w.writerow(["degraded_path", "restored_path", "mode", "n_faces"])
        for p in sorted(glob.glob(os.path.join(args.log_dir, "*.csv"))):
            if os.path.abspath(p) != os.path.abspath(pairs_csv):
                with open(p, newline="") as fh:
                    w.writerows(csv.reader(fh))
    print("CSV:", pairs_csv)


if __name__ == "__main__":
    main()

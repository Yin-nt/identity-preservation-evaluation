"""Build Wan2GP image-to-video generation queues.

Reads an index CSV of degraded images and packs them into chunked .zip files.
Each zip bundles the input images plus a `queue.json` manifest describing the
Wan i2v job for every image. Per-image fields (resolution, image_start) are
filled in automatically; the shared job configuration (prompt, sampler,
guidance, model, etc.) is loaded from an external JSON template so it stays out
of source control.

Provide the template via --template (default: queue_template.json). See
queue_template.example.json for the expected structure.

Input CSV : any column starting with `path_`, or a `Source_Path` column.
"""

import argparse
import json
import os
import zipfile

import pandas as pd
from PIL import Image

# --- Configuration ---
# Dataset root. Override with the IPE_DATA_ROOT environment variable.
DATA_ROOT = os.environ.get("IPE_DATA_ROOT", "data")
INPUT_CSV = os.path.join(DATA_ROOT, "csv/frontal/blurred10.csv")
OUTPUT_ZIP_PREFIX = "gaussian10"
CHUNK_SIZE = 250
DEFAULT_TEMPLATE = "queue_template.json"


def load_template(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Queue template not found: {path}\n"
            "Copy queue_template.example.json and fill in your Wan i2v settings."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect_image_paths(df):
    image_cols = [c for c in df.columns if c.startswith("path_") or c == "Source_Path"]
    paths = []
    for _, row in df.iterrows():
        for col in image_cols:
            value = row[col]
            if pd.notna(value) and str(value).strip():
                paths.append(value)
    return paths


def create_split_zips(template_params):
    if not os.path.exists(INPUT_CSV):
        print(f"Input CSV not found: {INPUT_CSV}")
        return

    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    all_tasks = collect_image_paths(df)

    total = len(all_tasks)
    num_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Total images: {total}. Splitting into {num_chunks} zip file(s).")

    for i in range(num_chunks):
        start = i * CHUNK_SIZE
        end = min((i + 1) * CHUNK_SIZE, total)
        chunk_paths = all_tasks[start:end]
        output_zip = f"{OUTPUT_ZIP_PREFIX}_{i + 1}.zip"

        print(f"\nCreating {output_zip}...")
        manifest = []

        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for rel_idx, img_path in enumerate(chunk_paths):
                global_idx = start + rel_idx
                if not os.path.exists(img_path):
                    print(f"  missing image: {img_path}")
                    continue

                zip_image_name = f"task_{global_idx}_{os.path.basename(img_path)}"

                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        with zf.open(zip_image_name, "w") as img_file:
                            img.save(img_file, format="PNG")
                except Exception as e:
                    print(f"  image error {img_path}: {e}")
                    continue

                params = dict(template_params)
                params["resolution"] = f"{w}x{h}"
                params["image_start"] = zip_image_name
                manifest.append({"id": global_idx + 1, "params": params})

            zf.writestr("queue.json", json.dumps(manifest, indent=4))

        print(f"Created {output_zip} ({len(manifest)} tasks)")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Wan2GP i2v queue zips.")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE, help="Wan i2v job template JSON.")
    args = parser.parse_args()

    create_split_zips(load_template(args.template))

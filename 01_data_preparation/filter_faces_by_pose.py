"""Select shared frontal/side identities from LFW.

For every identity with at least four usable photos, estimates head pose with
InsightFace and splits candidates into frontal vs side by yaw/pitch thresholds.
An identity qualifies once it has at least one off-axis (side) candidate and
at least three frontal candidates. Each qualifying identity then contributes
four distinct photos, with each pose protocol having its own independent
source + reference pair and no photo shared between the two protocols:

  * frontal source     - the sharpest frontal photo (degraded input under the
                          frontal protocol).
  * frontal reference   - the second-sharpest frontal photo (undegraded
                          identity anchor for the frontal protocol only).
  * side source         - the strongest off-axis photo (degraded input under
                          the side-profile protocol).
  * side reference       - the third-sharpest frontal photo (undegraded
                          identity anchor for the side protocol only).

Outputs one CSV per protocol (person, test_path, ref_path, ...) and copies the
selected Test/Ref images into OUTPUT_DIR.
"""

import os
import shutil

import cv2
import pandas as pd
from insightface.app import FaceAnalysis
from tqdm import tqdm

# --- Configuration ---
# Dataset root. Override with the IPE_DATA_ROOT environment variable.
DATA_ROOT = os.environ.get("IPE_DATA_ROOT", "data")
MODEL_NAME = "buffalo_l"
LFW_PATH = os.path.join(DATA_ROOT, "input/lfw/lfw-deepfunneled")
OUTPUT_DIR = os.path.join(DATA_ROOT, "Experiment_Data_Split")
TARGET_IDENTITIES = 500

# Pose thresholds (degrees).
YAW_THRESHOLD = 25       # |yaw| above this -> side candidate
PITCH_THRESHOLD = 20     # |pitch| above this -> side candidate
FRONTAL_THRESHOLD = 10   # |yaw| and |pitch| below this -> frontal candidate

app = FaceAnalysis(
    name=MODEL_NAME,
    root=os.path.abspath("."),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(640, 640))


def get_face_info(img_path):
    """Return pose/sharpness for the largest detected face, or None."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        faces = app.get(img)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        pitch, yaw, _roll = face.pose
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        return {"path": img_path, "yaw": yaw, "pitch": pitch, "sharpness": sharpness}
    except Exception:
        return None


def main():
    side_group = []
    frontal_group = []

    for person_name in tqdm(os.listdir(LFW_PATH), desc="Filtering identities"):
        if len(side_group) >= TARGET_IDENTITIES:
            break

        person_dir = os.path.join(LFW_PATH, person_name)
        if not os.path.isdir(person_dir):
            continue

        photos = os.listdir(person_dir)
        if len(photos) < 4:
            continue

        data_list = [
            info
            for info in (get_face_info(os.path.join(person_dir, p)) for p in photos)
            if info
        ]
        if not data_list:
            continue

        side_cands = [
            d for d in data_list
            if abs(d["yaw"]) > YAW_THRESHOLD or abs(d["pitch"]) > PITCH_THRESHOLD
        ]
        front_cands = [
            d for d in data_list
            if abs(d["yaw"]) < FRONTAL_THRESHOLD and abs(d["pitch"]) < FRONTAL_THRESHOLD
        ]

        # Need a side photo plus three distinct frontal photos: frontal
        # source, frontal reference, side reference (all different).
        if not side_cands or len(front_cands) < 3:
            continue

        side_source = max(side_cands, key=lambda x: max(abs(x["yaw"]), abs(x["pitch"])))
        sorted_front = sorted(front_cands, key=lambda x: x["sharpness"], reverse=True)
        frontal_source = sorted_front[0]
        frontal_reference = sorted_front[1]
        side_reference = sorted_front[2]

        side_group.append({
            "person": person_name,
            "test_path": side_source["path"],
            "ref_path": side_reference["path"],
            "yaw": side_source["yaw"],
        })
        frontal_group.append({
            "person": person_name,
            "test_path": frontal_source["path"],
            "ref_path": frontal_reference["path"],
            "sharpness": frontal_source["sharpness"],
        })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_side = pd.DataFrame(side_group)
    df_front = pd.DataFrame(frontal_group)
    df_side.to_csv(os.path.join(DATA_ROOT, "lfw_side_test.csv"), index=False)
    df_front.to_csv(os.path.join(DATA_ROOT, "lfw_frontal_test.csv"), index=False)

    for group_name, dataset in [("Side_Exp", side_group), ("Frontal_Exp", frontal_group)]:
        base = os.path.join(OUTPUT_DIR, group_name)
        os.makedirs(os.path.join(base, "Test"), exist_ok=True)
        os.makedirs(os.path.join(base, "Ref"), exist_ok=True)
        for item in dataset:
            shutil.copy2(item["test_path"], os.path.join(base, "Test", f"{item['person']}_test.jpg"))
            shutil.copy2(item["ref_path"], os.path.join(base, "Ref", f"{item['person']}_ref.jpg"))

    print(f"Done. Selected {len(df_side)} shared identities (side + frontal protocols, "
          f"4 distinct photos each, no photo shared between protocols).")


if __name__ == "__main__":
    main()

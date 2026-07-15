"""Extract identity cosine similarity with DeepFace / FaceNet512.

For every verification pair in the input CSV, computes cosine similarity against
the reference embedding for three conditions:

  * GroundTruth : the clean test image.
  * Baseline    : the degraded input image.
  * Wan2.1      : the best frame of the generated video (see frame selection below).

Wan videos are scanned over their central portion (the first/last SKIP_*_PERCENT
are dropped); the top-K frames most similar to the reference are averaged and
L2-normalised to form the restored embedding. Results and a score-distribution
plot are written to OUTPUT_DIR.
"""

import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from deepface import DeepFace
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# --- Configuration ---
# Dataset root. Override with the IPE_DATA_ROOT environment variable.
DATA_ROOT = os.environ.get("IPE_DATA_ROOT", "data")
INPUT_CSV = os.path.join(DATA_ROOT, "csv/side/negativeSample/downup_L1_negative.csv")
VIDEO_DIR = os.path.join(DATA_ROOT, "video/side/downup_L1")
OUTPUT_DIR = os.path.join(DATA_ROOT, "reports/deepface/side/downup_L1")

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"
RECOGNITION_THRESHOLD = 0.15

# Video frame-selection config.
SKIP_START_PERCENT = 0.25
SKIP_END_PERCENT = 0.25
FRAME_STRIDE = 5
TOP_K_AVERAGE = 3
SAVE_BEST_FRAMES = True

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class FaceObj:
    def __init__(self, bbox, conf):
        self.bbox = bbox
        self.conf = conf


def cosine_similarity(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 0.0
    v1, v2 = np.array(emb1), np.array(emb2)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)


def get_embedding_robust(img, ref_emb=None):
    """Return (embedding, FaceObj) for the best face in img, or (None, None)."""
    if img is None:
        return None, None
    try:
        results = DeepFace.represent(
            img_path=img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
        )
        if not results:
            return None, None

        # Drop the whole-image fallback detection (confidence == 0).
        valid_faces = [f for f in results if f.get("face_confidence", 0) > 0]
        if not valid_faces:
            return None, None

        if ref_emb is not None:
            # Pick the face closest to the reference identity.
            target = max(valid_faces, key=lambda x: cosine_similarity(x["embedding"], ref_emb))
        else:
            # Pick the largest face.
            target = max(valid_faces, key=lambda x: x["facial_area"]["w"] * x["facial_area"]["h"])

        emb = np.array(target["embedding"])
        area = target["facial_area"]
        bbox = np.array([area["x"], area["y"], area["x"] + area["w"], area["y"] + area["h"]])
        return emb, FaceObj(bbox, target.get("face_confidence", 0))
    except Exception:
        return None, None


def get_best_embedding_from_video(video_path, person_name, output_base, ref_emb):
    """Scan a video, keep frames closest to ref_emb, return the fused embedding."""
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_f = int(total_frames * SKIP_START_PERCENT)
    end_f = int(total_frames * (1 - SKIP_END_PERCENT))

    candidates = []
    for i in range(start_f, end_f, FRAME_STRIDE):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        emb, face_obj = get_embedding_robust(frame, ref_emb)
        if emb is not None:
            sim = cosine_similarity(emb, ref_emb)
            candidates.append((sim, emb, i, frame.copy(), face_obj))
    cap.release()

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)

    if SAVE_BEST_FRAMES:
        save_dir = os.path.join(output_base, "best_frames_debug", person_name.replace(" ", "_"))
        os.makedirs(save_dir, exist_ok=True)
        for rank, (_sim, _emb, f_idx, frame, _face) in enumerate(candidates[:1], 1):
            cv2.imwrite(os.path.join(save_dir, f"top{rank}_frame_{f_idx}.jpg"), frame)

    # Embedding fusion: average the top-K embeddings and L2-normalise.
    if len(candidates) >= TOP_K_AVERAGE:
        top_embs = [c[1] for c in candidates[:TOP_K_AVERAGE]]
    else:
        top_embs = [candidates[0][1]]
    avg_emb = np.mean(top_embs, axis=0)
    return avg_emb / np.linalg.norm(avg_emb)


def main():
    if not os.path.exists(INPUT_CSV):
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)

    results = []
    print(f"Identity analysis | model: {MODEL_NAME}")

    for idx, row in df.iterrows():
        label, person = row["label"], row["person"]
        ref_p, test_p, source_p = row["ref_path"], row["test_path"], row["Source_Path"]
        video_name = os.path.splitext(os.path.basename(source_p))[0]
        wan_p = os.path.join(VIDEO_DIR, f"{video_name}.mp4")

        emb_ref, _ = get_embedding_robust(cv2.imread(ref_p))
        if emb_ref is None:
            continue

        emb_gt, _ = get_embedding_robust(cv2.imread(test_p), emb_ref)
        s_gt = cosine_similarity(emb_gt, emb_ref)

        emb_base, _ = get_embedding_robust(cv2.imread(source_p), emb_ref)
        s_base = cosine_similarity(emb_base, emb_ref)

        emb_wan = get_best_embedding_from_video(wan_p, person, OUTPUT_DIR, emb_ref)
        s_wan = cosine_similarity(emb_wan, emb_ref)

        print(f"[{idx + 1}/{len(df)}] {person[:10]:<10} | GT:{s_gt:.2f} Base:{s_base:.2f} Wan:{s_wan:.2f}")
        results.append({
            "Person": person, "Label": label,
            "GroundTruth": s_gt, "Baseline": s_base, "Wan2.1": s_wan,
        })

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(OUTPUT_DIR, "identity_comparison_report.csv"), index=False)

    y_true = df_res["Label"].values
    methods = ["GroundTruth", "Baseline", "Wan2.1"]

    print("\n" + "=" * 110)
    print(f"{'Method':<15} | {'MeanCos':<8} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<6}")
    print("-" * 110)
    for m in methods:
        y_pred = [1 if s >= RECOGNITION_THRESHOLD else 0 for s in df_res[m]]
        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"{m:<15} | {df_res[m].mean():.3f}   | {acc:<10.2%} | {pre:<10.2%} | {rec:<10.2%} | {f1:.3f}")

    plt.figure(figsize=(12, 6))
    df_melt = df_res.melt(
        id_vars=["Person", "Label"], value_vars=methods,
        var_name="Method", value_name="Similarity",
    )
    sns.boxplot(x="Method", y="Similarity", hue="Label", data=df_melt)
    plt.axhline(y=RECOGNITION_THRESHOLD, color="r", linestyle="--", label="Threshold")
    plt.title("Similarity Distribution: Ground Truth vs Baseline vs Wan2.1")
    plt.savefig(os.path.join(OUTPUT_DIR, "score_distribution.png"))
    print(f"Report and plot saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

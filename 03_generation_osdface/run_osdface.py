import argparse
import copy
import glob
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as Fun
import torchvision.transforms.functional as F
from PIL import Image
from safetensors import safe_open
from torchvision import transforms
from tqdm import tqdm

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

# Requires a local OSDFace repo checkout on PYTHONPATH.
from utils.others import get_x0_from_noise
from utils.vaehook import perfcount
from lq_embed import TwoLayerConv1x1, vqvae_encoder

# Dataset root / pose. Override with the IPE_DATA_ROOT / IPE_POSE env vars.
DATA_ROOT = os.environ.get("IPE_DATA_ROOT", "data")
POSE = os.environ.get("IPE_POSE", "frontal")


# ==============================================================================
# 1. MODEL
# ==============================================================================
class OSDFaceRestorer(nn.Module):
    """One-step diffusion face restoration (OSDFace)."""

    def __init__(self, args, gpu_id, unet=None):
        super().__init__()
        self.args = args
        self.device = torch.device(f"cuda:{gpu_id}")

        self.noise_scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_model_path, subfolder="scheduler"
        )
        self.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")

        if args.merge_lora:
            self.unet = copy.deepcopy(unet)
        else:
            self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")

        self.weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

        self.load_ckpt(args.ckpt_path)

        self.img_encoder = vqvae_encoder(args).to(self.device, dtype=self.weight_dtype)
        self.unet.to(self.device, dtype=self.weight_dtype)
        self.vae.to(self.device, dtype=self.weight_dtype)
        self.img_encoder.to(self.device, dtype=self.weight_dtype)

        self.timesteps = 399

    def load_ckpt(self, ckpt_path):
        if not self.args.cat_prompt_embedding:
            self.embedding_change = TwoLayerConv1x1(512, 1024)
            self.embedding_change.load_state_dict(
                torch.load(os.path.join(ckpt_path, "embedding_change_weights.pth"), weights_only=False)
            )
            self.embedding_change.to(self.device, dtype=self.weight_dtype)

        if not self.args.merge_lora:
            pipe = StableDiffusionPipeline(self.vae, None, None, self.unet, self.noise_scheduler, None, None)
            pipe.load_lora_weights(ckpt_path)
            self.unet = pipe.unet

    @perfcount
    @torch.no_grad()
    def forward(self, lq):
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()

        with torch.cuda.stream(stream1):
            prompt_embeds = self.img_encoder(lq).reshape(lq.shape[0], 77, -1)
            if not self.args.cat_prompt_embedding:
                prompt_embeds = self.embedding_change(prompt_embeds)

        with torch.cuda.stream(stream2):
            lq_latent = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample() * self.vae.config.scaling_factor

        torch.cuda.synchronize()

        model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds).sample

        x_0 = get_x0_from_noise(
            lq_latent.double(), model_pred.double(), self.alphas_cumprod.double(), self.timesteps
        ).float()

        output_image = self.vae.decode(x_0.to(self.weight_dtype) / self.vae.config.scaling_factor).sample
        output_image = (output_image.clamp(-1, 1) * 0.5 + 0.5).clamp(0.0, 1.0)
        return output_image


def merge_lora_weights(args):
    """Fold trained LoRA weights into the base UNet (used when --merge-lora)."""
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")
    alpha = float(args.lora_alpha / args.lora_rank)
    processed_keys = set()

    with safe_open(os.path.join(args.ckpt_path, "pytorch_lora_weights.safetensors"), framework="pt") as f:
        state_dict = {key: f.get_tensor(key) for key in f.keys()}

    state_dict_unet = unet.state_dict()

    for key in state_dict.keys():
        if "lora_A" in key:
            lora_b_key = key.replace("lora_A", "lora_B")
            unet_key = key.replace(".lora_A.weight", ".weight").replace("unet.", "")
            W_A, W_B = state_dict[key], state_dict[lora_b_key]
            original_weight = state_dict_unet[unet_key]
            processed_keys.update([key, lora_b_key])

            if original_weight.dim() == 4 and W_A.dim() == 4 and W_B.dim() == 4:
                out_channels, in_channels, kH, kW = original_weight.shape
                rank = W_A.shape[0]
                delta_W = torch.matmul(W_B.view(out_channels, rank), W_A.view(rank, -1)).view(
                    out_channels, in_channels, kH, kW
                )
                merged_weight = original_weight + alpha * delta_W
            else:
                merged_weight = original_weight + alpha * torch.mm(W_B, W_A)
            state_dict_unet[unet_key] = merged_weight

        elif "lora.up.weight" in key:
            lora_down_key = key.replace("lora.up.weight", "lora.down.weight")
            original_key = key.replace(".lora.up.weight", ".weight").replace("unet.", "")
            W_up, W_down, W_orig = state_dict[key], state_dict[lora_down_key], state_dict_unet[original_key]
            processed_keys.update([key, lora_down_key])

            if W_orig.ndim != 2:
                print(f"Warning: unhandled weight shape for {original_key}")
                continue
            state_dict_unet[original_key] = W_orig + alpha * torch.matmul(W_up, W_down)

    remaining = [k for k in state_dict.keys() if k not in processed_keys]
    if remaining:
        print("Warning: unprocessed LoRA weights:", remaining)

    unet.load_state_dict(state_dict_unet)
    print("LoRA merge done.")
    return unet


# ==============================================================================
# 2. CSV DISCOVERY (same conventions as 04_evaluation/extract_similarity_*.py)
# ==============================================================================
def parse_groups(groups_text):
    if not groups_text.strip():
        return []
    return [g.strip().lower() for g in groups_text.split(",") if g.strip()]


def infer_experiment_name(csv_path):
    return Path(csv_path).stem


def group_match(exp_name, groups):
    if not groups:
        return True
    name = exp_name.lower()
    return any(name == g or name.startswith(g + "_") or name.startswith(g) or g in name for g in groups)


def discover_csvs(args):
    groups = parse_groups(args.groups)
    files = sorted(glob.glob(os.path.join(args.csv_dir, "*.csv")))
    return [f for f in files if group_match(infer_experiment_name(f), groups)]


# ==============================================================================
# 3. INFERENCE WORKERS (multi-GPU via a result Queue to avoid CSV race conditions)
# ==============================================================================
def worker(unet, gpu_id, jobs, weight_dtype, args, result_queue):
    torch.cuda.set_device(gpu_id)
    model = OSDFaceRestorer(args, gpu_id, unet).to(gpu_id)

    for row_idx, source_path, out_path in tqdm(jobs, desc=f"GPU{gpu_id}"):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        input_image = Image.open(source_path).convert("RGB")

        with torch.no_grad():
            lq = F.to_tensor(input_image).unsqueeze(0).to(gpu_id, dtype=weight_dtype) * 2 - 1
            if lq.shape[2] == lq.shape[3]:
                lq = Fun.interpolate(lq, (512, 512), mode="bilinear", align_corners=True)
            output_image = model(lq)
            transforms.ToPILImage()(output_image[0].cpu()).save(out_path)

        result_queue.put((row_idx, out_path))


def process_csv(csv_path, unet, args):
    exp_name = infer_experiment_name(csv_path)
    print("\n" + "#" * 90)
    print(f"OSDFace: {exp_name}")
    print(f"CSV: {csv_path}")
    print("#" * 90)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "Source_Path" not in df.columns:
        print("Skip (missing Source_Path column).")
        return

    if "OSDFace_path" not in df.columns:
        df["OSDFace_path"] = ""

    out_dir = os.path.join(DATA_ROOT, "image", POSE, "osdface", exp_name)

    jobs = []
    for idx, row in df.iterrows():
        source_path = str(row["Source_Path"])
        existing = row.get("OSDFace_path", "")
        if isinstance(existing, str) and existing and os.path.exists(existing):
            continue
        if not os.path.exists(source_path):
            print(f"  missing source image: {source_path}")
            continue
        out_path = os.path.join(out_dir, os.path.basename(source_path))
        jobs.append((idx, source_path, out_path))

    print(f"Rows to process: {len(jobs)} / {len(df)} (rest already restored)")
    if not jobs:
        return

    random.shuffle(jobs)
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

    num_gpus = len(args.gpu_ids)
    chunk_size = (len(jobs) + num_gpus - 1) // num_gpus
    result_queue = mp.Queue()
    processes = []

    for rank, gpu_id in enumerate(args.gpu_ids):
        subset = jobs[rank * chunk_size: (rank + 1) * chunk_size]
        if not subset:
            continue
        p = mp.Process(target=worker, args=(unet, gpu_id, subset, weight_dtype, args, result_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    for row_idx, out_path in results:
        df.at[row_idx, "OSDFace_path"] = os.path.abspath(out_path)

    df.to_csv(csv_path, index=False)
    print(f"Updated OSDFace_path for {len(results)} row(s) -> {csv_path}")


# ==============================================================================
# 4. MAIN
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run OSDFace restoration over a degradation index CSV.")

    parser.add_argument(
        "--csv-dir",
        default=os.path.join(DATA_ROOT, f"csv/{POSE}/negativeSample"),
        help="Folder of index CSVs, one per experiment (each needs a Source_Path column).",
    )
    parser.add_argument("--groups", default="", help="Filter experiments, e.g. motion or motion_L2,lowlight_L3.")

    parser.add_argument("--pretrained-model-path", required=True, dest="pretrained_model_path",
                         help="Path to the Stable-Diffusion-2.1-base checkpoint.")
    parser.add_argument("--ckpt-path", required=True, dest="ckpt_path", help="Path to the OSDFace LoRA/ckpt weights.")
    parser.add_argument("--mixed-precision", dest="mixed_precision", choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--merge-lora", dest="merge_lora", action="store_true", help="Fold LoRA weights into the base UNet before inference.")
    parser.add_argument("--lora-rank", dest="lora_rank", type=int, default=16)
    parser.add_argument("--lora-alpha", dest="lora_alpha", type=float, default=16)
    parser.add_argument("--cat-prompt-embedding", dest="cat_prompt_embedding", action="store_true",
                         help="Use concatenated prompt embedding instead of the embedding_change projection.")
    parser.add_argument("--gpu-ids", dest="gpu_ids", nargs="+", type=int, default=[0], help="GPU IDs to use.")
    parser.add_argument("--seed", type=int, default=114)

    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    unet = merge_lora_weights(args) if args.merge_lora else None

    csv_files = discover_csvs(args)
    print(f"Found {len(csv_files)} experiment CSV(s):")
    for f in csv_files:
        print(f"  - {f}")

    for csv_path in csv_files:
        process_csv(csv_path, unet, args)

    print("\nDone.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

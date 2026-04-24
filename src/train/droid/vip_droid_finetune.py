# =============================================================================
# vip_droid_finetune.py
#
# Fine-tunes the VIP ResNet-50 encoder on the DROID robot manipulation dataset,
# preserving VIP's original training objective (Algorithm 1 / Equation 6).
#
# Paper references:
#   Ma et al. 2023 — VIP: Value-Implicit Pre-training for Robot Learning (ICLR)
#   Khazatsky et al. 2024 — DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset
#
# KEY DESIGN: STREAMING DATA LOADER
#   Episodes are streamed from TFRecord files one at a time — never accumulated
#   in RAM. A small in-memory buffer (--buffer_size episodes) is maintained.
#   RAM usage stays flat regardless of dataset size.
#
# Usage:
#   python vip_droid_finetune.py \
#       --data_dir  ./droid_full/1.0.1 \
#       --out_dir   ./checkpoints \
#       --epochs    5 \
#       --lr        3e-5 \
#       --batch_size 16 \
#       --buffer_size 100 \
#       --steps_per_epoch 200
#
# Checkpoint loading in vip_metaworld_multitask.py:
#   from vip_droid_finetune import load_finetuned_vip
#   vip = load_finetuned_vip("checkpoints/vip_droid_finetuned.pt")
# =============================================================================

import os
import glob
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import io

from vip import load_vip

try:
    from tfrecord.reader import tfrecord_loader
    TFRECORD_AVAILABLE = True
except ImportError:
    TFRECORD_AVAILABLE = False
    print("[Warning] tfrecord not found. Install with: pip install tfrecord")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune VIP encoder on DROID TFRecord dataset (streaming)")
    p.add_argument("--data_dir",        type=str, required=True)
    p.add_argument("--out_dir",         type=str, default="./checkpoints")
    p.add_argument("--epochs",          type=int, default=5)
    p.add_argument("--lr",              type=float, default=3e-5)
    p.add_argument("--batch_size",      type=int, default=16)
    p.add_argument("--buffer_size",     type=int, default=100,
                   help="Episodes kept in RAM at once. 100 ~= 2.5GB RAM.")
    p.add_argument("--steps_per_epoch", type=int, default=200,
                   help="Training steps per epoch.")
    p.add_argument("--gamma",           type=float, default=0.98)
    p.add_argument("--num_negatives",   type=int, default=3)
    p.add_argument("--camera",          type=str, default="exterior_image_1_left",
                   choices=["exterior_image_1_left", "exterior_image_2_left"])
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--save_every",      type=int, default=1)
    p.add_argument("--resume",          type=str, default=None)
    p.add_argument("--finetune_mode",   type=str, default="projection_head",
                   choices=["projection_head", "full"],
                   help="projection_head: freeze backbone, tune only FC layer (recommended). "
                        "full: unfreeze entire encoder (risk of catastrophic forgetting).")
    return p.parse_args()


# =============================================================================
# Preprocessing
# =============================================================================

NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])


def preprocess_frame(frame_np):
    t = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0
    return T.functional.resize(t, [256])


def apply_crop(frame_t, i, j):
    return NORMALIZE(frame_t[:, i:i+224, j:j+224])


# =============================================================================
# Streaming episode reader
# =============================================================================

def episode_stream(tfrecord_files, camera_key, rng):
    """
    Generator that yields one episode dict at a time.
    Yields: { "frames": np.ndarray [T,H,W,3] }
    """
    assert TFRECORD_AVAILABLE, "pip install tfrecord"

    img_key     = f"steps/observation/{camera_key}"
    description = {img_key: "byte"}

    files = tfrecord_files.copy()
    rng.shuffle(files)

    for tf_path in files:
        try:
            loader = tfrecord_loader(tf_path, None, description)
            for record in loader:
                try:
                    raw_imgs = record.get(img_key)
                    if raw_imgs is None or len(raw_imgs) < 10:
                        continue

                    frames = []
                    for jpeg_bytes in raw_imgs:
                        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
                        frames.append(np.array(img, dtype=np.uint8))

                    if len(frames) < 10:
                        continue

                    yield {"frames": np.stack(frames)}

                except Exception:
                    continue
        except Exception as e:
            print(f"  [Skip shard] {e}")
            continue


class StreamingBuffer:
    """Rolling buffer — same as r3m_droid_finetune.py."""
    def __init__(self, stream_gen, buffer_size, rng):
        self.stream      = stream_gen
        self.buffer_size = buffer_size
        self.rng         = rng
        self.buffer      = []
        self._fill()

    def _fill(self):
        while len(self.buffer) < self.buffer_size:
            try:
                self.buffer.append(next(self.stream))
            except StopIteration:
                break

    def sample(self, n):
        if not self.buffer:
            return []
        sampled = self.rng.choices(self.buffer, k=n)
        for _ in range(min(n, len(self.buffer))):
            try:
                idx = self.rng.randint(0, len(self.buffer) - 1)
                self.buffer[idx] = next(self.stream)
            except StopIteration:
                break
        return sampled

    def size(self):
        return len(self.buffer)


# =============================================================================
# Sampling — VIP format: {o_t, o_T, o_k[N], o_k1[N]}
# =============================================================================

def sample_vip_frames(episode, num_negatives, rng):
    frames = episode["frames"]
    T_len  = frames.shape[0]

    early_end  = max(1, int(0.20 * T_len))
    late_start = min(T_len - 1, int(0.80 * T_len))

    t_idx = rng.randint(0, early_end - 1)
    T_idx = rng.randint(late_start, T_len - 1)
    if T_idx - t_idx < 3:
        T_idx = min(T_len - 1, t_idx + 3)

    neg_pairs = []
    for _ in range(num_negatives):
        k = rng.randint(t_idx + 1, max(t_idx + 2, T_idx - 2))
        neg_pairs.append((k, k + 1))

    all_needed = set([t_idx, T_idx])
    for k, k1 in neg_pairs:
        all_needed.add(k); all_needed.add(k1)

    sample_r = preprocess_frame(frames[t_idx])
    H_res, W_res = sample_r.shape[1], sample_r.shape[2]
    ci = rng.randint(0, max(0, H_res - 224))
    cj = rng.randint(0, max(0, W_res - 224))

    raw = {fi: apply_crop(preprocess_frame(frames[fi]), ci, cj)
           for fi in all_needed}

    return {
        "o_t":  raw[t_idx],
        "o_T":  raw[T_idx],
        "o_k":  torch.stack([raw[k]  for k, k1 in neg_pairs]),
        "o_k1": torch.stack([raw[k1] for k, k1 in neg_pairs]),
    }


def make_batch(buffer, batch_size, num_negatives, rng):
    episodes = buffer.sample(batch_size)
    if not episodes:
        return None
    items = [sample_vip_frames(ep, num_negatives, rng) for ep in episodes]
    return {
        "o_t":  torch.stack([it["o_t"]  for it in items]),
        "o_T":  torch.stack([it["o_T"]  for it in items]),
        "o_k":  torch.stack([it["o_k"]  for it in items]),
        "o_k1": torch.stack([it["o_k1"] for it in items]),
    }


# =============================================================================
# VIP Loss — Equation 6 / Algorithm 1
# =============================================================================

def vip_loss(encoder, batch, gamma, device):
    B = batch["o_t"].shape[0]
    N = batch["o_k"].shape[1]

    o_t  = batch["o_t"].to(device)
    o_T  = batch["o_T"].to(device)
    o_k  = batch["o_k"].to(device).view(B*N, 3, 224, 224)
    o_k1 = batch["o_k1"].to(device).view(B*N, 3, 224, 224)

    phi_t  = encoder(o_t)
    phi_T  = encoder(o_T)
    phi_k  = encoder(o_k).view(B, N, -1)
    phi_k1 = encoder(o_k1).view(B, N, -1)

    dist_t_T = torch.norm(phi_t - phi_T, dim=-1)
    term1    = (1.0 - gamma) * dist_t_T.mean()

    phi_T_exp = phi_T.unsqueeze(1).expand_as(phi_k)
    dist_k_T  = torch.norm(phi_k  - phi_T_exp, dim=-1)
    dist_k1_T = torch.norm(phi_k1 - phi_T_exp, dim=-1)

    td_exp = dist_k_T - (-1.0) - gamma * dist_k1_T
    log_N  = torch.log(torch.tensor(float(N), device=device))
    term2  = (torch.logsumexp(td_exp, dim=1) - log_N).mean()

    return term1 + term2


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Device: {device}  gamma={args.gamma}  "
          f"buffer={args.buffer_size}  steps/epoch={args.steps_per_epoch}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Stream setup ---
    tfrecord_files = sorted(glob.glob(os.path.join(args.data_dir, "*.tfrecord*")))
    if not tfrecord_files:
        raise FileNotFoundError(f"No TFRecord files in {args.data_dir}")
    print(f"[Dataset] {len(tfrecord_files)} shards — streaming mode")

    stream = episode_stream(tfrecord_files, args.camera, rng)
    buffer = StreamingBuffer(stream, args.buffer_size, rng)
    print(f"[Dataset] Buffer ready: {buffer.size()} episodes "
          f"(~{buffer.size()*26/1000:.1f} GB RAM)")

    # --- Model ---
    print("[Model] Loading VIP (ResNet-50, 1024-dim)...")
    vip_wrapper = load_vip()
    vip_wrapper.to(device)
    encoder = vip_wrapper.module

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(state["encoder"])
        print(f"[Model] Resumed from {args.resume}")

    # --- Fine-tune mode ---
    if args.finetune_mode == "projection_head":
        # Freeze entire backbone — only the final FC projection layer is updated.
        # This preserves VIP's value-implicit representation learned on Ego4D and
        # reduces the risk of catastrophic forgetting on a small dataset.
        print("[Model] Finetune mode: projection_head (backbone frozen)")

        # VIP mirrors R3M's architecture — fc may also be Identity().
        # Strategy: use real fc if it exists, else unfreeze layer4 (~8M params).
        for param in encoder.parameters():
            param.requires_grad = False

        proj_head = None
        for candidate_path in ["convnet.fc", "fc"]:
            parts = candidate_path.split(".")
            obj = encoder
            for p in parts:
                obj = getattr(obj, p, None)
                if obj is None:
                    break
            if obj is not None and sum(p.numel() for p in obj.parameters()) > 0:
                proj_head = obj
                print(f"[Model] Projection head: encoder.{candidate_path}")
                break

        if proj_head is not None:
            for param in proj_head.parameters():
                param.requires_grad = True
            proj_head.train()
            trainable_params = list(proj_head.parameters())
        else:
            layer4 = getattr(getattr(encoder, "convnet", encoder), "layer4", None)
            if layer4 is None:
                raise RuntimeError(
                    "Could not find a trainable projection head or layer4.")
            for param in layer4.parameters():
                param.requires_grad = True
            layer4.train()
            trainable_params = list(layer4.parameters())
            print("[Model] fc is Identity — unfreezing convnet.layer4 (~8M params)")
    else:
        # Full fine-tuning — entire ResNet-50 is updated.
        # Higher risk of catastrophic forgetting on small datasets.
        print("[Model] Finetune mode: full (entire backbone unfrozen)")
        encoder.train()
        trainable_params = list(encoder.parameters())

    print(f"[Model] Trainable parameters: "
          f"{sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    # --- Training ---
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_loss   = 0.0
        batches_seen = 0

        for _ in range(args.steps_per_epoch):
            batch = make_batch(buffer, args.batch_size, args.num_negatives, rng)
            if batch is None:
                break

            loss = vip_loss(encoder, batch, args.gamma, device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            epoch_loss   += loss.item()
            batches_seen += 1
            global_step  += 1

            if global_step % 50 == 0:
                print(f"  step {global_step:5d} | loss={loss.item():.4f}")

        if batches_seen > 0:
            print(f"\n[Epoch {epoch}/{args.epochs}]  "
                  f"avg_loss={epoch_loss/batches_seen:.4f}\n")

        if epoch % args.save_every == 0:
            ckpt = out_dir / f"vip_droid_ep{epoch:03d}.pt"
            torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                        "finetune_mode": args.finetune_mode,
                        "args": vars(args)}, ckpt)
            print(f"[Save] {ckpt}")

    final = out_dir / "vip_droid_finetuned.pt"
    torch.save({"epoch": args.epochs, "encoder": encoder.state_dict(),
                "finetune_mode": args.finetune_mode,
                "args": vars(args)}, final)
    print(f"\n[Done] → {final}")


# =============================================================================
# Drop-in loader
# =============================================================================

def load_finetuned_vip(checkpoint_path: str, device: str = "cuda"):
    """
    Drop-in for load_vip() in vip_metaworld_multitask.py:
        from vip_droid_finetune import load_finetuned_vip
        vip = load_finetuned_vip("checkpoints/vip_droid_finetuned.pt")
    VISUAL_DIM remains 1024.
    """
    from vip import load_vip
    wrapper = load_vip()
    state   = torch.load(checkpoint_path, map_location=device)
    wrapper.module.load_state_dict(state["encoder"])
    wrapper.eval().to(device)
    print(f"[load_finetuned_vip] Loaded from {checkpoint_path}")
    return wrapper


if __name__ == "__main__":
    main()
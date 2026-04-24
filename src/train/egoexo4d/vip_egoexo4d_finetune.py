# =============================================================================
# vip_egoexo4d_finetune.py
#
# Fine-tunes the VIP ResNet-50 encoder on exocentric clips from Ego-Exo4D,
# preserving VIP's original training objective (Algorithm 1 / Equation 6,
# Ma et al. 2023).
#
# Paper reference:
#   Ma et al. 2023 — VIP: Value-Implicit Pre-training for Robot Learning (ICLR)
#   Grauman et al. 2024 — Ego-Exo4D
#
# Motivation (Direction 3 in ME5250 paper):
#   VIP was pre-trained entirely on egocentric (first-person) Ego4D video.
#   The same egocentric bias that affects R3M applies here — robot observations
#   are third-person. Fine-tuning on Ego-Exo4D exocentric clips aims to close
#   this gap while preserving VIP's distinctive value-implicit structure, which
#   gives it smoother reward landscapes than R3M.
#
# How VIP's loss differs from R3M's TCN:
#   R3M: explicit InfoNCE time-contrastive (attract nearby frames, repel far).
#   VIP: dual value objective (Eq. 6) with:
#     Term 1 — (1-γ) * ||φ(o_t) - φ(o_T)||₂   "attract initial to goal"
#     Term 2 — log Σ exp(||φ(o_k) - φ(o_T)||₂ - δ̃(o_k) - γ||φ(o_{k+1}) - φ(o_T)||₂)
#              "one-step TD error in embedding space"
#   Negatives are the MIDDLE frames of the trajectory (not cross-clip as in TCN).
#   This is the implicit temporal contrastive structure described in Section 4.2.
#
# Output:
#   A .pt checkpoint whose encoder state_dict can be hot-swapped into
#   vip_metaworld_multitask.py via load_finetuned_vip() (see bottom of file).
#
# Data directory layout (same as r3m_egoexo4d_finetune.py — share one dataset):
#   exo_clips/
#     <take_uid>/
#       exo.mp4
#       narration.json   (optional — not used by VIP loss, no language term)
#
# Usage:
#   python vip_egoexo4d_finetune.py \
#       --data_dir  ./exo_clips \
#       --out_dir   ./checkpoints \
#       --epochs    5 \
#       --lr        3e-5 \
#       --batch_size 16 \
#       --gamma     0.98
# =============================================================================

import os
import json
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.io as tvio

from vip import load_vip


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune VIP encoder on Ego-Exo4D exocentric clips")
    p.add_argument("--data_dir",      type=str, required=True,
                   help="Root dir of exocentric clips (exo_clips/ layout)")
    p.add_argument("--out_dir",       type=str, default="./checkpoints")
    p.add_argument("--epochs",        type=int, default=5)
    p.add_argument("--lr",            type=float, default=3e-5,
                   help="Keep small — VIP is sensitive to catastrophic forgetting")
    p.add_argument("--batch_size",    type=int, default=16,
                   help="Clips per batch. VIP paper used 16 video clips.")
    p.add_argument("--gamma",         type=float, default=0.98,
                   help="Discount factor gamma in VIP objective (paper default: 0.98)")
    p.add_argument("--num_negatives", type=int, default=3,
                   help="Number of middle-frame (o_k, o_{k+1}) pairs per clip")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--max_clips",     type=int, default=None,
                   help="Cap on clips (None = all). Use ~500 for smoke-test.")
    p.add_argument("--save_every",    type=int, default=1)
    p.add_argument("--resume",        type=str, default=None,
                   help="Path to a previous .pt checkpoint to resume from")
    return p.parse_args()


# =============================================================================
# Preprocessing  (identical to VIP/R3M: 224x224 center-crop, ImageNet norm)
# =============================================================================

NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])


# =============================================================================
# Dataset
# =============================================================================

class EgoExo4DExoDataset(torch.utils.data.Dataset):
    """
    Loads exocentric MP4 clips and returns sub-trajectories in the format
    required by VIP Algorithm 1:

        { o_t, o_k, o_{k+1}, o_T }

    where:
        o_t       = initial frame  (sampled from first 20% of clip)
        o_T       = goal frame     (sampled from last  20% of clip)
        o_k       = middle frame   (sampled from interior of clip)
        o_{k+1}   = frame immediately after o_k  (consecutive pair)

    VIP samples sub-trajectories and treats their start/end as initial/goal.
    We treat each clip as a single sub-trajectory (reasonable for short clips).

    Multiple negatives: we sample `num_negatives` independent (o_k, o_{k+1})
    pairs from the interior and stack them — the loss loops over all of them.
    """

    def __init__(self, data_dir: str, num_negatives: int = 3,
                 max_clips: int = None, seed: int = 42):
        self.data_dir      = Path(data_dir)
        self.num_negatives = num_negatives
        self.rng           = random.Random(seed)

        self.clips = sorted(self.data_dir.rglob("*.mp4"))
        if max_clips:
            self.clips = self.clips[:max_clips]
        print(f"[Dataset] {len(self.clips)} exocentric clips found in {data_dir}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_path = self.clips[idx]

        try:
            vframes, _, _ = tvio.read_video(str(clip_path), pts_unit="sec")
        except Exception:
            return None

        T_len = vframes.shape[0]
        if T_len < 10:
            return None

        # --- Sample o_t (initial) and o_T (goal) ---
        early_end  = max(1, int(0.20 * T_len))
        late_start = min(T_len - 1, int(0.80 * T_len))

        t_idx = self.rng.randint(0, early_end - 1)
        T_idx = self.rng.randint(late_start, T_len - 1)

        # Need at least 2 interior frames for (o_k, o_{k+1}) pairs
        if T_idx - t_idx < 3:
            return None

        # --- Sample num_negatives consecutive (o_k, o_{k+1}) pairs ---
        neg_pairs = []
        for _ in range(self.num_negatives):
            # k must have k+1 < T_idx so o_{k+1} is still a middle frame
            k = self.rng.randint(t_idx + 1, T_idx - 2)
            neg_pairs.append((k, k + 1))

        # --- Video-level shared random crop (matches R3M/VIP training) ---
        all_needed = set([t_idx, T_idx])
        for k, k1 in neg_pairs:
            all_needed.add(k)
            all_needed.add(k1)

        crop_i = self.rng.randint(0, 256 - 224)
        crop_j = self.rng.randint(0, 256 - 224)

        raw_frames = {}
        for fi in all_needed:
            frame = vframes[fi].permute(2, 0, 1).float() / 255.0  # [3,H,W]
            frame = T.functional.resize(frame, [256])
            frame = frame[:, crop_i:crop_i + 224, crop_j:crop_j + 224]
            raw_frames[fi] = NORMALIZE(frame)

        o_t = raw_frames[t_idx]   # [3,224,224]
        o_T = raw_frames[T_idx]   # [3,224,224]

        o_k  = torch.stack([raw_frames[k]  for k, k1 in neg_pairs])  # [N,3,224,224]
        o_k1 = torch.stack([raw_frames[k1] for k, k1 in neg_pairs])  # [N,3,224,224]

        return {
            "o_t":  o_t,
            "o_T":  o_T,
            "o_k":  o_k,
            "o_k1": o_k1,
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        "o_t":  torch.stack([b["o_t"]  for b in batch]),   # [B,3,224,224]
        "o_T":  torch.stack([b["o_T"]  for b in batch]),   # [B,3,224,224]
        "o_k":  torch.stack([b["o_k"]  for b in batch]),   # [B,N,3,224,224]
        "o_k1": torch.stack([b["o_k1"] for b in batch]),   # [B,N,3,224,224]
    }


# =============================================================================
# VIP Loss  —  Equation 6 / Algorithm 1, Ma et al. 2023
# =============================================================================

def vip_loss(encoder: nn.Module, batch: dict, gamma: float,
             device: torch.device) -> torch.Tensor:
    """
    Implements VIP objective (Equation 6 / Algorithm 1).

    Given a sub-trajectory with initial o_t, goal o_T, and N middle-frame
    pairs (o_k, o_{k+1}):

        L(phi) = (1-gamma) * ||phi(o_t) - phi(o_T)||_2           [Term 1]
               + log( (1/N) * sum_k exp(TD_exponent_k) )          [Term 2]

    where:
        TD_exponent_k = ||phi(o_k)   - phi(o_T)||_2
                      - delta_tilde(o_k)          <- sparse reward = -1 for middle frames
                      - gamma * ||phi(o_{k+1}) - phi(o_T)||_2

    V*(phi(o), phi(g)) := -||phi(o) - phi(g)||_2  (negative L2, as in Section 4.3).

    Term 1 attracts the initial frame toward the goal in embedding space.
    Term 2 enforces one-step Bellman consistency on middle frames —
    repulsion of middle frames from the goal is an IMPLICIT emergent property
    of minimising the TD error, NOT an explicit contrastive push (Section 4.2).

    The (1/N) normalisation inside the log-sum-exp converts it to a
    log-mean-exp, matching the expectation over D in Eq. 6.
    """
    B = batch["o_t"].shape[0]
    N = batch["o_k"].shape[1]

    o_t  = batch["o_t"].to(device)                               # [B,3,224,224]
    o_T  = batch["o_T"].to(device)                               # [B,3,224,224]
    o_k  = batch["o_k"].to(device).view(B * N, 3, 224, 224)     # [B*N,3,224,224]
    o_k1 = batch["o_k1"].to(device).view(B * N, 3, 224, 224)    # [B*N,3,224,224]

    # Encode
    phi_t  = encoder(o_t)                    # [B, D]
    phi_T  = encoder(o_T)                    # [B, D]
    phi_k  = encoder(o_k).view(B, N, -1)    # [B, N, D]
    phi_k1 = encoder(o_k1).view(B, N, -1)   # [B, N, D]

    # --- Term 1: (1-gamma) * ||phi(o_t) - phi(o_T)||_2 ---
    dist_t_T = torch.norm(phi_t - phi_T, dim=-1)   # [B]
    term1    = (1.0 - gamma) * dist_t_T.mean()

    # --- Term 2: log-mean-exp of TD exponents ---
    phi_T_exp = phi_T.unsqueeze(1).expand_as(phi_k)     # [B,N,D]

    dist_k_T  = torch.norm(phi_k  - phi_T_exp, dim=-1)  # [B,N]
    dist_k1_T = torch.norm(phi_k1 - phi_T_exp, dim=-1)  # [B,N]

    # delta_tilde(o_k) = I(o_k == o_T) - 1 = 0 - 1 = -1 for all middle frames
    delta = -1.0

    # TD exponent: ||phi(o_k) - phi(o_T)||  - delta  - gamma * ||phi(o_{k+1}) - phi(o_T)||
    td_exp = dist_k_T - delta - gamma * dist_k1_T    # [B,N]

    # log-mean-exp = logsumexp - log(N)
    log_N  = torch.log(torch.tensor(float(N), device=device))
    term2  = (torch.logsumexp(td_exp, dim=1) - log_N).mean()  # scalar

    return term1 + term2


# =============================================================================
# Main fine-tuning loop
# =============================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Device: {device}   gamma={args.gamma}   lr={args.lr}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load VIP encoder
    # ------------------------------------------------------------------
    print("[Model] Loading pre-trained VIP (ResNet-50, 1024-dim output)...")
    vip_wrapper = load_vip()
    vip_wrapper.to(device)

    # VIP wraps the ResNet in vip_wrapper.module (same pattern as R3M)
    encoder = vip_wrapper.module
    encoder.train()

    if args.resume:
        print(f"[Model] Resuming from {args.resume}")
        state = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(state["encoder"])

    # ------------------------------------------------------------------
    # 2. Dataset & DataLoader
    # ------------------------------------------------------------------
    dataset = EgoExo4DExoDataset(
        args.data_dir,
        num_negatives=args.num_negatives,
        max_clips=args.max_clips,
        seed=args.seed,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"[Dataset] {len(dataset)} clips  →  {len(loader)} batches/epoch")

    # ------------------------------------------------------------------
    # 3. Optimiser  (same as original VIP: Adam, same LR as R3M paper)
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_loss   = 0.0
        batches_seen = 0

        for batch in loader:
            if batch is None:
                continue

            loss = vip_loss(encoder, batch, gamma=args.gamma, device=device)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping — important for TD-based objectives which can spike
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss   += loss.item()
            batches_seen += 1
            global_step  += 1

            if global_step % 50 == 0:
                print(f"  step {global_step:5d} | loss={loss.item():.4f}")

        if batches_seen > 0:
            avg = epoch_loss / batches_seen
            print(f"\n[Epoch {epoch}/{args.epochs}]  avg_loss={avg:.4f}\n")

        if epoch % args.save_every == 0:
            ckpt_path = out_dir / f"vip_egoexo4d_ep{epoch:03d}.pt"
            torch.save({
                "epoch":   epoch,
                "encoder": encoder.state_dict(),
                "args":    vars(args),
            }, ckpt_path)
            print(f"[Save] Checkpoint → {ckpt_path}")

    # --- Final save ---
    final_path = out_dir / "vip_egoexo4d_finetuned.pt"
    torch.save({
        "epoch":   args.epochs,
        "encoder": encoder.state_dict(),
        "args":    vars(args),
    }, final_path)
    print(f"\n[Done] Final fine-tuned encoder saved → {final_path}")
    print("To evaluate, use load_finetuned_vip() in vip_metaworld_multitask.py")


# =============================================================================
# Helper: load fine-tuned encoder back into VIP wrapper
# =============================================================================

def load_finetuned_vip(checkpoint_path: str, device: str = "cuda"):
    """
    Drop-in replacement for load_vip() that injects fine-tuned encoder weights.

    Usage in vip_metaworld_multitask.py:
        # Replace:
        #   from vip import load_vip
        #   vip = load_vip()
        # With:
        #   from vip_egoexo4d_finetune import load_finetuned_vip
        #   vip = load_finetuned_vip("checkpoints/vip_egoexo4d_finetuned.pt")

    Everything else in vip_metaworld_multitask.py stays unchanged.
    VISUAL_DIM remains 1024 — the embedding dimension does not change.
    """
    from vip import load_vip
    vip_wrapper = load_vip()
    state = torch.load(checkpoint_path, map_location=device)
    vip_wrapper.module.load_state_dict(state["encoder"])
    vip_wrapper.eval()
    vip_wrapper.to(device)
    print(f"[load_finetuned_vip] Loaded fine-tuned weights from {checkpoint_path}")
    return vip_wrapper


if __name__ == "__main__":
    main()

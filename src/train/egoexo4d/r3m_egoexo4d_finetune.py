# =============================================================================
# r3m_egoexo4d_finetune.py
#
# Fine-tunes the R3M ResNet-50 encoder on exocentric clips from Ego-Exo4D,
# preserving the original R3M training objective (TCN + language + L1/L2).
#
# Paper reference:
#   Nair et al. 2022 — R3M: A Universal Visual Representation for Robot Manipulation
#   Grauman et al. 2024 — Ego-Exo4D
#
# Motivation (Direction 1 in ME5250 paper):
#   R3M was pre-trained entirely on egocentric (first-person) Ego4D video.
#   MetaWorld / real robot observations are captured from THIRD-PERSON cameras.
#   Fine-tuning on Ego-Exo4D's exocentric clips attempts to close this viewpoint gap.
#
# What this script does:
#   1. Loads the pre-trained R3M ResNet-50 checkpoint (frozen by default in R3M).
#      We UNFREEZE it here and fine-tune with a small learning rate.
#   2. Reads exocentric MP4 clips from a local Ego-Exo4D subset directory.
#   3. Applies the same TCN + language (optional) + L1/L2 objective as R3M.
#      Language captions are optional — Ego-Exo4D has narrations but they are
#      structured differently from Ego4D; the script falls back to TCN-only
#      if no captions are found alongside the clip.
#   4. Saves the fine-tuned encoder weights to a .pt file that can be hot-swapped
#      into r3m_metaworld_multitask.py for A/B evaluation.
#
# Usage:
#   python r3m_egoexo4d_finetune.py \
#       --data_dir  /path/to/egoexo4d/exo_clips \
#       --out_dir   ./checkpoints \
#       --epochs    5 \
#       --lr        3e-5 \
#       --batch_size 8
#
# Data directory layout expected (see ego_exo4d_download_plan.md for how to get there):
#   exo_clips/
#     <take_uid>/
#       exo/
#         cam01.mp4   (third-person / exocentric)
#         cam02.mp4
#         ...
#       (optional) narration.json   — Ego-Exo4D narration file for this take
# =============================================================================

import os
import json
import argparse
import random
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.io as tvio

from r3m import load_r3m


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune R3M encoder on Ego-Exo4D exocentric clips")
    p.add_argument("--data_dir",   type=str, required=True,
                   help="Root dir of exocentric clips (see layout in header comment)")
    p.add_argument("--out_dir",    type=str, default="./checkpoints",
                   help="Where to save fine-tuned checkpoints")
    p.add_argument("--epochs",     type=int, default=5,
                   help="Fine-tuning epochs (5 is a reasonable starting point)")
    p.add_argument("--lr",         type=float, default=3e-5,
                   help="Learning rate — keep small to avoid catastrophic forgetting")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Number of video clips per batch (each clip → 5 frames)")
    p.add_argument("--frames_per_clip", type=int, default=5,
                   help="Frames sampled per clip: [I0, Ii, Ij, Ik, Ig] (matches R3M)")
    p.add_argument("--num_negatives", type=int, default=3,
                   help="Number of negative samples per anchor (R3M default: 3)")
    p.add_argument("--lambda1",    type=float, default=1.0,   help="TCN loss weight")
    p.add_argument("--lambda2",    type=float, default=1.0,   help="Language loss weight (0 to disable)")
    p.add_argument("--lambda3",    type=float, default=1e-5,  help="L1 regularisation weight")
    p.add_argument("--lambda4",    type=float, default=1e-5,  help="L2 regularisation weight")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--max_clips",  type=int, default=None,
                   help="Cap on number of clips to use (useful for quick smoke-tests)")
    p.add_argument("--save_every", type=int, default=1,
                   help="Save checkpoint every N epochs")
    p.add_argument("--resume",     type=str, default=None,
                   help="Path to a previous fine-tuned .pt checkpoint to resume from")
    return p.parse_args()


# =============================================================================
# Frame-level transforms  (identical to R3M preprocessing)
# =============================================================================

FRAME_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    # Random crop is applied at VIDEO level inside the dataset (all frames same crop)
])

NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

def to_tensor_normalized(frame_uint8: torch.Tensor) -> torch.Tensor:
    """Convert HWC uint8 [0,255] → CHW float [0,1] → normalize."""
    x = frame_uint8.permute(2, 0, 1).float() / 255.0  # CHW
    return NORMALIZE(x)


# =============================================================================
# Dataset
# =============================================================================

class EgoExo4DExoDataset(Dataset):
    """
    Iterates over exocentric MP4 clips from a local Ego-Exo4D subset.

    For each __getitem__ call, returns a dict with:
        frames  : Tensor [5, 3, 224, 224]  — sampled with video-level random crop
        caption : str or None
        clip_id : str (for debugging)

    The five frame slots follow R3M convention:
        idx 0 = I0   (initial,  first 20% of clip)
        idx 1 = Ii   (early)
        idx 2 = Ij   (mid,   j > i)
        idx 3 = Ik   (late,  k > j)
        idx 4 = Ig   (goal,   last 20% of clip)
    """

    def __init__(self, data_dir: str, max_clips: int = None, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.rng      = random.Random(seed)

        # Discover all exocentric mp4 files
        self.clips = sorted(self._discover_clips())
        if max_clips:
            self.clips = self.clips[:max_clips]

        print(f"[Dataset] Found {len(self.clips)} exocentric clips in {data_dir}")

        # Build caption lookup  { clip_path_str: caption_str }
        self.captions = self._load_captions()

    # ------------------------------------------------------------------
    def _discover_clips(self):
        """
        Recursively find all .mp4 files.  Accepts two layouts:
          Layout A: <take_uid>/exo/*.mp4        (standard Ego-Exo4D download)
          Layout B: flat  *.mp4                  (after user pre-processing)
        """
        clips = []
        for mp4 in self.data_dir.rglob("*.mp4"):
            clips.append(mp4)
        return clips

    def _load_captions(self):
        """
        Try to find narration.json files (Ego-Exo4D narration annotation).
        Falls back gracefully — TCN-only training is used when no captions exist.

        Ego-Exo4D narration format (simplified):
          { "narration_pass_1": { "narrations": [{"timestamp_sec": t, "narration_text": txt}, ...] } }

        We concatenate all narration texts for a take as a single caption string.
        """
        captions = {}
        for narr_file in self.data_dir.rglob("narration.json"):
            try:
                with open(narr_file) as f:
                    data = json.load(f)
                take_dir = narr_file.parent
                # Merge all narration passes into one string
                texts = []
                for pass_key, pass_val in data.items():
                    for entry in pass_val.get("narrations", []):
                        txt = entry.get("narration_text", "").strip()
                        if txt:
                            texts.append(txt)
                caption = " ".join(texts) if texts else None
                # Assign this caption to every exo clip in the same take directory
                for mp4 in take_dir.rglob("*.mp4"):
                    captions[str(mp4)] = caption
            except Exception as e:
                print(f"[Dataset] Warning: could not parse {narr_file}: {e}")
        return captions

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_path = self.clips[idx]

        # --- Load video frames with torchvision ---
        try:
            # vframes: Tensor [T, H, W, C] uint8
            vframes, _, _ = tvio.read_video(str(clip_path), pts_unit="sec")
        except Exception as e:
            # Return a dummy item on corrupted files — collate_fn will filter
            return {"frames": None, "caption": None, "clip_id": str(clip_path)}

        T_len = vframes.shape[0]
        if T_len < 10:
            return {"frames": None, "caption": None, "clip_id": str(clip_path)}

        # --- Sample 5 frame indices following R3M convention ---
        early_end = max(1, int(0.20 * T_len))    # first 20%  → I0, Ii
        late_start = min(T_len - 1, int(0.80 * T_len))  # last  20%  → Ig

        i0 = self.rng.randint(0, early_end - 1)
        ig = self.rng.randint(late_start, T_len - 1)

        # Sample Ii < Ij < Ik in (i0, ig) exclusive
        mid_range = list(range(i0 + 1, ig))
        if len(mid_range) < 3:
            mid_range = list(range(i0, ig + 1))
        three = sorted(self.rng.sample(mid_range, min(3, len(mid_range))))
        while len(three) < 3:
            three.append(three[-1])
        ii, ij, ik = three[0], three[1], three[2]

        indices = [i0, ii, ij, ik, ig]

        # --- Apply video-level random crop (same crop for all frames) ---
        # First resize each frame
        frames_raw = []
        for fi in indices:
            frame = vframes[fi]               # [H, W, C] uint8
            # Convert to CHW float for torchvision transforms
            frame_chw = frame.permute(2, 0, 1).float() / 255.0
            frame_chw = T.functional.resize(frame_chw, [256])
            frames_raw.append(frame_chw)

        # Compute a shared random crop offset (R3M: same crop within a video)
        i_crop = self.rng.randint(0, 256 - 224)
        j_crop = self.rng.randint(0, 256 - 224)
        frames_norm = []
        for frame_chw in frames_raw:
            cropped = frame_chw[:, i_crop:i_crop+224, j_crop:j_crop+224]
            frames_norm.append(NORMALIZE(cropped))

        frames_tensor = torch.stack(frames_norm)  # [5, 3, 224, 224]

        caption = self.captions.get(str(clip_path), None)

        return {
            "frames":   frames_tensor,
            "caption":  caption,
            "clip_id":  str(clip_path),
        }


def collate_fn(batch):
    """Filter out None frames (corrupted clips)."""
    batch = [b for b in batch if b["frames"] is not None]
    if not batch:
        return None
    frames   = torch.stack([b["frames"]  for b in batch])   # [B, 5, 3, 224, 224]
    captions = [b["caption"] for b in batch]
    clip_ids = [b["clip_id"] for b in batch]
    return {"frames": frames, "captions": captions, "clip_ids": clip_ids}


# =============================================================================
# R3M Loss Functions
# =============================================================================

def tcn_loss(embeddings: torch.Tensor, num_negatives: int = 3) -> torch.Tensor:
    """
    Time Contrastive (InfoNCE) loss — Equation 1 in R3M paper.

    embeddings : [B, 5, D]  where dim-1 order is [I0, Ii, Ij, Ik, Ig]
    We use the triplet (Ii, Ij, Ik):
        anchor   = Ii
        positive = Ij  (closer in time → should be closer in embedding space)
        hard neg = Ik  (farther in time)
        soft neg = random embeddings from OTHER clips in the batch

    Loss = -log( exp(-||zi - zj||) / Z )
    where Z sums over positive + Ik + num_negatives cross-clip negatives.
    """
    B, _, D = embeddings.shape
    zi = embeddings[:, 1, :]   # [B, D]  Ii
    zj = embeddings[:, 2, :]   # [B, D]  Ij
    zk = embeddings[:, 3, :]   # [B, D]  Ik

    # Similarity = negative L2 distance (as in R3M)
    sim_pos = -torch.norm(zi - zj, dim=-1)          # [B]
    sim_neg_k = -torch.norm(zi - zk, dim=-1)        # [B]

    # Cross-clip negatives: roll along the batch dimension
    cross_sims = []
    for n in range(1, min(num_negatives + 1, B)):
        z_neg = torch.roll(zi, shifts=n, dims=0)
        cross_sims.append(-torch.norm(zi - z_neg, dim=-1))  # [B]

    # Pad if batch too small to get num_negatives
    while len(cross_sims) < num_negatives:
        cross_sims.append(cross_sims[-1])

    # log-sum-exp denominator
    logits = torch.stack([sim_pos, sim_neg_k] + cross_sims, dim=1)  # [B, 2+N]
    loss = -sim_pos + torch.logsumexp(logits, dim=1)
    return loss.mean()


def l1_l2_reg(embeddings: torch.Tensor, lambda3: float, lambda4: float) -> torch.Tensor:
    """L1 + L2 sparsity regularisation (Equation 3, λ3 and λ4 terms)."""
    flat = embeddings.reshape(-1, embeddings.shape[-1])
    return lambda3 * flat.abs().mean() + lambda4 * flat.pow(2).mean()


# =============================================================================
# Language Head  (simplified — used only when captions are available)
# =============================================================================

class LanguageAlignmentHead(nn.Module):
    """
    5-layer MLP matching R3M's Gθ.
    Input:  [z0 || zg || l]  where z are 2048-dim, l is 768-dim (DistilBERT)
    Output: scalar score

    Only instantiated if transformers is installed and captions exist.
    """
    def __init__(self, embed_dim: int = 2048, lang_dim: int = 768):
        super().__init__()
        in_dim = 2 * embed_dim + lang_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 1024),  nn.ReLU(),
            nn.Linear(1024, 1024),  nn.ReLU(),
            nn.Linear(1024, 1024),  nn.ReLU(),
            nn.Linear(1024, 1),
        )

    def forward(self, z0, zg, lang_feat):
        x = torch.cat([z0, zg, lang_feat], dim=-1)
        return self.mlp(x).squeeze(-1)   # [B]


def try_load_language_model():
    """Attempt to load DistilBERT — gracefully return None if unavailable."""
    try:
        from transformers import DistilBertTokenizer, DistilBertModel
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model     = DistilBertModel.from_pretrained("distilbert-base-uncased")
        model.eval()
        print("[Language] DistilBERT loaded — language alignment loss ENABLED")
        return tokenizer, model
    except Exception as e:
        print(f"[Language] DistilBERT unavailable ({e}) — using TCN-only loss")
        return None, None


@torch.no_grad()
def encode_captions(captions, tokenizer, lang_model, device):
    """Return [B, 768] language embeddings. Returns None if lang_model is None."""
    if lang_model is None or all(c is None for c in captions):
        return None
    # Replace None captions with empty string
    texts = [c if c else "" for c in captions]
    tokens = tokenizer(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=64).to(device)
    out = lang_model(**tokens)
    return out.last_hidden_state[:, 0, :]  # CLS token → [B, 768]


# =============================================================================
# Main fine-tuning loop
# =============================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Setup] Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load R3M encoder  (ResNet-50, 2048-dim output)
    # ------------------------------------------------------------------
    print("[Model] Loading pre-trained R3M (ResNet-50)...")
    r3m_wrapper = load_r3m("resnet50")
    r3m_wrapper.to(device)

    # R3M wraps the encoder in r3m_wrapper.module  (the actual ResNet)
    # We fine-tune the FULL encoder (not frozen)
    encoder = r3m_wrapper.module   # nn.Module — ResNet-50 up to the embedding layer

    # If resuming from a previous fine-tune checkpoint
    if args.resume:
        print(f"[Model] Resuming from {args.resume}")
        state = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(state["encoder"])

    encoder.train()
    encoder.to(device)

    # ------------------------------------------------------------------
    # 2. Optional language head
    # ------------------------------------------------------------------
    tokenizer, lang_model = try_load_language_model()
    lang_head = None
    if lang_model is not None:
        lang_model = lang_model.to(device)
        lang_head  = LanguageAlignmentHead(embed_dim=2048, lang_dim=768).to(device)

    # ------------------------------------------------------------------
    # 3. Dataset & DataLoader
    # ------------------------------------------------------------------
    dataset = EgoExo4DExoDataset(args.data_dir, max_clips=args.max_clips, seed=args.seed)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    print(f"[Dataset] {len(dataset)} clips → {len(loader)} batches/epoch")

    # ------------------------------------------------------------------
    # 4. Optimiser
    # ------------------------------------------------------------------
    params = list(encoder.parameters())
    if lang_head is not None:
        params += list(lang_head.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_loss     = 0.0
        epoch_tcn      = 0.0
        epoch_lang     = 0.0
        epoch_reg      = 0.0
        batches_seen   = 0

        for batch in loader:
            if batch is None:
                continue

            frames   = batch["frames"].to(device)    # [B, 5, 3, 224, 224]
            captions = batch["captions"]              # list[str | None]

            B = frames.shape[0]

            # Encode all 5 frames per clip
            # Flatten to [B*5, 3, 224, 224], encode, reshape to [B, 5, D]
            flat_frames = frames.view(B * 5, 3, 224, 224)
            embeddings  = encoder(flat_frames)            # [B*5, 2048]
            embeddings  = embeddings.view(B, 5, -1)       # [B, 5, 2048]

            # --- TCN loss ---
            loss_tcn = tcn_loss(embeddings, num_negatives=args.num_negatives)

            # --- Regularisation ---
            loss_reg = l1_l2_reg(embeddings, args.lambda3, args.lambda4)

            # --- Language loss (optional) ---
            loss_lang = torch.tensor(0.0, device=device)
            if lang_head is not None and args.lambda2 > 0:
                lang_feats = encode_captions(captions, tokenizer, lang_model, device)
                if lang_feats is not None:
                    z0 = embeddings[:, 0, :]   # I0
                    zg = embeddings[:, 4, :]   # Ig (goal)

                    # Positive pair: (I0, Ig, caption)
                    score_pos = lang_head(z0, zg, lang_feats)

                    # Negative: rolled z0 from another clip
                    z0_neg   = torch.roll(z0, shifts=1, dims=0)
                    score_n1 = lang_head(z0_neg, zg, lang_feats)

                    # Negative: Ii (early frame, before task completion)
                    zi   = embeddings[:, 1, :]
                    score_n2 = lang_head(z0, zi, lang_feats)

                    logits_lang = torch.stack([score_pos, score_n1, score_n2], dim=1)
                    loss_lang   = F.cross_entropy(logits_lang,
                                                  torch.zeros(B, dtype=torch.long, device=device))

            # --- Total loss ---
            loss = (args.lambda1 * loss_tcn
                    + args.lambda2 * loss_lang
                    + loss_reg)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss   += loss.item()
            epoch_tcn    += loss_tcn.item()
            epoch_lang   += loss_lang.item()
            epoch_reg    += loss_reg.item()
            batches_seen += 1
            global_step  += 1

            if global_step % 50 == 0:
                print(f"  step {global_step:5d} | "
                      f"loss={loss.item():.4f}  "
                      f"tcn={loss_tcn.item():.4f}  "
                      f"lang={loss_lang.item():.4f}  "
                      f"reg={loss_reg.item():.6f}")

        # --- End of epoch ---
        if batches_seen > 0:
            print(f"\n[Epoch {epoch}/{args.epochs}] "
                  f"avg_loss={epoch_loss/batches_seen:.4f}  "
                  f"tcn={epoch_tcn/batches_seen:.4f}  "
                  f"lang={epoch_lang/batches_seen:.4f}  "
                  f"reg={epoch_reg/batches_seen:.6f}\n")

        # --- Checkpoint ---
        if epoch % args.save_every == 0:
            ckpt_path = out_dir / f"r3m_egoexo4d_ep{epoch:03d}.pt"
            torch.save({
                "epoch":     epoch,
                "encoder":   encoder.state_dict(),
                "lang_head": lang_head.state_dict() if lang_head else None,
                "args":      vars(args),
            }, ckpt_path)
            print(f"[Save] Checkpoint → {ckpt_path}")

    # --- Final save ---
    final_path = out_dir / "r3m_egoexo4d_finetuned.pt"
    torch.save({
        "epoch":     args.epochs,
        "encoder":   encoder.state_dict(),
        "lang_head": lang_head.state_dict() if lang_head else None,
        "args":      vars(args),
    }, final_path)
    print(f"\n[Done] Final fine-tuned encoder saved → {final_path}")
    print("To evaluate, use load_finetuned_r3m() in r3m_metaworld_multitask.py")


# =============================================================================
# Helper: load fine-tuned encoder back into R3M wrapper
# (paste this into r3m_metaworld_multitask.py to swap the encoder)
# =============================================================================

def load_finetuned_r3m(checkpoint_path: str, device: str = "cuda"):
    """
    Drop-in replacement for load_r3m("resnet50") that injects the
    fine-tuned encoder weights.

    Usage in r3m_metaworld_multitask.py:
        # Replace:
        #   r3m = load_r3m("resnet50")
        # With:
        #   from r3m_egoexo4d_finetune import load_finetuned_r3m
        #   r3m = load_finetuned_r3m("checkpoints/r3m_egoexo4d_finetuned.pt")
    """
    from r3m import load_r3m
    r3m_wrapper = load_r3m("resnet50")
    state = torch.load(checkpoint_path, map_location=device)
    r3m_wrapper.module.load_state_dict(state["encoder"])
    r3m_wrapper.eval()
    r3m_wrapper.to(device)
    print(f"[load_finetuned_r3m] Loaded fine-tuned weights from {checkpoint_path}")
    return r3m_wrapper


if __name__ == "__main__":
    main()

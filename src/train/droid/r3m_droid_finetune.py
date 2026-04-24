# =============================================================================
# r3m_droid_finetune.py
#
# Fine-tunes the R3M ResNet-50 encoder on the DROID robot manipulation dataset,
# preserving the original R3M training objective (TCN + language + L1/L2).
#
# Paper references:
#   Nair et al. 2022 — R3M: A Universal Visual Representation for Robot Manipulation
#   Khazatsky et al. 2024 — DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset
#
# Motivation (Direction 1 in ME5250 paper):
#   R3M was pre-trained on egocentric (first-person) Ego4D video.
#   DROID contains third-person robot manipulation video — exactly the viewpoint
#   used during MetaWorld evaluation. Fine-tuning on DROID directly closes the
#   egocentric bias without requiring Ego-Exo4D access.
#
# KEY DESIGN: STREAMING DATA LOADER
#   Episodes are streamed from TFRecord files one at a time — never accumulated
#   in RAM. A small in-memory buffer (--buffer_size episodes) is maintained and
#   batches are sampled from it. RAM usage stays flat regardless of dataset size.
#
# Usage:
#   python r3m_droid_finetune.py \
#       --data_dir  ./droid_full/1.0.1 \
#       --out_dir   ./checkpoints \
#       --epochs    5 \
#       --lr        3e-5 \
#       --batch_size 16 \
#       --buffer_size 100 \
#       --steps_per_epoch 200
#
# Checkpoint loading in r3m_metaworld_multitask.py:
#   from r3m_droid_finetune import load_finetuned_r3m
#   r3m = load_finetuned_r3m("checkpoints/r3m_droid_finetuned.pt")
# =============================================================================

import os
import glob
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import io

from r3m import load_r3m

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
        description="Fine-tune R3M encoder on DROID TFRecord dataset (streaming)")
    p.add_argument("--data_dir",        type=str, required=True)
    p.add_argument("--out_dir",         type=str, default="./checkpoints")
    p.add_argument("--epochs",          type=int, default=5)
    p.add_argument("--lr",              type=float, default=3e-5)
    p.add_argument("--batch_size",      type=int, default=16)
    p.add_argument("--buffer_size",     type=int, default=100,
                   help="Episodes kept in RAM at once. 100 ~= 2.5GB RAM.")
    p.add_argument("--steps_per_epoch", type=int, default=200,
                   help="Training steps per epoch.")
    p.add_argument("--num_negatives",   type=int, default=3)
    p.add_argument("--lambda1",         type=float, default=1.0)
    p.add_argument("--lambda2",         type=float, default=1.0)
    p.add_argument("--lambda3",         type=float, default=1e-5)
    p.add_argument("--lambda4",         type=float, default=1e-5)
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
    Streams directly from disk — never loads all episodes into RAM.
    Yields: { "frames": np.ndarray [T,H,W,3], "caption": str|None }
    """
    assert TFRECORD_AVAILABLE, "pip install tfrecord"

    img_key   = f"steps/observation/{camera_key}"
    lang_keys = ["steps/language_instruction",
                 "steps/language_instruction_2",
                 "steps/language_instruction_3"]
    description = {img_key: "byte", lang_keys[0]: "byte"}

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

                    frames = np.stack(frames)

                    caption = None
                    for lk in lang_keys:
                        raw = record.get(lk)
                        if raw is not None:
                            items = raw if hasattr(raw, '__iter__') else [raw]
                            for lb in items:
                                txt = lb.decode("utf-8") if isinstance(lb, bytes) else str(lb)
                                if txt.strip():
                                    caption = txt.strip()
                                    break
                        if caption:
                            break

                    yield {"frames": frames, "caption": caption}

                except Exception:
                    continue
        except Exception as e:
            print(f"  [Skip shard] {e}")
            continue


class StreamingBuffer:
    """
    Rolling buffer of episodes. Refills from stream as episodes are consumed.
    RAM usage = buffer_size x ~26MB per episode.
    """
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
# Sampling
# =============================================================================

def sample_frames_from_episode(episode, rng):
    frames = episode["frames"]
    T_len  = frames.shape[0]

    early_end  = max(1, int(0.20 * T_len))
    late_start = min(T_len - 1, int(0.80 * T_len))

    i0 = rng.randint(0, early_end - 1)
    ig = rng.randint(late_start, T_len - 1)

    interior = list(range(i0 + 1, ig))
    if len(interior) < 3:
        interior = list(range(max(0, i0 - 1), min(T_len, ig + 2)))
    three = sorted(rng.sample(interior, min(3, len(interior))))
    while len(three) < 3:
        three.append(three[-1])
    ii, ij, ik = three[0], three[1], three[2]

    resized = [preprocess_frame(frames[fi]) for fi in [i0, ii, ij, ik, ig]]
    H_res, W_res = resized[0].shape[1], resized[0].shape[2]
    ci = rng.randint(0, max(0, H_res - 224))
    cj = rng.randint(0, max(0, W_res - 224))

    return {
        "frames":  torch.stack([apply_crop(f, ci, cj) for f in resized]),
        "caption": episode["caption"],
    }


def make_batch(buffer, batch_size, rng):
    episodes = buffer.sample(batch_size)
    if not episodes:
        return None
    items = [sample_frames_from_episode(ep, rng) for ep in episodes]
    return {
        "frames":   torch.stack([it["frames"]  for it in items]),
        "captions": [it["caption"] for it in items],
    }


# =============================================================================
# R3M Loss Functions
# =============================================================================

def tcn_loss(embeddings, num_negatives=3):
    B, _, D = embeddings.shape
    zi = embeddings[:, 1, :]
    zj = embeddings[:, 2, :]
    zk = embeddings[:, 3, :]

    sim_pos   = -torch.norm(zi - zj, dim=-1)
    sim_neg_k = -torch.norm(zi - zk, dim=-1)

    cross_sims = []
    for n in range(1, min(num_negatives + 1, B)):
        cross_sims.append(-torch.norm(zi - torch.roll(zi, n, 0), dim=-1))
    while len(cross_sims) < num_negatives:
        cross_sims.append(cross_sims[-1])

    logits = torch.stack([sim_pos, sim_neg_k] + cross_sims, dim=1)
    return (-sim_pos + torch.logsumexp(logits, dim=1)).mean()


def l1_l2_reg(embeddings, lambda3, lambda4):
    flat = embeddings.reshape(-1, embeddings.shape[-1])
    return lambda3 * flat.abs().mean() + lambda4 * flat.pow(2).mean()


class LanguageAlignmentHead(nn.Module):
    def __init__(self, embed_dim=2048, lang_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2*embed_dim+lang_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1),
        )
    def forward(self, z0, zg, lf):
        return self.mlp(torch.cat([z0, zg, lf], dim=-1)).squeeze(-1)


def try_load_language_model():
    try:
        from transformers import DistilBertTokenizer, DistilBertModel
        tok   = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        model.eval()
        print("[Language] DistilBERT loaded")
        return tok, model
    except Exception as e:
        print(f"[Language] Unavailable ({e}) — TCN-only mode")
        return None, None


@torch.no_grad()
def encode_captions(captions, tokenizer, lang_model, device):
    if lang_model is None or all(c is None for c in captions):
        return None
    texts  = [c if c else "" for c in captions]
    tokens = tokenizer(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=64).to(device)
    return lang_model(**tokens).last_hidden_state[:, 0, :]


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
    print(f"[Setup] Device: {device}  buffer={args.buffer_size}  "
          f"steps/epoch={args.steps_per_epoch}")

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
    print("[Model] Loading R3M (ResNet-50)...")
    r3m_wrapper = load_r3m("resnet50")
    r3m_wrapper.to(device)
    encoder = r3m_wrapper.module

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(state["encoder"])
        print(f"[Model] Resumed from {args.resume}")

    # --- Fine-tune mode ---
    if args.finetune_mode == "projection_head":
        # Freeze entire backbone — only the final FC projection layer is updated.
        # This preserves the R3M representation learned on Ego4D and reduces the
        # risk of catastrophic forgetting when fine-tuning on a small dataset.
        print("[Model] Finetune mode: projection_head (backbone frozen)")

        # R3M replaces the standard ResNet fc with Identity() — no learnable
        # projection head exists. Strategy:
        #   1. If a real (non-empty) fc layer exists, unfreeze it only.
        #   2. Otherwise unfreeze convnet.layer4 (final residual block, ~8M params)
        #      — the standard "partial fine-tuning" middle ground.
        for param in encoder.parameters():
            param.requires_grad = False

        # Check for a real fc layer
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
            # No real fc — unfreeze layer4 (last residual block)
            layer4 = getattr(getattr(encoder, "convnet", encoder), "layer4", None)
            if layer4 is None:
                raise RuntimeError(
                    "Could not find a trainable projection head or layer4. "
                    "Check encoder architecture with encoder.named_modules()."
                )
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

    tokenizer, lang_model = try_load_language_model()
    lang_head = None
    if lang_model is not None:
        lang_model = lang_model.to(device)
        lang_head  = LanguageAlignmentHead().to(device)
        print("[Language] Language alignment loss ENABLED")
    else:
        print("[Language] TCN-only mode (install transformers to enable language loss)")

    params = trainable_params
    if lang_head:
        params = params + list(lang_head.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # --- Training ---
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_loss = epoch_tcn = epoch_lang = 0.0
        batches_seen = 0

        for _ in range(args.steps_per_epoch):
            batch = make_batch(buffer, args.batch_size, rng)
            if batch is None:
                break

            frames   = batch["frames"].to(device)
            captions = batch["captions"]
            B = frames.shape[0]

            embeddings = encoder(frames.view(B*5, 3, 224, 224)).view(B, 5, -1)

            loss_tcn  = tcn_loss(embeddings, args.num_negatives)
            loss_reg  = l1_l2_reg(embeddings, args.lambda3, args.lambda4)
            loss_lang = torch.tensor(0.0, device=device)

            if lang_head is not None and args.lambda2 > 0:
                lf = encode_captions(captions, tokenizer, lang_model, device)
                if lf is not None:
                    z0, zg, zi = embeddings[:,0,:], embeddings[:,4,:], embeddings[:,1,:]
                    logits = torch.stack([
                        lang_head(z0, zg, lf),
                        lang_head(torch.roll(z0,1,0), zg, lf),
                        lang_head(z0, zi, lf)
                    ], dim=1)
                    loss_lang = F.cross_entropy(
                        logits, torch.zeros(B, dtype=torch.long, device=device))

            loss = args.lambda1*loss_tcn + args.lambda2*loss_lang + loss_reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_tcn  += loss_tcn.item()
            epoch_lang += loss_lang.item()
            batches_seen += 1
            global_step  += 1

            if global_step % 50 == 0:
                print(f"  step {global_step:5d} | loss={loss.item():.4f}  "
                      f"tcn={loss_tcn.item():.4f}  lang={loss_lang.item():.4f}")

        if batches_seen > 0:
            print(f"\n[Epoch {epoch}/{args.epochs}] "
                  f"avg_loss={epoch_loss/batches_seen:.4f}  "
                  f"tcn={epoch_tcn/batches_seen:.4f}\n")

        if epoch % args.save_every == 0:
            ckpt = out_dir / f"r3m_droid_ep{epoch:03d}.pt"
            torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                        "finetune_mode": args.finetune_mode,
                        "args": vars(args)}, ckpt)
            print(f"[Save] {ckpt}")

    final = out_dir / "r3m_droid_finetuned.pt"
    torch.save({"epoch": args.epochs, "encoder": encoder.state_dict(),
                "finetune_mode": args.finetune_mode,
                "args": vars(args)}, final)
    print(f"\n[Done] → {final}")


# =============================================================================
# Drop-in loader
# =============================================================================

def load_finetuned_r3m(checkpoint_path: str, device: str = "cuda"):
    """
    Drop-in for load_r3m("resnet50") in r3m_metaworld_multitask.py:
        from r3m_droid_finetune import load_finetuned_r3m
        r3m = load_finetuned_r3m("checkpoints/r3m_droid_finetuned.pt")
    """
    from r3m import load_r3m
    wrapper = load_r3m("resnet50")
    state   = torch.load(checkpoint_path, map_location=device)
    wrapper.module.load_state_dict(state["encoder"])
    wrapper.eval().to(device)
    print(f"[load_finetuned_r3m] Loaded from {checkpoint_path}")
    return wrapper


if __name__ == "__main__":
    main()
# Fine-Tuning Script Reference

Scripts for fine-tuning R3M and VIP encoders on exocentric video datasets.

**Location:** `src/train/`

---

## Scripts

| Script | Encoder | Dataset | Location |
|---|---|---|---|
| `r3m_droid_finetune.py` | R3M | DROID | `src/train/droid/` |
| `vip_droid_finetune.py` | VIP | DROID | `src/train/droid/` |
| `r3m_egoexo4d_finetune.py` | R3M | Ego-Exo4D | `src/train/egoexo4d/` |
| `vip_egoexo4d_finetune.py` | VIP | Ego-Exo4D | `src/train/egoexo4d/` |

---

## DROID Fine-Tuning

### Flags — R3M and VIP (shared)

| Flag | Type | Default | Description |
|---|---|---|---|
| `--data_dir` | `str` | **required** | Path to directory containing DROID TFRecord shards |
| `--out_dir` | `str` | `./checkpoints` | Directory to save checkpoint files |
| `--epochs` | `int` | `5` | Number of fine-tuning epochs |
| `--lr` | `float` | `3e-5` | Learning rate. Keep small to avoid catastrophic forgetting |
| `--batch_size` | `int` | `16` | Episodes per training batch |
| `--buffer_size` | `int` | `100` | Episodes held in RAM at once. 100 episodes ≈ 2.5 GB RAM |
| `--steps_per_epoch` | `int` | `200` | Training steps per epoch |
| `--num_negatives` | `int` | `3` | Number of negative frame pairs per anchor |
| `--camera` | `str` | `exterior_image_1_left` | Which DROID camera stream to use. Choices: `exterior_image_1_left`, `exterior_image_2_left` |
| `--seed` | `int` | `42` | Random seed |
| `--save_every` | `int` | `1` | Save a checkpoint every N epochs |
| `--resume` | `str` | `None` | Path to a previous `.pt` checkpoint to resume training from |
| `--finetune_mode` | `str` | `projection_head` | Fine-tuning strategy. Choices: `projection_head`, `full`. See below |

### R3M-only flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--lambda1` | `float` | `1.0` | TCN (time-contrastive) loss weight |
| `--lambda2` | `float` | `1.0` | Language alignment loss weight. Set to `0` to disable |
| `--lambda3` | `float` | `1e-5` | L1 sparsity regularisation weight |
| `--lambda4` | `float` | `1e-5` | L2 regularisation weight |

### VIP-only flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--gamma` | `float` | `0.98` | Discount factor in the VIP dual RL objective (paper default: 0.98) |

---

### Fine-tuning modes

**`projection_head`** (recommended): Freezes the ResNet-50 backbone entirely and updates only the final FC projection layer. Preserves the Ego4D pretraining features and minimises the risk of catastrophic forgetting. Used in the frozen backbone condition in the paper.

**`full`**: Updates the entire ResNet-50 encoder including the convolutional backbone. Higher risk of catastrophic forgetting, especially on small datasets. Used in the full fine-tune condition in the paper.

---

### Checkpoints

Per-epoch checkpoints are saved as `r3m_droid_ep001.pt`, `r3m_droid_ep002.pt`, etc. The final checkpoint after all epochs is saved as `r3m_droid_finetuned.pt` (or `vip_droid_finetuned.pt`). Pass the final checkpoint path to the evaluation scripts using `--checkpoint`.

---

### DROID Examples

### Generic template

```bash
# Replace <model> with r3m or vip
# Replace <data_dir> with path to your TFRecord shard directory
# Replace <out_dir> with desired checkpoint output directory
# Replace <mode> with projection_head or full
# Replace <num_epochs> with number of training epochs
# Replace <buffer_size> with episodes to hold in RAM (100 ≈ 2.5 GB)
# Replace <steps> with training steps per epoch
# Replace <batch> with batch size
# Replace <lr> with learning rate (keep small, e.g. 3e-5)

python3 src/train/droid/<model>_droid_finetune.py \
    --data_dir <data_dir> \
    --out_dir <out_dir> \
    --finetune_mode <mode> \
    --epochs <num_epochs> \
    --buffer_size <buffer_size> \
    --steps_per_epoch <steps> \
    --batch_size <batch> \
    --lr <lr>
```

---

#### Projection head fine-tune (paper condition: frozen backbone)
```bash
python3 src/train/droid/r3m_droid_finetune.py \
    --data_dir ./droid_sample/1.0.0 \
    --out_dir ./checkpoints/r3m \
    --finetune_mode projection_head \
    --epochs 5 \
    --buffer_size 100 \
    --steps_per_epoch 200 \
    --batch_size 16 \
    --lr 3e-5

python3 src/train/droid/vip_droid_finetune.py \
    --data_dir ./droid_sample/1.0.0 \
    --out_dir ./checkpoints/vip \
    --finetune_mode projection_head \
    --epochs 5 \
    --buffer_size 100 \
    --steps_per_epoch 200 \
    --batch_size 16 \
    --lr 3e-5
```

#### Full backbone fine-tune (paper condition: full fine-tune)
```bash
python3 src/train/droid/r3m_droid_finetune.py \
    --data_dir ./droid_sample/1.0.0 \
    --out_dir ./checkpoints/r3m \
    --finetune_mode full \
    --epochs 5 \
    --buffer_size 100 \
    --steps_per_epoch 200 \
    --batch_size 16 \
    --lr 3e-5
```

#### Resume from a checkpoint
```bash
python3 src/train/droid/r3m_droid_finetune.py \
    --data_dir ./droid_sample/1.0.0 \
    --out_dir ./checkpoints/r3m \
    --finetune_mode projection_head \
    --epochs 10 \
    --resume ./checkpoints/r3m/r3m_droid_ep005.pt
```

#### Quick smoke test (small buffer, few steps)
```bash
python3 src/train/droid/r3m_droid_finetune.py \
    --data_dir ./droid_sample/1.0.0 \
    --out_dir ./checkpoints/r3m \
    --epochs 1 \
    --buffer_size 20 \
    --steps_per_epoch 5 \
    --batch_size 4
```

---

## Ego-Exo4D Fine-Tuning

### Flags — R3M

| Flag | Type | Default | Description |
|---|---|---|---|
| `--data_dir` | `str` | **required** | Root directory of exocentric clips |
| `--out_dir` | `str` | `./checkpoints` | Directory to save checkpoint files |
| `--epochs` | `int` | `5` | Number of fine-tuning epochs |
| `--lr` | `float` | `3e-5` | Learning rate |
| `--batch_size` | `int` | `8` | Video clips per batch (each clip → 5 frames) |
| `--frames_per_clip` | `int` | `5` | Frames sampled per clip |
| `--num_negatives` | `int` | `3` | Negative samples per anchor (R3M default: 3) |
| `--lambda1` | `float` | `1.0` | TCN loss weight |
| `--lambda2` | `float` | `1.0` | Language loss weight. Set to `0` to disable |
| `--lambda3` | `float` | `1e-5` | L1 regularisation weight |
| `--lambda4` | `float` | `1e-5` | L2 regularisation weight |
| `--seed` | `int` | `42` | Random seed |
| `--max_clips` | `int` | `None` | Cap on number of clips to use. Useful for quick smoke tests |
| `--save_every` | `int` | `1` | Save a checkpoint every N epochs |
| `--resume` | `str` | `None` | Path to a previous `.pt` checkpoint to resume from |

### Flags — VIP

| Flag | Type | Default | Description |
|---|---|---|---|
| `--data_dir` | `str` | **required** | Root directory of exocentric clips |
| `--out_dir` | `str` | `./checkpoints` | Directory to save checkpoint files |
| `--epochs` | `int` | `5` | Number of fine-tuning epochs |
| `--lr` | `float` | `3e-5` | Learning rate |
| `--batch_size` | `int` | `16` | Clips per batch (VIP paper default: 16) |
| `--gamma` | `float` | `0.98` | Discount factor in VIP objective |
| `--num_negatives` | `int` | `3` | Middle-frame pairs per clip |
| `--seed` | `int` | `42` | Random seed |
| `--max_clips` | `int` | `None` | Cap on clips. Use ~500 for a smoke test |
| `--save_every` | `int` | `1` | Save a checkpoint every N epochs |
| `--resume` | `str` | `None` | Path to a previous `.pt` checkpoint to resume from |

---

### Ego-Exo4D Examples

### Generic template

```bash
# Replace <model> with r3m or vip
# Replace <data_dir> with root directory of your exocentric clips
# Replace <out_dir> with desired checkpoint output directory
# Replace <num_epochs> with number of training epochs
# Replace <batch> with batch size (r3m default: 8, vip default: 16)
# Replace <lr> with learning rate (keep small, e.g. 3e-5)
# Replace <max_clips> with a clip cap for smoke tests, or omit for all clips

python3 src/train/egoexo4d/<model>_egoexo4d_finetune.py \
    --data_dir <data_dir> \
    --out_dir <out_dir> \
    --epochs <num_epochs> \
    --batch_size <batch> \
    --lr <lr> \
    --max_clips <max_clips>
```

---

#### Standard fine-tune
```bash
python3 src/train/egoexo4d/r3m_egoexo4d_finetune.py \
    --data_dir ./egoexo4d_raw \
    --out_dir ./checkpoints/r3m \
    --epochs 5 \
    --batch_size 8 \
    --lr 3e-5

python3 src/train/egoexo4d/vip_egoexo4d_finetune.py \
    --data_dir ./egoexo4d_raw \
    --out_dir ./checkpoints/vip \
    --epochs 5 \
    --batch_size 16 \
    --lr 3e-5
```

#### Smoke test with capped clips
```bash
python3 src/train/egoexo4d/r3m_egoexo4d_finetune.py \
    --data_dir ./egoexo4d_raw \
    --out_dir ./checkpoints/r3m \
    --epochs 1 \
    --max_clips 50 \
    --batch_size 4
```

---

## After Fine-Tuning: Running Evaluation

Once fine-tuning completes, pass the checkpoint to the eval scripts. See `docs/EVAL_REFERENCE.md` for full evaluation options.

```bash
# R3M DROID fine-tune → evaluate
python3 src/evals/metaworld/r3m_metaworld_multitask.py \
    --encoder droid \
    --checkpoint ./checkpoints/r3m/r3m_droid_finetuned.pt \
    --tag droid_full

# VIP Ego-Exo4D fine-tune → evaluate
python3 src/evals/metaworld/vip_metaworld_multitask.py \
    --encoder egoexo4d \
    --checkpoint ./checkpoints/vip/vip_egoexo4d_finetuned.pt \
    --tag egoexo4d
```

# ego_exo_analysis_26

Reproduction and extension of R3M and VIP visual representations for robot manipulation, with exocentric fine-tuning experiments using DROID and Ego-Exo4D.

This repository accompanies the paper:
**"Fine-Tuning Egocentric Visual Encoders on Exocentric Data for Robot Manipulation"**
David Ross, Northeastern University

---

> **Quick setup:** A setup script is available that automates the virtual environment creation, package installation, and Hydra/Python 3.12 compatibility patch. Run it from the repository root:
> ```bash
> chmod +x setup.sh
> ./setup.sh
> ```
> Run `./setup.sh --help` for all options, including `--env-name` to customize the venv folder name and `--prompt` to step through each stage interactively.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [System Requirements](#system-requirements)
- [Environment Setup](#environment-setup)
- [Installing R3M and VIP](#installing-r3m-and-vip)
- [Known Compatibility Issues and Fixes](#known-compatibility-issues-and-fixes)
- [Datasets](#datasets)
  - [MetaWorld (no download required)](#metaworld-no-download-required)
  - [DROID](#droid)
  - [Ego-Exo4D](#ego-exo4d)
- [Running Experiments](#running-experiments)
  - [MetaWorld Baseline Evaluation](#metaworld-baseline-evaluation)
  - [DROID Fine-tuning](#droid-fine-tuning)
  - [Ego-Exo4D Fine-tuning](#ego-exo4d-fine-tuning)
- [Results Data](#results-data)
- [Further Documentation](#further-documentation)

---

## Overview

This project investigates whether fine-tuning the visual encoders of R3M and VIP on exocentric (third-person) video data can improve their downstream behavioral cloning performance on robot manipulation tasks. The pipeline has three stages:

1. **Baseline evaluation** — frozen R3M and VIP encoders evaluated on five MetaWorld tasks across 3 camera viewpoints, 3 demonstration sizes, and 3 seeds.
2. **DROID fine-tuning** — projection head fine-tuning on a subset of the DROID robot manipulation dataset, then re-evaluation.
3. **Ego-Exo4D fine-tuning** — projection head fine-tuning on exocentric video from the Ego-Exo4D dataset (cooking and bike repair scenarios), then re-evaluation.

---

## Repository Structure

```
ego_exo_analysis_26/
├── README.md
├── docs/                              # Additional documentation
│   ├── DROID_DOWNLOAD.md              # Detailed DROID download guide
│   └── EGOEXO4D_ACCESS.md             # Ego-Exo4D access and download guide
├── r3m/                               # R3M submodule (facebookresearch/r3m)
├── vip/                               # VIP submodule (facebookresearch/vip)
├── src/
│   ├── evals/
│   │   ├── metaworld/
│   │   │   ├── r3m_metaworld_multitask.py  # R3M baseline MetaWorld evaluation
│   │   │   └── vip_metaworld_multitask.py  # VIP baseline MetaWorld evaluation
│   │   ├── franka_kitchen/
│   │   │   └── r3m_franka_kitchen_multitask.py
│   │   └── adroit/
│   │       └── r3m_adroit.py
│   └── train/
│       ├── droid/
│       │   ├── r3m_droid_finetune.py  # R3M fine-tuning on DROID
│       │   └── vip_droid_finetune.py  # VIP fine-tuning on DROID
│       └── egoexo4d/
│           ├── r3m_egoexo4d_finetune.py  # R3M fine-tuning on Ego-Exo4D
│           └── vip_egoexo4d_finetune.py  # VIP fine-tuning on Ego-Exo4D
├── checkpoints/                       # Fine-tuned encoder checkpoints
│   ├── r3m/                           # Full fine-tune (backbone + projection head)
│   ├── r3m_frozen/                    # Projection-head-only fine-tune
│   ├── vip/
│   └── vip_frozen/
├── results/
│   ├── r3m/
│   │   ├── baseline/metaworld/        # Frozen baseline CSV results
│   │   └── droid_finetune/metaworld/
│   │       ├── frozen_1/              # Projection-head-only fine-tune results
│   │       └── unfrozen_1/            # Full fine-tune results
│   └── vip/
│       ├── baseline/metaworld/
│       └── droid_finetune/metaworld/
│           ├── frozen_1/
│           └── unfrozen_1/
└── video/
    └── samples/                       # Sample policy rollout videos
        ├── metaworld/
        ├── franka_kitchen/
        └── adroit/
```

---

## System Requirements

- **OS:** Linux (Ubuntu 22.04 or 24.04 recommended). Not tested on macOS or Windows.
- **Python:** 3.12 (used throughout; see compatibility notes below for VIP)
- **GPU:** CUDA-capable GPU strongly recommended. All experiments were run on a single GPU. CPU-only is possible but impractically slow for fine-tuning.
- **Storage:**
  - MetaWorld baseline only: ~5 GB
  - With DROID sample (droid_100, 100 episodes): ~7 GB total
  - With larger DROID subset: up to several hundred GB (see [DROID section](#droid))
  - With Ego-Exo4D (cooking + bike repair exo): ~80-150 GB depending on resolution selected
- **System binaries required:**
  - `libgl1-mesa-glx` or equivalent (for MetaWorld rendering)
  - `libglib2.0-0`
  - `gsutil` (for DROID download via Google Cloud Storage)
  - `awscli` (for Ego-Exo4D download)

Install system binaries on Ubuntu:
```bash
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

---

## Environment Setup

Clone the repository including submodules:

```bash
git clone --recurse-submodules https://github.com/DavidRoss85/ego_exo_analysis_26.git
cd ego_exo_analysis_26
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

You will need to activate this environment every time you open a new terminal before running any scripts:

```bash
source .venv/bin/activate
```

Install the core Python dependencies:

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy pillow tqdm wandb transformers
pip install metaworld
pip install tensorflow tensorflow-datasets
pip install tfrecord
pip install ego4d
```

> **Note on protobuf:** TensorFlow 2.21 requires protobuf >= 6.31.1. If you hit protobuf conflicts, run:
> ```bash
> pip install "protobuf>=6.31.1,<8.0.0"
> ```

> **Note on TensorFlow/PyTorch GPU conflict:** The fine-tuning scripts handle this automatically by hiding the GPU from TensorFlow before PyTorch loads. Do not import TensorFlow before PyTorch in your own scripts or this will cause GPU memory conflicts.

---

## Installing R3M and VIP

R3M and VIP are included as git submodules. Install them as editable packages into your virtual environment:

```bash
# Make sure your venv is active first
source .venv/bin/activate

# Install R3M
pip install -e ./r3m

# Install VIP
pip install -e ./vip
```

Verify both installed correctly:

```bash
python3 -c "from r3m import load_r3m; print('R3M OK')"
python3 -c "from vip import load_vip; print('VIP OK')"
```

Pre-trained model weights will be downloaded automatically the first time you call `load_r3m()` or `load_vip()` and cached in `~/.r3m` and `~/.vip` respectively. Each model is approximately 200-400 MB.

---

## Known Compatibility Issues and Fixes

### VIP and Python 3.12: Hydra dataclass error

VIP depends on `hydra-core 1.3.2`, which is not officially compatible with Python 3.12. Running VIP on Python 3.12 will produce this error on import:

```
ValueError: mutable default <class 'hydra.conf.JobConf.JobConfig.OverrideDirname'>
for field override_dirname is not allowed: use default_factory
```

**Fix:** Patch the two offending lines in the hydra configuration file. First, find the file:

```bash
find .venv -path "*/hydra/conf/__init__.py"
```

Open the file and locate the `JobConfig` class (around line 70). Change these two lines:

```python
# Before (broken on Python 3.12):
override_dirname: OverrideDirname = OverrideDirname()
# ...
config: JobConfig = JobConfig()

# After (fixed):
override_dirname: OverrideDirname = field(default_factory=OverrideDirname)
# ...
config: JobConfig = field(default_factory=JobConfig)
```

Make sure `field` is imported at the top of that file (`from dataclasses import dataclass, field`). There may be additional mutable default errors in other hydra files — apply the same `field(default_factory=...)` pattern to each one until the import succeeds.

If you prefer to avoid this altogether, you can use Python 3.9 or 3.10 for your virtual environment, which are fully compatible with VIP and hydra without patching.

### MetaWorld environment version: v2 vs v3

This repository uses MetaWorld v3 environments. If you encounter:

```
ValueError: button-press-v2 is not a V3 environment
```

Ensure you are using `-v3` suffixes in all task names (e.g., `button-press-v3`, not `button-press-v2`). The scripts in this repo already use v3 naming.

### MetaWorld rendering: unexpected keyword argument 'mode'

Newer MetaWorld versions changed the render API. The scripts in this repo use the updated gymnasium-style `env.render()` call without a `mode` argument. If you run into render errors, check that you are running the scripts from this repo rather than code adapted from older MetaWorld examples.

---

## Datasets

### MetaWorld (no download required)

MetaWorld is installed as a Python package and the simulation environment runs locally. No dataset download is needed. Expert demonstrations are generated automatically by the evaluation scripts using MetaWorld's built-in scripted policies.

The Assembly task required a custom IK heuristic because the built-in policy is incompatible with the v3 environment. This is already implemented in the evaluation scripts.

### DROID

DROID is a large-scale robot manipulation dataset stored in Google Cloud Storage as TFRecord shards. Access requires a Google Cloud account with `gsutil` installed.

Install the Google Cloud CLI (which includes `gsutil`):

```bash
# Follow instructions at https://cloud.google.com/sdk/docs/install
# Then authenticate:
gcloud auth login
```

**Option 1 — DROID-100 sample (recommended starting point)**

This is a 100-episode sample (~2 GB total, 31 shards) that requires no account permissions beyond public bucket access:

```bash
mkdir -p ./droid_sample/1.0.0

# Download all metadata files first
gsutil cp gs://gresearch/robotics/droid_100/1.0.0/dataset_info.json ./droid_sample/1.0.0/
gsutil cp gs://gresearch/robotics/droid_100/1.0.0/features.json ./droid_sample/1.0.0/

# Download all 31 shards (~2 GB)
gsutil -m cp \
  "gs://gresearch/robotics/droid_100/1.0.0/r2d2_faceblur-train.tfrecord-*-of-00031" \
  ./droid_sample/1.0.0/
```

To download only a subset (e.g., 5 shards for a quick smoke test):

```bash
gsutil -m cp \
  "gs://gresearch/robotics/droid_100/1.0.0/r2d2_faceblur-train.tfrecord-0000[0-4]-of-00031" \
  ./droid_sample/1.0.0/
```

**Option 2 — Larger DROID subset from the full dataset**

The full DROID dataset is ~1.7 TB across thousands of shards. You can download a partial subset by selecting specific shard index ranges. Each shard is approximately 60-100 MB and contains roughly 3-4 episodes. See `docs/DROID_DOWNLOAD.md` for a full guide on how to enumerate and selectively download shards at scale.

**Space requirements by subset size:**

| Subset | Approx size | Approx episodes |
|---|---|---|
| droid_100 (all 31 shards) | ~2 GB | 100 |
| 50 shards from full dataset | ~3-5 GB | ~150-200 |
| 500 shards | ~30-50 GB | ~1,500-2,000 |
| Full dataset | ~1.7 TB | 76,000 |

### Ego-Exo4D

Ego-Exo4D requires a signed license agreement and approved access request. This is a separate process from regular Ego4D access, even if you have previously been approved for that dataset.

**Step 1 — Request access**

Go to [ego4d.dev/request/ego4d](https://ego4d.dev/request/ego4d) and complete the access request form. Select Ego-Exo4D specifically. You will receive AWS credentials by email once approved (typically within a few days).

**Step 2 — Configure AWS credentials**

```bash
pip install ego4d awscli

aws configure --profile ego4d
# Enter your Access Key ID and Secret Access Key from the approval email
# Leave region and output format blank (press Enter)
```

**Step 3 — Download using the egoexo CLI**

The `egoexo` command is installed as part of the `ego4d` package:

```bash
# List available takes to confirm access works
egoexo --list-takes --profile ego4d

# Download only exocentric video for cooking and bike repair
egoexo \
  -o ./egoexo4d_raw \
  --parts takes \
  --scenarios cooking,bike_repair \
  --views exo \
  --s3_profile ego4d
```

**Space requirements:**

Cooking + bike repair exo clips at standard resolution: approximately 80-150 GB depending on the number of takes selected. You can reduce this further by passing `--max_takes N` if your CLI version supports it, or by downloading one scenario at a time and filtering locally.

See `docs/EGOEXO4D_ACCESS.md` for a more detailed walkthrough including how to troubleshoot the common "could not get manifests" error and how to verify your credentials are provisioned for Ego-Exo4D specifically (as opposed to regular Ego4D).

---

## Running Experiments

### MetaWorld Baseline Evaluation

Evaluate frozen R3M across all five tasks, three camera viewpoints, and a given number of demonstrations:

```bash
python3 src/evals/metaworld/r3m_metaworld_multitask.py \
  --demo_episodes 10 \
  --camera_id 0 \
  --train_steps 20000 \
  --batch_size 32 \
  --eval_episodes 20 \
  --seed 42
```

For VIP:

```bash
python3 src/evals/metaworld/vip_metaworld_multitask.py \
  --demo_episodes 10 \
  --camera_id 0 \
  --train_steps 20000 \
  --batch_size 32 \
  --eval_episodes 20 \
  --seed 42
```

Key arguments:
- `--demo_episodes`: Number of expert demonstrations (5, 10, or 25)
- `--camera_id`: Camera viewpoint (0 = front, 1 = side, 2 = top-down)
- `--seed`: Random seed (used 42, 123, 456 in the paper)

Results are saved as CSV files in the working directory.

### DROID Fine-tuning

Fine-tune the R3M projection head on DROID data, then run MetaWorld evaluation:

```bash
python3 src/train/droid/r3m_droid_finetune.py \
  --data_dir ./droid_sample/1.0.0 \
  --out_dir ./checkpoints/r3m_frozen \
  --epochs 5 \
  --max_episodes 100 \
  --batch_size 16
```

For VIP:

```bash
python3 src/train/droid/vip_droid_finetune.py \
  --data_dir ./droid_sample/1.0.0 \
  --out_dir ./checkpoints/vip_frozen \
  --epochs 5 \
  --max_episodes 100 \
  --batch_size 16
```

After fine-tuning completes, pass the checkpoint path to the evaluation scripts with `--encoder_checkpoint ./checkpoints/r3m_droid_finetuned.pt` (exact flag name may vary; check script `--help` output).

### Ego-Exo4D Fine-tuning

Fine-tune on Ego-Exo4D exocentric video:

```bash
python3 src/train/egoexo4d/r3m_egoexo4d_finetune.py \
  --data_dir ./egoexo4d_raw \
  --out_dir ./checkpoints/r3m_frozen \
  --epochs 5 \
  --batch_size 16
```

```bash
python3 src/train/egoexo4d/vip_egoexo4d_finetune.py \
  --data_dir ./egoexo4d_raw \
  --out_dir ./checkpoints/vip_frozen \
  --epochs 5 \
  --batch_size 16
```

---

## Results Data

The fine-tuned encoder checkpoints used in the experiments reported in the paper are included in the `checkpoints/` directory. This means you can run the MetaWorld evaluation scripts directly against the paper's exact fine-tuned models without needing to re-run the fine-tuning pipeline yourself. Each subdirectory contains per-epoch checkpoints as well as the final `*_finetuned.pt` file used for evaluation.

All raw CSV result files are also included in the `results/` directory, organized by model, experiment type, and fine-tuning condition:

```
results/
├── r3m/baseline/metaworld/          # Frozen R3M baseline
├── r3m/droid_finetune/metaworld/
│   ├── frozen_1/                    # Projection-head-only fine-tune
│   └── unfrozen_1/                  # Full backbone fine-tune
└── vip/  (same structure as r3m/)
```

Column definitions:
- `task`: MetaWorld task name
- `demo_episodes`: Number of expert demonstrations used
- `camera_id`: Camera viewpoint (0/1/2)
- `train_steps`: Gradient steps during BC training
- `batch_size`: Training batch size
- `eval_episodes`: Number of evaluation rollouts
- `seed`: Random seed
- `success_rate`: Per-seed success rate
- `mean`: Mean success rate across seeds
- `std`: Standard deviation across seeds
- `row_type`: Whether this row is a per-seed result or the aggregate summary

---

## Further Documentation

Additional guides are in the `docs/` folder:

- `docs/DROID_DOWNLOAD.md` — How to enumerate and selectively download shards from the full DROID dataset at scale, including bandwidth and storage planning
- `docs/EGOEXO4D_ACCESS.md` — Detailed Ego-Exo4D access walkthrough, credential setup, common CLI errors and fixes, and how to select specific scenarios and camera streams
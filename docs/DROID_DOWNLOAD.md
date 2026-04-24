# DROID Dataset Download Guide

DROID (Diverse Robot Open-world Dataset) is stored as TFRecord shards in a public Google Cloud Storage bucket. This guide covers how to download it selectively, plan storage, and integrate it with the fine-tuning scripts.

---

## Prerequisites

Install the Google Cloud CLI, which includes `gsutil`:

```bash
# Ubuntu/Debian
sudo apt-get install apt-transport-https ca-certificates gnupg
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update && sudo apt-get install google-cloud-cli
```

Authenticate:

```bash
gcloud auth login
```

DROID is a public bucket so authentication is optional for read access, but it is required to avoid rate limits on large downloads.

---

## Dataset Structure

The DROID bucket has two main datasets:

```
gs://gresearch/robotics/droid_100/   # 100-episode sample, ~2 GB
gs://gresearch/robotics/droid/       # Full dataset, ~1.7 TB, 76,000+ episodes
```

Each dataset is versioned at `1.0.0/` and stored as TFRecord shards:

```
1.0.0/dataset_info.json
1.0.0/features.json
1.0.0/r2d2_faceblur-train.tfrecord-NNNNN-of-TOTAL
```

The `droid_100` sample has 31 shards. The full `droid` dataset has roughly 800+ shards.

---

## Option 1: DROID-100 Sample (Recommended Starting Point)

This is the easiest entry point. The full 100-episode sample is about 2 GB.

```bash
mkdir -p ./droid_sample/1.0.0

# Metadata files
gsutil cp gs://gresearch/robotics/droid_100/1.0.0/dataset_info.json ./droid_sample/1.0.0/
gsutil cp gs://gresearch/robotics/droid_100/1.0.0/features.json ./droid_sample/1.0.0/

# All 31 shards in parallel (-m flag enables parallel download)
gsutil -m cp \
  "gs://gresearch/robotics/droid_100/1.0.0/r2d2_faceblur-train.tfrecord-*-of-00031" \
  ./droid_sample/1.0.0/
```

To download only a few shards for a quick smoke test (e.g., shards 0 through 4):

```bash
gsutil -m cp \
  "gs://gresearch/robotics/droid_100/1.0.0/r2d2_faceblur-train.tfrecord-0000[0-4]-of-00031" \
  ./droid_sample/1.0.0/
```

Pass the metadata files even when downloading partial shards -- the fine-tuning scripts use them to parse the schema.

---

## Option 2: Selective Download from the Full Dataset

### Step 1: List available shards

```bash
gsutil ls gs://gresearch/robotics/droid/1.0.0/ | grep tfrecord | head -20
```

This shows shard names like `r2d2_faceblur-train.tfrecord-00000-of-00823`.

### Step 2: Download a contiguous range of shards

Use zero-padded shard indices. For shards 0 through 49 (approximately 3-5 GB):

```bash
mkdir -p ./droid_full_subset/1.0.0

gsutil cp gs://gresearch/robotics/droid/1.0.0/dataset_info.json ./droid_full_subset/1.0.0/
gsutil cp gs://gresearch/robotics/droid/1.0.0/features.json ./droid_full_subset/1.0.0/

gsutil -m cp \
  "gs://gresearch/robotics/droid/1.0.0/r2d2_faceblur-train.tfrecord-0000[0-9]-of-00823" \
  "gs://gresearch/robotics/droid/1.0.0/r2d2_faceblur-train.tfrecord-0004[0-9]-of-00823" \
  ./droid_full_subset/1.0.0/
```

You can also generate a list of shard paths and pass it to gsutil:

```bash
# Generate list of shard paths for shards 0-99
python3 -c "
for i in range(100):
    print(f'gs://gresearch/robotics/droid/1.0.0/r2d2_faceblur-train.tfrecord-{i:05d}-of-00823')
" > shard_list.txt

gsutil -m cp $(cat shard_list.txt | tr '\n' ' ') ./droid_full_subset/1.0.0/
```

### Step 3: Tell the fine-tuning scripts where to find the data

The fine-tuning scripts accept `--data_dir` pointing to the directory containing the shards and metadata files. Point it at whichever directory you downloaded into:

```bash
python3 src/r3m_droid_finetune.py \
  --data_dir ./droid_full_subset/1.0.0 \
  --out_dir ./checkpoints \
  --max_episodes 500
```

---

## Storage Planning

| Scenario | Shards | Approx size | Approx episodes |
|---|---|---|---|
| Smoke test (5 shards, droid_100) | 5 | ~300 MB | ~15 |
| Full droid_100 sample | 31 | ~2 GB | 100 |
| 50 shards from full dataset | 50 | ~3-5 GB | ~150-200 |
| 200 shards | 200 | ~12-20 GB | ~600-800 |
| 500 shards | 500 | ~30-50 GB | ~1,500-2,000 |
| Full dataset | ~823 | ~1.7 TB | 76,000+ |

Each shard is roughly 60-100 MB and contains approximately 3-4 episodes. These are rough estimates since episode lengths vary.

---

## TFRecord Reading

The fine-tuning scripts use the pure-Python `tfrecord` package to read DROID shards without requiring TensorFlow (which would conflict with PyTorch for GPU allocation). Install it with:

```bash
pip install tfrecord
```

If you see schema parsing errors when loading shards from the full `droid` dataset after only downloading the metadata from `droid_100` (or vice versa), make sure the `dataset_info.json` and `features.json` files match the shard source.

---

## Resuming Interrupted Downloads

`gsutil -m cp` does not resume partial downloads automatically. If a download is interrupted, re-run the same command -- gsutil will skip files that already exist at the destination. For very large downloads, consider running in a `tmux` or `screen` session so the transfer continues if your terminal disconnects.
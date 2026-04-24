# Evaluation Script Reference

Scripts for evaluating R3M and VIP encoders on the MetaWorld benchmark.

**Location:** `src/evals/metaworld/`

---

## Scripts

| Script | Encoder |
|---|---|
| `r3m_metaworld_multitask.py` | R3M (ResNet-50, 2048-dim) |
| `vip_metaworld_multitask.py` | VIP (ResNet-50, 1024-dim) |

Both scripts have identical flags and behavior. The only difference is the encoder used.

---

## Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--encoder` | `str` | `baseline` | Which encoder to evaluate. Choices: `baseline`, `droid`, `egoexo4d` |
| `--checkpoint` | `str` | `None` | Path to a fine-tuned `.pt` checkpoint file. Required when `--encoder` is not `baseline` |
| `--single_run` | flag | off | Run a single camera/demo combination instead of the full sweep |
| `--demos` | `int` | `10` | Number of expert demonstrations. Only used in `--single_run` mode |
| `--camera` | `int` | `0` | Camera viewpoint ID. Only used in `--single_run` mode. See camera reference below |
| `--tag` | `str` | `None` | String appended to the output CSV filename to distinguish runs (e.g. `frozen`, `full`) |
| `--visualize` | flag | off | Record policy rollout videos for each task |

---

## Camera Viewpoints

| ID | Viewpoint |
|---|---|
| `0` | Top-down |
| `1` | Front |
| `2` | Side |

---

## Behavior

**Without `--single_run`** (default): runs the full sweep -- all 5 tasks × 3 cameras × 3 demo sizes (5, 10, 25) × 3 seeds. This is the protocol used to generate the paper results. Produces 9 CSV files per encoder.

**With `--single_run`**: runs a single configuration -- all 5 tasks at the specified `--camera` and `--demos` values, 3 seeds. Useful for quick verification that an encoder works before committing to a full sweep.

Results are saved as timestamped CSV files in the working directory.

---

## Examples

### Generic template

```bash
# Replace <model> with r3m or vip
# Replace <encoder> with baseline, droid, or egoexo4d
# Replace <num_demos> with 5, 10, or 25
# Replace <camera_id> with 0, 1, or 2
# Replace <checkpoint_path> with path to your .pt file
# Replace <tag> with a label for this run (e.g. baseline, droid_full, egoexo4d)

python3 src/evals/metaworld/<model>_metaworld_multitask.py \
    --encoder <encoder> \
    --checkpoint <checkpoint_path> \
    --single_run \
    --demos <num_demos> \
    --camera <camera_id> \
    --tag <tag>
```

> Omit `--checkpoint` when using `--encoder baseline`.
> Omit `--single_run`, `--demos`, and `--camera` to run the full 3×3 sweep.

---

### Baseline evaluation (full sweep)
```bash
python3 src/evals/metaworld/r3m_metaworld_multitask.py --encoder baseline
python3 src/evals/metaworld/vip_metaworld_multitask.py --encoder baseline
```

### Quick single-run verification
```bash
python3 src/evals/metaworld/r3m_metaworld_multitask.py \
    --encoder baseline \
    --single_run \
    --demos 10 \
    --camera 0
```

### Evaluate a DROID fine-tuned encoder (full sweep)
```bash
python3 src/evals/metaworld/r3m_metaworld_multitask.py \
    --encoder droid \
    --checkpoint ./checkpoints/r3m/r3m_droid_finetuned.pt \
    --tag droid_full

python3 src/evals/metaworld/vip_metaworld_multitask.py \
    --encoder droid \
    --checkpoint ./checkpoints/vip/vip_droid_finetuned.pt \
    --tag droid_full
```

### Evaluate a DROID fine-tuned encoder (single run, side camera)
```bash
python3 src/evals/metaworld/r3m_metaworld_multitask.py \
    --encoder droid \
    --checkpoint ./checkpoints/r3m/r3m_droid_finetuned.pt \
    --single_run \
    --demos 25 \
    --camera 2 \
    --tag droid_full
```

### Evaluate an Ego-Exo4D fine-tuned encoder
```bash
python3 src/evals/metaworld/r3m_metaworld_multitask.py \
    --encoder egoexo4d \
    --checkpoint ./checkpoints/r3m/r3m_egoexo4d_finetuned.pt \
    --tag egoexo4d

python3 src/evals/metaworld/vip_metaworld_multitask.py \
    --encoder egoexo4d \
    --checkpoint ./checkpoints/vip/vip_egoexo4d_finetuned.pt \
    --tag egoexo4d
```

### Record rollout videos during evaluation
```bash
python3 src/evals/metaworld/r3m_metaworld_multitask.py \
    --encoder baseline \
    --single_run \
    --demos 25 \
    --camera 0 \
    --visualize
```

---

## Output CSV Format

Each run produces a timestamped CSV file, e.g. `r3m_results_baseline_demos10_cam0_20260419_133403.csv`.

| Column | Description |
|---|---|
| `task` | MetaWorld task name |
| `demo_episodes` | Number of expert demonstrations used |
| `camera_id` | Camera viewpoint (0/1/2) |
| `train_steps` | Gradient steps during BC training |
| `batch_size` | Training batch size |
| `eval_episodes` | Number of evaluation rollouts per seed |
| `seed` | Random seed |
| `success_rate` | Per-seed success rate |
| `mean` | Mean success rate across seeds |
| `std` | Standard deviation across seeds |
| `row_type` | `per_seed` for individual seed rows, `summary` for the aggregate |

---

## Tasks

The five MetaWorld tasks evaluated in this work:

| Task key | Environment | Description |
|---|---|---|
| `button-press` | `button-press-v3` | Press a button mounted on a surface |
| `drawer-open` | `drawer-open-v3` | Open a drawer by grasping the handle |
| `bin-picking` | `bin-picking-v3` | Pick a block and place it in the target bin |
| `hammer` | `hammer-v3` | Hammer a nail into a block |
| `assembly` | `assembly-v3` | Place a ring onto a peg (hardest task) |

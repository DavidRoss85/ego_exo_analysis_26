# =============================================================================
# vip_metaworld_multitask.py
#
# Reproduces the MetaWorld evaluation from the VIP paper (Ma et al. 2023)
# across all 5 tasks:
#   - button-press-v3    (our working baseline)
#   - assembly-v3        (ring onto peg)
#   - bin-picking-v3     (block between bins)
#   - drawer-open-v3     (open drawer)
#   - hammer-v3          (hammer nail)
#
# Architecture mirrors the R3M paper setup, but substitutes the VIP encoder:
#   - VIP (ResNet-50) frozen visual encoder → 1024-dim embedding
#     (Note: VIP FC output dim is 1024, vs R3M's 2048 — see Table 2 of the paper)
#   - Proprioception concatenated (task-specific dim)
#   - 2-layer MLP [256, 256] policy head (Behavioral Cloning)
#   - 10 expert demos per task (paper uses 5/10/25; we default to 10)
#   - 500-step horizon per episode
#
# Structure is intentionally identical to r3m_metaworld_multitask.py so that
# results are directly comparable. The only meaningful difference is the encoder.
#
# CLI usage:
#   # Baseline (original pre-trained VIP):
#   python3 vip_metaworld_multitask.py --encoder baseline
#
#   # Fine-tuned on DROID:
#   python3 vip_metaworld_multitask.py --encoder droid \
#       --checkpoint ./checkpoints/vip/vip_droid_finetuned.pt
#
#   # Fine-tuned on Ego-Exo4D (when available):
#   python3 vip_metaworld_multitask.py --encoder egoexo4d \
#       --checkpoint ./checkpoints/vip/vip_egoexo4d_finetuned.pt
#
#   # Quick single run instead of full sweep:
#   python3 vip_metaworld_multitask.py --encoder droid \
#       --checkpoint ./checkpoints/vip/vip_droid_finetuned.pt \
#       --single_run --demos 10 --camera 0
# =============================================================================

import argparse
import random
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import csv, datetime

# ---------- VIP ----------
from vip import load_vip

# ---------- MetaWorld ----------
import metaworld
import metaworld.policies as mwp


# =============================================================================
# Shared setup
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Encoder is loaded after CLI args are parsed — see load_encoder() below
vip = None


def load_encoder(encoder_type: str = "baseline", checkpoint: str = None):
    """
    Load the VIP encoder based on CLI flag.

    encoder_type:
        "baseline"  — original pre-trained VIP (no fine-tuning)
        "droid"     — VIP fine-tuned on DROID (requires --checkpoint)
        "egoexo4d"  — VIP fine-tuned on Ego-Exo4D (requires --checkpoint)
    """
    global vip

    if encoder_type == "baseline":
        print("[Encoder] Loading baseline VIP (pre-trained, no fine-tuning)")
        vip = load_vip()
    else:
        if checkpoint is None:
            raise ValueError(f"--checkpoint required for encoder type '{encoder_type}'")
        print(f"[Encoder] Loading fine-tuned VIP ({encoder_type}) from {checkpoint}")
        vip = load_vip()
        state = torch.load(checkpoint, map_location=DEVICE)
        vip.module.load_state_dict(state["encoder"])
        print(f"[Encoder] Fine-tuned weights loaded from {checkpoint}")

    vip.eval()
    vip.to(DEVICE)
    return vip


# VIP uses the same preprocessing pipeline as R3M (both trained on Ego4D
# with 224x224 center-cropped frames). Input expected in [0, 255] range.
TRANSFORMS = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
])

VISUAL_DIM = 1024   # VIP ResNet-50 FC output (R3M is 2048; VIP is 1024)
ACTION_DIM  = 4     # All MetaWorld Sawyer tasks: [dx, dy, dz, gripper]


# =============================================================================
# Task registry
# =============================================================================

TASK_REGISTRY = {
    "button-press": {
        "env_name":     "button-press-v3",
        "policy_class": None,
        "proprio_dim":  39,
        "max_steps":    500,
        "camera_id":    0,
        "notes":        "Custom heuristic — built-in policy reads wrong obs indices",
    },
    "assembly": {
        "env_name":     "assembly-v3",
        "policy_class": mwp.SawyerAssemblyV3Policy,
        "proprio_dim":  39,
        "max_steps":    500,
        "camera_id":    0,
        "notes":        "Ring onto peg",
    },
    "bin-picking": {
        "env_name":     "bin-picking-v3",
        "policy_class": mwp.SawyerBinPickingV3Policy,
        "proprio_dim":  39,
        "max_steps":    500,
        "camera_id":    0,
        "notes":        "Pick and place block between bins",
    },
    "drawer-open": {
        "env_name":     "drawer-open-v3",
        "policy_class": mwp.SawyerDrawerOpenV3Policy,
        "proprio_dim":  39,
        "max_steps":    500,
        "camera_id":    0,
        "notes":        "Open drawer",
    },
    "hammer": {
        "env_name":     "hammer-v3",
        "policy_class": mwp.SawyerHammerV3Policy,
        "proprio_dim":  39,
        "max_steps":    500,
        "camera_id":    0,
        "notes":        "Hammer nail",
    },
}


# =============================================================================
# Custom expert policies
# =============================================================================

class ButtonPressHeuristic:
    """
    3-phase heuristic for button-press-v3.
    Confirmed obs layout (39-dim Sawyer):
      [0:3]  EEF xyz   [3] gripper   [4:7] button xyz   [7:10] goal xyz
    """
    def get_action(self, obs):
        eef    = obs[0:3]
        button = obs[4:7]
        above  = button.copy(); above[2] += 0.04
        delta   = above - eef
        dist_xy = np.linalg.norm(delta[:2])
        if dist_xy > 0.02:
            ax = np.clip(delta[0] * 8, -1, 1)
            ay = np.clip(delta[1] * 8, -1, 1)
            az = np.clip(delta[2] * 4, -1, 1)
            action = np.array([ax, ay, az, 0.0])
        elif abs(delta[2]) > 0.02:
            action = np.array([0.0, 0.0, np.clip(delta[2] * 4, -1, 1), 0.0])
        else:
            action = np.array([0.0, 0.0, -1.0, 0.0])
        return np.clip(action, -1.0, 1.0)


class AssemblyHeuristic:
    """
    6-phase explicit state machine for assembly-v3.
    The peg position is read directly from MuJoCo via env.data.body('peg').xpos.
    """
    XY_THRESH       = 0.01
    Z_THRESH        = 0.02
    GRASP_Z         = 0.03
    CARRY_Z         = 0.20
    GAIN_XY         = 8.0
    GAIN_Z          = 6.0
    HANDLE_OFFSET_X = 0.13

    def __init__(self, env):
        self.env   = env
        self.phase = 1

    def reset(self):
        self.phase = 1

    def get_action(self, obs):
        eef     = obs[0:3]
        gripper = obs[3]
        ring    = obs[4:7]
        peg_pos = self.env.data.body('peg').xpos
        peg_x, peg_y, peg_z = peg_pos[0], peg_pos[1], peg_pos[2]

        if self.phase == 1:
            above_ring    = ring.copy()
            above_ring[2] = ring[2] + self.GRASP_Z + 0.05
            delta_xy = above_ring[:2] - eef[:2]
            if np.linalg.norm(delta_xy) < self.XY_THRESH:
                self.phase = 2
            ax = np.clip(delta_xy[0] * self.GAIN_XY, -1, 1)
            ay = np.clip(delta_xy[1] * self.GAIN_XY, -1, 1)
            az = np.clip((above_ring[2] - eef[2]) * self.GAIN_Z, -1, 1)
            return np.clip([ax, ay, az, -1.0], -1, 1)
        if self.phase == 2:
            grasp_z = ring[2] + self.GRASP_Z
            if eef[2] <= grasp_z + self.Z_THRESH:
                self.phase = 3
            az = np.clip((grasp_z - eef[2]) * self.GAIN_Z, -1, 1)
            return np.clip([0.0, 0.0, az, -1.0], -1, 1)
        if self.phase == 3:
            if gripper <= 0.6:
                self.phase = 4
            return np.array([0.0, 0.0, -0.1, 1.0])
        if self.phase == 4:
            if eef[2] >= self.CARRY_Z - self.Z_THRESH:
                self.phase = 5
            az = np.clip((self.CARRY_Z - eef[2]) * self.GAIN_Z, -1, 1)
            return np.clip([0.0, 0.0, az, 1.0], -1, 1)
        if self.phase == 5:
            target_x = peg_x + self.HANDLE_OFFSET_X
            target_y = peg_y
            dx = target_x - eef[0]
            dy = target_y - eef[1]
            if abs(dx) < self.XY_THRESH and abs(dy) < self.XY_THRESH:
                self.phase = 6
            ax = np.clip(dx * self.GAIN_XY, -1, 1)
            ay = np.clip(dy * self.GAIN_XY, -1, 1)
            return np.clip([ax, ay, 0.0, 1.0], -1, 1)
        return np.array([0.0, 0.0, -1.0, -1.0])


CUSTOM_POLICIES = {
    "button-press": ButtonPressHeuristic(),
}


# =============================================================================
# Policy network
# =============================================================================

class VIPPolicy(nn.Module):
    """
    Visual + proprioceptive behavioural cloning policy.
    Input  : VIP embedding (1024) concatenated with proprioception (proprio_dim)
    Output : action (4-dim for all MetaWorld Sawyer tasks)
    """
    def __init__(self, proprio_dim: int, visual_dim: int = VISUAL_DIM,
                 action_dim: int = ACTION_DIM, hidden: int = 256):
        super().__init__()
        input_dim = visual_dim + proprio_dim
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, visual_features, proprioception):
        x = torch.cat([visual_features, proprioception], dim=1)
        return self.net(x)


# =============================================================================
# Per-task runner
# =============================================================================

class MetaWorldTaskRunner:

    def __init__(self, task_key: str, proprio_dim: int = None):
        cfg = TASK_REGISTRY[task_key]
        self.task_key    = task_key
        self.env_name    = cfg["env_name"]
        self.max_steps   = cfg["max_steps"]
        self.proprio_dim = proprio_dim or cfg["proprio_dim"]
        self.camera_id   = cfg["camera_id"]

        self.ml1 = metaworld.ML1(self.env_name)
        self.env = self.ml1.train_classes[self.env_name]()
        self.env.render_mode = "rgb_array"
        self.env.mujoco_renderer.camera_id = self.camera_id

        if task_key == "assembly":
            self.expert = AssemblyHeuristic(self.env)
        elif task_key in CUSTOM_POLICIES:
            self.expert = CUSTOM_POLICIES[task_key]
        else:
            self.expert = cfg["policy_class"]()

        self.policy = None

        print(f"[{task_key}] Loaded '{self.env_name}'  "
              f"proprio_dim={self.proprio_dim}  max_steps={self.max_steps}  "
              f"camera_id={self.camera_id}")
        if cfg["notes"]:
            print(f"  note: {cfg['notes']}")

    @staticmethod
    def _encode(rgb_array):
        """
        Encode an RGB frame using the frozen VIP encoder.
        VIP expects pixel values in [0, 255] — same convention as R3M.
        Returns a (1, 1024) CPU tensor.
        """
        img = Image.fromarray(rgb_array)
        t   = TRANSFORMS(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = vip(t * 255.0)
        return feat.cpu()

    def collect_demos(self, num_episodes: int = 10, debug: bool = False) -> list:
        demo_data  = []
        successes  = 0
        attempt    = 0

        print(f"\n[{self.task_key}] Collecting {num_episodes} successful demos …")

        while successes < num_episodes and attempt < num_episodes * 5:
            attempt += 1
            task = random.choice(self.ml1.train_tasks)
            self.env.set_task(task)
            obs, info = self.env.reset()
            if hasattr(self.expert, 'reset'):
                self.expert.reset()

            trajectory = []
            final_info = {}

            for step in range(self.max_steps):
                rgb         = self.env.render()
                visual_feat = self._encode(rgb)
                action      = self.expert.get_action(obs)

                trajectory.append({
                    "visual_features": visual_feat,
                    "proprioception":  obs.copy(),
                    "action":          action.copy(),
                })

                obs, _, done, truncated, info = self.env.step(action)
                final_info = info

                if debug and attempt == 1 and step < 3:
                    print(f"  step {step}  info={info}")

                if info.get("success", False):
                    break
                if done or truncated:
                    break

            succeeded = bool(final_info.get("success", False))
            status    = "✓" if succeeded else "✗"
            print(f"  attempt {attempt:3d}: {len(trajectory):3d} steps  {status}")

            if succeeded:
                demo_data.extend(trajectory)
                successes += 1

        print(f"[{self.task_key}] Collected {len(demo_data)} transitions "
              f"from {successes}/{num_episodes} successful episodes")
        return demo_data

    def train(self, demo_data: list, num_steps: int = 20000,
              batch_size: int = 32, lr: float = 1e-3):
        """Behavioural Cloning — MSE loss on (visual+proprio) → action.
        Matches R3M/VIP paper: fixed gradient steps, logging every 1000 steps.
        """
        self.policy = VIPPolicy(proprio_dim=self.proprio_dim).to(DEVICE)
        criterion   = nn.MSELoss()
        optimizer   = torch.optim.Adam(self.policy.parameters(), lr=lr)

        print(f"\n[{self.task_key}] Training VIP BC policy for {num_steps} steps …")

        self.policy.train()
        loss_history = []
        indices = list(range(len(demo_data)))

        for step in range(num_steps):
            batch_idx = random.choices(indices, k=batch_size)
            batch     = [demo_data[i] for i in batch_idx]

            vis  = torch.cat([b["visual_features"] for b in batch]).to(DEVICE)
            prop = torch.tensor(
                np.stack([b["proprioception"] for b in batch])).float().to(DEVICE)
            act  = torch.tensor(
                np.stack([b["action"] for b in batch])).float().to(DEVICE)

            pred = self.policy(vis, prop)
            loss = criterion(pred, act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if step % 1000 == 0 or step == num_steps - 1:
                print(f"  step {step:5d}/{num_steps}  loss={loss.item():.5f}")

        print(f"[{self.task_key}] Training complete.")
        return loss_history

    def evaluate(self, num_episodes: int = 20) -> float:
        assert self.policy is not None, "Call train() before evaluate()"
        self.policy.eval()

        successes  = 0
        tasks_used = []

        print(f"\n[{self.task_key}] Evaluating over {num_episodes} episodes …")

        for ep in range(num_episodes):
            task = random.choice(self.ml1.train_tasks)
            tasks_used.append(task)
            self.env.set_task(task)
            obs, info = self.env.reset()
            if hasattr(self.expert, 'reset'):
                self.expert.reset()

            ep_success = False

            for step in range(self.max_steps):
                rgb  = self.env.render()
                feat = self._encode(rgb)
                prop = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    action_t  = self.policy(feat.to(DEVICE), prop)
                action_np = action_t.cpu().numpy().squeeze()

                obs, _, done, truncated, info = self.env.step(action_np)

                if info.get("success", False):
                    ep_success = True
                    break
                if done or truncated:
                    break

            status = "✓" if ep_success else "✗"
            print(f"  ep {ep + 1:3d}  {status}")
            if ep_success:
                successes += 1

        rate = successes / num_episodes
        print(f"[{self.task_key}] Success rate: {rate:.1%}  ({successes}/{num_episodes})")
        return rate, tasks_used

    def visualize(self, num_episodes: int = 5, playback_speed: int = 2,
                  tasks=None, save_path: str = None):
        try:
            import cv2
        except ImportError:
            print("OpenCV not found — skipping visualisation")
            return

        assert self.policy is not None, "Call train() + evaluate() first"
        self.policy.eval()

        save_path = save_path or f"vip_{self.task_key}_policy.mp4"
        fps       = max(1, 10 * playback_speed) if playback_speed > 0 else 30
        delay_ms  = max(1, 50 // playback_speed) if playback_speed > 0 else 1
        fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
        out       = cv2.VideoWriter(save_path, fourcc, float(fps), (480, 480))

        if tasks is None:
            rng   = random.Random(42)
            tasks = [rng.choice(self.ml1.train_tasks) for _ in range(num_episodes)]

        vis_successes = 0

        for ep, task in enumerate(tasks):
            self.env.set_task(task)
            obs, info = self.env.reset()

            ep_success   = False
            total_reward = 0.0

            for step in range(self.max_steps):
                rgb  = self.env.render()
                feat = self._encode(rgb)
                prop = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    action_t  = self.policy(feat.to(DEVICE), prop)
                action_np = action_t.cpu().numpy().squeeze()

                obs, reward, done, truncated, info = self.env.step(action_np)
                total_reward += reward

                if info.get("success", False):
                    ep_success = True

                frame = rgb.copy()
                color = (0, 255, 0) if ep_success else (255, 255, 255)
                label = "SUCCESS!" if ep_success else f"Step {step}"
                cv2.putText(frame, f"VIP  {self.task_key}  Ep {ep+1}  {label}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"reward={total_reward:.1f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1)

                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow(f"VIP — {self.task_key}", bgr)
                out.write(bgr)

                if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                    out.release(); cv2.destroyAllWindows(); return

                if ep_success:
                    cv2.waitKey(1500)
                    break
                if done or truncated:
                    break

            s = "✓" if ep_success else "✗"
            print(f"  vis ep {ep+1}: {s}  reward={total_reward:.1f}")
            if ep_success:
                vis_successes += 1
            cv2.waitKey(800)

        out.release()
        cv2.destroyAllWindows()
        print(f"[{self.task_key}] Vis done: {vis_successes}/{num_episodes}  saved → {save_path}")


# =============================================================================
# CSV logging
# =============================================================================

def save_results_csv(results: dict, demo_episodes: int, camera_ids: dict,
                     eval_episodes: int, train_steps: int, batch_size: int,
                     encoder_type: str = "baseline",
                     tag: str = None,
                     csv_path: str = None):
    if csv_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path  = f"vip_results_{encoder_type}_demos{demo_episodes}_{timestamp}.csv"

    rows = []

    for task_key, rate in results.items():
        cam_id = camera_ids.get(task_key, 0)
        if rate is None:
            rows.append({
                "task": task_key, "demo_episodes": demo_episodes,
                "camera_id": cam_id, "train_steps": train_steps,
                "batch_size": batch_size, "eval_episodes": eval_episodes,
                "encoder": f"{encoder_type}_{tag}" if tag else encoder_type, "seed": "N/A",
                "success_rate": "SKIPPED", "mean": "SKIPPED",
                "std": "SKIPPED", "row_type": "per_seed",
            })
        else:
            for seed_idx, seed_rate in enumerate(rate["seeds"]):
                rows.append({
                    "task": task_key, "demo_episodes": demo_episodes,
                    "camera_id": cam_id, "train_steps": train_steps,
                    "batch_size": batch_size, "eval_episodes": eval_episodes,
                    "encoder": f"{encoder_type}_{tag}" if tag else encoder_type, "seed": seed_idx,
                    "success_rate": round(seed_rate, 4),
                    "mean": round(rate["mean"], 4),
                    "std":  round(rate["std"],  4),
                    "row_type": "per_seed",
                })

    valid = {k: v for k, v in results.items() if v is not None}
    if valid:
        overall_mean = np.mean([v["mean"] for v in valid.values()])
        overall_std  = np.mean([v["std"]  for v in valid.values()])
        rows.append({
            "task": "AVERAGE", "demo_episodes": demo_episodes,
            "camera_id": "all", "train_steps": train_steps,
            "batch_size": batch_size, "eval_episodes": eval_episodes,
            "encoder": f"{encoder_type}_{tag}" if tag else encoder_type, "seed": "all",
            "success_rate": "N/A",
            "mean": round(overall_mean, 4),
            "std":  round(overall_std,  4),
            "row_type": "summary",
        })

    fieldnames = ["task", "demo_episodes", "camera_id", "train_steps",
                  "batch_size", "eval_episodes", "encoder", "seed",
                  "success_rate", "mean", "std", "row_type"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[CSV] Results saved → {csv_path}")
    return csv_path


# =============================================================================
# Multi-task orchestrator
# =============================================================================

def run_all_tasks(
    tasks_to_run   = None,
    demo_episodes  : int  = 10,
    train_epochs   : int  = 200,
    batch_size     : int  = 32,
    eval_episodes  : int  = 20,
    visualize      : bool = False,
    playback_speed : int  = 0,
    train_steps    : int  = 20000,
    num_seeds      : int  = 3,
    encoder_type   : str  = "baseline",
    tag            : str  = None,
):
    if tasks_to_run is None:
        tasks_to_run = list(TASK_REGISTRY.keys())

    results = {}

    for task_key in tasks_to_run:
        print("\n" + "=" * 60)
        print(f"  TASK: {task_key}")
        print("=" * 60)

        runner    = MetaWorldTaskRunner(task_key)
        demo_data = runner.collect_demos(num_episodes=demo_episodes)

        if not demo_data:
            print(f"[{task_key}] WARNING: no successful demos — skipping task")
            results[task_key] = None
            continue

        seed_rates = []
        final_loss = None

        for seed in range(num_seeds):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            loss_history     = runner.train(demo_data, num_steps=train_steps,
                                            batch_size=batch_size)
            final_loss       = loss_history[-1] if loss_history else None
            rate, tasks_used = runner.evaluate(num_episodes=eval_episodes)
            seed_rates.append(rate)
            print(f"  [{task_key}] seed {seed}: {rate:.1%}")

        avg_rate = np.mean(seed_rates)
        std_rate = np.std(seed_rates)
        print(f"  [{task_key}] mean={avg_rate:.1%}  std={std_rate:.1%}")
        results[task_key] = {"mean": avg_rate, "std": std_rate,
                             "seeds": seed_rates, "final_loss": final_loss}

        if visualize:
            runner.visualize(
                num_episodes=5,
                playback_speed=playback_speed,
                save_path=(f"vip_demo-size-{demo_episodes}_cam-"
                           f"{TASK_REGISTRY[task_key]['camera_id']}_"
                           f"{encoder_type}_{task_key}_policy.mp4"),
                tasks=tasks_used[:5])

    # Summary table
    print("\n" + "=" * 60)
    print(f"  VIP RESULTS SUMMARY  [{encoder_type}]")
    print(f"  demos={demo_episodes}  steps={train_steps}  eval_eps={eval_episodes}")
    print("=" * 60)
    total_success = 0
    total_tasks   = 0
    for task_key, rate in results.items():
        if rate is None:
            print(f"  {task_key:<20s}  SKIPPED")
        else:
            mean = rate["mean"]
            std  = rate["std"]
            bar  = "█" * int(mean * 20) + "░" * (20 - int(mean * 20))
            print(f"  {task_key:<20s}  {mean:6.1%} ± {std:5.1%}  |{bar}|")
            total_success += mean
            total_tasks   += 1

    if total_tasks:
        print(f"  {'AVERAGE':<20s}  {total_success/total_tasks:6.1%}")
    print("=" * 60)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cam_id    = TASK_REGISTRY[tasks_to_run[-1]]["camera_id"]
    tag_str   = f"_{tag}" if tag else ""
    csv_path  = (f"vip_results_{encoder_type}{tag_str}_demos{demo_episodes}"
                 f"_cam{cam_id}_{timestamp}.csv")

    save_results_csv(
        results       = results,
        demo_episodes = demo_episodes,
        camera_ids    = {k: TASK_REGISTRY[k]["camera_id"] for k in tasks_to_run},
        eval_episodes = eval_episodes,
        train_steps   = train_steps,
        batch_size    = batch_size,
        encoder_type  = encoder_type,
        tag           = tag,
        csv_path      = csv_path,
    )

    return results


# =============================================================================
# Sanity check helper
# =============================================================================

def sanity_check_task(task_key: str, num_steps: int = 60):
    print(f"\n========== SANITY CHECK: {task_key} ==========")
    runner = MetaWorldTaskRunner(task_key)
    task   = runner.ml1.train_tasks[0]
    runner.env.set_task(task)
    obs, info = runner.env.reset()

    print(f"obs shape: {obs.shape}")
    print(f"obs at reset: EEF={obs[0:3].round(3)}  gripper={obs[3]:.3f}  "
          f"obj_A={obs[4:7].round(3)}  obj_B={obs[7:10].round(3)}")

    for step in range(num_steps):
        action = runner.expert.get_action(obs)
        obs, reward, done, truncated, info = runner.env.step(action)
        if step % 10 == 0 or info.get("near_object", 0) > 0.5:
            print(f"  step {step:3d}  success={info.get('success')}  "
                  f"near={info.get('near_object', 0):.2f}  "
                  f"dist={info.get('obj_to_target', 0):.4f}  "
                  f"reward={reward:.3f}")
        if done or truncated or info.get("success", False):
            print(f"  → done at step {step}  success={info.get('success')}")
            break
    print("=" * 48 + "\n")


# =============================================================================
# execute_evaluation wrapper
# =============================================================================

def execute_evaluation(
    tasks_to_run   = None,
    demo_episodes  : int  = 10,
    train_epochs   : int  = 200,
    train_steps    : int  = 20000,
    batch_size     : int  = 32,
    eval_episodes  : int  = 20,
    visualize      : bool = False,
    playback_speed : int  = 0,
    camera_ids     : dict = None,
    encoder_type   : str  = "baseline",
    tag            : str  = None,
):
    TASKS_TO_RUN = tasks_to_run or list(TASK_REGISTRY.keys())

    if camera_ids is None:
        camera_ids = {k: 0 for k in TASKS_TO_RUN}

    for task_key, cam_id in camera_ids.items():
        if task_key in TASK_REGISTRY:
            TASK_REGISTRY[task_key]["camera_id"] = cam_id

    return run_all_tasks(
        tasks_to_run  = TASKS_TO_RUN,
        demo_episodes = demo_episodes,
        train_epochs  = train_epochs,
        batch_size    = batch_size,
        eval_episodes = eval_episodes,
        visualize     = visualize,
        playback_speed= playback_speed,
        train_steps   = train_steps,
        encoder_type  = encoder_type,
        tag           = tag,
    )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="VIP MetaWorld evaluation — baseline or fine-tuned encoder")
    parser.add_argument("--encoder", type=str, default="baseline",
                        choices=["baseline", "droid", "egoexo4d"],
                        help="Which encoder to use (default: baseline)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to fine-tuned .pt file (required if encoder != baseline)")
    parser.add_argument("--single_run", action="store_true",
                        help="Run one camera/demo combo instead of the full 3x3 sweep")
    parser.add_argument("--demos", type=int, default=10,
                        help="Demo episodes for single_run mode (default: 10)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera ID for single_run mode (default: 0)")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional tag appended to CSV filename "
                             "(e.g. 'frozen', 'full', 'baseline') to avoid "
                             "overwriting previous results.")
    args = parser.parse_args()

    # Load encoder — must happen before any MetaWorldTaskRunner is instantiated
    load_encoder(encoder_type=args.encoder, checkpoint=args.checkpoint)

    if args.single_run:
        cameras = {k: args.camera for k in TASK_REGISTRY}
        execute_evaluation(
            demo_episodes = args.demos,
            eval_episodes = 20,
            visualize     = args.visualize,
            playback_speed= 0,
            camera_ids    = cameras,
            encoder_type  = args.encoder,
            tag           = args.tag,
        )
    else:
        # Full sweep matching baseline protocol (3 cameras x 3 demo counts)
        for cam_id in [0, 1, 2]:
            cameras = {k: cam_id for k in TASK_REGISTRY}
            for demo_eps in [5, 10, 25]:
                execute_evaluation(
                    demo_episodes = demo_eps,
                    eval_episodes = 20,
                    visualize     = args.visualize,
                    playback_speed= 0,
                    camera_ids    = cameras,
                    encoder_type  = args.encoder,
                    tag           = args.tag,
                )
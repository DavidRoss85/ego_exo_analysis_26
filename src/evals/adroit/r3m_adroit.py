# =============================================================================
# r3m_adroit.py
#
# Reproduces the Adroit evaluation from the R3M paper (Nair et al. 2022):
#   - pen:      reorient a pen to a specified orientation (AdroitHandPen-v1)
#   - relocate: pick and move a ball to a specified position (AdroitHandRelocate-v1)
#
# Architecture mirrors the paper:
#   - R3M (ResNet-50) frozen visual encoder → 2048-dim embedding
#   - Proprioception concatenated
#   - 2-layer MLP [256, 256] policy head (Behavioral Cloning)
#   - 25 human demos per task (from D4RL/pen/human-v2, D4RL/relocate/human-v2)
#   - Paper evaluates at 5, 10, 25 demos — use --demos flag to select
#
# Datasets (downloaded automatically via Minari):
#   D4RL/pen/human-v2       — 25 human demos, ~2.9 MB
#   D4RL/relocate/human-v2  — 25 human demos, ~5.0 MB
#
# Usage:
#   python3 r3m_adroit.py                         # all tasks, 25 demos
#   python3 r3m_adroit.py --tasks pen             # single task
#   python3 r3m_adroit.py --demos 5               # 5-demo condition
#   python3 r3m_adroit.py --demos 10              # 10-demo condition
#   python3 r3m_adroit.py --demos 25 --viz        # full run + video
#   python3 r3m_adroit.py --show-demos            # inspect demos before training
# =============================================================================

import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import minari
import gymnasium
import gymnasium_robotics

# ---------- R3M ----------
from r3m import load_r3m

# =============================================================================
# Global setup
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

r3m_encoder = load_r3m("resnet50")
r3m_encoder.eval()
r3m_encoder.to(DEVICE)

TRANSFORMS = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
])

VISUAL_DIM = 2048   # ResNet-50 R3M output
HIDDEN_DIM = 256


# =============================================================================
# Task registry
# =============================================================================

TASK_REGISTRY = {
    "pen": {
        "env_id":      "AdroitHandPen-v1",
        "dataset_id":  "D4RL/pen/human-v2",
        "action_dim":  24,    # 24-DOF ShadowHand
        "proprio_dim": 45,    # pen obs: hand joints + pen pose + target
        "max_steps":   200,
        "notes":       "Reorient pen to target orientation",
    },
    "relocate": {
        "env_id":      "AdroitHandRelocate-v1",
        "dataset_id":  "D4RL/relocate/human-v2",
        "action_dim":  30,    # 24 hand + 6 arm DOF
        "proprio_dim": 39,    # ball pos + target pos + hand joints
        "max_steps":   400,
        "notes":       "Pick up ball and move to target position",
    },
}


# =============================================================================
# Policy network  (matches R3M paper: 2-layer MLP [256, 256])
# =============================================================================

class R3MPolicy(nn.Module):
    def __init__(self, visual_dim: int, proprio_dim: int, action_dim: int,
                 hidden: int = HIDDEN_DIM):
        super().__init__()
        in_dim = visual_dim + proprio_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )

    def forward(self, visual_feat, proprio):
        x = torch.cat([visual_feat, proprio], dim=-1)
        return self.net(x)


# =============================================================================
# Task runner
# =============================================================================

class AdroitTaskRunner:

    def __init__(self, task_key: str):
        self.task_key  = task_key
        self.cfg       = TASK_REGISTRY[task_key]
        self.policy    = None
        self._demo_episodes = []   # list of trajectory dicts for show_demos

        gymnasium_robotics  # ensure envs are registered
        self.env = gymnasium.make(
            self.cfg["env_id"],
            max_episode_steps=self.cfg["max_steps"],
            render_mode="rgb_array",
        )

        print(f"[{task_key}] Loaded  env={self.cfg['env_id']}  "
              f"action_dim={self.cfg['action_dim']}  "
              f"proprio_dim={self.cfg['proprio_dim']}")
        print(f"  note: {self.cfg['notes']}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _render(self) -> np.ndarray:
        frame = self.env.render()
        if frame is None:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        return frame

    @torch.no_grad()
    def _encode(self, rgb: np.ndarray) -> torch.Tensor:
        img  = Image.fromarray(rgb)
        inp  = TRANSFORMS(img).unsqueeze(0).to(DEVICE)
        feat = r3m_encoder(inp * 255.0)
        return feat.cpu()

    def _proprio(self, obs) -> np.ndarray:
        """Extract flat proprioception from obs (dict or array)."""
        if isinstance(obs, dict):
            obs = obs.get("observation", list(obs.values())[0])
        obs = np.array(obs, dtype=np.float32).flatten()
        dim = self.cfg["proprio_dim"]
        if len(obs) < dim:
            obs = np.pad(obs, (0, dim - len(obs)))
        return obs[:dim]

    def _success(self, info: dict) -> bool:
        return bool(info.get("success", False) or
                    info.get("is_success", False))

    # ------------------------------------------------------------------
    # Demo collection (Minari replay)
    # ------------------------------------------------------------------

    def collect_demos(self, num_episodes: int = 25,
                      debug: bool = False) -> list:
        """
        Collect demos by replaying Minari human demonstrations.

        Key advantages over Franka Kitchen:
        - Single-task episodes — no multi-task noise or windowing needed
        - All 25 episodes are valid demos, just pick the first num_episodes
        - Same start state per episode (deterministic reset)
        """
        dataset_id = self.cfg["dataset_id"]
        print(f"\n[{self.task_key}] Collecting {num_episodes} demos "
              f"from '{dataset_id}' …")

        # Download if needed
        try:
            dataset = minari.load_dataset(dataset_id)
        except FileNotFoundError:
            print(f"  Downloading '{dataset_id}' …")
            minari.download_dataset(dataset_id)
            dataset = minari.load_dataset(dataset_id)

        total = dataset.total_episodes
        print(f"  Dataset: {total} episodes available")

        if num_episodes > total:
            print(f"  WARNING: requested {num_episodes} but only {total} available. "
                  f"Using all {total}.")
            num_episodes = total

        demo_data = []
        self._demo_episodes = []

        for ep_idx, ep in enumerate(dataset.iterate_episodes()):
            if ep_idx >= num_episodes:
                break

            actions  = np.array(ep.actions)         # (T, action_dim)
            obs_ep   = ep.observations
            # Adroit obs may be flat array or dict
            if isinstance(obs_ep, dict):
                obs_arr = np.array(obs_ep.get("observation",
                          list(obs_ep.values())[0]))
            else:
                obs_arr = np.array(obs_ep)
            T = len(actions)

            if debug:
                print(f"  ep {ep_idx}: {T} steps  "
                      f"obs shape={obs_arr.shape}  "
                      f"action shape={actions.shape}")

            # Seed env to match dataset's initial state.
            #
            # _get_obs() = [qpos[0:30], palm-obj, palm-target, obj-target]
            #
            # Inversion:
            #   hand joints:   qpos[0:30] = obs[0:30]  (direct)
            #   obj position:  obj_pos    = palm - obs[30:33]
            #   target pos:    target_pos = palm - obs[33:36]
            #   target is mocap[0], set via data.mocap_pos[0]
            obs_dict, info = self.env.reset()
            try:
                import mujoco
                m  = self.env.unwrapped.model
                d  = self.env.unwrapped.data
                uw = self.env.unwrapped

                # 1. Set hand joints: obs[0:30] = qpos[0:30]
                d.qpos[:30] = obs_arr[0, :30]
                d.qvel[:]   = 0.0
                mujoco.mj_forward(m, d)

                # 2. Get palm position at this hand config
                palm_pos = d.site_xpos[uw.S_grasp_site_id].copy()

                # 3. Recover and set object position
                obj_pos = palm_pos - obs_arr[0, 30:33]
                d.qpos[30:33] = obj_pos

                # 4. Recover and set target position (mocap body)
                target_pos = palm_pos - obs_arr[0, 33:36]
                d.mocap_pos[0] = target_pos

                d.qvel[:] = 0.0
                mujoco.mj_forward(m, d)
                obs_dict = uw._get_obs()

                if debug:
                    print(f"    Seeded: obj={obj_pos.round(3)}  "
                          f"target={target_pos.round(3)}")
            except Exception as e:
                if debug:
                    print(f"    (seeding failed: {e})")
                obs_dict, info = self.env.reset()

            # Store full sim state for show_demos
            try:
                init_qpos     = self.env.unwrapped.data.qpos.copy()
                init_mocap_pos = self.env.unwrapped.data.mocap_pos.copy()
            except Exception:
                init_qpos      = None
                init_mocap_pos = None

            trajectory = []
            ep_success = False

            for t in range(T):
                rgb    = self._render()
                feat   = self._encode(rgb)
                action = actions[t]

                trajectory.append({
                    "visual_features": feat,
                    "proprioception":  self._proprio(obs_dict),
                    "action":          action.copy(),
                })

                obs_dict, reward, terminated, truncated, info = \
                    self.env.step(action)

                if debug and t < 3:
                    print(f"    t={t}  r={reward:.4f}  success={self._success(info)}")

                if self._success(info):
                    ep_success = True

                if terminated or truncated:
                    break

            status = "✓" if ep_success else "–"
            print(f"  ep {ep_idx+1:2d}/{num_episodes}  "
                  f"{len(trajectory):3d} steps  {status}")

            demo_data.extend(trajectory)
            self._demo_episodes.append({
                "trajectory":   trajectory,
                "actions":      actions,
                "init_qpos":    init_qpos,
                "init_mocap":   init_mocap_pos,
            })

        print(f"[{self.task_key}] Collected {len(demo_data)} transitions "
              f"from {min(ep_idx+1, num_episodes)} episodes.")
        return demo_data

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, demo_data: list, num_epochs: int = 200,
              batch_size: int = 32, lr: float = 1e-3):
        action_dim  = self.cfg["action_dim"]
        proprio_dim = self.cfg["proprio_dim"]
        self.policy = R3MPolicy(VISUAL_DIM, proprio_dim, action_dim).to(DEVICE)
        criterion   = nn.MSELoss()
        optimizer   = torch.optim.Adam(self.policy.parameters(), lr=lr)

        print(f"\n[{self.task_key}] Training BC  "
              f"epochs={num_epochs}  transitions={len(demo_data)} …")

        loss_history = []
        for epoch in range(num_epochs):
            self.policy.train()
            np.random.shuffle(demo_data)
            epoch_loss  = 0.0
            num_batches = 0

            for i in range(0, len(demo_data) - batch_size, batch_size):
                batch = demo_data[i:i + batch_size]
                vis   = torch.cat([b["visual_features"] for b in batch]).to(DEVICE)
                prop  = torch.tensor(
                    np.stack([b["proprioception"] for b in batch])
                ).float().to(DEVICE)
                act   = torch.tensor(
                    np.stack([b["action"] for b in batch])
                ).float().to(DEVICE)

                loss = criterion(self.policy(vis, prop), act)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss  += loss.item()
                num_batches += 1

            avg = epoch_loss / max(num_batches, 1)
            loss_history.append(avg)
            if epoch % 20 == 0 or epoch == num_epochs - 1:
                print(f"  epoch {epoch:4d}/{num_epochs}  loss={avg:.5f}")

        print(f"[{self.task_key}] Training complete.")
        return loss_history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, num_episodes: int = 20) -> float:
        assert self.policy is not None, "Call train() first"
        self.policy.eval()

        successes = 0
        print(f"\n[{self.task_key}] Evaluating over {num_episodes} episodes …")

        for ep in range(num_episodes):
            obs_dict, info = self.env.reset()
            ep_success = False

            for step in range(self.cfg["max_steps"]):
                rgb  = self._render()
                feat = self._encode(rgb)
                prop = torch.from_numpy(
                    self._proprio(obs_dict)
                ).float().unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    action_np = self.policy(
                        feat.to(DEVICE), prop
                    ).cpu().numpy().squeeze()

                obs_dict, _, terminated, truncated, info = \
                    self.env.step(action_np)

                if self._success(info):
                    ep_success = True
                    break
                if terminated or truncated:
                    break

            status = "✓" if ep_success else "✗"
            print(f"  ep {ep+1:3d}  {status}")
            if ep_success:
                successes += 1

        rate = successes / num_episodes
        print(f"[{self.task_key}] Success rate: {rate:.1%}  "
              f"({successes}/{num_episodes})")
        return rate

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize(self, num_episodes: int = 5, playback_speed: int = 1,
                  save_path: str = None):
        try:
            import cv2
        except ImportError:
            print("pip install opencv-python  to enable visualisation")
            return

        assert self.policy is not None
        self.policy.eval()

        save_path = save_path or f"{self.task_key}_policy.mp4"
        fps       = max(1, 10 * playback_speed) if playback_speed > 0 else 30
        fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
        out       = cv2.VideoWriter(save_path, fourcc, float(fps), (480, 480))

        for ep in range(num_episodes):
            obs_dict, info = self.env.reset()
            ep_success     = False
            total_reward   = 0.0

            for step in range(self.cfg["max_steps"]):
                rgb  = self._render()
                feat = self._encode(rgb)
                prop = torch.from_numpy(
                    self._proprio(obs_dict)
                ).float().unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    action_np = self.policy(
                        feat.to(DEVICE), prop
                    ).cpu().numpy().squeeze()

                obs_dict, reward, terminated, truncated, info = \
                    self.env.step(action_np)
                total_reward += reward

                if self._success(info):
                    ep_success = True

                frame = np.array(Image.fromarray(rgb).resize((480, 480)))
                color = (0, 255, 0) if ep_success else (255, 255, 255)
                cv2.putText(frame,
                            f"{self.task_key}  Ep {ep+1}  "
                            f"{'SUCCESS!' if ep_success else f'step {step}'}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"reward={total_reward:.1f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (200, 200, 200), 1)
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                if ep_success or terminated or truncated:
                    break

            s = "✓" if ep_success else "✗"
            print(f"  vis ep {ep+1}: {s}  reward={total_reward:.1f}")

        out.release()
        print(f"[{self.task_key}] Policy video → {save_path}")

    def show_demos(self, demo_data: list, save_path: str = None):
        """Render collected demo episodes to MP4 for inspection."""
        try:
            import cv2
        except ImportError:
            print("pip install opencv-python  to enable demo visualisation")
            return

        if not self._demo_episodes:
            print("No demos collected yet.")
            return

        save_path = save_path or f"{self.task_key}_demos.mp4"
        fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
        out       = cv2.VideoWriter(save_path, fourcc, 20.0, (480, 480))

        print(f"[{self.task_key}] Rendering {len(self._demo_episodes)} "
              f"demo episodes → {save_path} …")

        for ep_idx, ep_meta in enumerate(self._demo_episodes):
            # Each Adroit episode starts from a clean env.reset() —
            # no warm-up needed, the dataset episodes are single-task.
            obs_dict, _ = self.env.reset()
            actions      = ep_meta["actions"]
            init_qpos  = ep_meta.get("init_qpos")
            init_mocap = ep_meta.get("init_mocap")
            if init_qpos is not None:
                try:
                    import mujoco
                    d  = self.env.unwrapped.data
                    m  = self.env.unwrapped.model
                    d.qpos[:]     = init_qpos
                    d.qvel[:]     = 0.0
                    if init_mocap is not None:
                        d.mocap_pos[:] = init_mocap
                    mujoco.mj_forward(m, d)
                    obs_dict = self.env.unwrapped._get_obs()
                except Exception:
                    pass
                try:
                    import mujoco
                    self.env.unwrapped.data.qpos[:] = init_qpos
                    self.env.unwrapped.data.qvel[:] = 0.0
                    mujoco.mj_forward(self.env.unwrapped.model,
                                      self.env.unwrapped.data)
                    obs_dict = self.env.unwrapped._get_obs()
                except Exception:
                    pass

            for t, action in enumerate(actions):
                rgb   = self._render()
                frame = np.array(Image.fromarray(rgb).resize((480, 480)))
                cv2.putText(frame,
                            f"Ep {ep_idx+1}/{len(self._demo_episodes)}  "
                            f"step {t+1}/{len(actions)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 255, 0), 2)
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                obs_dict, _, terminated, truncated, _ = self.env.step(action)
                if terminated or truncated:
                    break

        out.release()
        print(f"[{self.task_key}] Demo video → {save_path}")

    def close(self):
        self.env.close()


# =============================================================================
# Run all tasks
# =============================================================================

def run_all_tasks(
    tasks_to_run   = None,
    demo_episodes  : int  = 25,
    train_epochs   : int  = 200,
    batch_size     : int  = 32,
    eval_episodes  : int  = 20,
    visualize      : bool = False,
    playback_speed : int  = 1,
    show_demos     : bool = False,
    debug_demos    : bool = False,
) -> dict:

    tasks_to_run = tasks_to_run or list(TASK_REGISTRY.keys())
    results      = {}

    for task_key in tasks_to_run:
        if task_key not in TASK_REGISTRY:
            print(f"Unknown task '{task_key}' — skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"  TASK: {task_key}")
        print(f"{'='*60}")

        runner    = AdroitTaskRunner(task_key)
        demo_data = runner.collect_demos(
            num_episodes=demo_episodes,
            debug=debug_demos,
        )

        if not demo_data:
            print(f"[{task_key}] No demo data — skipping.")
            results[task_key] = None
            runner.close()
            continue

        if show_demos:
            runner.show_demos(demo_data)

        runner.train(demo_data,
                     num_epochs=train_epochs,
                     batch_size=batch_size)

        rate = runner.evaluate(num_episodes=eval_episodes)
        results[task_key] = rate

        if visualize:
            runner.visualize(num_episodes=5,
                             playback_speed=playback_speed)

        runner.close()

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ADROIT — R3M RESULTS SUMMARY")
    print(f"  demos={demo_episodes}  epochs={train_epochs}  eval_eps={eval_episodes}")
    print(f"{'='*60}")

    total, n = 0.0, 0
    for task_key, rate in results.items():
        if rate is None:
            print(f"  {task_key:<20s}  SKIPPED")
        else:
            bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
            print(f"  {task_key:<20s}  {rate:6.1%}  |{bar}|")
            total += rate; n += 1

    avg = total / n if n else 0.0
    if n:
        print(f"  {'AVERAGE':<20s}  {avg:6.1%}")
    print(f"{'='*60}")
    print(f"  R3M paper (25 demos): pen ~28%, relocate ~18%, avg ~23%")
    print(f"{'='*60}")

    # ── Matplotlib chart ─────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = list(results.keys())
        values = [r * 100 if r is not None else 0.0 for r in results.values()]
        paper  = {"pen": 28.0, "relocate": 18.0}   # R3M paper @25 demos
        colors = ["#2ecc71" if v >= 15 else "#e74c3c" for v in values]

        fig, ax = plt.subplots(figsize=(7, 5))
        x   = np.arange(len(labels))
        w   = 0.35
        ax.bar(x - w/2, values, w, color=colors,
               label="Our results", edgecolor="white", zorder=3)
        ax.bar(x + w/2,
               [paper.get(k, 0) for k in labels], w,
               color="#3498db", alpha=0.7,
               label="R3M paper (@25 demos)", edgecolor="white", zorder=3)

        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylim(0, 60)
        ax.set_ylabel("Success Rate (%)", fontsize=12)
        ax.set_title(
            f"R3M Adroit — BC Policy  ({demo_episodes} demos, {train_epochs} epochs)",
            fontsize=13, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        for i, v in enumerate(values):
            ax.text(x[i] - w/2, v + 1, f"{v:.0f}%",
                    ha="center", fontsize=11, fontweight="bold")

        plt.tight_layout()
        chart = f"r3m_adroit_results_{demo_episodes}demos.png"
        plt.savefig(chart, dpi=150)
        plt.close()
        print(f"\n  Chart saved → {chart}")
    except Exception as e:
        print(f"  (chart skipped: {e})")

    return results


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    # ── CONFIG BLOCK ──────────────────────────────────────────────────
    TASKS_TO_RUN   = list(TASK_REGISTRY.keys())   # ["pen", "relocate"]
    DEMO_EPISODES  = 25    # paper uses 5, 10, or 25
    TRAIN_EPOCHS   = 200
    BATCH_SIZE     = 32
    EVAL_EPISODES  = 20
    VISUALIZE      = False
    PLAYBACK_SPEED = 1
    SHOW_DEMOS     = False
    DEBUG_DEMOS    = False
    # ─────────────────────────────────────────────────────────────────

    parser = argparse.ArgumentParser(
        description="R3M Adroit imitation learning — pen and relocate tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 r3m_adroit.py                          # both tasks, 25 demos
  python3 r3m_adroit.py --tasks pen              # pen only
  python3 r3m_adroit.py --demos 5                # 5-demo condition
  python3 r3m_adroit.py --demos 10               # 10-demo condition
  python3 r3m_adroit.py --demos 25 --viz         # full run + video
  python3 r3m_adroit.py --show-demos             # inspect demos first
  python3 r3m_adroit.py --debug                  # verbose demo logging
        """,
    )
    parser.add_argument("--tasks",      nargs="+", default=None)
    parser.add_argument("--demos",      type=int,  default=None)
    parser.add_argument("--epochs",     type=int,  default=None)
    parser.add_argument("--batch",      type=int,  default=None)
    parser.add_argument("--eval",       type=int,  default=None)
    parser.add_argument("--viz",        action="store_true", default=None)
    parser.add_argument("--speed",      type=int,  default=None)
    parser.add_argument("--show-demos", action="store_true", default=None,
                        dest="show_demos")
    parser.add_argument("--debug",      action="store_true", default=None)
    args = parser.parse_args()

    if args.tasks      is not None: TASKS_TO_RUN   = args.tasks
    if args.demos      is not None: DEMO_EPISODES  = args.demos
    if args.epochs     is not None: TRAIN_EPOCHS   = args.epochs
    if args.batch      is not None: BATCH_SIZE     = args.batch
    if args.eval       is not None: EVAL_EPISODES  = args.eval
    if args.viz        is not None: VISUALIZE      = args.viz
    if args.speed      is not None: PLAYBACK_SPEED = args.speed
    if args.show_demos is not None: SHOW_DEMOS     = args.show_demos
    if args.debug      is not None: DEBUG_DEMOS    = args.debug

    results = run_all_tasks(
        tasks_to_run   = TASKS_TO_RUN,
        demo_episodes  = DEMO_EPISODES,
        train_epochs   = TRAIN_EPOCHS,
        batch_size     = BATCH_SIZE,
        eval_episodes  = EVAL_EPISODES,
        visualize      = VISUALIZE,
        playback_speed = PLAYBACK_SPEED,
        show_demos     = SHOW_DEMOS,
        debug_demos    = DEBUG_DEMOS,
    )
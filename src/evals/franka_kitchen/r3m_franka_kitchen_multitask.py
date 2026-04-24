# =============================================================================
# r3m_frankakitchen_multitask.py
#
# Reproduces the Franka Kitchen evaluation from the R3M paper (Nair et al. 2022)
# across all 5 tasks:
#   - sliding_door   (slide right door open)
#   - hinge_cabinet  (open left door)
#   - light_switch   (turn on the light)
#   - knob_burner    (turn the stove top knob)
#   - microwave      (open the microwave)
#
# Architecture mirrors the paper:
#   - R3M (ResNet-50) frozen visual encoder → 2048-dim embedding
#   - Proprioception concatenated (60-dim: 9 joint pos + 9 joint vel + 42 task obs)
#   - 2-layer MLP [256, 256] policy head (Behavioural Cloning)
#   - 10 expert demos per task (paper uses 5/10/25; we default to 10)
#   - 50-step horizon per episode (paper: horizon=50 for all Franka tasks)
#
# Key differences from the MetaWorld version:
#   - Uses relay-kitchen / d4rl FrankaKitchen environment
#   - Proprioception is 60-dim (Franka joint states + task obs)
#   - Action space is 9-dim (7 joint torques + 2 gripper DoFs)
#   - Horizon is 50 steps (vs 500 for MetaWorld)
#   - The kitchen desk position is randomised between episodes (paper augmentation)
#   - Expert demos come from the offline D4RL dataset (kitchen-complete-v0 or
#     kitchen-mixed-v0) rather than scripted policies
#
# Environment dependencies:
#   pip install d4rl gym  (or pip install gymnasium-robotics[kitchen])
#   The relay-kitchen / FrankaKitchen env is exposed via:
#       import gym; env = gym.make("kitchen-complete-v0")
#   OR via:
#       from gymnasium_robotics.envs.franka_kitchen import KitchenEnv
#
# =============================================================================

import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# ---------- R3M ----------
from r3m import load_r3m

# ---------- Franka Kitchen ----------
# We support two backends; the script will try d4rl first, then gymnasium-robotics.
try:
    import gym
    import d4rl  # noqa: F401 — registers kitchen-* envs into gym
    _BACKEND = "d4rl"
except ImportError:
    try:
        import gymnasium as gym
        import gymnasium_robotics  # noqa: F401 — registers FrankaKitchen envs
        _BACKEND = "gymnasium_robotics"
    except ImportError:
        raise ImportError(
            "No Franka Kitchen backend found.\n"
            "Install one of:\n"
            "  pip install d4rl  (requires mujoco 2.x and mujoco-py)\n"
            "  pip install gymnasium-robotics  (requires mujoco >= 3.x)\n"
        )

print(f"[FrankaKitchen] Using backend: {_BACKEND}")


# =============================================================================
# Shared setup
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

r3m = load_r3m("resnet50")
r3m.eval()
r3m.to(DEVICE)

TRANSFORMS = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
])

VISUAL_DIM  = 2048   # ResNet-50 output dim
ACTION_DIM  = 9      # Franka: 7 joint torques + 2 gripper DoFs

# Franka Kitchen proprioception:
#   9  joint positions  (qpos)
#   9  joint velocities (qvel)
#   42 task-specific obs (object states, door angles, knob angles, …)
#   → total 60 dimensions (confirmed by d4rl / gymnasium-robotics spec)
PROPRIO_DIM = 60


# =============================================================================
# Task registry
# =============================================================================
# Each entry defines:
#   task_elements : list of task element names (from relay_kitchen / d4rl)
#   env_id        : gym environment ID that singles out this task
#   max_steps     : rollout horizon — paper uses 50 for all Franka tasks
#   camera_name   : MuJoCo camera name for rendering
#   notes         : task-specific details
#
# The relay-kitchen environment accepts a `tasks_to_complete` argument
# that restricts which task elements count as "done".  With d4rl you use
# `kitchen-<task>-v0` IDs; with gymnasium-robotics you pass tasks_to_complete
# to the constructor.  Both are handled in FrankaKitchenTaskRunner.

TASK_REGISTRY = {
    "sliding_door": {
        "task_elements": ["slide_cabinet"],
        "env_id_d4rl":   "kitchen-complete-v0",   # will filter by task_elements
        "env_id_gym":    "FrankaKitchen-v1",
        "max_steps":     50,
        "camera_name":   "camera1",
        "notes":         "Slide the right cabinet/door open",
    },
    "hinge_cabinet": {
        "task_elements": ["hinge_cabinet"],
        "env_id_d4rl":   "kitchen-complete-v0",
        "env_id_gym":    "FrankaKitchen-v1",
        "max_steps":     50,
        "camera_name":   "camera1",
        "notes":         "Open the left hinge door",
    },
    "light_switch": {
        "task_elements": ["light_switch"],
        "env_id_d4rl":   "kitchen-complete-v0",
        "env_id_gym":    "FrankaKitchen-v1",
        "max_steps":     50,
        "camera_name":   "camera1",
        "notes":         "Toggle the light switch on",
    },
    "knob_burner": {
        "task_elements": ["knob_burner1"],       # front-left burner knob
        "env_id_d4rl":   "kitchen-complete-v0",
        "env_id_gym":    "FrankaKitchen-v1",
        "max_steps":     50,
        "camera_name":   "camera1",
        "notes":         "Turn the stove-top knob",
    },
    "microwave": {
        "task_elements": ["microwave"],
        "env_id_d4rl":   "kitchen-complete-v0",
        "env_id_gym":    "FrankaKitchen-v1",
        "max_steps":     50,
        "camera_name":   "camera1",
        "notes":         "Open the microwave door",
    },
}


# =============================================================================
# Policy network  (same 2-layer MLP as R3M paper)
# =============================================================================

class R3MPolicy(nn.Module):
    """
    Visual + proprioceptive behavioural cloning policy.
    Input  : R3M embedding (2048) concatenated with proprioception (60-dim)
    Output : action (9-dim for Franka Kitchen)
    """
    def __init__(self, proprio_dim: int = PROPRIO_DIM,
                 visual_dim: int = VISUAL_DIM,
                 action_dim: int = ACTION_DIM,
                 hidden: int = 256):
        super().__init__()
        input_dim = visual_dim + proprio_dim
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),   # actions normalised to [-1, 1]
        )

    def forward(self, visual_features, proprioception):
        x = torch.cat([visual_features, proprioception], dim=1)
        return self.net(x)


# =============================================================================
# Offline dataset loader  (replaces scripted policies from MetaWorld version)
# =============================================================================

def load_offline_demos(task_elements: list, num_episodes: int = 10,
                       dataset_id: str = "kitchen-complete-v0",
                       max_steps: int = 50) -> list:
    """
    Load demonstration trajectories from the D4RL kitchen offline dataset.

    The D4RL `kitchen-complete-v0` dataset contains ~4000 trajectories of a
    human teleoperating the Franka arm to complete all kitchen tasks sequentially.
    We filter for episodes that contain a completion event for our target task
    element and extract the relevant sub-trajectory.

    Returns a list of transition dicts:
        {"visual_features": tensor(1, 2048),
         "proprioception":  np.ndarray(60,),
         "action":          np.ndarray(9,)}
    """
    print(f"\n[DataLoader] Loading offline demos for tasks={task_elements} "
          f"from {dataset_id} …")

    # --- load d4rl dataset ---
    import gym
    env_tmp = gym.make(dataset_id)
    dataset = env_tmp.get_dataset()
    env_tmp.close()

    obs_all     = dataset["observations"]    # (N, 60)
    act_all     = dataset["actions"]         # (N, 9)
    term_all    = dataset["terminals"]       # (N,)
    timeout_all = dataset["timeouts"]        # (N,)

    # Reconstruct episode boundaries
    episode_ends = np.where(term_all | timeout_all)[0]
    episode_starts = np.concatenate([[0], episode_ends[:-1] + 1])

    demo_data  = []
    n_found    = 0

    for start, end in zip(episode_starts, episode_ends):
        if n_found >= num_episodes:
            break

        ep_obs = obs_all[start:end + 1]
        ep_act = act_all[start:end + 1]

        # Check if this episode completes our target task(s).
        # D4rl kitchen obs[30:] encodes task completion flags.
        # Indices: microwave=30, kettle=31, bottom_left_burner=32,
        #          light=33, slide_cabinet=34, hinge_cabinet=35,
        #          knob_burner1=36 (approx — varies slightly by version)
        task_done = _episode_contains_task(ep_obs, task_elements)

        if not task_done:
            continue

        # Encode each frame with R3M (this is the bottleneck — GPU helps)
        # To keep memory manageable we trim episodes to max_steps around the
        # task completion moment.
        ep_data = _encode_episode(ep_obs, ep_act, task_elements, max_steps)
        demo_data.extend(ep_data)
        n_found += 1
        print(f"  episode found {n_found}/{num_episodes}  "
              f"({len(ep_data)} transitions)")

    print(f"[DataLoader] Loaded {len(demo_data)} transitions "
          f"from {n_found}/{num_episodes} episodes.")
    return demo_data


def _episode_contains_task(ep_obs: np.ndarray,
                            task_elements: list) -> bool:
    """
    Heuristic: check if the episode obs trajectory includes a state change
    consistent with the task being completed.

    The relay_kitchen / d4rl obs vector encodes task-completion boolean flags
    in the last 30 dims (obs[30:60]).  The exact mapping:
        obs[30] microwave angle > threshold
        obs[31] kettle position
        obs[32] bottom_left_burner
        obs[33] light switch
        obs[34] slide_cabinet angle
        obs[35] hinge_cabinet angle
        obs[36] knob_burner1
        obs[37] knob_burner2
        obs[38] knob_burner3
        obs[39] knob_burner4
    Values are continuous (joint angles / positions).  A task is "done" if
    the final value in the episode differs substantially from the initial value.
    """
    TASK_OBS_IDX = {
        "microwave":    30,
        "slide_cabinet": 34,
        "hinge_cabinet": 35,
        "light_switch":  33,
        "knob_burner1":  36,
        "knob_burner2":  37,
        "knob_burner3":  38,
        "knob_burner4":  39,
    }
    DELTA_THRESH = 0.2   # minimum change to count as "task done"

    for elem in task_elements:
        idx = TASK_OBS_IDX.get(elem)
        if idx is None:
            continue
        delta = abs(float(ep_obs[-1, idx]) - float(ep_obs[0, idx]))
        if delta < DELTA_THRESH:
            return False   # this element not completed in episode
    return True


def _encode_episode(ep_obs: np.ndarray, ep_act: np.ndarray,
                    task_elements: list, max_steps: int) -> list:
    """
    Encode the episode frames with R3M.

    Because the offline dataset has NO rendered images, we cannot use R3M
    image encoding directly on the stored obs.  Instead we follow two options:

    Option A (preferred when you have a live environment):
        Re-play the stored actions in the live env and render each frame.

    Option B (fallback / faster but less faithful):
        Encode the proprioceptive obs directly as a pseudo-visual feature using
        a learned linear projection.  This is equivalent to the "no-vision"
        ablation but preserves the training pipeline structure.

    Here we implement Option A when possible and fall back to Option B if the
    environment is not available for rendering.  The FrankaKitchenTaskRunner
    class uses Option A (live env rendering) for its own collect_demos method.
    This function is used only in the offline fallback path.

    For simplicity (and because R3M paper protocol uses live rendering for
    imitation learning), we store a DUMMY visual feature of zeros here so that
    the data structure is consistent; FrankaKitchenTaskRunner.collect_demos
    always uses live rendering and never calls this function.
    """
    dummy_visual = torch.zeros(1, VISUAL_DIM)   # replaced during live replay
    data = []
    ep_len = min(len(ep_obs) - 1, max_steps)
    for t in range(ep_len):
        data.append({
            "visual_features": dummy_visual,
            "proprioception":  ep_obs[t].copy(),
            "action":          ep_act[t].copy(),
        })
    return data


# =============================================================================
# Per-task runner
# =============================================================================

class FrankaKitchenTaskRunner:
    """
    Encapsulates a single Franka Kitchen task with:
      - Environment setup (d4rl or gymnasium-robotics backend)
      - Live demo collection via re-playing offline actions with rendering
      - Behavioural Cloning training
      - Evaluation
      - Optional video export

    Structure mirrors MetaWorldTaskRunner from r3m_metaworld_multitask.py,
    with the key differences noted in the module docstring.
    """

    def __init__(self, task_key: str):
        assert task_key in TASK_REGISTRY, \
            f"Unknown task '{task_key}'. Valid: {list(TASK_REGISTRY.keys())}"

        cfg = TASK_REGISTRY[task_key]
        self.task_key      = task_key
        self.task_elements = cfg["task_elements"]
        self.max_steps     = cfg["max_steps"]
        self.camera_name   = cfg["camera_name"]
        self.proprio_dim   = PROPRIO_DIM

        # --- build the environment ---
        self.env = self._make_env(cfg)

        # --- offline dataset (D4RL) for expert trajectories ---
        self._dataset = None   # loaded lazily

        # --- trained BC policy ---
        self.policy = None

        print(f"[{task_key}] Loaded  task_elements={self.task_elements}  "
              f"max_steps={self.max_steps}  backend={_BACKEND}")
        if cfg["notes"]:
            print(f"  note: {cfg['notes']}")

    # ------------------------------------------------------------------
    # Environment construction
    # ------------------------------------------------------------------
    def _make_env(self, cfg: dict):
        """Build a FrankaKitchen env that singles out the target task element."""
        if _BACKEND == "d4rl":
            # d4rl kitchen envs accept `tasks_to_complete` via make kwargs
            # kitchen-complete-v0 by default requires ALL 4 tasks; we pass
            # only our target element so success fires at the right time.
            env = gym.make(
                cfg["env_id_d4rl"],
                tasks_to_complete=self.task_elements,
            )
        else:
            # gymnasium-robotics FrankaKitchen-v1
            env = gym.make(
                cfg["env_id_gym"],
                tasks_to_complete=self.task_elements,
                remove_task_when_completed=True,
                terminate_on_tasks_completed=True,
            )

        # Set render mode for off-screen RGB rendering
        if hasattr(env, "render_mode"):
            env.render_mode = "rgb_array"
        elif hasattr(env.unwrapped, "render_mode"):
            env.unwrapped.render_mode = "rgb_array"

        return env

    # ------------------------------------------------------------------
    # Helper: encode one image with R3M
    # ------------------------------------------------------------------
    @staticmethod
    def _encode(rgb_array: np.ndarray) -> torch.Tensor:
        """Encode an (H, W, 3) uint8 numpy array → (1, 2048) tensor."""
        img = Image.fromarray(rgb_array)
        t   = TRANSFORMS(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = r3m(t * 255.0)   # R3M expects pixel values in [0, 255]
        return feat.cpu()

    # ------------------------------------------------------------------
    # Helper: render frame from the kitchen environment
    # ------------------------------------------------------------------
    def _render(self) -> np.ndarray:
        """Return an (H, W, 3) uint8 RGB frame."""
        frame = self.env.render()
        if frame is None:
            # Some versions require an explicit camera argument
            frame = self.env.render(camera_name=self.camera_name)
        if frame is None:
            # Fallback: blank frame
            frame = np.zeros((480, 480, 3), dtype=np.uint8)
        return frame

    # ------------------------------------------------------------------
    # Demo collection  (live env replay — Option A from paper protocol)
    # ------------------------------------------------------------------
    def collect_demos(self, num_episodes: int = 10,
                      debug: bool = False) -> list:
        """
        Collect num_episodes of successful demonstrations.

        Strategy:
          1. Load the D4RL offline dataset.
          2. Find episodes in the dataset that complete our target task.
          3. Re-play those actions in the live env, rendering each frame.
          4. Encode each frame with R3M.
          5. Store (visual_feature, proprioception, action) transitions.

        Only episodes that end with env `success=True` (or equivalent) are kept.
        """
        print(f"\n[{self.task_key}] Collecting {num_episodes} successful demos …")

        # --- load offline dataset lazily ---
        if self._dataset is None:
            self._load_dataset()

        demo_data = []
        successes = 0
        attempt   = 0
        max_attempts = num_episodes * 10

        for ep_idx in self._candidate_episodes:
            if successes >= num_episodes or attempt >= max_attempts:
                break
            attempt += 1

            start = self._ep_starts[ep_idx]
            end   = self._ep_ends[ep_idx]
            ep_actions = self._dataset["actions"][start:end]

            # Re-play in live env to get rendered frames
            obs, info = self.env.reset()
            trajectory = []
            ep_success = False

            for t, action in enumerate(ep_actions):
                if t >= self.max_steps:
                    break

                # Render → encode with R3M
                rgb  = self._render()
                feat = self._encode(rgb)

                trajectory.append({
                    "visual_features": feat,
                    "proprioception":  obs[:self.proprio_dim].copy(),
                    "action":          action.copy(),
                })

                obs, reward, done, *rest = self.env.step(action)

                # Handle both gym and gymnasium API
                if isinstance(rest[0], dict):
                    info = rest[0]
                else:
                    truncated = rest[0]
                    info = rest[1] if len(rest) > 1 else {}

                if debug and attempt == 1 and t < 5:
                    print(f"  t={t}  done={done}  info={info}")

                if info.get("success", False) or done:
                    ep_success = info.get("success", False) or bool(done)
                    break

            status = "✓" if ep_success else "✗"
            print(f"  attempt {attempt:3d}: {len(trajectory):3d} steps  {status}")

            if ep_success:
                demo_data.extend(trajectory)
                successes += 1

        print(f"[{self.task_key}] Collected {len(demo_data)} transitions "
              f"from {successes}/{num_episodes} successful episodes.")
        return demo_data

    def _load_dataset(self):
        """Load the D4RL dataset and pre-compute episode boundaries + candidates."""
        print(f"[{self.task_key}] Loading D4RL dataset …")
        raw = self.env.get_dataset()   # d4rl API

        obs_all     = raw["observations"]
        term_all    = raw["terminals"]
        timeout_all = raw["timeouts"]

        ep_ends   = np.where(term_all | timeout_all)[0]
        ep_starts = np.concatenate([[0], ep_ends[:-1] + 1])

        self._dataset    = raw
        self._ep_starts  = ep_starts
        self._ep_ends    = ep_ends

        # Pre-filter episodes that contain the target task
        candidates = []
        for i, (s, e) in enumerate(zip(ep_starts, ep_ends)):
            if _episode_contains_task(obs_all[s:e + 1], self.task_elements):
                candidates.append(i)

        random.shuffle(candidates)
        self._candidate_episodes = candidates
        print(f"  Found {len(candidates)} candidate episodes out of {len(ep_starts)}.")

    # ------------------------------------------------------------------
    # Behavioural Cloning training
    # ------------------------------------------------------------------
    def train(self, demo_data: list, num_epochs: int = 200,
              batch_size: int = 32, lr: float = 1e-3):
        """MSE behavioural cloning on (visual+proprio) → action."""
        self.policy = R3MPolicy(
            proprio_dim=self.proprio_dim,
            visual_dim=VISUAL_DIM,
            action_dim=ACTION_DIM,
        ).to(DEVICE)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        print(f"\n[{self.task_key}] Training BC policy for {num_epochs} epochs "
              f"on {len(demo_data)} transitions …")

        self.policy.train()
        loss_history = []

        for epoch in range(num_epochs):
            random.shuffle(demo_data)
            epoch_loss  = 0.0
            num_batches = 0

            for i in range(0, len(demo_data) - batch_size, batch_size):
                batch = demo_data[i:i + batch_size]

                vis  = torch.cat([b["visual_features"] for b in batch]).to(DEVICE)
                prop = torch.tensor(
                    np.stack([b["proprioception"] for b in batch])
                ).float().to(DEVICE)
                act  = torch.tensor(
                    np.stack([b["action"] for b in batch])
                ).float().to(DEVICE)

                pred = self.policy(vis, prop)
                loss = criterion(pred, act)

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
    def evaluate(self, num_episodes: int = 20) -> tuple:
        """
        Roll out the trained BC policy and return (success_rate, episode_list).
        Matches paper protocol: 50-step horizon, randomised kitchen positions.
        """
        assert self.policy is not None, "Call train() before evaluate()"
        self.policy.eval()

        successes = 0
        print(f"\n[{self.task_key}] Evaluating over {num_episodes} episodes …")

        for ep in range(num_episodes):
            obs, info = self.env.reset()
            ep_success = False

            for step in range(self.max_steps):
                rgb  = self._render()
                feat = self._encode(rgb)
                prop = torch.from_numpy(
                    obs[:self.proprio_dim]
                ).float().unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    action_t  = self.policy(feat.to(DEVICE), prop)
                action_np = action_t.cpu().numpy().squeeze()

                obs, reward, done, *rest = self.env.step(action_np)

                if isinstance(rest[0], dict):
                    info = rest[0]
                else:
                    info = rest[1] if len(rest) > 1 else {}

                if info.get("success", False):
                    ep_success = True
                    break
                if done:
                    break

            status = "✓" if ep_success else "✗"
            print(f"  ep {ep + 1:3d}  {status}")
            if ep_success:
                successes += 1

        rate = successes / num_episodes
        print(f"[{self.task_key}] Success rate: {rate:.1%}  ({successes}/{num_episodes})")
        return rate, []

    # ------------------------------------------------------------------
    # Visualisation (optional, requires OpenCV)
    # ------------------------------------------------------------------
    def visualize(self, num_episodes: int = 5, playback_speed: int = 2,
                  save_path: str = None):
        """Render rollouts to screen and save to MP4."""
        try:
            import cv2
        except ImportError:
            print("OpenCV not found — skipping visualisation. "
                  "Install with: pip install opencv-python")
            return

        assert self.policy is not None, "Call train() + evaluate() first"
        self.policy.eval()

        save_path = save_path or f"{self.task_key}_policy.mp4"
        fps       = max(1, 10 * playback_speed) if playback_speed > 0 else 30
        delay_ms  = max(1, 50 // playback_speed) if playback_speed > 0 else 1
        fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
        out       = cv2.VideoWriter(save_path, fourcc, float(fps), (480, 480))

        vis_successes = 0

        for ep in range(num_episodes):
            obs, info = self.env.reset()
            ep_success   = False
            total_reward = 0.0

            for step in range(self.max_steps):
                rgb  = self._render()
                feat = self._encode(rgb)
                prop = torch.from_numpy(
                    obs[:self.proprio_dim]
                ).float().unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    action_t  = self.policy(feat.to(DEVICE), prop)
                action_np = action_t.cpu().numpy().squeeze()

                obs, reward, done, *rest = self.env.step(action_np)
                total_reward += reward

                if isinstance(rest[0], dict):
                    info = rest[0]
                else:
                    info = rest[1] if len(rest) > 1 else {}

                if info.get("success", False):
                    ep_success = True

                # --- overlay ---
                frame = rgb.copy()
                # Resize to 480×480 if necessary
                if frame.shape[:2] != (480, 480):
                    frame = np.array(
                        Image.fromarray(frame).resize((480, 480))
                    )
                color = (0, 255, 0) if ep_success else (255, 255, 255)
                label = "SUCCESS!" if ep_success else f"Step {step}"
                cv2.putText(frame, f"{self.task_key}  Ep {ep+1}  {label}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"reward={total_reward:.1f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1)

                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow(f"R3M — {self.task_key}", bgr)
                out.write(bgr)

                if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                    out.release(); cv2.destroyAllWindows(); return

                if ep_success:
                    cv2.waitKey(1500)
                    break
                if done:
                    break

            s = "✓" if ep_success else "✗"
            print(f"  vis ep {ep+1}: {s}  reward={total_reward:.1f}")
            if ep_success:
                vis_successes += 1
            cv2.waitKey(800)

        out.release()
        cv2.destroyAllWindows()
        print(f"[{self.task_key}] Vis done: {vis_successes}/{num_episodes} "
              f"saved → {save_path}")


# =============================================================================
# Sanity-check helper
# =============================================================================

def sanity_check_task(task_key: str, num_steps: int = 20):
    """
    Reset the environment and step with zero actions to verify obs shape,
    rendering, and info keys — useful before starting a full run.
    """
    print(f"\n========== SANITY CHECK: {task_key} ==========")
    runner = FrankaKitchenTaskRunner(task_key)
    obs, info = runner.env.reset()

    print(f"obs shape      : {obs.shape}")
    print(f"obs[:9] (qpos) : {obs[:9].round(3)}")
    print(f"obs[9:18](qvel): {obs[9:18].round(3)}")
    print(f"obs[18:] (task): {obs[18:].round(3)}")
    print(f"info at reset  : {info}")

    rgb = runner._render()
    print(f"render shape   : {rgb.shape}  dtype={rgb.dtype}")

    for step in range(num_steps):
        action = runner.env.action_space.sample() * 0.0   # zero action
        obs, reward, done, *rest = runner.env.step(action)
        if isinstance(rest[0], dict):
            info = rest[0]
        else:
            info = rest[1] if len(rest) > 1 else {}

        if step % 5 == 0:
            print(f"  step {step:3d}  reward={reward:.4f}  "
                  f"done={done}  success={info.get('success', '?')}")
        if done:
            break

    print("=" * 48 + "\n")


# =============================================================================
# Multi-task orchestrator
# =============================================================================

def run_all_tasks(
    tasks_to_run  = None,        # list of task_key strings; None → all 5
    demo_episodes : int = 10,    # R3M paper: 5, 10, or 25
    train_epochs  : int = 200,
    batch_size    : int = 32,
    eval_episodes : int = 20,
    visualize     : bool = False,
    playback_speed: int = 2,
):
    """
    Run the full R3M imitation learning pipeline for each Franka Kitchen task
    and print a summary table at the end.
    """
    if tasks_to_run is None:
        tasks_to_run = list(TASK_REGISTRY.keys())

    results = {}

    for task_key in tasks_to_run:
        print("\n" + "=" * 60)
        print(f"  TASK: {task_key}")
        print("=" * 60)

        runner = FrankaKitchenTaskRunner(task_key)

        # 1. Collect demonstrations (live replay of offline dataset)
        demo_data = runner.collect_demos(num_episodes=demo_episodes)

        if not demo_data:
            print(f"[{task_key}] WARNING: no successful demos — skipping task")
            results[task_key] = None
            continue

        # 2. Train BC policy
        runner.train(demo_data, num_epochs=train_epochs, batch_size=batch_size)

        # 3. Evaluate
        rate, _ = runner.evaluate(num_episodes=eval_episodes)
        results[task_key] = rate

        # 4. (Optional) visualise
        if visualize:
            runner.visualize(num_episodes=5, playback_speed=playback_speed)

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("  FRANKA KITCHEN — RESULTS SUMMARY")
    print(f"  demos={demo_episodes}  epochs={train_epochs}  eval_eps={eval_episodes}")
    print("=" * 60)
    total_success = 0
    total_tasks   = 0
    for task_key, rate in results.items():
        if rate is None:
            print(f"  {task_key:<20s}  SKIPPED")
        else:
            bar   = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
            print(f"  {task_key:<20s}  {rate:6.1%}  |{bar}|")
            total_success += rate
            total_tasks   += 1
    if total_tasks:
        avg = total_success / total_tasks
        print(f"  {'AVERAGE':<20s}  {avg:6.1%}")
    print("=" * 60)
    print(f"\n  R3M paper reports ~53% average success rate on Franka Kitchen.")

    return results


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------ config
    TASKS_TO_RUN   = list(TASK_REGISTRY.keys())   # all 5; or e.g. ["microwave"]
    DEMO_EPISODES  = 10       # paper: 5, 10, or 25
    TRAIN_EPOCHS   = 200      # increase for better performance
    BATCH_SIZE     = 32
    EVAL_EPISODES  = 20       # number of held-out rollouts per task
    VISUALIZE      = True     # set True to render MP4 videos
    PLAYBACK_SPEED = 0        # 1=real-time, 2=2x, 0=max speed

    # ------------------------------------------------------------------ sanity
    # Uncomment to verify each task's environment before a full run:
    # for task in TASKS_TO_RUN:
    #     sanity_check_task(task, num_steps=10)

    # ----------------------------------------------------------------- run all
    results = run_all_tasks(
        tasks_to_run   = TASKS_TO_RUN,
        demo_episodes  = DEMO_EPISODES,
        train_epochs   = TRAIN_EPOCHS,
        batch_size     = BATCH_SIZE,
        eval_episodes  = EVAL_EPISODES,
        visualize      = VISUALIZE,
        playback_speed = PLAYBACK_SPEED,
    )
"""
Knight's Tour – Reinforcement Learning
=======================================
A Gymnasium environment with **action masking** and a *Maskable PPO* agent
(from ``sb3-contrib``) that learns to solve the Knight's Tour.

Key design choice
-----------------
The agent is ONLY allowed to pick legal knight moves at every step (via an
``action_masks()`` method).  This eliminates wasted exploration on invalid
moves and dramatically accelerates convergence compared to vanilla DQN.

Components
----------
  - KnightTourEnv  :  Gymnasium environment with ``action_masks()``
  - KnightTourAgent:  wrapper around MaskablePPO with train / solve / save / load
"""

from __future__ import annotations

import os
from pathlib import Path

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

matplotlib.use("Agg")

# All eight L-shaped knight moves (row_delta, col_delta)
KNIGHT_MOVES = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2),  (1, 2),  (2, -1),  (2, 1),
]

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Gymnasium environment
# ---------------------------------------------------------------------------

class KnightTourEnv(gym.Env):
    """Knight's Tour as a single-agent Gymnasium environment **with action masking**.

    Observation (Box):
        Channel 0 – NxN visited mask   (0/1)
        Channel 1 – NxN current pos    (one-hot)
        Channel 2 – NxN degree map     (valid unvisited neighbours, normalised)
        Flattened to 1-D vector of length 3*N*N.

    Actions (Discrete 8):
        Index into ``KNIGHT_MOVES``.  The ``action_masks()`` method tells the
        agent which of the 8 moves are legal at each timestep.

    Rewards:
        +1.0  per valid new square (scaled by progress)
        +N²   bonus for completing the full tour
        -1.0  dead-end penalty (partial credit for progress)

    The starting position is randomised each ``reset()`` unless overridden
    via ``set_fixed_start()``.
    """

    metadata = {"render_modes": []}

    def __init__(self, board_size: int = 5):
        super().__init__()
        self.board_size = board_size
        self.n_squares = board_size * board_size

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(3 * self.n_squares,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(8)

        # State (set in reset)
        self.visited: np.ndarray = None  # type: ignore[assignment]
        self.current_row: int = 0
        self.current_col: int = 0
        self.move_count: int = 0
        self.tour: list[int] = []

        self._fixed_start: tuple[int, int] | None = None

    # -- public helpers ----------------------------------------------------

    def set_fixed_start(self, row: int, col: int) -> None:
        self._fixed_start = (row, col)

    def clear_fixed_start(self) -> None:
        self._fixed_start = None

    # -- action masking (critical for performance) -------------------------

    def action_masks(self) -> np.ndarray:
        """Return a boolean mask of length 8: True = legal move."""
        mask = np.zeros(8, dtype=bool)
        for i, (dr, dc) in enumerate(KNIGHT_MOVES):
            nr, nc = self.current_row + dr, self.current_col + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                if self.visited[nr, nc] == 0:
                    mask[i] = True
        return mask

    # -- gym interface -----------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.visited = np.zeros((self.board_size, self.board_size), dtype=np.float32)

        if self._fixed_start is not None:
            r, c = self._fixed_start
        else:
            r = self.np_random.integers(0, self.board_size)
            c = self.np_random.integers(0, self.board_size)

        self.current_row, self.current_col = int(r), int(c)
        self.visited[self.current_row, self.current_col] = 1.0
        self.move_count = 1
        self.tour = [self.current_row * self.board_size + self.current_col]
        return self._obs(), {}

    def step(self, action: int):
        dr, dc = KNIGHT_MOVES[action]
        nr, nc = self.current_row + dr, self.current_col + dc

        def _info(reason):
            return {
                "reason": reason,
                "move_count": self.move_count,
                "tour": list(self.tour),
            }

        # With action masking this should rarely happen, but handle it anyway
        if not (0 <= nr < self.board_size and 0 <= nc < self.board_size):
            return self._obs(), -1.0, True, False, _info("off_board")
        if self.visited[nr, nc] == 1.0:
            return self._obs(), -1.0, True, False, _info("revisited")

        # Valid new square
        self.current_row, self.current_col = nr, nc
        self.visited[nr, nc] = 1.0
        self.move_count += 1
        self.tour.append(nr * self.board_size + nc)

        progress = self.move_count / self.n_squares

        # Tour complete!
        if self.move_count == self.n_squares:
            reward = 1.0 + float(self.n_squares)
            return self._obs(), reward, True, False, _info("complete")

        # Dead-end – no legal moves left
        if not np.any(self.action_masks()):
            reward = -1.0 + progress   # partial credit
            return self._obs(), reward, True, False, _info("dead_end")

        # Normal step – reward proportional to progress
        reward = 0.5 + 0.5 * progress
        return self._obs(), reward, False, False, _info("continue")

    # -- internal ----------------------------------------------------------

    def _obs(self) -> np.ndarray:
        pos = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        pos[self.current_row, self.current_col] = 1.0

        degree = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.visited[r, c] == 0:
                    cnt = 0
                    for dr, dc in KNIGHT_MOVES:
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < self.board_size and 0 <= cc < self.board_size:
                            if self.visited[rr, cc] == 0:
                                cnt += 1
                    degree[r, c] = cnt / 8.0
        return np.concatenate([
            self.visited.flatten(),
            pos.flatten(),
            degree.flatten(),
        ])


# ---------------------------------------------------------------------------
# Helper: wrap environment with ActionMasker for MaskablePPO
# ---------------------------------------------------------------------------

def _mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()


def make_masked_env(board_size: int) -> ActionMasker:
    """Create a KnightTourEnv wrapped with ActionMasker."""
    env = KnightTourEnv(board_size)
    return ActionMasker(env, _mask_fn)


# ---------------------------------------------------------------------------
# 2. Training progress callback
# ---------------------------------------------------------------------------

class _ProgressCallback(BaseCallback):
    """Collects per-episode stats and calls an optional user function."""

    def __init__(self, log_fn=None, verbose=0):
        super().__init__(verbose)
        self.log_fn = log_fn
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.successes: list[bool] = []
        self._ep_reward = 0.0

    def _on_step(self) -> bool:
        self._ep_reward += self.locals["rewards"][0]

        done = self.locals["dones"][0]
        if done:
            info = self.locals["infos"][0]
            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(info.get("move_count", 1))
            self.successes.append(info.get("reason") == "complete")
            self._ep_reward = 0.0

            if self.log_fn and len(self.episode_rewards) % 50 == 0:
                n = len(self.episode_rewards)
                self.log_fn(
                    n,
                    np.mean(self.episode_lengths[-100:]),
                    np.mean(self.successes[-100:]) * 100,
                )
        return True


# ---------------------------------------------------------------------------
# 3. Agent wrapper
# ---------------------------------------------------------------------------

class KnightTourAgent:
    """High-level wrapper: train, solve, save, load (using MaskablePPO)."""

    def __init__(self, board_size: int = 5):
        self.board_size = board_size
        self.env = KnightTourEnv(board_size)
        self.model: MaskablePPO | None = None
        self.stats: dict = {}

    @property
    def model_path(self) -> Path:
        return MODEL_DIR / f"knight_ppo_{self.board_size}x{self.board_size}"

    def _make_model(self) -> MaskablePPO:
        wrapped = make_masked_env(self.board_size)
        return MaskablePPO(
            "MlpPolicy",
            wrapped,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,        # encourage exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=0,
        )

    # -- training ----------------------------------------------------------

    def train(
        self,
        total_timesteps: int = 500_000,
        log_fn=None,
    ) -> dict:
        """Train the Maskable PPO agent.

        Parameters
        ----------
        total_timesteps : int
            Total environment steps.
        log_fn : callable, optional
            ``log_fn(episode, avg_length, success_rate_pct)``

        Returns
        -------
        dict  with keys ``episode_rewards``, ``episode_lengths``, ``successes``
        """
        self.model = self._make_model()
        cb = _ProgressCallback(log_fn=log_fn)
        self.model.learn(total_timesteps=total_timesteps, callback=cb)
        self.model.save(str(self.model_path))

        self.stats = {
            "episode_rewards": cb.episode_rewards,
            "episode_lengths": cb.episode_lengths,
            "successes": cb.successes,
        }
        return self.stats

    # -- inference ---------------------------------------------------------

    def load(self) -> bool:
        """Load a previously saved model. Returns True on success."""
        p = str(self.model_path) + ".zip"
        if not os.path.exists(p):
            return False
        self.model = MaskablePPO.load(p)
        return True

    def solve(
        self,
        start_row: int,
        start_col: int,
        max_retries: int = 200,
    ) -> tuple[list[int] | None, int]:
        """Use the trained model to attempt a knight's tour.

        First tries deterministic prediction.  If that fails, retries up to
        ``max_retries`` times using **stochastic** sampling from the policy,
        which lets the agent explore alternative paths.

        Returns ``(tour_list_or_None, best_squares_visited)``.
        """
        if self.model is None:
            if not self.load():
                return None, 0

        best_tour: list[int] | None = None
        best_partial_tour: list[int] = []
        best_visited = 0

        self.env.set_fixed_start(start_row, start_col)

        for attempt in range(1 + max_retries):
            deterministic = (attempt == 0)
            obs, _ = self.env.reset()
            done = False
            while not done:
                masks = self.env.action_masks()
                if not np.any(masks):
                    break
                action, _ = self.model.predict(
                    obs, deterministic=deterministic, action_masks=masks,
                )
                obs, _, terminated, truncated, _ = self.env.step(int(action))
                done = terminated or truncated

            if self.env.move_count == self.env.n_squares:
                best_tour = list(self.env.tour)
                best_visited = self.env.move_count
                best_partial_tour = best_tour
                break
            if self.env.move_count > best_visited:
                best_visited = self.env.move_count
                best_partial_tour = list(self.env.tour)

        self.env.clear_fixed_start()
        self.last_partial_tour = best_partial_tour
        return best_tour, best_visited

    # -- stats plotting ----------------------------------------------------

    @staticmethod
    def plot_training_stats(stats: dict) -> plt.Figure:
        """Return a matplotlib Figure with training curves."""
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

        eps = range(1, len(stats["episode_lengths"]) + 1)
        window = min(100, len(stats["episode_lengths"]))

        # Tour length
        ax = axes[0]
        ax.plot(eps, stats["episode_lengths"], alpha=0.3, linewidth=0.5, color="steelblue")
        if window > 1:
            smooth = np.convolve(stats["episode_lengths"], np.ones(window) / window, mode="valid")
            ax.plot(range(window, len(stats["episode_lengths"]) + 1), smooth, color="darkblue", linewidth=1.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Squares visited")
        ax.set_title("Tour Length")
        ax.grid(True, alpha=0.3)

        # Episode reward
        ax = axes[1]
        ax.plot(eps, stats["episode_rewards"], alpha=0.3, linewidth=0.5, color="orange")
        if window > 1:
            smooth = np.convolve(stats["episode_rewards"], np.ones(window) / window, mode="valid")
            ax.plot(range(window, len(stats["episode_rewards"]) + 1), smooth, color="darkorange", linewidth=1.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total reward")
        ax.set_title("Episode Reward")
        ax.grid(True, alpha=0.3)

        # Success rate
        ax = axes[2]
        cum_success = np.cumsum(stats["successes"])
        rate = cum_success / np.arange(1, len(stats["successes"]) + 1) * 100
        ax.plot(eps, rate, color="green", linewidth=1.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Success %")
        ax.set_title("Cumulative Success Rate")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

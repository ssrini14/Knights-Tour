# ♞ Knight's Tour Solver

An interactive Knight's Tour solver and visualiser featuring both a classical **Warnsdorff's heuristic** and a **Reinforcement Learning (Masked PPO)** agent. Built with Python, Gradio, and Stable-Baselines3.

![Knights Tour Layout](Knights%20Tour_Layout.png)

## What is the Knight's Tour?

The Knight's Tour is a classic chess puzzle: move a knight across an N×N chessboard so that it visits **every square exactly once**. A tour that ends on a square from which the knight can return to its starting position is called a **closed tour** (Hamiltonian cycle).

## Features

- **Board sizes 3×3 through 8×8** — select any starting square
- **Warnsdorff's Heuristic** — greedy solver with lookahead tie-breaking (converted from a MATLAB connectivity project)
- **Reinforcement Learning (Masked PPO)** — a Gymnasium environment with action masking trained via `sb3-contrib`'s `MaskablePPO`
- **Step-through animation** — drag a slider to replay the tour move-by-move
- **Knight Tour Matrix** — view the visit-order grid for the solved tour
- **Pre-trained models** included for all board sizes (3–8)

## Project Structure

```
├── app.py               # Gradio web application
├── knight_tour.py       # Core algorithms (adjacency matrix, Warnsdorff's heuristic, visualisation)
├── rl_knight_tour.py    # Gymnasium environment, Masked PPO agent (train / solve / save / load)
├── train_all.py         # Script to train RL models for board sizes 3–8
├── requirements.txt     # Python dependencies
├── uv.toml              # uv package manager config
└── models/              # Pre-trained MaskablePPO models
    ├── knight_ppo_3x3.zip
    ├── knight_ppo_4x4.zip
    ├── knight_ppo_5x5.zip
    ├── knight_ppo_6x6.zip
    ├── knight_ppo_7x7.zip
    └── knight_ppo_8x8.zip
```

## Getting Started

### Prerequisites

- Python 3.10+

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd knight_tour

# Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### Launch the App

```bash
python app.py
```

The Gradio interface will open in your browser. Select a board size, pick a starting square, choose an algorithm, and click **Solve**.

## Algorithms

### Warnsdorff's Heuristic

A greedy approach that always moves to the neighbour with the **fewest onward moves**. Ties are broken with a multi-step lookahead. It finds a valid tour almost instantly for most starting positions on boards 5×5 and larger.

### Reinforcement Learning (Masked PPO)

| Component | Details |
|---|---|
| **Environment** | Custom `KnightTourEnv` (Gymnasium) with 3-channel observation: visited mask, current position one-hot, and normalised degree map |
| **Action Space** | Discrete(8) — the eight possible L-shaped moves |
| **Action Masking** | Only legal moves are presented to the policy via `action_masks()`, eliminating wasted exploration |
| **Algorithm** | MaskablePPO from `sb3-contrib` (PPO with invalid-action masking) |
| **Reward** | +0.5–1.0 per step (scaled by progress), +N² bonus for tour completion, −1.0 for dead-ends |
| **Network** | MLP with two hidden layers of 256 units |

### Training

Pre-trained models are included in `models/`. To retrain from scratch:

```bash
python train_all.py
```

Training timesteps per board size:

| Board | Timesteps |
|-------|-----------|
| 3×3 | 200,000 |
| 4×4 | 200,000 |
| 5×5 | 500,000 |
| 6×6 | 1,000,000 |
| 7×7 | 1,500,000 |
| 8×8 | 2,000,000 |

## Dependencies

- **numpy** — numerical computation
- **matplotlib** — chessboard rendering
- **gradio** — interactive web UI
- **gymnasium** — RL environment framework
- **stable-baselines3** — RL algorithm implementations
- **sb3-contrib** — MaskablePPO (action-masked PPO)

## References

1. Philip, A. (2013). A generalized pseudo-knight's tour algorithm for encryption of an image. *IEEE Potentials*, 32(6), 10–16.
2. Philip, A. (2014). A novel pseudo-knight's tour algorithm for encryption of an image. *IEEE Potentials*, 33(1), 10–16.
3. Squirrel, D., & Çull, P. (1996). A Warnsdorff-Rule Algorithm for Knight's Tours.

## License

This project is provided as-is for educational purposes.

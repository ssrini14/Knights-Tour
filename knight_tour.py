"""
This python file is used for core Knight's Tour algorithms. Includes adjacency
matrix construction, Warnsdorff's heuristic with lookahead tie-breaking,
tour validation utilities, and chessboard visualisation with matplotlib.
Converted from MATLAB (Grad School Project) to Python.
####################################################################
## Personal Project - Srinivas Sridharan
####################################################################

Author: Srinivas Sridharan
Copyright: 2026
Project: knight_tour

License: Personal Project
Version: 0.0.1
Maintainer: Srinivas Sridharan

Status: Development

Other dependencies:
    numpy, matplotlib
"""

from __future__ import annotations

import random
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


matplotlib.use("Agg")  # non-interactive backend for Gradio


# ---------------------------------------------------------------------------
# 1. Adjacency matrix
# ---------------------------------------------------------------------------

def create_adj_matrix(board_size: int) -> np.ndarray:
    """Create an adjacency matrix for all valid knight moves on a board_size x board_size board.

    Nodes are numbered 0 .. board_size²-1 in row-major order.
    """
    n = board_size * board_size
    board = np.zeros((n, n), dtype=np.int8)

    for i in range(n):
        r1, c1 = divmod(i, board_size)
        for j in range(i + 1, n):
            r2, c2 = divmod(j, board_size)
            dr, dc = abs(r1 - r2), abs(c1 - c2)
            if (dr == 1 and dc == 2) or (dr == 2 and dc == 1):
                board[i, j] = 1
                board[j, i] = 1

    return board


# ---------------------------------------------------------------------------
# 2. Validation helpers
# ---------------------------------------------------------------------------

def check_knight_validity(tour: list[int], board_size: int) -> bool:
    """Return True if every consecutive pair in *tour* is a valid knight move."""
    for k in range(len(tour) - 1):
        r1, c1 = divmod(tour[k], board_size)
        r2, c2 = divmod(tour[k + 1], board_size)
        dr, dc = abs(r1 - r2), abs(c1 - c2)
        if not ((dr == 1 and dc == 2) or (dr == 2 and dc == 1)):
            return False
    return True

def is_hamiltonian_cycle(tour: list[int], board_size: int) -> bool:
    """Check whether the last node can reach the first via a knight move (closed tour)."""
    r1, c1 = divmod(tour[0], board_size)
    r2, c2 = divmod(tour[-1], board_size)
    dr, dc = abs(r1 - r2), abs(c1 - c2)
    return (dr == 1 and dc == 2) or (dr == 2 and dc == 1)


def validate_tours(tours: list[list[int]], board_size: int) -> dict:
    """Validate a list of tours and return summary info."""
    n = board_size * board_size
    valid_tours = []
    hamiltonian_cycles = []

    for tour in tours:
        if len(tour) != n:
            continue
        if len(set(tour)) != n:
            continue
        if not check_knight_validity(tour, board_size):
            continue
        valid_tours.append(tour)
        if is_hamiltonian_cycle(tour, board_size):
            hamiltonian_cycles.append(tour)

    return {
        "total": len(tours),
        "valid": len(valid_tours),
        "hamiltonian_cycles": len(hamiltonian_cycles),
        "tours": valid_tours,
        "cycles": hamiltonian_cycles,
    }


# ---------------------------------------------------------------------------
# 3. Warnsdorff's Heuristic  (single-tour, fast)
# ---------------------------------------------------------------------------

def _get_edge_connectivity(adj: np.ndarray, node: int) -> tuple[list[int], list[int]]:
    """Return (neighbours, degree-of-each-neighbour) for *node*."""
    neighbours = list(np.where(adj[node] == 1)[0])
    degrees = [int(adj[n].sum()) for n in neighbours]
    return neighbours, degrees


def _check_paths(adj: np.ndarray, candidates: list[int], lookahead: int = 3) -> int:
    """Choose the next move among *candidates* using multi-step lookahead (tie-breaking).

    This mirrors the MATLAB ``checkPaths.m`` logic.
    """
    path_counts = []
    for cand in candidates:
        total = 0
        current_nodes = [cand]
        for _ in range(lookahead):
            next_nodes = []
            for cn in current_nodes:
                neighbours, degrees = _get_edge_connectivity(adj, cn)
                total += sum(degrees)
                next_nodes.extend(neighbours)
            current_nodes = next_nodes
        path_counts.append(total)

    max_val = max(path_counts)
    best_indices = [i for i, v in enumerate(path_counts) if v == max_val]
    chosen = random.choice(best_indices)
    return candidates[chosen]


def warnsdorff_tour(board_size: int, start_row: int, start_col: int) -> Optional[list[int]]:
    """Find a knight's tour using Warnsdorff's rule with lookahead tie-breaking.

    Parameters
    ----------
    board_size : int
        Board dimension (e.g. 8 for 8×8).
    start_row, start_col : int
        0-based row and column of the starting square.

    Returns
    -------
    list[int] | None
        Node indices of the tour in visit order, or None if no tour found.
    """
    adj = create_adj_matrix(board_size)
    n = board_size * board_size
    current = start_row * board_size + start_col
    tour = [current]

    while adj.sum() != 0:
        neighbours = list(np.where(adj[current] == 1)[0])
        if not neighbours:
            if len(tour) == n:
                break
            return None  # dead end

        # Warnsdorff: pick neighbour with fewest onward moves
        edge_counts = np.array([adj[nb].sum() for nb in neighbours])
        min_count = edge_counts.min()
        best = [neighbours[i] for i, c in enumerate(edge_counts) if c == min_count]

        if len(best) == 1:
            new_node = best[0]
        else:
            new_node = _check_paths(adj, best)

        # Remove current node from the board
        adj[current, :] = 0
        adj[:, current] = 0

        current = new_node
        tour.append(current)

    return tour if len(tour) == n else None


# ---------------------------------------------------------------------------
# 4.  High-level solver  (used by GUI)
# ---------------------------------------------------------------------------

def solve_knight_tour(
    board_size: int,
    start_row: int,
    start_col: int,
) -> dict:
    """Run Warnsdorff's heuristic and return results.

    Returns
    -------
    dict with keys:
        tour : list[int] | None       – the tour to display
        is_cycle : bool                – whether the tour is a Hamiltonian cycle
        message : str                  – human-readable summary
    """
    tour = warnsdorff_tour(board_size, start_row, start_col)
    if tour is None:
        # On odd-sized boards, minority-colour squares can't have a tour
        if board_size % 2 == 1 and (start_row + start_col) % 2 == 1:
            reason = (
                "No valid Knight's Tour exists from this starting position.\n"
                f"On a {board_size}×{board_size} board the path needs more "
                "minority-colour squares than exist (parity constraint)."
            )
        else:
            reason = "No valid Knight's Tour found from this starting position."
        return {
            "tour": None,
            "is_cycle": False,
            "message": reason,
        }
    cycle = is_hamiltonian_cycle(tour, board_size)
    return {
        "tour": tour,
        "is_cycle": cycle,
        "message": (
            f"Knight's Tour found ({'Hamiltonian Cycle ✓' if cycle else 'Open tour'}).\n"
            f"Visited all {board_size*board_size} squares."
        ),
    }


# ---------------------------------------------------------------------------
# 6. Chessboard visualisation
# ---------------------------------------------------------------------------

_LIGHT = "#F0D9B5"
_DARK = "#B58863"
_GREEN = "#7FC97F"
_YELLOW = "#FFD700"
_RED = "#FF4500"
_KNIGHT = "♞"


def generate_chessboard_image(
    tour: list[int],
    board_size: int,
    show_animation_frame: int | None = None,
) -> plt.Figure:
    """Render the knight tour on a chessboard.

    If *show_animation_frame* is None, draw the full completed tour.
    Otherwise draw only the first *show_animation_frame* steps (for animation).
    """
    steps = len(tour) if show_animation_frame is None else show_animation_frame
    steps = min(steps, len(tour))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_xlim(0, board_size)
    ax.set_ylim(0, board_size)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Draw squares (original chessboard pattern preserved)
    for r in range(board_size):
        for c in range(board_size):
            colour = _LIGHT if (r + c) % 2 == 0 else _DARK
            ax.add_patch(patches.Rectangle((c, r), 1, 1, facecolor=colour, edgecolor="none"))

    # Highlight only start (green border) and end (red border) squares
    if steps > 0:
        sr, sc = divmod(tour[0], board_size)
        ax.add_patch(patches.Rectangle(
            (sc, sr), 1, 1, facecolor="none",
            edgecolor=_GREEN, linewidth=3,
        ))
    if steps == len(tour) and steps > 1:
        er, ec = divmod(tour[-1], board_size)
        ax.add_patch(patches.Rectangle(
            (ec, er), 1, 1, facecolor="none",
            edgecolor=_RED, linewidth=3,
        ))

    # Draw arrows for path
    for idx in range(steps - 1):
        r1, c1 = divmod(tour[idx], board_size)
        r2, c2 = divmod(tour[idx + 1], board_size)
        ax.annotate(
            "",
            xy=(c2 + 0.5, r2 + 0.5),
            xytext=(c1 + 0.5, r1 + 0.5),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5),
        )

    # Number each visited square
    for idx in range(steps):
        node = tour[idx]
        r, c = divmod(node, board_size)
        ax.text(
            c + 0.5, r + 0.5, str(idx + 1),
            ha="center", va="center", fontsize=max(8, 20 - board_size),
            fontweight="bold", color="black",
        )

    # Place knight symbol on the last visited square
    if steps > 0:
        last = tour[steps - 1]
        lr, lc = divmod(last, board_size)
        ax.text(
            lc + 0.5, lr + 0.15, _KNIGHT,
            ha="center", va="center", fontsize=max(12, 28 - board_size),
            color="#222",
        )

    # Axis labels
    row_labels = [chr(ord("A") + i) for i in range(board_size)]
    col_labels = [str(i + 1) for i in range(board_size)]
    ax.set_xticks([i + 0.5 for i in range(board_size)])
    ax.set_xticklabels(col_labels, fontsize=12)
    ax.set_yticks([i + 0.5 for i in range(board_size)])
    ax.set_yticklabels(row_labels, fontsize=12)
    ax.tick_params(length=0)

    # Hamiltonian cycle indicator line (last → first)
    if steps == len(tour) and is_hamiltonian_cycle(tour, board_size):
        r1, c1 = divmod(tour[-1], board_size)
        r2, c2 = divmod(tour[0], board_size)
        ax.annotate(
            "",
            xy=(c2 + 0.5, r2 + 0.5),
            xytext=(c1 + 0.5, r1 + 0.5),
            arrowprops=dict(arrowstyle="->", color="blue", lw=2.5, linestyle="dashed"),
        )

    ax.set_title(
        f"Knight's Tour  {board_size}×{board_size}  "
        f"(step {steps}/{len(tour)})",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Knight Tour Matrix
# ---------------------------------------------------------------------------

def tour_to_matrix(tour: list[int], board_size: int) -> np.ndarray:
    """Convert a tour (list of node indices) to an NxN matrix where
    ``mat[r][c]`` is the visit order (1-based)."""
    mat = np.zeros((board_size, board_size), dtype=int)
    for idx, node in enumerate(tour):
        r, c = divmod(node, board_size)
        mat[r, c] = idx + 1
    return mat


# ---------------------------------------------------------------------------
# 8. Empty board with knight placement
# ---------------------------------------------------------------------------

def generate_empty_board(
    board_size: int,
    knight_row: int | None = None,
    knight_col: int | None = None,
) -> plt.Figure:
    """Render an empty chessboard, optionally placing a knight on (knight_row, knight_col).

    Parameters are 0-based.
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_xlim(0, board_size)
    ax.set_ylim(0, board_size)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Draw squares
    for r in range(board_size):
        for c in range(board_size):
            colour = _LIGHT if (r + c) % 2 == 0 else _DARK
            ax.add_patch(patches.Rectangle((c, r), 1, 1, facecolor=colour, edgecolor="none"))

    # Place knight if position is valid
    if (
        knight_row is not None
        and knight_col is not None
        and 0 <= knight_row < board_size
        and 0 <= knight_col < board_size
    ):
        # Highlight the square
        ax.add_patch(patches.Rectangle(
            (knight_col, knight_row), 1, 1,
            facecolor="none", edgecolor=_GREEN, linewidth=3,
        ))
        # Knight symbol
        ax.text(
            knight_col + 0.5, knight_row + 0.5, _KNIGHT,
            ha="center", va="center",
            fontsize=max(16, 36 - board_size), color="#222",
        )

    # Axis labels
    row_labels = [chr(ord("A") + i) for i in range(board_size)]
    col_labels = [str(i + 1) for i in range(board_size)]
    ax.set_xticks([i + 0.5 for i in range(board_size)])
    ax.set_xticklabels(col_labels, fontsize=12)
    ax.set_yticks([i + 0.5 for i in range(board_size)])
    ax.set_yticklabels(row_labels, fontsize=12)
    ax.tick_params(length=0)

    ax.set_title(
        f"Chessboard  {board_size}×{board_size}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    return fig

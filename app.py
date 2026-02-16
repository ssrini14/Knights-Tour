"""
This python file is used for Gradio web application for the Knight's Tour
solver. Provides an interactive GUI to select board size, starting square,
algorithm (Warnsdorff / RL), visualise the tour, and view the visit-order matrix.
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
    gradio, numpy, knight_tour, rl_knight_tour
"""

from __future__ import annotations

import gradio as gr
import numpy as np

from knight_tour import (
    generate_chessboard_image,
    generate_empty_board,
    solve_knight_tour,
    tour_to_matrix,
)
from rl_knight_tour import KnightTourAgent


# ── Shared state ──────────────────────────────────────────────────────────
_last_result: dict = {}
_rl_agents: dict[int, KnightTourAgent] = {}   # board_size → agent


def _get_agent(board_size: int) -> KnightTourAgent:
    if board_size not in _rl_agents:
        _rl_agents[board_size] = KnightTourAgent(board_size)
    return _rl_agents[board_size]


# ── Helpers ───────────────────────────────────────────────────────────────

ROW_LETTERS = "ABCDEFGH"


def _is_parity_impossible(board_size: int, row: int, col: int) -> bool:
    """On odd-sized boards a knight's tour from a minority-colour square
    is mathematically impossible (the path needs more squares of that
    colour than exist on the board)."""
    if board_size % 2 == 0:
        return False
    # Minority colour = black = (r+c) odd, which has (n²-1)/2 squares
    return (row + col) % 2 == 1


def _row_choices(board_size: int) -> list[str]:
    return [ROW_LETTERS[i] for i in range(board_size)]


def _col_choices(board_size: int) -> list[str]:
    return [str(i + 1) for i in range(board_size)]


# ── Callbacks ─────────────────────────────────────────────────────────────

def on_board_size_change(board_size: int):
    """Update row/column dropdowns and render empty board when board size changes."""
    rows = _row_choices(board_size)
    cols = _col_choices(board_size)
    fig = generate_empty_board(int(board_size), knight_row=0, knight_col=0)
    return (
        gr.update(choices=rows, value=rows[0]),
        gr.update(choices=cols, value=cols[0]),
        fig,
    )


def on_position_change(board_size: int, row_letter: str, col_str: str):
    """Re-render the board with the knight at the selected position."""
    if not row_letter or not col_str:
        return generate_empty_board(int(board_size))
    row = ROW_LETTERS.index(row_letter)
    col = int(col_str) - 1
    return generate_empty_board(int(board_size), knight_row=row, knight_col=col)


def run_solver(
    board_size: int,
    row_letter: str,
    col_str: str,
    algorithm: str,
):
    """Execute the chosen solver and return visualisation + info."""
    global _last_result

    board_size = int(board_size)
    row = ROW_LETTERS.index(row_letter)
    col = int(col_str) - 1

    if algorithm == "Reinforcement Learning (Masked PPO)":
        agent = _get_agent(board_size)
        if agent.model is None and not agent.load():
            empty_fig = generate_empty_board(board_size, knight_row=row, knight_col=col)
            _last_result = {}
            return (
                empty_fig,
                "No trained RL model found for this board size.\n"
                "Go to the 'RL Training' section below to train one first.",
                "",
                gr.update(maximum=1, value=1, visible=False),
            )

        tour, visited = agent.solve(row, col)
        n = board_size * board_size
        if tour is not None:
            _last_result = {"tour": tour}
            fig = generate_chessboard_image(tour, board_size)
            mat = tour_to_matrix(tour, board_size)
            matrix_text = _format_matrix(mat, board_size)
            return (
                fig,
                f"RL agent completed the tour!\nVisited all {n} squares.",
                matrix_text,
                gr.update(minimum=1, maximum=n, value=n, visible=True),
            )
        else:
            # RL failed – check with Warnsdorff whether a tour even exists
            warnsdorff_result = solve_knight_tour(board_size, row, col)
            warnsdorff_possible = warnsdorff_result["tour"] is not None

            # Show the RL agent's partial tour on the board
            partial_tour = agent.last_partial_tour if hasattr(agent, 'last_partial_tour') and agent.last_partial_tour else [row * board_size + col]
            _last_result = {"tour": partial_tour}
            fig = generate_chessboard_image(partial_tour, board_size)

            if not warnsdorff_possible:
                msg = (
                    f"⚠ A Knight's Tour is NOT possible from this starting square.\n"
                    f"Verified with Warnsdorff's heuristic.\n"
                    f"RL agent visited {visited}/{n} squares (best achievable)."
                )
            else:
                msg = (
                    f"RL agent got stuck after {visited}/{n} squares.\n"
                    f"A tour IS possible (Warnsdorff confirms it).\n"
                    "Try training for more timesteps to improve the RL agent."
                )
            return (
                fig,
                msg,
                "",
                gr.update(minimum=1, maximum=len(partial_tour), value=len(partial_tour), visible=True),
            )
    else:
        # Warnsdorff
        result = solve_knight_tour(board_size, row, col)
        _last_result = result

        if result["tour"] is None:
            empty_fig = generate_chessboard_image([0], board_size, show_animation_frame=0)
            return (
                empty_fig,
                result["message"],
                "",
                gr.update(maximum=1, value=1, visible=False),
            )

        tour = result["tour"]
        n = board_size * board_size
        fig = generate_chessboard_image(tour, board_size)

        mat = tour_to_matrix(tour, board_size)
        matrix_text = _format_matrix(mat, board_size)

        return (
            fig,
            result["message"],
            matrix_text,
            gr.update(minimum=1, maximum=n, value=n, visible=True),
        )


def _format_matrix(mat: np.ndarray, board_size: int) -> str:
    lines = []
    header = "    " + "  ".join(f"{i+1:>3}" for i in range(board_size))
    lines.append(header)
    lines.append("    " + "----" * board_size)
    for r in range(board_size):
        label = ROW_LETTERS[r]
        vals = "  ".join(f"{mat[r, c]:>3}" for c in range(board_size))
        lines.append(f" {label} | {vals}")
    return "\n".join(lines)


def on_slider_change(step: int):
    """Re-render board at a particular animation step."""
    result = _last_result
    if not result or result.get("tour") is None:
        return None

    tour = result["tour"]
    board_size = int(np.sqrt(len(tour)))
    fig = generate_chessboard_image(tour, board_size, show_animation_frame=int(step))
    return fig


# ── Gradio UI ─────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Knight's Tour Solver",
    ) as app:
        gr.Markdown(
            "# ♞ Knight's Tour Solver\n"
            "Find and visualise a Knight's Tour on an N×N chessboard.  \n"
            "Choose board size, starting position, algorithm, then click **Solve**."
        )

        with gr.Row():
            # ── Left column: controls ──
            with gr.Column(scale=1):
                board_size = gr.Slider(
                    minimum=3, maximum=8, step=1, value=5,
                    label="Board Size",
                )
                row_dd = gr.Dropdown(
                    choices=_row_choices(5), value="A",
                    label="Starting Row",
                )
                col_dd = gr.Dropdown(
                    choices=_col_choices(5), value="1",
                    label="Starting Column",
                )
                algo_dd = gr.Radio(
                    choices=["Warnsdorff's Heuristic", "Reinforcement Learning (Masked PPO)"],
                    value="Warnsdorff's Heuristic",
                    label="Algorithm",
                )
                solve_btn = gr.Button("Solve", variant="primary", size="lg")

                status_box = gr.Textbox(label="Result", lines=3, interactive=False)

                step_slider = gr.Slider(
                    minimum=1, maximum=25, step=1, value=25,
                    label="Animation Step (drag to step through)",
                    visible=False,
                )

            # ── Right column: board visualisation ──
            with gr.Column(scale=2):
                board_plot = gr.Plot(
                    label="Chessboard",
                    value=generate_empty_board(5, knight_row=0, knight_col=0),
                )

        # ── Tour matrix ──
        with gr.Accordion("Knight Tour Matrix", open=False):
            matrix_box = gr.Code(label="Visit-Order Matrix", language=None, lines=12)

        # ── Wiring ──
        board_size.change(
            on_board_size_change,
            inputs=[board_size],
            outputs=[row_dd, col_dd, board_plot],
        )

        row_dd.change(
            on_position_change,
            inputs=[board_size, row_dd, col_dd],
            outputs=[board_plot],
        )

        col_dd.change(
            on_position_change,
            inputs=[board_size, row_dd, col_dd],
            outputs=[board_plot],
        )

        solve_btn.click(
            run_solver,
            inputs=[board_size, row_dd, col_dd, algo_dd],
            outputs=[board_plot, status_box, matrix_box, step_slider],
        )

        step_slider.change(
            on_slider_change,
            inputs=[step_slider],
            outputs=[board_plot],
        )

    return app


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())

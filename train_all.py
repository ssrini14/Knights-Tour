"""Train RL (MaskablePPO) models for all board sizes (3–8)."""

import time
import numpy as np
from rl_knight_tour import KnightTourAgent

STEPS_BY_SIZE = {
    3: 200_000,
    4: 200_000,
    5: 500_000,
    6: 1_000_000,
    7: 1_500_000,
    8: 2_000_000,
}

for N in range(3, 9):
    steps = STEPS_BY_SIZE[N]
    print(f"\n{'=' * 50}")
    print(f"Training {N}x{N} board  ({steps:,} timesteps)")
    print(f"{'=' * 50}")

    agent = KnightTourAgent(N)
    t0 = time.time()
    stats = agent.train(
        total_timesteps=steps,
        log_fn=lambda ep, l, s: print(
            f"  ep={ep:>6d}  avg_len={l:5.1f}  success={s:5.1f}%"
        ),
    )
    elapsed = time.time() - t0

    n_ep = len(stats["successes"])
    n_ok = sum(stats["successes"])
    last = stats["successes"][-100:] if n_ep >= 100 else stats["successes"]
    print(
        f"  Done in {elapsed:.0f}s  |  {n_ep} episodes  |  "
        f"{n_ok} successes ({n_ok / max(n_ep, 1) * 100:.1f}%)  |  "
        f"last-100: {np.mean(last) * 100:.1f}%"
    )

    # Quick solve test
    ok = 0
    total = N * N
    for r in range(N):
        for c in range(N):
            tour, _ = agent.solve(r, c)
            if tour is not None:
                ok += 1
    print(f"  Solve: {ok}/{total} starting positions")

print("\n\nAll models trained and saved!")

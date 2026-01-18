"""RL Opponent Agent for 007 Pose Game."""

import numpy as np
from dataclasses import dataclass, field

MAX_BULLETS = 10
ACTIONS = ['L', 'B', 'S']
HISTORY_LEN = 5


@dataclass
class GameContext:
    """Current game state for decision making."""
    my_bullets: int
    opp_bullets: int
    opp_history: list[str] = field(default_factory=list)
    my_history: list[str] = field(default_factory=list)


def to_features(ctx: GameContext) -> np.ndarray:
    """Convert game context to neural net input (32 features)."""
    features = [
        ctx.my_bullets / MAX_BULLETS,
        ctx.opp_bullets / MAX_BULLETS,
    ]

    # One-hot encode last N opponent actions (3 values each: L/B/S)
    for i in range(HISTORY_LEN):
        if i < len(ctx.opp_history):
            action = ctx.opp_history[-(i + 1)]
            features.extend([
                float(action == 'L'),
                float(action == 'B'),
                float(action == 'S'),
            ])
        else:
            features.extend([0.0, 0.0, 0.0])

    # One-hot encode last N AI actions
    for i in range(HISTORY_LEN):
        if i < len(ctx.my_history):
            action = ctx.my_history[-(i + 1)]
            features.extend([
                float(action == 'L'),
                float(action == 'B'),
                float(action == 'S'),
            ])
        else:
            features.extend([0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)

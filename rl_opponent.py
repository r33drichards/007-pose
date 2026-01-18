"""RL Opponent Agent for 007 Pose Game."""

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_BULLETS = 10
ACTIONS = ['L', 'B', 'S']
HISTORY_LEN = 5
INPUT_SIZE = 32
HIDDEN_SIZE = 64


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


class OpponentPredictor(nn.Module):
    """Predicts probability distribution over opponent's next action."""

    def __init__(self, input_size: int = INPUT_SIZE, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 3),  # L, B, S logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for [L, B, S]."""
        return self.net(x)

    def predict_probs(self, x: torch.Tensor) -> np.ndarray:
        """Returns probability distribution [P(L), P(B), P(S)]."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1).cpu().numpy()


class OpponentModel:
    """Manages opponent predictor with online learning and persistence."""

    def __init__(self, model_path: str = "opponent_model.pt"):
        self.model_path = Path(model_path)
        self.model = OpponentPredictor()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # Experience buffer for mini-batch updates
        self.buffer: list[tuple[np.ndarray, int]] = []
        self.buffer_size = 500

        self.load()

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict action probabilities for given features."""
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        probs = self.model.predict_probs(x)
        return probs[0]

    def update(self, features: np.ndarray, actual_action: str):
        """Update model after observing opponent's action."""
        action_idx = ACTIONS.index(actual_action)
        self.buffer.append((features.copy(), action_idx))

        # Keep buffer bounded
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        # Train on mini-batch from buffer
        self._train_step()

    def _train_step(self, batch_size: int = 16, epochs: int = 2):
        """Single training step on recent experience."""
        if len(self.buffer) < 4:
            return

        if len(self.buffer) < batch_size:
            batch = self.buffer
        else:
            batch = random.sample(self.buffer, batch_size)

        X = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
        y = torch.tensor([b[1] for b in batch], dtype=torch.long)

        self.model.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            self.optimizer.step()
        self.model.eval()

    def save(self):
        """Save model and buffer to disk."""
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'buffer': self.buffer,
        }, self.model_path)

    def load(self):
        """Load model and buffer from disk if exists."""
        if self.model_path.exists():
            try:
                data = torch.load(self.model_path, weights_only=False)
                self.model.load_state_dict(data['model'])
                self.optimizer.load_state_dict(data['optimizer'])
                self.buffer = data.get('buffer', [])
            except Exception as e:
                print(f"Could not load opponent model: {e}")

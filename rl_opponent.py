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


# Payoff matrix: PAYOFF[ai_action][opp_action] = AI's reward
PAYOFF = {
    'L': {'L': 0.0,  'B': 0.0,  'S': -1.0},   # AI loads: dies if opp shoots
    'B': {'L': 0.0,  'B': 0.0,  'S': 0.1},    # AI blocks: safe, slight bonus if blocked shot
    'S': {'L': 1.0,  'B': -0.1, 'S': 0.0},    # AI shoots: kills loader, wastes on blocker
}


def get_valid_actions(bullets: int) -> list[str]:
    """Return valid actions given bullet count."""
    if bullets > 0:
        return ['L', 'B', 'S']
    return ['L', 'B']


def select_action(
    opp_probs: np.ndarray,
    ai_bullets: int,
    opp_bullets: int,
    exploration_rate: float = 0.1
) -> str:
    """Select best action given predicted opponent distribution."""
    p_L, p_B, p_S = opp_probs

    # Adjust if opponent can't shoot (0 bullets)
    if opp_bullets <= 0:
        # Redistribute S probability to L and B
        p_L = p_L + p_S * 0.7  # Most likely reload
        p_B = p_B + p_S * 0.3  # Maybe block
        p_S = 0.0

    # Calculate expected value for each AI action
    expected_values = {}
    for ai_action in ACTIONS:
        ev = (p_L * PAYOFF[ai_action]['L'] +
              p_B * PAYOFF[ai_action]['B'] +
              p_S * PAYOFF[ai_action]['S'])
        expected_values[ai_action] = ev

    # Filter to valid actions
    valid_actions = get_valid_actions(ai_bullets)

    # Add exploration
    if random.random() < exploration_rate:
        return random.choice(valid_actions)

    # Pick best valid action
    return max(valid_actions, key=lambda a: expected_values[a])


class RLOpponent:
    """Complete RL opponent with persistent + session learning."""

    def __init__(self, model_path: str = "opponent_model.pt"):
        self.model_path = Path(model_path)

        # Persistent opponent model (loads from disk)
        self.persistent_model = OpponentModel(model_path)

        # Session-specific adaptation
        self.session_buffer: list[tuple[np.ndarray, int]] = []
        self.session_model: OpponentPredictor | None = None

        # Game state tracking
        self.my_history: list[str] = []
        self.opp_history: list[str] = []

        # Stats
        self.rounds_played = 0

    def reset_game(self):
        """Reset for new game (keep session learning)."""
        self.my_history = []
        self.opp_history = []

    def reset_session(self):
        """Reset for new play session (keep persistent model)."""
        self.session_buffer = []
        self.session_model = None
        self.reset_game()

    def reset_all(self):
        """Full reset - clear all learned data and delete saved model."""
        # Delete saved model file
        if self.model_path.exists():
            self.model_path.unlink()

        # Reinitialize persistent model (fresh weights, empty buffer)
        self.persistent_model = OpponentModel(str(self.model_path))

        # Clear session data
        self.reset_session()
        self.rounds_played = 0

    def get_action(self, my_bullets: int, opp_bullets: int) -> str:
        """Get AI's action for this round."""
        ctx = GameContext(
            my_bullets=my_bullets,
            opp_bullets=opp_bullets,
            opp_history=self.opp_history,
            my_history=self.my_history,
        )
        features = to_features(ctx)

        # Get predictions from persistent model
        persistent_probs = self.persistent_model.predict(features)

        # Blend with session model if available
        if self.session_model is not None and len(self.session_buffer) >= 10:
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            session_probs = self.session_model.predict_probs(x)[0]

            # Weight session more as it grows (max 60%)
            session_weight = min(0.6, len(self.session_buffer) / 50)
            probs = (1 - session_weight) * persistent_probs + session_weight * session_probs
        else:
            probs = persistent_probs

        return select_action(probs, my_bullets, opp_bullets)

    def update(self, my_action: str, opp_action: str):
        """Update after round completes."""
        # Validate actions
        if my_action not in ACTIONS:
            raise ValueError(f"Invalid my_action: {my_action}. Must be one of {ACTIONS}")
        if opp_action not in ACTIONS:
            raise ValueError(f"Invalid opp_action: {opp_action}. Must be one of {ACTIONS}")

        # Build features from state BEFORE this round
        ctx = GameContext(
            my_bullets=0,  # Not used for history encoding
            opp_bullets=0,
            opp_history=self.opp_history,
            my_history=self.my_history,
        )
        features = to_features(ctx)

        # Update histories
        self.my_history.append(my_action)
        self.opp_history.append(opp_action)
        self.rounds_played += 1

        # Update persistent model
        self.persistent_model.update(features, opp_action)

        # Update session buffer
        action_idx = ACTIONS.index(opp_action)
        self.session_buffer.append((features.copy(), action_idx))

        # Create/update session model after enough data
        if len(self.session_buffer) >= 10:
            self._update_session_model()

    def _update_session_model(self):
        """Train session-specific model on current session data."""
        if self.session_model is None:
            self.session_model = OpponentPredictor()

        X = torch.tensor(np.array([b[0] for b in self.session_buffer]), dtype=torch.float32)
        y = torch.tensor([b[1] for b in self.session_buffer], dtype=torch.long)

        optimizer = torch.optim.Adam(self.session_model.parameters(), lr=0.05)
        self.session_model.train()
        for _ in range(10):
            optimizer.zero_grad()
            loss = F.cross_entropy(self.session_model(X), y)
            loss.backward()
            optimizer.step()
        self.session_model.eval()

    def save(self):
        """Save persistent model to disk."""
        self.persistent_model.save()

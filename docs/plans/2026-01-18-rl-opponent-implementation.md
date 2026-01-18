# RL Opponent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace LearningAI with sample-efficient RL opponent that learns player patterns and persists across sessions.

**Architecture:** Two-layer opponent model (persistent + session) predicts player actions from game context (bullets + history), then game-theoretic action selection picks optimal counter. Single new file `rl_opponent.py` contains all components.

**Tech Stack:** PyTorch (already in use), numpy, dataclasses

---

### Task 1: GameContext and Feature Extraction

**Files:**
- Create: `rl_opponent.py`
- Test: `test_rl_opponent.py`

**Step 1: Write the failing test for GameContext**

```python
# test_rl_opponent.py
import pytest
import numpy as np
from rl_opponent import GameContext, to_features, MAX_BULLETS

def test_game_context_creation():
    ctx = GameContext(
        my_bullets=3,
        opp_bullets=5,
        opp_history=['L', 'B', 'S'],
        my_history=['L', 'L'],
    )
    assert ctx.my_bullets == 3
    assert ctx.opp_bullets == 5
    assert ctx.opp_history == ['L', 'B', 'S']
    assert ctx.my_history == ['L', 'L']

def test_to_features_shape():
    ctx = GameContext(my_bullets=0, opp_bullets=0, opp_history=[], my_history=[])
    features = to_features(ctx)
    assert features.shape == (32,)
    assert features.dtype == np.float32

def test_to_features_bullet_normalization():
    ctx = GameContext(my_bullets=5, opp_bullets=10, opp_history=[], my_history=[])
    features = to_features(ctx)
    assert features[0] == 5 / MAX_BULLETS  # my_bullets normalized
    assert features[1] == 10 / MAX_BULLETS  # opp_bullets normalized

def test_to_features_history_encoding():
    ctx = GameContext(my_bullets=0, opp_bullets=0, opp_history=['S'], my_history=['L'])
    features = to_features(ctx)
    # opp_history[-1] = 'S' -> [0, 0, 1] at indices 2-4
    assert features[2] == 0  # L
    assert features[3] == 0  # B
    assert features[4] == 1  # S
    # my_history[-1] = 'L' -> [1, 0, 0] at indices 17-19
    assert features[17] == 1  # L
    assert features[18] == 0  # B
    assert features[19] == 0  # S
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_rl_opponent.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'rl_opponent'"

**Step 3: Write minimal implementation**

```python
# rl_opponent.py
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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_rl_opponent.py -v`
Expected: PASS (4 passed)

**Step 5: Commit**

```bash
git add rl_opponent.py test_rl_opponent.py
git commit -m "feat: add GameContext and to_features for RL opponent"
```

---

### Task 2: OpponentPredictor Neural Network

**Files:**
- Modify: `rl_opponent.py`
- Modify: `test_rl_opponent.py`

**Step 1: Write the failing test**

```python
# Add to test_rl_opponent.py
import torch
from rl_opponent import OpponentPredictor

def test_opponent_predictor_output_shape():
    model = OpponentPredictor()
    x = torch.randn(1, 32)
    logits = model(x)
    assert logits.shape == (1, 3)

def test_opponent_predictor_batch():
    model = OpponentPredictor()
    x = torch.randn(16, 32)
    logits = model(x)
    assert logits.shape == (16, 3)

def test_opponent_predictor_predict_probs():
    model = OpponentPredictor()
    x = torch.randn(1, 32)
    probs = model.predict_probs(x)
    assert probs.shape == (1, 3)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)
    assert (probs >= 0).all()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_rl_opponent.py::test_opponent_predictor_output_shape -v`
Expected: FAIL with "ImportError: cannot import name 'OpponentPredictor'"

**Step 3: Write minimal implementation**

```python
# Add to rl_opponent.py after existing code
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SIZE = 32
HIDDEN_SIZE = 64


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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_rl_opponent.py -v -k "predictor"`
Expected: PASS (3 passed)

**Step 5: Commit**

```bash
git add rl_opponent.py test_rl_opponent.py
git commit -m "feat: add OpponentPredictor neural network"
```

---

### Task 3: OpponentModel with Persistence

**Files:**
- Modify: `rl_opponent.py`
- Modify: `test_rl_opponent.py`

**Step 1: Write the failing test**

```python
# Add to test_rl_opponent.py
import tempfile
import os
from rl_opponent import OpponentModel

def test_opponent_model_predict():
    model = OpponentModel()
    ctx = GameContext(my_bullets=0, opp_bullets=0, opp_history=[], my_history=[])
    features = to_features(ctx)
    probs = model.predict(features)
    assert probs.shape == (3,)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)

def test_opponent_model_update():
    model = OpponentModel()
    ctx = GameContext(my_bullets=0, opp_bullets=0, opp_history=[], my_history=[])
    features = to_features(ctx)
    # Should not raise
    model.update(features, 'L')
    model.update(features, 'B')
    model.update(features, 'S')
    assert len(model.buffer) == 3

def test_opponent_model_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model.pt")

        # Create and train model
        model1 = OpponentModel(path)
        ctx = GameContext(my_bullets=5, opp_bullets=3, opp_history=['L'], my_history=['B'])
        features = to_features(ctx)
        for _ in range(10):
            model1.update(features, 'S')
        model1.save()

        # Load in new instance
        model2 = OpponentModel(path)
        assert len(model2.buffer) == 10

        # Predictions should be similar
        probs1 = model1.predict(features)
        probs2 = model2.predict(features)
        assert np.allclose(probs1, probs2, atol=0.1)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_rl_opponent.py::test_opponent_model_predict -v`
Expected: FAIL with "ImportError: cannot import name 'OpponentModel'"

**Step 3: Write minimal implementation**

```python
# Add to rl_opponent.py after OpponentPredictor
import random
from pathlib import Path


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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_rl_opponent.py -v -k "opponent_model"`
Expected: PASS (3 passed)

**Step 5: Commit**

```bash
git add rl_opponent.py test_rl_opponent.py
git commit -m "feat: add OpponentModel with persistence"
```

---

### Task 4: Game-Theoretic Action Selection

**Files:**
- Modify: `rl_opponent.py`
- Modify: `test_rl_opponent.py`

**Step 1: Write the failing test**

```python
# Add to test_rl_opponent.py
from rl_opponent import select_action, PAYOFF

def test_payoff_matrix_structure():
    # Verify payoff matrix has correct structure
    for ai_action in ['L', 'B', 'S']:
        assert ai_action in PAYOFF
        for opp_action in ['L', 'B', 'S']:
            assert opp_action in PAYOFF[ai_action]

def test_select_action_shoots_loader():
    # If opponent will definitely load, AI should shoot (if has bullets)
    probs = np.array([1.0, 0.0, 0.0])  # 100% Load
    action = select_action(probs, ai_bullets=1, opp_bullets=0)
    assert action == 'S'

def test_select_action_blocks_shooter():
    # If opponent will definitely shoot, AI should block
    probs = np.array([0.0, 0.0, 1.0])  # 100% Shoot
    action = select_action(probs, ai_bullets=1, opp_bullets=1)
    assert action == 'B'

def test_select_action_loads_against_blocker():
    # If opponent will definitely block, AI should load
    probs = np.array([0.0, 1.0, 0.0])  # 100% Block
    action = select_action(probs, ai_bullets=1, opp_bullets=1)
    assert action == 'L'

def test_select_action_no_bullets_cant_shoot():
    # AI with no bullets can't shoot even if optimal
    probs = np.array([1.0, 0.0, 0.0])  # 100% Load (shoot would be optimal)
    action = select_action(probs, ai_bullets=0, opp_bullets=0)
    assert action in ['L', 'B']  # Must load or block

def test_select_action_adjusts_for_opp_no_bullets():
    # If opponent has no bullets, their shoot probability should be ignored
    probs = np.array([0.0, 0.0, 1.0])  # Predicts 100% Shoot
    # But opponent has 0 bullets, so can't shoot - AI should not block
    action = select_action(probs, ai_bullets=1, opp_bullets=0)
    # With adjusted probs, opponent will L or B, so AI should S or L
    assert action in ['S', 'L']
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_rl_opponent.py::test_payoff_matrix_structure -v`
Expected: FAIL with "ImportError: cannot import name 'select_action'"

**Step 3: Write minimal implementation**

```python
# Add to rl_opponent.py after OpponentModel

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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_rl_opponent.py -v -k "select_action or payoff"`
Expected: PASS (6 passed)

**Step 5: Commit**

```bash
git add rl_opponent.py test_rl_opponent.py
git commit -m "feat: add game-theoretic action selection"
```

---

### Task 5: RLOpponent Main Class

**Files:**
- Modify: `rl_opponent.py`
- Modify: `test_rl_opponent.py`

**Step 1: Write the failing test**

```python
# Add to test_rl_opponent.py
from rl_opponent import RLOpponent

def test_rl_opponent_get_action():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model.pt")
        ai = RLOpponent(path)

        action = ai.get_action(my_bullets=0, opp_bullets=0)
        assert action in ['L', 'B', 'S']

def test_rl_opponent_update():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model.pt")
        ai = RLOpponent(path)

        ai.get_action(my_bullets=0, opp_bullets=0)
        ai.update(my_action='L', opp_action='L')

        assert len(ai.my_history) == 1
        assert len(ai.opp_history) == 1
        assert ai.rounds_played == 1

def test_rl_opponent_reset_game():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model.pt")
        ai = RLOpponent(path)

        ai.get_action(my_bullets=0, opp_bullets=0)
        ai.update(my_action='L', opp_action='L')
        ai.reset_game()

        assert len(ai.my_history) == 0
        assert len(ai.opp_history) == 0
        # rounds_played persists across games
        assert ai.rounds_played == 1

def test_rl_opponent_session_adaptation():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model.pt")
        ai = RLOpponent(path)

        # Play enough rounds to trigger session model
        for _ in range(15):
            ai.get_action(my_bullets=1, opp_bullets=1)
            ai.update(my_action='L', opp_action='S')  # Opponent always shoots

        assert len(ai.session_buffer) == 15
        assert ai.session_model is not None

def test_rl_opponent_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model.pt")

        ai1 = RLOpponent(path)
        for _ in range(10):
            ai1.get_action(my_bullets=1, opp_bullets=1)
            ai1.update(my_action='B', opp_action='L')
        ai1.save()

        ai2 = RLOpponent(path)
        assert len(ai2.persistent_model.buffer) == 10
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_rl_opponent.py::test_rl_opponent_get_action -v`
Expected: FAIL with "ImportError: cannot import name 'RLOpponent'"

**Step 3: Write minimal implementation**

```python
# Add to rl_opponent.py at the end

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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_rl_opponent.py -v -k "rl_opponent"`
Expected: PASS (5 passed)

**Step 5: Commit**

```bash
git add rl_opponent.py test_rl_opponent.py
git commit -m "feat: add RLOpponent main class with two-layer learning"
```

---

### Task 6: Integration with collect_poses.py

**Files:**
- Modify: `collect_poses.py`

**Step 1: Write integration test (manual)**

This is a manual test - run the game and verify AI works.

**Step 2: Update imports and initialization**

In `collect_poses.py`, find line ~8 and add import:

```python
# Add after existing imports
from rl_opponent import RLOpponent
```

**Step 3: Replace LearningAI with RLOpponent**

In `main()` function around line 608, replace:

```python
# Old:
ai = LearningAI()

# New:
ai = RLOpponent(os.path.join(script_dir, "opponent_model.pt"))
```

**Step 4: Update run_game() to use new interface**

In `run_game()` around line 436, replace:

```python
# Old:
ai.reset()

# New:
ai.reset_game()
```

Around line 511, replace:

```python
# Old:
ai_action = ai.get_action(state.p2_bullets)

# New:
ai_action = ai.get_action(state.p2_bullets, state.p1_bullets)
```

Around line 517, replace:

```python
# Old:
ai.update(ai_action, player_action)

# New:
ai.update(ai_action, player_action)  # Same signature, but now tracks features internally
```

**Step 5: Add save on quit**

In `main()` after the game loop ends (around line 644), add:

```python
# After the while True loop, before cap.release()
ai.save()
```

**Step 6: Remove old LearningAI code**

Delete or comment out the entire `LearningAI` class and its helper functions (lines ~82-196):
- `ideal_response`
- `options`
- `select_proportional`
- `select_maximum`
- `random_reply`
- `single_event_proportional`
- `single_event_greedy`
- `digraph_event_proportional`
- `digraph_event_greedy`
- `class LearningAI`

**Step 7: Run manual test**

Run: `python collect_poses.py`
Expected: Game starts, AI makes decisions, model saves on quit

**Step 8: Commit**

```bash
git add collect_poses.py
git commit -m "feat: integrate RLOpponent into game loop"
```

---

### Task 7: Run All Tests

**Step 1: Run full test suite**

Run: `python -m pytest test_rl_opponent.py -v`
Expected: All tests pass

**Step 2: Final commit**

```bash
git add -A
git commit -m "test: verify all RL opponent tests pass"
```

---

## Summary

| Task | Description |
|------|-------------|
| 1 | GameContext + to_features |
| 2 | OpponentPredictor neural net |
| 3 | OpponentModel with persistence |
| 4 | Game-theoretic action selection |
| 5 | RLOpponent main class |
| 6 | Integration with collect_poses.py |
| 7 | Run all tests |

Total: ~7 commits, incremental TDD approach.

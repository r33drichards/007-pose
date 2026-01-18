# RL Opponent Agent Design

## Overview

Replace the existing `LearningAI` multi-arm bandit with a sample-efficient RL opponent that:
- Learns player patterns using full game context (bullet counts + action history)
- Persists learned model to disk across sessions
- Adapts within-session to playstyle changes
- Uses game-theoretic action selection for optimal counter-play

## Game Rules

Standard 007:
- Both players start with 0 bullets
- RELOAD (L): Gain 1 bullet (max 10)
- SHOOT (S): Spend 1 bullet, kills opponent if they're reloading
- SHIELD (B): Blocks a shot (no cost)
- First to kill wins the game (first to 3 kills wins match)
- Simultaneous reveal with countdown sync

## Architecture

```
┌─────────────────────────┐     ┌─────────────────────────┐
│   OpponentPredictor     │────▶│   Action Selection      │
│   (persistent model)    │     │   (game theory)         │
└─────────────────────────┘     └─────────────────────────┘
         +                                │
┌─────────────────────────┐               │
│   Session Model         │───────────────┘
│   (within-session)      │
└─────────────────────────┘
```

## State Representation

32 features total:

```python
features = [
    my_bullets / MAX_BULLETS,       # AI bullets (normalized)
    opp_bullets / MAX_BULLETS,      # Player bullets (normalized)
    *one_hot(opp_history[-1]),      # Last 5 opponent actions (3 each)
    *one_hot(opp_history[-2]),
    *one_hot(opp_history[-3]),
    *one_hot(opp_history[-4]),
    *one_hot(opp_history[-5]),
    *one_hot(my_history[-1]),       # Last 5 AI actions (3 each)
    *one_hot(my_history[-2]),
    *one_hot(my_history[-3]),
    *one_hot(my_history[-4]),
    *one_hot(my_history[-5]),
]
```

Captures:
- Bullet-dependent patterns ("always reloads at 0")
- Sequential patterns ("shoots after two reloads")
- Reactive patterns ("shields when AI has bullets")

## Opponent Predictor

Small neural net predicting P(player_action | game_context):

```
Input (32) → Linear(64) → ReLU → Dropout(0.1)
          → Linear(64) → ReLU → Dropout(0.1)
          → Linear(3) → Softmax → [P(L), P(B), P(S)]
```

Training:
- Online learning after each round
- Experience buffer (500 samples) for mini-batch updates
- High learning rate (0.01) for fast adaptation
- Cross-entropy loss

## Action Selection

Game-theoretic best response using payoff matrix:

```
PAYOFF[ai_action][player_action]:

              Player
AI        L      B      S
L         0      0     -1    (die if they shoot)
B         0      0    +0.1   (safe, slight waste)
S        +1    -0.1    0     (kill if they load)
```

Selection:
1. Get P(L), P(B), P(S) from predictor
2. Calculate expected value for each AI action
3. Adjust if opponent can't shoot (0 bullets)
4. Pick highest EV action
5. 10% exploration to avoid predictability

## Two-Layer Learning

**Persistent Model:**
- Saves to `opponent_model.pt`
- Contains experience buffer + model weights
- Learns general player tendencies over days/weeks

**Session Model:**
- Fresh each play session
- Trained only on current session data
- Adapts to today's playstyle changes

**Blending:**
| Session rounds | Persistent weight | Session weight |
|----------------|-------------------|----------------|
| 0-9            | 100%              | 0%             |
| 10-25          | ~80%              | ~20%           |
| 25-50          | ~50%              | ~50%           |
| 50+            | 40%               | 60%            |

## Files

```
007-pose/
├── collect_poses.py      # Existing - minor integration changes
├── rl_opponent.py        # NEW - all RL opponent code
├── opponent_model.pt     # NEW - persistent model (auto-created)
```

## Integration

Replace in `collect_poses.py`:

```python
# Old:
ai = LearningAI()
ai.reset()
ai_action = ai.get_action(state.p2_bullets)
ai.update(ai_action, player_action)

# New:
ai = RLOpponent("opponent_model.pt")
ai.reset_game()
ai_action = ai.get_action(state.p2_bullets, state.p1_bullets)
ai.update(ai_action, player_action, features)
ai.save()  # on quit
```

## Sample Efficiency

- Useful predictions after ~20-30 rounds
- Good adaptation after ~50-100 rounds
- Continuous improvement over hundreds of rounds
- No convergence issues (not using DQN/PPO)

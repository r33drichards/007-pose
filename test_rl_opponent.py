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


# OpponentPredictor tests
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

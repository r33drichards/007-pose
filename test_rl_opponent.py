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
import tempfile
import os
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


# OpponentModel tests
from rl_opponent import OpponentModel

def test_opponent_model_predict():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model.pt")
        model = OpponentModel(path)
        ctx = GameContext(my_bullets=0, opp_bullets=0, opp_history=[], my_history=[])
        features = to_features(ctx)
        probs = model.predict(features)
        assert probs.shape == (3,)
        assert np.isclose(probs.sum(), 1.0, atol=1e-5)

def test_opponent_model_update():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model.pt")
        model = OpponentModel(path)
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


# Action Selection tests
from rl_opponent import select_action, PAYOFF, get_valid_actions

def test_payoff_matrix_structure():
    # Verify payoff matrix has correct structure
    for ai_action in ['L', 'B', 'S']:
        assert ai_action in PAYOFF
        for opp_action in ['L', 'B', 'S']:
            assert opp_action in PAYOFF[ai_action]

def test_get_valid_actions_with_bullets():
    actions = get_valid_actions(1)
    assert set(actions) == {'L', 'B', 'S'}

def test_get_valid_actions_no_bullets():
    actions = get_valid_actions(0)
    assert set(actions) == {'L', 'B'}

def test_select_action_shoots_loader():
    # If opponent will definitely load, AI should shoot (if has bullets)
    probs = np.array([1.0, 0.0, 0.0])  # 100% Load
    action = select_action(probs, ai_bullets=1, opp_bullets=0, exploration_rate=0.0)
    assert action == 'S'

def test_select_action_blocks_shooter():
    # If opponent will definitely shoot, AI should block
    probs = np.array([0.0, 0.0, 1.0])  # 100% Shoot
    action = select_action(probs, ai_bullets=1, opp_bullets=1, exploration_rate=0.0)
    assert action == 'B'

def test_select_action_loads_against_blocker():
    # If opponent will definitely block, AI should load
    probs = np.array([0.0, 1.0, 0.0])  # 100% Block
    action = select_action(probs, ai_bullets=1, opp_bullets=1, exploration_rate=0.0)
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

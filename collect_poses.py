#!/usr/bin/env python3
"""Pose collection with visual countdown GUI."""

import cv2
import mediapipe as mp
import os
import time
from collections import Counter
from random import choices, choice
from itertools import chain
from dataclasses import dataclass
from typing import Optional

from pose_classifier import PoseClassifier, POSE_LABELS
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Game Definition ################################

MAX_BULLETS = 10

@dataclass
class GameState:
    """State of the Load-Block-Shoot game."""
    p1_bullets: int = 0
    p2_bullets: int = 0
    winner: Optional[int] = None  # None=ongoing, 0=draw, 1=p1 wins, 2=p2 wins

    @property
    def is_terminal(self) -> bool:
        return self.winner is not None


def get_valid_actions(bullets: int) -> list[str]:
    """Return list of valid actions given bullet count."""
    if bullets > 0:
        return ['L', 'B', 'S']
    return ['L', 'B']


def step(state: GameState, p1_action: str, p2_action: str) -> GameState:
    """Execute one turn of simultaneous actions, return new state."""
    if state.is_terminal:
        return state

    p1_bullets = state.p1_bullets
    p2_bullets = state.p2_bullets
    winner = None

    # Determine if shots are fired (must have bullets)
    p1_shot = p1_action == 'S' and p1_bullets > 0
    p2_shot = p2_action == 'S' and p2_bullets > 0
    p1_blocked = p1_action == 'B'
    p2_blocked = p2_action == 'B'

    # Update bullet counts
    if p1_action == 'L':
        p1_bullets = min(p1_bullets + 1, MAX_BULLETS)
    elif p1_shot:
        p1_bullets -= 1

    if p2_action == 'L':
        p2_bullets = min(p2_bullets + 1, MAX_BULLETS)
    elif p2_shot:
        p2_bullets -= 1

    # Determine hits
    p1_hit = p2_shot and not p1_blocked
    p2_hit = p1_shot and not p2_blocked

    if p1_hit and p2_hit:
        winner = 0  # Draw
    elif p1_hit:
        winner = 2  # P2 wins
    elif p2_hit:
        winner = 1  # P1 wins

    return GameState(p1_bullets=p1_bullets, p2_bullets=p2_bullets, winner=winner)


# Learning AI ####################################

# L=Load, B=Block, S=Shoot
# If opponent Loads, Shoot wins. If opponent Shoots, Block saves. If opponent Blocks, Load is free.
ideal_response = {'L': 'S', 'B': 'L', 'S': 'B'}
options = ['L', 'B', 'S']


def select_proportional(events, baseline=()):
    if not events and not baseline:
        return choice(options)
    rel_freq = Counter(chain(baseline, events))
    population, weights = zip(*rel_freq.items())
    return choices(population, weights)[0]


def select_maximum(events, baseline=()):
    if not events and not baseline:
        return choice(options)
    rel_freq = Counter(chain(baseline, events))
    return rel_freq.most_common(1)[0][0]


# Strategies
def random_reply(p1hist, p2hist):
    return choice(options)


def single_event_proportional(p1hist, p2hist):
    prediction = select_proportional(p2hist, options)
    return ideal_response[prediction]


def single_event_greedy(p1hist, p2hist):
    prediction = select_maximum(p2hist, options)
    return ideal_response[prediction]


def digraph_event_proportional(p1hist, p2hist):
    if not p2hist:
        return choice(options)
    recent_play = p2hist[-1]
    digraphs = list(zip(p2hist, p2hist[1:]))
    followers = [b for a, b in digraphs if a == recent_play]
    if not followers:
        return single_event_proportional(p1hist, p2hist)
    prediction = select_proportional(followers, options)
    return ideal_response[prediction]


def digraph_event_greedy(p1hist, p2hist):
    if not p2hist:
        return choice(options)
    recent_play = p2hist[-1]
    digraphs = list(zip(p2hist, p2hist[1:]))
    followers = [b for a, b in digraphs if a == recent_play]
    if not followers:
        return single_event_greedy(p1hist, p2hist)
    prediction = select_maximum(followers, options)
    return ideal_response[prediction]


class LearningAI:
    """Multi-arm bandit AI that learns opponent patterns."""

    strategies = [
        random_reply,
        single_event_proportional,
        single_event_greedy,
        digraph_event_proportional,
        digraph_event_greedy,
    ]

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset for a new game."""
        self.weights = [1] * len(self.strategies)
        self.p1hist = []  # AI history
        self.p2hist = []  # Opponent (player) history

    def get_action(self, ai_bullets: int) -> str:
        """Get AI's action given its bullet count."""
        strategy_range = range(len(self.strategies))

        # Get moves from all strategies
        our_moves = [s(self.p1hist, self.p2hist) for s in self.strategies]

        # Choose strategy based on weights
        i = choices(strategy_range, self.weights)[0]
        move = our_moves[i]

        # Can't shoot without bullets - fall back to Load or Block
        if move == 'S' and ai_bullets <= 0:
            move = choice(['L', 'B'])

        return move

    def update(self, ai_action: str, player_action: str):
        """Update history and strategy weights after a round."""
        self.p1hist.append(ai_action)
        self.p2hist.append(player_action)

        # Reward strategies that would have won
        our_moves = [s(self.p1hist[:-1], self.p2hist[:-1]) for s in self.strategies]
        for i, move in enumerate(our_moves):
            # Check if this strategy's move would have beaten the player
            if move == 'S' and player_action == 'L':
                self.weights[i] += 1
            elif move == 'B' and player_action == 'S':
                self.weights[i] += 1
            elif move == 'L' and player_action == 'B':
                self.weights[i] += 1


def draw_centered_text(frame, text, y_offset=0, font_scale=3, color=(255, 255, 255), thickness=4):
    """Draw text centered on the frame."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (w - text_w) // 2
    y = (h + text_h) // 2 + y_offset
    # Shadow
    cv2.putText(frame, text, (x + 3, y + 3), font, font_scale, (0, 0, 0), thickness + 2)
    # Main text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def draw_button(frame, text, y, selected=False):
    """Draw a menu button."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3

    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    btn_w, btn_h = text_w + 60, text_h + 40
    x = (w - btn_w) // 2

    if selected:
        # Selected: bright cyan background, thick yellow border
        cv2.rectangle(frame, (x, y), (x + btn_w, y + btn_h), (200, 200, 0), -1)
        cv2.rectangle(frame, (x - 4, y - 4), (x + btn_w + 4, y + btn_h + 4), (0, 255, 255), 4)

        # Arrow indicator
        arrow_x = x - 50
        arrow_y = y + btn_h // 2
        cv2.putText(frame, ">>", (arrow_x, arrow_y + 10), font, 1.2, (0, 255, 255), 3)
    else:
        # Unselected: dark gray background
        cv2.rectangle(frame, (x, y), (x + btn_w, y + btn_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, y), (x + btn_w, y + btn_h), (100, 100, 100), 2)

    # Button text
    text_color = (0, 0, 0) if selected else (150, 150, 150)
    text_x = x + (btn_w - text_w) // 2
    text_y = y + (btn_h + text_h) // 2 - 5
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)

    return (x, y, x + btn_w, y + btn_h)


def show_menu(cap, classifier):
    """Show main menu. Returns 'start', 'calibrate', or None for quit."""
    selected = 0  # 0 = Start, 1 = Calibrate

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        # Darken background
        frame = (frame * 0.3).astype('uint8')

        # Title
        draw_centered_text(frame, "007 POSE", y_offset=-200, font_scale=3, color=(0, 255, 255))

        # Model status
        if classifier.is_ready():
            status = "Model: Ready"
            status_color = (0, 255, 0)
        else:
            status = "Model: Not calibrated"
            status_color = (0, 0, 255)

        h, w = frame.shape[:2]
        cv2.putText(frame, status, (w//2 - 120, h//2 - 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # Buttons
        draw_button(frame, "START", h//2 - 40, selected == 0)
        draw_button(frame, "CALIBRATE", h//2 + 60, selected == 1)

        # Instructions
        cv2.putText(frame, "UP/DOWN to select, ENTER to confirm, Q to quit",
                   (w//2 - 280, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)

        cv2.imshow("007 Pose Game", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return None
        elif key in [ord('w'), 82]:  # W or Up arrow
            selected = (selected - 1) % 2
        elif key in [ord('s'), 84]:  # S or Down arrow
            selected = (selected + 1) % 2
        elif key in [13, 10]:  # Enter
            return 'start' if selected == 0 else 'calibrate'


def run_calibration(cap, landmarker, classifier):
    """Run calibration mode. Returns True if successful."""
    classifier.clear_training_data()
    record_duration = 3.0
    countdown_duration = 5.0  # Time to get into position

    for pose_idx, pose_name in enumerate(POSE_LABELS):
        # === COUNTDOWN PHASE ===
        countdown_start = time.time()
        while True:
            elapsed = time.time() - countdown_start
            remaining = countdown_duration - elapsed

            if remaining <= 0:
                break

            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            draw_centered_text(frame, f"Get ready: {pose_name}", y_offset=-100, font_scale=1.8, color=(0, 255, 255))

            # Show countdown number
            num = int(remaining) + 1
            draw_centered_text(frame, str(num), y_offset=20, font_scale=4, color=(255, 255, 255))

            draw_centered_text(frame, f"Pose {pose_idx + 1}/{len(POSE_LABELS)}", y_offset=150, font_scale=0.8, color=(150, 150, 150))

            cv2.putText(frame, "[ESC] Cancel", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

            cv2.imshow("007 Pose Game", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                return False

        # === RECORDING PHASE ===
        record_start = time.time()
        samples = 0

        while True:
            elapsed = time.time() - record_start
            remaining = record_duration - elapsed

            if remaining <= 0:
                break

            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            results = landmarker.detect(mp_image)

            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                landmarks = results.pose_landmarks[0]
                classifier.add_sample(landmarks, pose_name)
                samples += 1

            draw_centered_text(frame, f"Recording {pose_name}", y_offset=-50, font_scale=1.5, color=(0, 255, 0))
            draw_centered_text(frame, f"{remaining:.1f}s", y_offset=30, font_scale=2, color=(0, 255, 0))
            draw_centered_text(frame, f"Samples: {samples}", y_offset=100, font_scale=0.8, color=(200, 200, 200))

            cv2.imshow("007 Pose Game", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                return False

    # Train model
    print("Training model...")
    if classifier.train():
        # Show success
        for _ in range(60):  # ~2 seconds
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame = (frame * 0.3).astype('uint8')
                draw_centered_text(frame, "Calibration Complete!", font_scale=1.5, color=(0, 255, 0))
                cv2.imshow("007 Pose Game", frame)
                cv2.waitKey(33)
        return True
    else:
        print("Training failed")
        return False


def draw_game_state(frame, state, round_num):
    """Draw bullet counts and round number."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Your bullets (left side)
    cv2.putText(frame, "YOU", (30, 50), font, 0.8, (0, 255, 255), 2)
    bullets_str = "|" * state.p1_bullets + "." * (MAX_BULLETS - state.p1_bullets)
    cv2.putText(frame, f"[{bullets_str}]", (30, 85), font, 0.6, (0, 255, 0), 2)

    # AI bullets (right side)
    cv2.putText(frame, "AI", (w - 120, 50), font, 0.8, (0, 0, 255), 2)
    bullets_str = "|" * state.p2_bullets + "." * (MAX_BULLETS - state.p2_bullets)
    cv2.putText(frame, f"[{bullets_str}]", (w - 200, 85), font, 0.6, (0, 255, 0), 2)

    # Round number
    cv2.putText(frame, f"Round {round_num}", (w // 2 - 60, 40), font, 0.8, (255, 255, 255), 2)


def pose_to_action(pose, bullets):
    """Convert pose name to game action (L/B/S)."""
    mapping = {
        "SHOOT": 'S',
        "SHIELD": 'B',
        "RELOAD": 'L',
    }
    action = mapping.get(pose)
    # Can't shoot without bullets
    if action == 'S' and bullets <= 0:
        return None
    return action


# Display names for actions
ACTION_NAMES = {'L': "RELOAD", 'B': "SHIELD", 'S': "SHOOT"}


def run_game(cap, landmarker, classifier, ai):
    """Run the pose collection game against AI. Returns when game ends."""
    countdown_seconds = 2
    pause_after_capture = 2

    pose_colors = {
        "SHOOT": (0, 0, 255),      # Red
        "SHIELD": (255, 165, 0),   # Orange
        "RELOAD": (0, 255, 255),   # Yellow
        "NEUTRAL": (128, 128, 128),
        "NO POSE": (100, 100, 100),
    }

    # Reset AI for new game
    ai.reset()

    state = GameState()
    round_num = 0

    while not state.is_terminal:
        round_num += 1

        # === COUNTDOWN PHASE ===
        countdown_start = time.time()
        while True:
            elapsed = time.time() - countdown_start
            remaining = countdown_seconds - elapsed

            if remaining <= 0:
                break

            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            draw_game_state(frame, state, round_num)

            # Draw countdown number
            num = int(remaining) + 1
            if remaining < 0.5:
                draw_centered_text(frame, "POSE!", font_scale=4, color=(0, 255, 0))
            else:
                draw_centered_text(frame, str(num), font_scale=5, color=(0, 255, 255))

            # Show valid moves hint
            valid = get_valid_actions(state.p1_bullets)
            valid_names = [ACTION_NAMES[a] for a in valid]
            cv2.putText(frame, f"Valid: {', '.join(valid_names)}", (10, frame.shape[0] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, "[ESC] Quit", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

            cv2.imshow("007 Pose Game", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                return None

        # === CAPTURE PHASE - wait for valid non-neutral pose ===
        pose = "NEUTRAL"
        player_action = None
        while player_action is None:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            results = landmarker.detect(mp_image)

            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                landmarks = results.pose_landmarks[0]
                if classifier.is_ready():
                    pose = classifier.predict(landmarks)
                    player_action = pose_to_action(pose, state.p1_bullets)
                else:
                    pose = "NO MODEL"
                    return None

            # Show waiting state
            draw_game_state(frame, state, round_num)
            draw_centered_text(frame, "POSE!", font_scale=4, color=(0, 255, 0))
            if pose != "NEUTRAL" and player_action is None:
                draw_centered_text(frame, "No bullets!", y_offset=100, font_scale=1, color=(0, 0, 255))
            cv2.imshow("007 Pose Game", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                return None

        # Get AI action
        ai_action = ai.get_action(state.p2_bullets)

        # Execute turn
        new_state = step(state, player_action, ai_action)

        # Update AI with what happened
        ai.update(ai_action, player_action)

        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] Round {round_num}: You={ACTION_NAMES[player_action]} vs AI={ACTION_NAMES[ai_action]}")

        # === RESULT DISPLAY PHASE ===
        color = pose_colors.get(pose, (255, 255, 255))
        result_start = time.time()

        while time.time() - result_start < pause_after_capture:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            draw_game_state(frame, state, round_num)

            # Show both moves
            draw_centered_text(frame, f"You: {ACTION_NAMES[player_action]}", y_offset=-80, font_scale=1.5, color=color)
            draw_centered_text(frame, "vs", y_offset=-20, font_scale=1, color=(150, 150, 150))
            draw_centered_text(frame, f"AI: {ACTION_NAMES[ai_action]}", y_offset=40, font_scale=1.5, color=(100, 100, 255))

            cv2.imshow("007 Pose Game", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                return None

        state = new_state

    # === GAME OVER ===
    if state.winner == 0:
        result_text = "DRAW!"
        result_color = (0, 255, 255)
        result = 0
    elif state.winner == 1:
        result_text = "YOU WIN!"
        result_color = (0, 255, 0)
        result = 1
    else:
        result_text = "AI WINS!"
        result_color = (0, 0, 255)
        result = -1

    # Show game over for 3 seconds
    result_start = time.time()
    while time.time() - result_start < 3:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame = (frame * 0.4).astype('uint8')

        draw_centered_text(frame, result_text, y_offset=-50, font_scale=3, color=result_color)
        draw_centered_text(frame, f"Game ended in {round_num} rounds", y_offset=50, font_scale=1, color=(200, 200, 200))

        cv2.imshow("007 Pose Game", frame)
        cv2.waitKey(1)

    return result


def main():
    # Get model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "pose_landmarker_lite.task")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Initialize pose landmarker
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize pose classifier
    classifier = PoseClassifier(os.path.join(script_dir, "pose_model.pt"))

    # Initialize learning AI
    ai = LearningAI()

    print("007 Pose Game")
    print("AI uses pattern recognition to learn your moves!")

    # Track score
    wins, losses, draws = 0, 0, 0

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            choice = show_menu(cap, classifier)

            if choice is None:  # Quit
                break
            elif choice == 'start':
                if not classifier.is_ready():
                    # Show warning briefly
                    for _ in range(60):
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.flip(frame, 1)
                            frame = (frame * 0.3).astype('uint8')
                            draw_centered_text(frame, "Calibrate first!", font_scale=1.5, color=(0, 0, 255))
                            cv2.imshow("007 Pose Game", frame)
                            cv2.waitKey(33)
                else:
                    result = run_game(cap, landmarker, classifier, ai)
                    if result == 1:
                        wins += 1
                    elif result == -1:
                        losses += 1
                    elif result == 0:
                        draws += 1
                    print(f"Score: You {wins} - {losses} AI (Draws: {draws})")
            elif choice == 'calibrate':
                run_calibration(cap, landmarker, classifier)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

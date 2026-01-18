#!/usr/bin/env python3
"""Pose collection with visual countdown GUI."""

import cv2
import json
import mediapipe as mp
import numpy as np
import os
import sounddevice as sd
import time
from dataclasses import dataclass
from typing import Optional

from pose_classifier import PoseClassifier, POSE_LABELS
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from rl_opponent import RLOpponent


# Settings #######################################

DEFAULT_SETTINGS = {
    "show_bullets": False,
}

def get_settings_path():
    """Get path to settings file in same directory as script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "settings.json")

def load_settings():
    """Load settings from JSON file, return defaults if not found."""
    settings_path = get_settings_path()
    try:
        with open(settings_path, 'r') as f:
            saved = json.load(f)
            # Merge with defaults to handle new settings
            return {**DEFAULT_SETTINGS, **saved}
    except (FileNotFoundError, json.JSONDecodeError):
        return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    """Save settings to JSON file."""
    settings_path = get_settings_path()
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=2)


# Sound Effects ##################################

SAMPLE_RATE = 44100

def play_tone(frequency: float, duration: float = 0.15):
    """Play a sine wave tone (non-blocking)."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    # Sine wave with envelope to avoid clicks
    envelope = np.exp(-t * 8)  # Decay envelope
    tone = np.sin(2 * np.pi * frequency * t) * envelope * 0.5
    sd.play(tone.astype(np.float32), SAMPLE_RATE)

def play_countdown_tone():
    """Play tone for countdown numbers (2, 1)."""
    play_tone(440, 0.12)  # A4 note

def play_pose_tone():
    """Play tone for POSE! moment."""
    play_tone(880, 0.2)  # A5 note (higher pitch)

def play_win_sound():
    """Play ascending victory jingle."""
    for freq in [523, 659, 784, 1047]:  # C5, E5, G5, C6 (major arpeggio)
        play_tone(freq, 0.15)
        sd.wait()

def play_lose_sound():
    """Play descending defeat sound."""
    for freq in [392, 349, 311, 262]:  # G4, F4, Eb4, C4 (descending)
        play_tone(freq, 0.2)
        sd.wait()

def play_draw_sound():
    """Play neutral draw sound."""
    play_tone(440, 0.3)
    sd.wait()
    play_tone(440, 0.3)

def play_compare_sound():
    """Play sound when comparing moves."""
    play_tone(660, 0.1)  # E5 - quick reveal sound


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
    """Show main menu. Returns 'start', 'settings', or None for quit."""
    selected = 0  # 0 = Start, 1 = Settings, 2 = Quit
    num_options = 3

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
        cv2.putText(frame, status, (w//2 - 120, h//2 - 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # Buttons
        draw_button(frame, "START", h//2 - 80, selected == 0)
        draw_button(frame, "SETTINGS", h//2 + 20, selected == 1)
        draw_button(frame, "QUIT", h//2 + 120, selected == 2)

        # Instructions
        cv2.putText(frame, "UP/DOWN to select, ENTER to confirm",
                   (w//2 - 220, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)

        cv2.imshow("007 Pose Game", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return None
        elif key in [ord('w'), ord('W'), 0]:  # W or Up arrow
            selected = (selected - 1) % num_options
        elif key in [ord('s'), ord('S'), 1]:  # S or Down arrow
            selected = (selected + 1) % num_options
        elif key in [13, 10]:  # Enter
            if selected == 0:
                return 'start'
            elif selected == 1:
                return 'settings'
            else:
                return None  # Quit


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


def show_settings(cap, settings, ai, landmarker, classifier):
    """Show settings menu. Modifies settings dict in place."""
    selected = 0  # 0 = Show Bullets, 1 = Calibrate, 2 = Reset AI, 3 = Back
    num_options = 4
    confirm_reset = False  # Two-step confirmation for reset

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        # Darken background
        frame = (frame * 0.3).astype('uint8')

        # Title
        draw_centered_text(frame, "SETTINGS", y_offset=-180, font_scale=2, color=(0, 255, 255))

        h, w = frame.shape[:2]

        # Show Bullets toggle
        show_bullets_text = "Show Bullets: " + ("ON" if settings["show_bullets"] else "OFF")
        draw_button(frame, show_bullets_text, h//2 - 110, selected == 0)

        # Calibrate button
        draw_button(frame, "CALIBRATE", h//2 - 10, selected == 1)

        # Reset AI button
        reset_text = "CONFIRM RESET?" if confirm_reset else "Reset AI"
        draw_button(frame, reset_text, h//2 + 90, selected == 2)

        # Back button
        draw_button(frame, "BACK", h//2 + 190, selected == 3)

        # Instructions
        cv2.putText(frame, "UP/DOWN to select, ENTER to toggle/confirm",
                   (w//2 - 250, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)

        cv2.imshow("007 Pose Game", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Q or ESC
            return
        elif key in [ord('w'), ord('W'), 0]:  # W or Up arrow
            selected = (selected - 1) % num_options
            confirm_reset = False  # Cancel confirmation on navigation
        elif key in [ord('s'), ord('S'), 1]:  # S or Down arrow
            selected = (selected + 1) % num_options
            confirm_reset = False  # Cancel confirmation on navigation
        elif key in [13, 10]:  # Enter
            if selected == 0:
                # Toggle show_bullets
                settings["show_bullets"] = not settings["show_bullets"]
                save_settings(settings)
            elif selected == 1:
                # Calibrate
                run_calibration(cap, landmarker, classifier)
            elif selected == 2:
                # Reset AI (two-step confirmation)
                if confirm_reset:
                    ai.reset_all()
                    confirm_reset = False
                    # Show confirmation message briefly
                    for _ in range(45):  # ~1.5 seconds
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.flip(frame, 1)
                            frame = (frame * 0.3).astype('uint8')
                            draw_centered_text(frame, "AI Reset!", font_scale=1.5, color=(0, 255, 0))
                            cv2.imshow("007 Pose Game", frame)
                            cv2.waitKey(33)
                else:
                    confirm_reset = True
            else:
                # Back
                return


def draw_game_state(frame, state, round_num, show_bullets=True):
    """Draw bullet counts and round number."""
    _, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    if show_bullets:
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


def run_game(cap, landmarker, classifier, ai, settings):
    """Run the pose collection game against AI. Returns when game ends."""
    countdown_seconds = 2
    pause_after_capture = 2
    show_bullets = settings.get("show_bullets", False)

    pose_colors = {
        "SHOOT": (0, 0, 255),      # Red
        "SHIELD": (255, 165, 0),   # Orange
        "RELOAD": (0, 255, 255),   # Yellow
        "NEUTRAL": (128, 128, 128),
        "NO POSE": (100, 100, 100),
    }

    # Reset AI for new game
    ai.reset_game()

    state = GameState()
    round_num = 0

    while not state.is_terminal:
        round_num += 1

        # === COUNTDOWN PHASE ===
        countdown_start = time.time()
        last_display = None  # Track what's being shown to trigger sounds
        while True:
            elapsed = time.time() - countdown_start
            remaining = countdown_seconds - elapsed

            if remaining <= 0:
                break

            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            draw_game_state(frame, state, round_num, show_bullets)

            # Draw countdown number and play sounds on transitions
            num = int(remaining) + 1
            if remaining < 0.5:
                current_display = "POSE"
                if last_display != current_display:
                    play_pose_tone()
                    last_display = current_display
                draw_centered_text(frame, "POSE!", font_scale=4, color=(0, 255, 0))
            else:
                current_display = num
                if last_display != current_display:
                    play_countdown_tone()
                    last_display = current_display
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
                    pose = classifier.predict(landmarks, debug=True)
                    player_action = pose_to_action(pose, state.p1_bullets)
                else:
                    pose = "NO MODEL"
                    return None

            # Show waiting state
            draw_game_state(frame, state, round_num, show_bullets)
            draw_centered_text(frame, "POSE!", font_scale=4, color=(0, 255, 0))
            if pose != "NEUTRAL" and player_action is None:
                draw_centered_text(frame, "No bullets!", y_offset=100, font_scale=1, color=(0, 0, 255))
            cv2.imshow("007 Pose Game", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                return None

        # Get AI action
        ai_action = ai.get_action(state.p2_bullets, state.p1_bullets)

        # Execute turn
        new_state = step(state, player_action, ai_action)

        # Update AI with what happened
        ai.update(ai_action, player_action)

        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] Round {round_num}: You={ACTION_NAMES[player_action]} vs AI={ACTION_NAMES[ai_action]}")

        # === RESULT DISPLAY PHASE ===
        play_compare_sound()
        color = pose_colors.get(pose, (255, 255, 255))
        result_start = time.time()

        while time.time() - result_start < pause_after_capture:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            draw_game_state(frame, state, round_num, show_bullets)

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
        play_draw_sound()
    elif state.winner == 1:
        result_text = "YOU WIN!"
        result_color = (0, 255, 0)
        result = 1
        play_win_sound()
    else:
        result_text = "AI WINS!"
        result_color = (0, 0, 255)
        result = -1
        play_lose_sound()

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

    # Initialize RL opponent
    ai = RLOpponent(os.path.join(script_dir, "opponent_model.pt"))

    # Load settings
    settings = load_settings()

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
                    result = run_game(cap, landmarker, classifier, ai, settings)
                    if result == 1:
                        wins += 1
                    elif result == -1:
                        losses += 1
                    elif result == 0:
                        draws += 1
                    print(f"Score: You {wins} - {losses} AI (Draws: {draws})")
            elif choice == 'settings':
                show_settings(cap, settings, ai, landmarker, classifier)

    ai.save()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

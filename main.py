#!/usr/bin/env python3
"""007 Pose Detector - Simple and dumb pose detection for the 007 schoolyard game."""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

from pose_classifier import PoseClassifier, POSE_LABELS
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class CalibrationState:
    """Manages calibration mode state."""

    def __init__(self):
        self.active = False
        self.current_pose_idx = 0
        self.recording = False
        self.record_start_time = 0
        self.record_duration = 3.0  # seconds

    def start(self):
        """Enter calibration mode."""
        self.active = True
        self.current_pose_idx = 0
        self.recording = False

    def cancel(self):
        """Exit calibration mode."""
        self.active = False
        self.recording = False

    def start_recording(self):
        """Start recording samples for current pose."""
        self.recording = True
        self.record_start_time = time.time()

    def get_current_pose(self) -> str:
        """Get the pose currently being calibrated."""
        return POSE_LABELS[self.current_pose_idx]

    def next_pose(self) -> bool:
        """Move to next pose. Returns False if all poses done."""
        self.current_pose_idx += 1
        self.recording = False
        return self.current_pose_idx < len(POSE_LABELS)

    def get_recording_elapsed(self) -> float:
        """Get seconds elapsed since recording started."""
        return time.time() - self.record_start_time

    def is_recording_done(self) -> bool:
        """Check if recording duration has elapsed."""
        return self.get_recording_elapsed() >= self.record_duration


# Landmark indices (same as legacy API)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

# Pose connections for drawing
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
    (11, 23), (12, 24), (23, 24),  # Torso
    (23, 25), (25, 27), (24, 26), (26, 28),  # Legs
]


def get_landmark(landmarks, idx):
    """Extract x, y, z, visibility from landmark."""
    lm = landmarks[idx]
    return lm.x, lm.y, lm.z, lm.visibility


def draw_landmarks(image, landmarks):
    """Draw pose landmarks and connections on image."""
    h, w = image.shape[:2]

    # Draw connections
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            if start.visibility > 0.5 and end.visibility > 0.5:
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv2.line(image, start_point, end_point, (0, 255, 0), 2)

    # Draw landmarks
    for lm in landmarks:
        if lm.visibility > 0.5:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)


def draw_pose_text(image, pose, color):
    """Draw large pose text on the image."""
    h, w = image.shape[:2]

    symbols = {
        "SHOOT": "SHOOT",
        "SHIELD": "SHIELD",
        "RELOAD": "RELOAD",
        "NEUTRAL": "..."
    }

    text = symbols.get(pose, pose)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5
    thickness = 5

    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    x = (w - text_w) // 2
    y = h - 50

    cv2.putText(image, text, (x + 3, y + 3), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness)


def draw_calibration_overlay(image, calibration, classifier):
    """Draw calibration mode UI overlay."""
    h, w = image.shape[:2]

    # Semi-transparent overlay
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title
    cv2.putText(image, "CALIBRATION MODE", (w // 2 - 200, 60),
                font, 1.5, (0, 255, 255), 3)

    pose = calibration.get_current_pose()

    if calibration.recording:
        # Show recording countdown
        elapsed = calibration.get_recording_elapsed()
        remaining = max(0, calibration.record_duration - elapsed)

        cv2.putText(image, f"Recording {pose}...", (w // 2 - 150, h // 2 - 50),
                    font, 1.2, (0, 255, 0), 2)
        cv2.putText(image, f"{remaining:.1f}s", (w // 2 - 50, h // 2 + 50),
                    font, 2.0, (0, 255, 0), 3)

        samples = classifier.get_sample_count(pose)
        cv2.putText(image, f"Samples: {samples}", (w // 2 - 80, h // 2 + 120),
                    font, 0.8, (255, 255, 255), 2)
    else:
        # Show prompt
        cv2.putText(image, f"Hold {pose} pose", (w // 2 - 150, h // 2 - 50),
                    font, 1.2, (255, 255, 255), 2)
        cv2.putText(image, "Press SPACE to start recording", (w // 2 - 220, h // 2 + 50),
                    font, 0.9, (200, 200, 200), 2)

    # Instructions
    cv2.putText(image, "[ESC] Cancel", (10, h - 20),
                font, 0.6, (150, 150, 150), 1)


def main():
    # Get model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "pose_landmarker_lite.task")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please download the model first.")
        return

    # Initialize pose landmarker
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Give camera time to warm up
    print("Warming up camera...")
    time.sleep(1.0)
    # Discard first few frames
    for _ in range(5):
        cap.read()

    colors = {
        "SHOOT": (0, 0, 255),      # Red
        "SHIELD": (255, 165, 0),   # Orange
        "RELOAD": (0, 255, 255),   # Yellow
        "NEUTRAL": (128, 128, 128) # Gray
    }

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        print("007 Pose Detector")
        print("Press Q to quit")
        print("-" * 30)

        frame_timestamp_ms = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Skip bad frames instead of crashing
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect pose
            frame_timestamp_ms += 33  # ~30 fps
            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            current_pose = "NEUTRAL"

            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                landmarks = results.pose_landmarks[0]
                draw_landmarks(frame, landmarks)
                current_pose = classify_pose(landmarks)
                if current_pose != "NEUTRAL":
                    print({
                        "current_pose": current_pose,
                        "landmarks": landmarks,
                    })
            else:
                current_pose = "NO PLAYER"

            color = colors.get(current_pose, (128, 128, 128))
            draw_pose_text(frame, current_pose, color)

            cv2.putText(frame, "[Q] Quit", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("007 Pose Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

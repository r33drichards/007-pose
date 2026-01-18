#!/usr/bin/env python3
"""007 Pose Detector - Simple and dumb pose detection for the 007 schoolyard game."""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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


def classify_pose(landmarks):
    """Classify pose based on keypoint positions.

    SHOOT: Wrists forward (z-depth ahead of shoulders), arms extended
    SHIELD: X arms - wrists crossed in front of chest
    RELOAD: Guns to sky - wrists above elbows, elbows bent and down
    """
    # Extract relevant landmarks
    l_shoulder = get_landmark(landmarks, LEFT_SHOULDER)
    r_shoulder = get_landmark(landmarks, RIGHT_SHOULDER)
    l_elbow = get_landmark(landmarks, LEFT_ELBOW)
    r_elbow = get_landmark(landmarks, RIGHT_ELBOW)
    l_wrist = get_landmark(landmarks, LEFT_WRIST)
    r_wrist = get_landmark(landmarks, RIGHT_WRIST)

    # Check visibility - need decent confidence on key points
    min_visibility = 0.5
    if (l_wrist[3] < min_visibility or r_wrist[3] < min_visibility or
        l_elbow[3] < min_visibility or r_elbow[3] < min_visibility):
        return "NEUTRAL"

    # Calculate body center x (for crossing detection)
    body_center_x = (l_shoulder[0] + r_shoulder[0]) / 2

    # Shoulder width for relative measurements
    shoulder_width = abs(r_shoulder[0] - l_shoulder[0])

    # === SHIELD: X arms - wrists crossed ===
    # Left wrist should be on the right side, right wrist on the left side
    left_wrist_crossed = l_wrist[0] > body_center_x
    right_wrist_crossed = r_wrist[0] < body_center_x

    # Wrists should be in front of chest (between shoulders vertically)
    chest_top = min(l_shoulder[1], r_shoulder[1])
    chest_bottom = (l_shoulder[1] + r_shoulder[1]) / 2 + shoulder_width
    wrists_at_chest = (chest_top - 0.1 < l_wrist[1] < chest_bottom + 0.1 and
                       chest_top - 0.1 < r_wrist[1] < chest_bottom + 0.1)

    if left_wrist_crossed and right_wrist_crossed and wrists_at_chest:
        return "SHIELD"

    # === RELOAD: Guns to sky - wrists above elbows, elbows down ===
    # Wrists should be above elbows (lower y value = higher on screen)
    left_wrist_above_elbow = l_wrist[1] < l_elbow[1]
    right_wrist_above_elbow = r_wrist[1] < r_elbow[1]

    # Elbows should be relatively low (near or below shoulders)
    left_elbow_down = l_elbow[1] > l_shoulder[1] - 0.05
    right_elbow_down = r_elbow[1] > r_shoulder[1] - 0.05

    # Wrists should be near shoulder height or above
    wrists_up = l_wrist[1] < l_shoulder[1] + 0.1 and r_wrist[1] < r_shoulder[1] + 0.1

    if (left_wrist_above_elbow and right_wrist_above_elbow and
        left_elbow_down and right_elbow_down and wrists_up):
        return "RELOAD"

    # === SHOOT: Arms extended forward ===
    # Wrists should be forward (more negative z = closer to camera)
    # In MediaPipe, z is depth relative to hips, negative = closer to camera
    z_threshold = -0.15  # Wrists should be notably in front
    left_wrist_forward = l_wrist[2] < l_shoulder[2] + z_threshold
    right_wrist_forward = r_wrist[2] < r_shoulder[2] + z_threshold

    # Arms should be somewhat extended (wrists away from shoulders horizontally)
    # and at roughly shoulder height
    wrists_extended = (abs(l_wrist[0] - l_shoulder[0]) > shoulder_width * 0.3 or
                       abs(r_wrist[0] - r_shoulder[0]) > shoulder_width * 0.3)
    wrists_at_shoulder_height = (abs(l_wrist[1] - l_shoulder[1]) < 0.2 and
                                  abs(r_wrist[1] - r_shoulder[1]) < 0.2)

    if left_wrist_forward and right_wrist_forward and wrists_at_shoulder_height:
        return "SHOOT"

    return "NEUTRAL"


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

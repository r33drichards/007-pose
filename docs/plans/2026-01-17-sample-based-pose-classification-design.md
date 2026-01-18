# Sample-Based Pose Classification Design

## Overview

Replace heuristic-based pose classification with a sample-based approach where users demonstrate each pose and a neural network learns to classify them.

## Requirements

- User provides examples of each pose (SHOOT, SHIELD, RELOAD) via calibration mode
- Continuous recording (~3 seconds per pose) captures natural variation
- PyTorch MLP classifier trained on collected samples
- Model persists to disk for reuse across sessions
- No heuristic fallback - NEUTRAL shown until calibrated

## Architecture

### Calibration Flow

1. User presses `C` to enter calibration mode
2. For each pose (SHOOT, SHIELD, RELOAD):
   - Display prompt: "Hold [POSE] and press SPACE to start"
   - User presses SPACE, record for 3 seconds
   - Display confirmation with sample count
3. Train MLP on collected samples
4. Save model to `pose_model.pt`
5. Return to detection mode

Press `ESC` to cancel calibration.

### Feature Extraction

8 landmarks tracked: shoulders, elbows, wrists, hips.

Normalization per frame:
1. Compute body center (midpoint of shoulders)
2. Compute shoulder width (distance between shoulders)
3. Normalize each landmark: subtract body center, divide by shoulder width
4. Flatten to 24 features (8 landmarks x 3 coords)

Frames with low-confidence landmarks (visibility < 0.5) are skipped.

### Neural Network

```
Input (24 features)
  -> Linear(24, 64) + ReLU
  -> Linear(64, 32) + ReLU
  -> Linear(32, 3)
  -> Softmax
```

Training:
- ~60-90 frames per pose (~180-270 total samples)
- CrossEntropyLoss, Adam optimizer
- 100-200 epochs
- Confidence threshold: softmax < 0.7 outputs NEUTRAL

### Classification

If `pose_model.pt` exists:
- Load model, use for classification
- Output NEUTRAL if confidence below threshold

If no model:
- Output NEUTRAL for all poses until calibrated

## File Structure

```
007-pose/
├── main.py              # UI/camera loop, calibration mode trigger
├── pose_classifier.py   # PoseClassifier class (features, MLP, train/predict)
├── pose_model.pt        # Saved model (generated after calibration)
└── models/
    └── pose_landmarker_lite.task
```

## UI Changes

- Add `[C] Calibrate` hint to display
- Calibration mode overlay with prompts and countdown
- Training progress indicator

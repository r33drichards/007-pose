# Sample-Based Pose Classification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace heuristic pose classification with a PyTorch MLP trained on user-provided samples.

**Architecture:** A separate `PoseClassifier` class handles feature extraction, training, and inference. Main loop triggers calibration mode on keypress, collects samples, trains model, saves to disk. On startup, loads existing model if available.

**Tech Stack:** Python, PyTorch, MediaPipe, OpenCV

---

### Task 1: Create PoseClassifier with Feature Extraction

**Files:**
- Create: `pose_classifier.py`

**Step 1: Create the PoseClassifier class with constants and init**

```python
#!/usr/bin/env python3
"""Pose classifier using PyTorch MLP trained on user samples."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

LANDMARK_INDICES = [
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
]

POSE_LABELS = ["SHOOT", "SHIELD", "RELOAD"]
NUM_FEATURES = len(LANDMARK_INDICES) * 3  # 8 landmarks * 3 coords (x, y, z) = 24


class PoseMLP(nn.Module):
    """Simple MLP for pose classification."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(NUM_FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(POSE_LABELS)),
        )

    def forward(self, x):
        return self.layers(x)


class PoseClassifier:
    """Manages pose classification with user-trained MLP."""

    def __init__(self, model_path: str = "pose_model.pt"):
        self.model_path = Path(model_path)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = 0.7
        self.min_visibility = 0.5

        # Training data storage
        self.training_samples = []  # List of (features, label_idx)

        # Try to load existing model
        self.load_model()

    def load_model(self) -> bool:
        """Load model from disk if exists. Returns True if loaded."""
        if self.model_path.exists():
            self.model = PoseMLP().to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            return True
        return False

    def save_model(self):
        """Save trained model to disk."""
        if self.model is not None:
            torch.save(self.model.state_dict(), self.model_path)
```

**Step 2: Run Python syntax check**

Run: `python -m py_compile pose_classifier.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add pose_classifier.py
git commit -m "feat: add PoseClassifier skeleton with MLP architecture"
```

---

### Task 2: Add Feature Extraction to PoseClassifier

**Files:**
- Modify: `pose_classifier.py`

**Step 1: Add extract_features method**

Add after `save_model` method:

```python
    def extract_features(self, landmarks) -> np.ndarray | None:
        """Extract normalized features from landmarks.

        Returns None if landmarks have insufficient visibility.
        """
        # Check visibility of key landmarks
        for idx in LANDMARK_INDICES:
            if landmarks[idx].visibility < self.min_visibility:
                return None

        # Get shoulder positions for normalization
        l_shoulder = landmarks[LEFT_SHOULDER]
        r_shoulder = landmarks[RIGHT_SHOULDER]

        # Body center (midpoint of shoulders)
        center_x = (l_shoulder.x + r_shoulder.x) / 2
        center_y = (l_shoulder.y + r_shoulder.y) / 2
        center_z = (l_shoulder.z + r_shoulder.z) / 2

        # Shoulder width for scale normalization
        shoulder_width = np.sqrt(
            (r_shoulder.x - l_shoulder.x) ** 2 +
            (r_shoulder.y - l_shoulder.y) ** 2
        )

        if shoulder_width < 0.01:  # Too small, invalid
            return None

        # Extract and normalize features
        features = []
        for idx in LANDMARK_INDICES:
            lm = landmarks[idx]
            features.extend([
                (lm.x - center_x) / shoulder_width,
                (lm.y - center_y) / shoulder_width,
                (lm.z - center_z) / shoulder_width,
            ])

        return np.array(features, dtype=np.float32)
```

**Step 2: Run Python syntax check**

Run: `python -m py_compile pose_classifier.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add pose_classifier.py
git commit -m "feat: add feature extraction with body-relative normalization"
```

---

### Task 3: Add Training Data Collection Methods

**Files:**
- Modify: `pose_classifier.py`

**Step 1: Add sample collection methods**

Add after `extract_features` method:

```python
    def clear_training_data(self):
        """Clear all collected training samples."""
        self.training_samples = []

    def add_sample(self, landmarks, pose_label: str) -> bool:
        """Add a training sample. Returns True if sample was valid and added."""
        features = self.extract_features(landmarks)
        if features is None:
            return False

        label_idx = POSE_LABELS.index(pose_label)
        self.training_samples.append((features, label_idx))
        return True

    def get_sample_count(self, pose_label: str = None) -> int:
        """Get count of samples, optionally filtered by pose."""
        if pose_label is None:
            return len(self.training_samples)

        label_idx = POSE_LABELS.index(pose_label)
        return sum(1 for _, idx in self.training_samples if idx == label_idx)
```

**Step 2: Run Python syntax check**

Run: `python -m py_compile pose_classifier.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add pose_classifier.py
git commit -m "feat: add training sample collection methods"
```

---

### Task 4: Add Model Training

**Files:**
- Modify: `pose_classifier.py`

**Step 1: Add train method**

Add after `get_sample_count` method:

```python
    def train(self, epochs: int = 200, lr: float = 0.001) -> bool:
        """Train the model on collected samples. Returns True if successful."""
        if len(self.training_samples) < 10:
            return False

        # Prepare data
        X = np.array([s[0] for s in self.training_samples])
        y = np.array([s[1] for s in self.training_samples])

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        # Create and train model
        self.model = PoseMLP().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        self.model.eval()
        self.save_model()
        return True
```

**Step 2: Run Python syntax check**

Run: `python -m py_compile pose_classifier.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add pose_classifier.py
git commit -m "feat: add model training with PyTorch"
```

---

### Task 5: Add Prediction Method

**Files:**
- Modify: `pose_classifier.py`

**Step 1: Add predict method**

Add after `train` method:

```python
    def predict(self, landmarks) -> str:
        """Predict pose from landmarks. Returns pose label or 'NEUTRAL'."""
        if self.model is None:
            return "NEUTRAL"

        features = self.extract_features(landmarks)
        if features is None:
            return "NEUTRAL"

        with torch.no_grad():
            X = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            outputs = self.model(X)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

            if confidence.item() < self.confidence_threshold:
                return "NEUTRAL"

            return POSE_LABELS[predicted.item()]

    def is_ready(self) -> bool:
        """Check if classifier has a trained model."""
        return self.model is not None
```

**Step 2: Run Python syntax check**

Run: `python -m py_compile pose_classifier.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add pose_classifier.py
git commit -m "feat: add pose prediction with confidence threshold"
```

---

### Task 6: Update main.py - Add Imports and Classifier Init

**Files:**
- Modify: `main.py`

**Step 1: Add import for PoseClassifier**

After line 8 (`import time`), add:

```python
from pose_classifier import PoseClassifier, POSE_LABELS
```

**Step 2: Remove old classify_pose function**

Delete the entire `classify_pose` function (lines 37-111).

**Step 3: Run Python syntax check**

Run: `python -m py_compile main.py`
Expected: Error about classify_pose not defined (expected, we'll fix in next task)

**Step 4: Commit**

```bash
git add main.py
git commit -m "refactor: remove heuristic classify_pose, add PoseClassifier import"
```

---

### Task 7: Add Calibration State to main.py

**Files:**
- Modify: `main.py`

**Step 1: Add calibration state class**

After the imports, before `get_landmark` function, add:

```python
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
```

**Step 2: Run Python syntax check**

Run: `python -m py_compile main.py`
Expected: Error (still missing classify_pose usage fix)

**Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add CalibrationState class for managing calibration flow"
```

---

### Task 8: Add Calibration UI Drawing Functions

**Files:**
- Modify: `main.py`

**Step 1: Add calibration overlay function**

After `draw_pose_text` function, add:

```python
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
```

**Step 2: Run Python syntax check**

Run: `python -m py_compile main.py`
Expected: Still has error (expected)

**Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add calibration overlay drawing function"
```

---

### Task 9: Update Main Loop - Initialize Classifier and Calibration

**Files:**
- Modify: `main.py`

**Step 1: Update main() to initialize classifier and calibration state**

In the `main()` function, after the `colors` dictionary (around line 201), add:

```python
    # Initialize pose classifier and calibration state
    classifier = PoseClassifier(os.path.join(script_dir, "pose_model.pt"))
    calibration = CalibrationState()

    if classifier.is_ready():
        print("Loaded trained pose model")
    else:
        print("No trained model - press C to calibrate")
```

**Step 2: Update the print instructions**

Change the print statements around line 204-206:

```python
        print("007 Pose Detector")
        print("Press C to calibrate, Q to quit")
        print("-" * 30)
```

**Step 3: Run Python syntax check**

Run: `python -m py_compile main.py`
Expected: Still has error about classify_pose

**Step 4: Commit**

```bash
git add main.py
git commit -m "feat: initialize PoseClassifier and CalibrationState in main"
```

---

### Task 10: Update Main Loop - Handle Calibration Mode

**Files:**
- Modify: `main.py`

**Step 1: Replace the pose detection and display logic in the main loop**

Find the section starting with `current_pose = "NEUTRAL"` (around line 226) and replace everything up to and including `draw_pose_text(frame, current_pose, color)` with:

```python
            current_pose = "NEUTRAL"
            landmarks = None

            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                landmarks = results.pose_landmarks[0]
                draw_landmarks(frame, landmarks)

            if calibration.active:
                # Calibration mode
                if calibration.recording and landmarks:
                    classifier.add_sample(landmarks, calibration.get_current_pose())

                    if calibration.is_recording_done():
                        if calibration.next_pose():
                            # More poses to record
                            pass
                        else:
                            # All poses recorded, train model
                            print("Training model...")
                            if classifier.train():
                                print("Calibration complete!")
                            else:
                                print("Training failed - not enough samples")
                            calibration.cancel()

                draw_calibration_overlay(frame, calibration, classifier)
            else:
                # Normal detection mode
                if landmarks:
                    current_pose = classifier.predict(landmarks)
                else:
                    current_pose = "NO PLAYER"

                color = colors.get(current_pose, (128, 128, 128))
                draw_pose_text(frame, current_pose, color)
```

**Step 2: Run Python syntax check**

Run: `python -m py_compile main.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add calibration mode handling in main loop"
```

---

### Task 11: Update Main Loop - Handle Key Inputs

**Files:**
- Modify: `main.py`

**Step 1: Update UI hints**

Replace the `cv2.putText(frame, "[Q] Quit"...` line with:

```python
            if not calibration.active:
                cv2.putText(frame, "[C] Calibrate  [Q] Quit", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
```

**Step 2: Update key handling**

Replace the key handling at the end of the loop:

```python
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c') and not calibration.active:
                calibration.start()
                classifier.clear_training_data()
                print("Entering calibration mode...")
            elif key == ord(' ') and calibration.active and not calibration.recording:
                calibration.start_recording()
                print(f"Recording {calibration.get_current_pose()}...")
            elif key == 27 and calibration.active:  # ESC
                calibration.cancel()
                print("Calibration cancelled")
```

**Step 3: Run Python syntax check**

Run: `python -m py_compile main.py`
Expected: No output (success)

**Step 4: Commit**

```bash
git add main.py
git commit -m "feat: add keyboard handling for calibration controls"
```

---

### Task 12: Clean Up - Remove Unused Code

**Files:**
- Modify: `main.py`

**Step 1: Remove unused imports and constants**

Remove these lines from the top of main.py (they're now in pose_classifier.py):
- The landmark index constants (LEFT_SHOULDER through RIGHT_HIP, lines 14-21)
- The `get_landmark` function (if not used elsewhere)

Keep the POSE_CONNECTIONS constant as it's used for drawing.

**Step 2: Verify the get_landmark function**

Check if `get_landmark` is used anywhere. If not, delete it. Search the file for "get_landmark".

**Step 3: Run the application to verify**

Run: `python main.py`
Expected: Application starts, shows "No trained model - press C to calibrate"

**Step 4: Test calibration flow**

1. Press C to enter calibration
2. Hold SHOOT pose, press SPACE
3. Wait 3 seconds
4. Repeat for SHIELD and RELOAD
5. Verify model trains and saves

**Step 5: Commit**

```bash
git add main.py
git commit -m "refactor: clean up unused constants and functions"
```

---

### Task 13: Final Testing and Verification

**Files:**
- All files

**Step 1: Verify pose_model.pt is created after calibration**

Run: `ls -la pose_model.pt`
Expected: File exists with recent timestamp

**Step 2: Restart application and verify model loads**

Run: `python main.py`
Expected: Shows "Loaded trained pose model"

**Step 3: Test all three poses work correctly**

Verify SHOOT, SHIELD, RELOAD are detected based on your calibrated samples.

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete sample-based pose classification implementation"
```

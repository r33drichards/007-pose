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
NUM_FEATURES = len(LANDMARK_INDICES) * 4  # 8 landmarks * (3 coords + 1 presence flag) = 32


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
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
            self.model.eval()
            return True
        return False

    def save_model(self):
        """Save trained model to disk."""
        if self.model is not None:
            torch.save(self.model.state_dict(), self.model_path)

    def extract_features(self, landmarks) -> np.ndarray | None:
        """Extract normalized features from landmarks.

        Returns features with presence flags for each landmark.
        Missing landmarks (low visibility) get zeros for coords and 0.0 presence flag.
        Returns None only if shoulders aren't visible (can't normalize).
        """
        # Get shoulder positions for normalization
        l_shoulder = landmarks[LEFT_SHOULDER]
        r_shoulder = landmarks[RIGHT_SHOULDER]

        # Need at least shoulders visible to normalize
        if l_shoulder.visibility < self.min_visibility or r_shoulder.visibility < self.min_visibility:
            return None

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

        # Extract and normalize features with presence flags
        features = []
        for idx in LANDMARK_INDICES:
            lm = landmarks[idx]
            if lm.visibility >= self.min_visibility:
                # Landmark is present
                features.extend([
                    (lm.x - center_x) / shoulder_width,
                    (lm.y - center_y) / shoulder_width,
                    (lm.z - center_z) / shoulder_width,
                    1.0,  # presence flag
                ])
            else:
                # Landmark is missing - use zeros
                features.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)

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
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
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
            self.model.eval()
            return True
        return False

    def save_model(self):
        """Save trained model to disk."""
        if self.model is not None:
            torch.save(self.model.state_dict(), self.model_path)

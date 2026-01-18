#!/usr/bin/env python3
"""Tests for CalibrationState."""

import pytest
import sys
import os

# Add project root to path so we can import from main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import CalibrationState
from pose_classifier import POSE_LABELS


class TestCalibrationState:
    """Tests for CalibrationState class."""

    def test_get_current_pose_after_cancel_when_all_poses_done(self):
        """Reproduce bug: IndexError when calling get_current_pose after cancel when all poses recorded.

        Bug scenario:
        1. Start calibration
        2. Record all poses (next_pose() called until it returns False)
        3. Call cancel() (which doesn't reset current_pose_idx)
        4. Call get_current_pose() -> IndexError: list index out of range
        """
        calibration = CalibrationState()
        calibration.start()

        # Simulate recording all poses
        while calibration.next_pose():
            pass

        # At this point current_pose_idx is beyond POSE_LABELS length
        assert calibration.current_pose_idx == len(POSE_LABELS)

        # Cancel calibration (this is what happens after training completes/fails)
        calibration.cancel()

        # This should NOT raise IndexError
        # Bug: currently raises IndexError because cancel() doesn't reset current_pose_idx
        pose = calibration.get_current_pose()
        assert pose in POSE_LABELS

    def test_cancel_resets_state_completely(self):
        """Cancel should reset all state including current_pose_idx."""
        calibration = CalibrationState()
        calibration.start()

        # Advance through some poses
        calibration.next_pose()
        calibration.next_pose()
        calibration.start_recording()

        calibration.cancel()

        assert calibration.active is False
        assert calibration.recording is False
        assert calibration.current_pose_idx == 0

    def test_start_resets_state(self):
        """Start should reset all state."""
        calibration = CalibrationState()

        # Mess up state
        calibration.current_pose_idx = 5
        calibration.recording = True

        calibration.start()

        assert calibration.active is True
        assert calibration.current_pose_idx == 0
        assert calibration.recording is False

    def test_next_pose_returns_false_when_done(self):
        """next_pose returns False when all poses have been recorded."""
        calibration = CalibrationState()
        calibration.start()

        # Should be able to advance through all poses
        for i in range(len(POSE_LABELS) - 1):
            assert calibration.next_pose() is True

        # Final next_pose should return False
        assert calibration.next_pose() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

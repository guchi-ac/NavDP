import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from utils_tasks.goal_capture import GoalCaptureStore


class GoalCaptureStoreTests(unittest.TestCase):
    def test_capture_requires_a_d435_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            store = GoalCaptureStore(Path(directory))

            with self.assertRaisesRegex(RuntimeError, "No D435 RGB frame yet"):
                store.capture()

            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_capture_preserves_resolution_and_updates_latest_path(self):
        with tempfile.TemporaryDirectory() as directory:
            goal_dir = Path(directory)
            store = GoalCaptureStore(goal_dir)
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            image[:, :, 1] = 180
            store.update(image)

            archive, latest = store.capture(
                datetime(2026, 7, 17, 14, 15, 30, 123000)
            )

            self.assertEqual(
                archive.name,
                "captured_goal_20260717_141530_123.jpg",
            )
            self.assertEqual(latest.name, "captured_goal_latest.jpg")
            self.assertEqual(
                cv2.imread(str(archive), cv2.IMREAD_COLOR).shape,
                image.shape,
            )
            self.assertEqual(
                cv2.imread(str(latest), cv2.IMREAD_COLOR).shape,
                image.shape,
            )


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class GoalCaptureStore:
    def __init__(self, goal_dir: Path):
        self.goal_dir = Path(goal_dir)
        self.lock = threading.Lock()
        self.frame = None

    def update(self, frame_bgr: np.ndarray) -> None:
        with self.lock:
            self.frame = np.asarray(frame_bgr).copy()

    def capture(
        self,
        captured_at: Optional[datetime] = None,
    ) -> Tuple[Path, Path]:
        with self.lock:
            if self.frame is None:
                raise RuntimeError("No D435 RGB frame yet")
            frame = self.frame.copy()

        encoded, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        if not encoded:
            raise RuntimeError("OpenCV JPEG encoding failed")

        timestamp = (captured_at or datetime.now()).strftime(
            "%Y%m%d_%H%M%S_%f"
        )[:-3]
        archive = self.goal_dir / f"captured_goal_{timestamp}.jpg"
        latest = self.goal_dir / "captured_goal_latest.jpg"
        self._atomic_write(archive, jpeg.tobytes())
        self._atomic_write(latest, jpeg.tobytes())
        return archive, latest

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as output:
                temporary = Path(output.name)
                output.write(data)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

"""Line-crossing vehicle counter — uses YOLOv8 built-in track IDs."""

@dataclass
class CounterConfig:
    line_y: int                              # Y-coordinate of the counting line
    line_color: tuple = (0, 255, 255)        # BGR yellow
    line_thickness: int = 2

 
class VehicleCounter:
    """
    Counts vehicles crossing a horizontal line.
    """
    def __init__(self, config: CounterConfig):
        self.cfg = config
        self._last_y: dict[int, float] = {}          # track_id → last center_y
        self._crossed: set[int] = set()              # track_ids that already crossed
        self._counts: dict[str, dict[str, int]] = defaultdict(lambda: {"in": 0, "out": 0})

    def update(self, results) -> None:
        """
        Feed a model.track() result for one frame.
        results: single Results object from model.track(frame, persist=True)[0]
        """

        if results.boxes.id is None:
            return  # no tracked objects this frame
        
        for box, track_id, cls_id in zip(results.boxes.xyxy, results.boxes.id, results.cls, ):
            track_id = int(track_id)
            class_name = results.names[int(cls_id)]

            x1, y1, x2, y2 = box
            cy = float((y1 + y2) / 2)

            last_y = self._last_y.get(track_id)

            if last_y is not None and track_id not in self._crossed:
                line = self.cfg.line_y
                if last_y < line <= cy:
                    self._counts[class_name]["in"] += 1
                    self._crossed.add(track_id)
                elif last_y >= line > cy:
                    self._counts[class_name]["out"] += 1
                    self._crossed.add(track_id)
            self._last_y[track_id] = cy
    @property
    def total_in(self) -> int:
        return sum(v["in"] for v in self._counts.values())
    
    @property
    def total_out(self) -> int:
        return sum(v["out"] for v in self._counts.values())
    
    @property
    def count(self) -> dict:
        return {cls: dict(dirs) for cls, dirs in self._counts.items()}
    
    def summary(self) -> dict:
        result = {"total_in": self.total_in, "total_out": self.total_out}
        for cls, dirs in self._counts.items():
            result[f"{cls}_in"] = dirs["in"]
            result[f"{cls}_out"] = dirs["out"]
        return result

def draw_overlay(self, frame: np.ndarray) -> None:
    """Draw counting line and scoreboard onto frame (in-place)."""
    h,w = frame.shape[:2]

    # Counting line
    cv2.line(frame, (0, self.cfg.line_y), (w, self.cfg.line_y), self.cfg.line_color, self.cfg.line_thickness)
    cv2.putText(frame, "COUNT LINE",  (8, self.cfg.line_y - 6),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.cfg.line_color, 1, cv2.LINE_AA)

    # Scoreboard
    lines = [f"IN : {self.total_in}", f"OUT: {self.total_out}", "---"]
    for cls, dirs in self._counts.items():
        lines.append(f"{cls:12s} +{dirs['in']} -{dirs['out']}")

    x0, y0, pad, line_h = 10, 10, 6, 22
    board_w, board_h = 220, len(lines) * line_h + pad * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + board_w, y0 + board_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for i, text in enumerate(lines):
        color = (0, 255, 255) if text.startswith(("IN", "OUT")) else (200, 200, 200)
        cv2.putText(frame, text, (x0 + pad, y0 + pad + (i + 1) * line_h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
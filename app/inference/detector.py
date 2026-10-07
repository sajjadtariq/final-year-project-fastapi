"""Ultralytics YOLO11m adapter returning normalized independent detections."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..errors import ModelExecutionError


class YOLODetector:
    """Load a local YOLO checkpoint and expose a small framework-neutral result."""

    expected_classes = {"acne", "bcc", "melanoma", "nevus"}

    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
        image_size: int,
        confidence: float,
        iou: float,
        maximum_detections: int,
        version: str,
    ) -> None:
        if not checkpoint_path.is_file():
            raise ModelExecutionError(f"YOLO checkpoint not found: {checkpoint_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelExecutionError(
                "ultralytics is required to load the YOLO detector"
            ) from exc

        self.model = YOLO(str(checkpoint_path))
        if self.model.task != "detect":
            raise ModelExecutionError(
                f"Expected a YOLO detection checkpoint, got {self.model.task!r}"
            )
        self.names = {
            int(index): str(name).lower()
            for index, name in self.model.names.items()
        }
        available = set(self.names.values())
        if available != self.expected_classes:
            raise ModelExecutionError(
                "YOLO classes must be exactly acne, bcc, melanoma, and nevus; "
                f"found {sorted(available)}"
            )
        self.supported_classes = frozenset(available)
        self.device = device
        self.ultralytics_device: int | str = 0 if device.type == "cuda" else "cpu"
        self.image_size = image_size
        self.confidence = confidence
        self.iou = iou
        self.maximum_detections = maximum_detections
        self.model_version = version

    def predict(self, image: Image.Image) -> tuple[list[dict[str, Any]], float]:
        started = time.perf_counter()
        results = self.model.predict(
            source=np.asarray(image, dtype=np.uint8),
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            max_det=self.maximum_detections,
            device=self.ultralytics_device,
            verbose=False,
        )
        detections: list[dict[str, Any]] = []
        if results:
            boxes = results[0].boxes
            if boxes is not None:
                coordinates = boxes.xyxyn.detach().cpu().tolist()
                confidences = boxes.conf.detach().cpu().tolist()
                classes = boxes.cls.detach().cpu().tolist()
                for coordinates_row, confidence, class_index in zip(
                    coordinates,
                    confidences,
                    classes,
                    strict=True,
                ):
                    detections.append(
                        {
                            "class_name": self.names[int(class_index)],
                            "confidence": float(confidence),
                            "box_normalized": tuple(
                                min(1.0, max(0.0, float(value)))
                                for value in coordinates_row
                            ),
                        }
                    )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return detections, elapsed_ms

    def warmup(self) -> None:
        dummy = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        self.model.predict(
            source=dummy,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            max_det=1,
            device=self.ultralytics_device,
            verbose=False,
        )


# File summary: YOLODetector validates the four-class detection checkpoint, runs its
# own 640px inference, and returns normalized boxes without using classifier scores.

"""Construct the real local pipeline from environment-backed settings."""

from __future__ import annotations

from ..config import Settings
from .classifier import EfficientNetClassifier, select_device
from .detector import YOLODetector
from .pipeline import SkinAssessmentPipeline


def build_pipeline(settings: Settings) -> SkinAssessmentPipeline:
    """Load both checkpoints once, warm them, and return the coordinator."""
    device = select_device(settings.device)
    classifier = EfficientNetClassifier(
        settings.efficientnet_checkpoint,
        device,
    )
    detector = YOLODetector(
        checkpoint_path=settings.yolo_checkpoint,
        device=device,
        image_size=settings.yolo_image_size,
        confidence=settings.yolo_confidence,
        iou=settings.yolo_iou,
        maximum_detections=settings.yolo_max_detections,
        version=settings.detector_version,
    )
    classifier.warmup()
    detector.warmup()
    return SkinAssessmentPipeline(
        classifier=classifier,
        detector=detector,
        maximum_file_size_mb=settings.maximum_file_size_mb,
        pipeline_version=settings.pipeline_version,
    )


# File summary: build_pipeline creates and warms both local model adapters exactly
# once so request handlers never reload weights per assessment.

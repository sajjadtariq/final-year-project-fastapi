"""Coordinate quality checks, classification, localization, and safe decisions."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

from PIL import Image

from ..schemas import AssessmentResponse
from .preprocessing import validate_and_prepare_image


class ClassifierAdapter(Protocol):
    classes: tuple[str, ...]
    model_version: str
    registry_version: str | None
    mlflow_run_id: str | None
    image_size: int

    def predict(self, image: Image.Image) -> tuple[dict[str, Any], float]: ...


class DetectorAdapter(Protocol):
    supported_classes: frozenset[str]
    model_version: str

    def predict(self, image: Image.Image) -> tuple[list[dict[str, Any]], float]: ...


DISPLAY_NAMES = {
    "nevus": "nevus",
    "melanoma": "melanoma",
    "bcc": "basal cell carcinoma",
    "acne": "acne",
    "eczema": "eczema",
    "psoriasis": "psoriasis",
}

LOGGER = logging.getLogger("skinhealth.inference.pipeline")


def _decision_for(classification: dict[str, Any]) -> dict[str, Any]:
    disclaimer = "This is an AI-assisted assessment, not a medical diagnosis."
    if not classification["accepted"]:
        return {
            "result_type": "insufficient_confidence",
            "recommended_action": "retake_or_dermatologist_review",
            "review_priority": "not_assessed",
            "action_label": "Retake the photo or see a dermatologist",
            "severity": None,
            "title": "We could not assess this image reliably",
            "explanation": (
                "Try another clear, well-lit close-up photo or consult a "
                "dermatologist."
            ),
            "disclaimer": disclaimer,
        }

    class_name = str(classification["class_name"])
    display_name = DISPLAY_NAMES[class_name]
    if class_name in {"melanoma", "bcc"}:
        action = "urgent_dermatologist_review"
        review_priority = "as_soon_as_possible"
        action_label = "See a dermatologist as soon as possible"
        explanation = (
            f"This image has features associated with {display_name}. "
            "See a qualified dermatologist as soon as possible."
        )
    elif class_name == "nevus":
        action = "routine_dermatologist_review"
        review_priority = "routine"
        action_label = "Book a dermatologist check"
        explanation = (
            "This image has features associated with a nevus. Book a "
            "dermatologist check, especially if the area is new, changing, "
            "bleeding, or itching. This AI result cannot confirm that the area "
            "is harmless or rule out skin cancer."
        )
    else:
        action = "dermatologist_review"
        review_priority = "recommended"
        action_label = "Dermatologist review recommended"
        explanation = (
            f"This image has features associated with {display_name}. "
            "A clinician can confirm the cause and discuss suitable care."
        )
    return {
        "result_type": "accepted",
        "recommended_action": action,
        "review_priority": review_priority,
        "action_label": action_label,
        "severity": None,
        "title": f"Features consistent with {display_name}",
        "explanation": explanation,
        "disclaimer": disclaimer,
    }


class SkinAssessmentPipeline:
    """Stateless two-model service suitable for dependency injection and testing."""

    def __init__(
        self,
        classifier: ClassifierAdapter,
        detector: DetectorAdapter,
        maximum_file_size_mb: float,
        pipeline_version: str,
    ) -> None:
        self.classifier = classifier
        self.detector = detector
        self.maximum_file_size_mb = maximum_file_size_mb
        self.pipeline_version = pipeline_version
        self.device = str(getattr(classifier, "device", "unknown"))

    @property
    def model_versions(self) -> dict[str, Any]:
        return {
            "classifier": self.classifier.model_version,
            "classifier_registry_version": self.classifier.registry_version,
            "classifier_mlflow_run_id": self.classifier.mlflow_run_id,
            "detector": self.detector.model_version,
            "preprocessing": f"quality-v1-{self.classifier.image_size}px",
            "pipeline": self.pipeline_version,
        }

    def analyze(
        self,
        image_bytes: bytes,
        filename: str,
        declared_mime_type: str | None,
        body_area: str,
        symptoms: str | None = None,
        duration: str | None = None,
        assessment_id: str | None = None,
    ) -> AssessmentResponse:
        total_started = time.perf_counter()
        resolved_assessment_id = assessment_id or f"asm_{uuid4().hex}"
        preprocessing_started = time.perf_counter()
        image, quality = validate_and_prepare_image(
            image_bytes,
            filename,
            declared_mime_type,
            self.maximum_file_size_mb,
        )
        preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000

        try:
            classification, classifier_ms = self.classifier.predict(image)
            class_name = classification["class_name"]
            if classification["accepted"]:
                classification["display_label"] = (
                    f"Features consistent with {DISPLAY_NAMES[str(class_name)]}"
                )
            else:
                classification["display_label"] = (
                    "We could not assess this image reliably"
                )

            localization, detector_ms = self._localize(
                image,
                classification,
                resolved_assessment_id,
            )
            decision = _decision_for(classification)
            public_classification = dict(classification)
            public_classification.pop("raw_predicted_class", None)
            if not public_classification["accepted"]:
                public_classification["probabilities"] = {}
        finally:
            image.close()

        total_ms = (time.perf_counter() - total_started) * 1000
        return AssessmentResponse.model_validate(
            {
                "assessment_id": resolved_assessment_id,
                "status": "completed" if classification["accepted"] else "refused",
                "created_at": datetime.now(timezone.utc),
                "submitted_details": {
                    "body_area": body_area,
                    "symptoms": symptoms,
                    "duration": duration,
                },
                "image_quality": quality,
                "classification": public_classification,
                "localization": localization,
                "decision": decision,
                "model_versions": self.model_versions,
                "timings": {
                    "preprocessing_ms": preprocessing_ms,
                    "classifier_ms": classifier_ms,
                    "detector_ms": detector_ms,
                    "total_ms": total_ms,
                },
            }
        )

    def _localize(
        self,
        image: Image.Image,
        classification: dict[str, Any],
        assessment_id: str,
    ) -> tuple[dict[str, Any], float]:
        if not classification["accepted"]:
            return (
                {
                    "status": "classifier_refused",
                    "supported": False,
                    "class_conflict_logged": False,
                    "displayed_region_count": 0,
                    "raw_detection_count": 0,
                    "detections": [],
                    "message": "Localization is hidden when classification is refused.",
                },
                0.0,
            )

        predicted = str(classification["class_name"])
        if predicted not in self.detector.supported_classes:
            return (
                {
                    "status": "detector_not_supported",
                    "supported": False,
                    "class_conflict_logged": False,
                    "displayed_region_count": 0,
                    "raw_detection_count": 0,
                    "detections": [],
                    "message": (
                        f"The current detector does not localize {predicted} regions."
                    ),
                },
                0.0,
            )

        try:
            detections, detector_ms = self.detector.predict(image)
        except Exception:
            return (
                {
                    "status": "temporarily_unavailable",
                    "supported": True,
                    "class_conflict_logged": False,
                    "displayed_region_count": 0,
                    "raw_detection_count": 0,
                    "detections": [],
                    "message": "Image highlighting is temporarily unavailable.",
                },
                0.0,
            )

        detector_classes = sorted(
            {str(detection["class_name"]) for detection in detections}
        )
        class_conflict = any(
            detector_class != predicted for detector_class in detector_classes
        )
        if class_conflict:
            LOGGER.warning(
                "model_class_conflict assessment_id=%s classifier_class=%s "
                "detector_classes=%s raw_detection_count=%d",
                assessment_id,
                predicted,
                ",".join(detector_classes),
                len(detections),
            )

        visible_regions = [
            {
                "confidence": detection["confidence"],
                "box_normalized": detection["box_normalized"],
            }
            for detection in detections
        ]
        if visible_regions:
            status = "regions_detected"
            message = "Regions were detected and can be highlighted."
        else:
            status = "no_regions_detected"
            message = "No regions met the detector confidence threshold."

        return (
            {
                "status": status,
                "supported": True,
                "class_conflict_logged": class_conflict,
                "displayed_region_count": len(visible_regions),
                "raw_detection_count": len(detections),
                "detections": visible_regions,
                "message": message,
            },
            detector_ms,
        )


# File summary:
# - SkinAssessmentPipeline validates once, runs EfficientNet first, conditionally
#   invokes YOLO, returns class-neutral regions, and logs internal label conflicts.
# - _decision_for supplies deterministic, non-diagnostic next actions while
#   keeping clinical severity unavailable.

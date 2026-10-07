"""Contract tests for two-model coordination using deterministic fake adapters."""

from __future__ import annotations

import unittest
from io import BytesIO
from typing import Any

import numpy as np
import torch
from PIL import Image

from app.inference.pipeline import SkinAssessmentPipeline


def make_jpeg() -> bytes:
    generator = np.random.default_rng(42)
    pixels = generator.integers(0, 256, size=(256, 320, 3), dtype=np.uint8)
    output = BytesIO()
    Image.fromarray(pixels).save(output, format="JPEG")
    return output.getvalue()


class FakeClassifier:
    classes = ("nevus", "melanoma", "bcc", "acne", "eczema", "psoriasis")
    model_version = "classifier-test"
    registry_version = "7"
    mlflow_run_id = "run-test"
    image_size = 384
    device = torch.device("cpu")

    def __init__(self, class_name: str | None, raw_class: str = "acne") -> None:
        self.class_name = class_name
        self.raw_class = raw_class

    def predict(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        accepted = self.class_name is not None
        return (
            {
                "accepted": accepted,
                "class_name": self.class_name,
                "raw_predicted_class": self.raw_class,
                "confidence": 0.82 if accepted else 0.51,
                "threshold_used": 0.60,
                "probabilities": {
                    "nevus": 0.02,
                    "melanoma": 0.01,
                    "bcc": 0.01,
                    "acne": 0.82 if accepted else 0.51,
                    "eczema": 0.10,
                    "psoriasis": 0.04,
                },
            },
            2.0,
        )


class FakeDetector:
    supported_classes = frozenset({"acne", "bcc", "melanoma", "nevus"})
    model_version = "detector-test"

    def __init__(self, class_name: str = "acne") -> None:
        self.class_name = class_name
        self.calls = 0

    def predict(self, image: Image.Image) -> tuple[list[dict[str, Any]], float]:
        self.calls += 1
        return (
            [
                {
                    "class_name": self.class_name,
                    "confidence": 0.78,
                    "box_normalized": (0.1, 0.2, 0.4, 0.6),
                }
            ],
            3.0,
        )


def build_pipeline(
    classifier_class: str | None,
    detector_class: str = "acne",
    raw_class: str = "acne",
) -> tuple[SkinAssessmentPipeline, FakeDetector]:
    detector = FakeDetector(detector_class)
    pipeline = SkinAssessmentPipeline(
        classifier=FakeClassifier(classifier_class, raw_class),
        detector=detector,
        maximum_file_size_mb=15,
        pipeline_version="pipeline-test",
    )
    return pipeline, detector


class PipelineTests(unittest.TestCase):
    def test_detector_returns_class_neutral_normalized_regions(self) -> None:
        pipeline, detector = build_pipeline("acne")
        result = pipeline.analyze(
            make_jpeg(),
            "sample.jpg",
            "image/jpeg",
            "face",
            assessment_id="asm_test",
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.localization.status, "regions_detected")
        self.assertEqual(result.localization.displayed_region_count, 1)
        self.assertFalse(result.localization.class_conflict_logged)
        self.assertNotIn(
            "class_name",
            result.localization.detections[0].model_dump(),
        )
        self.assertEqual(detector.calls, 1)
        self.assertIsNone(result.decision.severity)
        self.assertEqual(result.model_versions.classifier, "classifier-test")

    def test_refusal_skips_detector_and_hides_class(self) -> None:
        pipeline, detector = build_pipeline(None)
        result = pipeline.analyze(
            make_jpeg(),
            "sample.jpg",
            "image/jpeg",
            "face",
        )
        self.assertEqual(result.status, "refused")
        self.assertIsNone(result.classification.class_name)
        self.assertEqual(result.classification.probabilities, {})
        self.assertEqual(result.localization.status, "classifier_refused")
        self.assertEqual(detector.calls, 0)
        self.assertEqual(result.decision.review_priority, "not_assessed")
        self.assertEqual(
            result.decision.action_label,
            "Retake the photo or see a dermatologist",
        )

    def test_cancer_classes_recommend_review_as_soon_as_possible(self) -> None:
        for class_name in ("melanoma", "bcc"):
            with self.subTest(class_name=class_name):
                pipeline, _ = build_pipeline(class_name, detector_class=class_name)
                result = pipeline.analyze(
                    make_jpeg(),
                    "sample.jpg",
                    "image/jpeg",
                    "face",
                )
                self.assertEqual(
                    result.decision.recommended_action,
                    "urgent_dermatologist_review",
                )
                self.assertEqual(
                    result.decision.review_priority,
                    "as_soon_as_possible",
                )
                self.assertEqual(
                    result.decision.action_label,
                    "See a dermatologist as soon as possible",
                )
                self.assertIn(
                    "as soon as possible",
                    result.decision.explanation,
                )
                self.assertNotIn("prompt", result.decision.explanation.lower())

    def test_nevus_is_not_presented_as_cancer_or_urgent(self) -> None:
        pipeline, _ = build_pipeline("nevus", detector_class="nevus")
        result = pipeline.analyze(
            make_jpeg(),
            "sample.jpg",
            "image/jpeg",
            "arm",
        )
        self.assertEqual(
            result.decision.recommended_action,
            "routine_dermatologist_review",
        )
        self.assertEqual(result.decision.review_priority, "routine")
        self.assertEqual(
            result.decision.action_label,
            "Book a dermatologist check",
        )
        self.assertIn("cannot confirm", result.decision.explanation)
        self.assertIn("rule out skin cancer", result.decision.explanation)
        self.assertIsNone(result.decision.severity)

    def test_inflammatory_class_recommends_dermatologist_review(self) -> None:
        pipeline, _ = build_pipeline("eczema")
        result = pipeline.analyze(
            make_jpeg(),
            "sample.jpg",
            "image/jpeg",
            "arm",
        )
        self.assertEqual(result.decision.review_priority, "recommended")
        self.assertEqual(
            result.decision.action_label,
            "Dermatologist review recommended",
        )

    def test_unsupported_classifier_class_skips_detector(self) -> None:
        pipeline, detector = build_pipeline("eczema")
        result = pipeline.analyze(
            make_jpeg(),
            "sample.jpg",
            "image/jpeg",
            "arm",
        )
        self.assertEqual(result.localization.status, "detector_not_supported")
        self.assertEqual(detector.calls, 0)

    def test_detector_class_conflict_is_logged_without_hiding_boxes(self) -> None:
        pipeline, _ = build_pipeline("acne", detector_class="bcc")
        with self.assertLogs(
            "skinhealth.inference.pipeline",
            level="WARNING",
        ) as captured:
            result = pipeline.analyze(
                make_jpeg(),
                "sample.jpg",
                "image/jpeg",
                "face",
                assessment_id="asm_conflict_test",
            )
        self.assertEqual(result.localization.status, "regions_detected")
        self.assertTrue(result.localization.class_conflict_logged)
        self.assertEqual(result.localization.displayed_region_count, 1)
        self.assertEqual(result.localization.raw_detection_count, 1)
        self.assertEqual(len(result.localization.detections), 1)
        self.assertEqual(
            result.localization.detections[0].box_normalized,
            (0.1, 0.2, 0.4, 0.6),
        )
        log_output = "\n".join(captured.output)
        self.assertIn("assessment_id=asm_conflict_test", log_output)
        self.assertIn("classifier_class=acne", log_output)
        self.assertIn("detector_classes=bcc", log_output)


# File summary: Fake adapters verify confidence refusal, action wording, class-neutral
# YOLO regions, conflict logging, version fields, and the absence of severity.

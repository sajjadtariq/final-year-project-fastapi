"""Stable request-independent response contracts shared with the frontend team."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ImageQualityResult(BaseModel):
    accepted: bool = True
    original_width: int
    original_height: int
    file_size_bytes: int
    detected_mime_type: str
    brightness: float
    blur_score: float
    focus_status: Literal["acceptable", "warning"]
    lighting_status: Literal["acceptable", "enhanced"]
    resolution_status: Literal["acceptable", "warning"]
    enhancement_applied: Literal["none", "clahe"]
    warnings: list[str] = Field(default_factory=list)


class ClassificationResult(BaseModel):
    accepted: bool
    class_name: str | None
    display_label: str
    confidence: float
    threshold_used: float
    probabilities: dict[str, float]


class DetectionResult(BaseModel):
    confidence: float
    box_normalized: tuple[float, float, float, float]


class LocalizationResult(BaseModel):
    status: Literal[
        "regions_detected",
        "no_regions_detected",
        "detector_not_supported",
        "classifier_refused",
        "temporarily_unavailable",
    ]
    supported: bool
    class_conflict_logged: bool = False
    displayed_region_count: int
    raw_detection_count: int
    detections: list[DetectionResult] = Field(default_factory=list)
    message: str


class DecisionResult(BaseModel):
    result_type: Literal["accepted", "insufficient_confidence"]
    recommended_action: Literal[
        "urgent_dermatologist_review",
        "dermatologist_review",
        "routine_dermatologist_review",
        "retake_or_dermatologist_review",
    ]
    review_priority: Literal[
        "as_soon_as_possible",
        "recommended",
        "routine",
        "not_assessed",
    ]
    action_label: str
    severity: None = None
    title: str
    explanation: str
    disclaimer: str


class ModelVersions(BaseModel):
    classifier: str
    classifier_registry_version: str | None = None
    classifier_mlflow_run_id: str | None = None
    detector: str
    preprocessing: str
    pipeline: str


class TimingResult(BaseModel):
    preprocessing_ms: float
    classifier_ms: float
    detector_ms: float
    total_ms: float


class AssessmentResponse(BaseModel):
    assessment_id: str
    status: Literal["completed", "refused"]
    created_at: datetime
    submitted_details: dict[str, str | None]
    image_quality: ImageQualityResult
    classification: ClassificationResult
    localization: LocalizationResult
    decision: DecisionResult
    model_versions: ModelVersions
    timings: TimingResult


class HealthResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    models_loaded: bool
    device: str
    pipeline_version: str


class ModelInfoResponse(BaseModel):
    models_loaded: bool
    model_versions: ModelVersions | None = None
    classifier_classes: list[str]
    detector_classes: list[str]


# File summary:
# - Pydantic models define the exact frontend handoff contract for quality,
#   classification, localization, action-oriented decisions, timing, health,
#   and model identity.

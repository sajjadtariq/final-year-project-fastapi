"""Readiness and immutable model-identity endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ..schemas import HealthResponse, ModelInfoResponse
from ..security import require_internal_key

router = APIRouter(tags=["service"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    pipeline = request.app.state.pipeline
    settings = request.app.state.settings
    return HealthResponse(
        status="ready" if pipeline is not None else "not_ready",
        models_loaded=pipeline is not None,
        device=pipeline.device if pipeline is not None else settings.device,
        pipeline_version=settings.pipeline_version,
    )


@router.get(
    "/models",
    response_model=ModelInfoResponse,
    dependencies=[Depends(require_internal_key)],
)
def models(request: Request) -> ModelInfoResponse:
    pipeline = request.app.state.pipeline
    return ModelInfoResponse(
        models_loaded=pipeline is not None,
        model_versions=(pipeline.model_versions if pipeline is not None else None),
        classifier_classes=[
            "nevus",
            "melanoma",
            "bcc",
            "acne",
            "eczema",
            "psoriasis",
        ],
        detector_classes=["acne", "bcc", "melanoma", "nevus"],
    )


# File summary: /health reports readiness without private paths; /models exposes the
# class coverage and exact model versions needed for frontend/database audit fields.

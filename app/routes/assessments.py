"""Stateless synchronous assessment endpoint for local frontend integration."""

from __future__ import annotations

from functools import partial

import anyio
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from ..errors import ImageValidationError
from ..schemas import AssessmentResponse
from ..security import require_internal_key

router = APIRouter(prefix="/assessments", tags=["assessments"])


@router.post(
    "/analyze",
    response_model=AssessmentResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"description": "The uploaded image failed a hard requirement."},
        503: {"description": "Local model service is not ready."},
    },
)
async def analyze_assessment(
    request: Request,
    image: UploadFile = File(...),
    body_area: str = Form(..., min_length=1, max_length=80),
    symptoms: str | None = Form(default=None, max_length=500),
    duration: str | None = Form(default=None, max_length=120),
    assessment_id: str | None = Form(
        default=None,
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9_-]+$",
    ),
    _: None = Depends(require_internal_key),
) -> AssessmentResponse:
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "MODELS_NOT_LOADED",
                "message": "The local model service is not ready.",
            },
        )

    settings = request.app.state.settings
    maximum_bytes = round(settings.maximum_file_size_mb * 1024 * 1024)
    payload = await image.read(maximum_bytes + 1)
    await image.close()
    if len(payload) > maximum_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "FILE_TOO_LARGE",
                "message": (
                    f"The image exceeds the {settings.maximum_file_size_mb:g} MB limit"
                ),
            },
        )

    normalized_body_area = body_area.strip()
    if not normalized_body_area:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "INVALID_BODY_AREA",
                "message": "body_area cannot contain only whitespace",
            },
        )

    operation = partial(
        pipeline.analyze,
        image_bytes=payload,
        filename=image.filename or "upload",
        declared_mime_type=image.content_type,
        body_area=normalized_body_area,
        symptoms=symptoms.strip() if symptoms else None,
        duration=duration.strip() if duration else None,
        assessment_id=assessment_id,
    )
    try:
        async with request.app.state.inference_semaphore:
            return await anyio.to_thread.run_sync(operation)
    except ImageValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        request.app.state.logger.exception("Assessment inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "INFERENCE_FAILED",
                "message": "The assessment could not be completed.",
            },
        ) from exc


# File summary: POST /assessments/analyze bounds uploads, serializes local GPU work,
# runs the stateless pipeline off the event loop, and returns stable safe errors.

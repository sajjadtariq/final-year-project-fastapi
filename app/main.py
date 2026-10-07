"""FastAPI application factory for the standalone local SkinHealth backend."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .config import Settings
from .inference.factory import build_pipeline
from .inference.pipeline import SkinAssessmentPipeline
from .routes.assessments import router as assessment_router
from .routes.health import router as health_router


def create_app(
    settings: Settings | None = None,
    pipeline_override: SkinAssessmentPipeline | None = None,
) -> FastAPI:
    """Create an app that can use real models or an injected test pipeline."""
    resolved_settings = settings or Settings.from_env()
    logger = logging.getLogger("skinhealth.api")

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.settings = resolved_settings
        application.state.logger = logger
        application.state.inference_semaphore = asyncio.Semaphore(
            resolved_settings.max_concurrency
        )
        if pipeline_override is not None:
            application.state.pipeline = pipeline_override
        elif resolved_settings.load_models:
            logger.info("Loading local EfficientNet and YOLO checkpoints")
            application.state.pipeline = await asyncio.to_thread(
                build_pipeline,
                resolved_settings,
            )
            logger.info("Local inference models are ready")
        else:
            application.state.pipeline = None
            logger.warning("Model loading is disabled; inference returns HTTP 503")
        yield
        application.state.pipeline = None

    application = FastAPI(
        title="SkinHealth Local Inference API",
        version=__version__,
        description=(
            "Stateless, non-diagnostic EfficientNet-B3 and YOLO11m assessment API."
        ),
        lifespan=lifespan,
        docs_url="/docs" if resolved_settings.docs_enabled else None,
        redoc_url="/redoc" if resolved_settings.docs_enabled else None,
        openapi_url="/openapi.json" if resolved_settings.docs_enabled else None,
    )
    if resolved_settings.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved_settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "X-Internal-Key", "X-Request-ID"],
        )
    application.include_router(health_router, prefix="/api/v1")
    application.include_router(assessment_router, prefix="/api/v1")
    return application


app = create_app()


# File summary:
# - create_app wires CORS, routes, a one-request inference semaphore, and dependency
#   injection for tests or real locally loaded checkpoints.
# - lifespan loads and warms each model once rather than once per HTTP request.

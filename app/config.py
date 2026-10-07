"""Environment-based configuration for the standalone local inference API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    """All settings required to run the copyable backend folder."""

    load_models: bool
    efficientnet_checkpoint: Path
    yolo_checkpoint: Path
    device: str
    cors_origins: tuple[str, ...]
    maximum_file_size_mb: float
    yolo_image_size: int
    yolo_confidence: float
    yolo_iou: float
    yolo_max_detections: int
    detector_version: str
    pipeline_version: str
    internal_api_key: str
    docs_enabled: bool
    max_concurrency: int

    @classmethod
    def from_env(cls) -> "Settings":
        origins = tuple(
            origin.strip()
            for origin in os.getenv(
                "SKINHEALTH_CORS_ORIGINS",
                "",
            ).split(",")
            if origin.strip()
        )
        settings = cls(
            load_models=_env_bool("SKINHEALTH_LOAD_MODELS", True),
            efficientnet_checkpoint=_env_path(
                "SKINHEALTH_EFFICIENTNET_CHECKPOINT",
                "models/efficientnet-b3/checkpoints/best.pt",
            ),
            yolo_checkpoint=_env_path(
                "SKINHEALTH_YOLO_CHECKPOINT",
                "models/yolo11m.pt",
            ),
            device=os.getenv("SKINHEALTH_DEVICE", "auto").strip().lower(),
            cors_origins=origins,
            maximum_file_size_mb=float(os.getenv("SKINHEALTH_MAX_FILE_SIZE_MB", "15")),
            yolo_image_size=int(os.getenv("SKINHEALTH_YOLO_IMAGE_SIZE", "640")),
            yolo_confidence=float(os.getenv("SKINHEALTH_YOLO_CONFIDENCE", "0.45")),
            yolo_iou=float(os.getenv("SKINHEALTH_YOLO_IOU", "0.70")),
            yolo_max_detections=int(os.getenv("SKINHEALTH_YOLO_MAX_DETECTIONS", "20")),
            detector_version=os.getenv(
                "SKINHEALTH_DETECTOR_VERSION",
                "yolo11m-skin-v1-baseline-aug",
            ),
            pipeline_version=os.getenv(
                "SKINHEALTH_PIPELINE_VERSION",
                "local-pipeline-v1",
            ),
            internal_api_key=os.getenv("INTERNAL_API_KEY", "").strip(),
            docs_enabled=_env_bool("DOCS_ENABLED", False),
            max_concurrency=int(os.getenv("SKINHEALTH_MAX_CONCURRENCY", "1")),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("SKINHEALTH_DEVICE must be auto, cpu, cuda, or mps")
        if self.maximum_file_size_mb <= 0:
            raise ValueError("Maximum file size must be positive")
        if self.yolo_image_size <= 0 or self.yolo_max_detections <= 0:
            raise ValueError("YOLO size and maximum detections must be positive")
        if not 0 <= self.yolo_confidence <= 1:
            raise ValueError("YOLO confidence must be between 0 and 1")
        if not 0 <= self.yolo_iou <= 1:
            raise ValueError("YOLO IoU must be between 0 and 1")
        if len(self.internal_api_key) < 32:
            raise ValueError("INTERNAL_API_KEY must contain at least 32 characters")
        if self.max_concurrency <= 0:
            raise ValueError("SKINHEALTH_MAX_CONCURRENCY must be positive")


# File summary:
# - Settings.from_env builds a portable configuration without project-root imports.
# - validate rejects invalid device, upload-size, and YOLO threshold settings.

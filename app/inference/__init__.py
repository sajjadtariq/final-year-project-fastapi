"""Model-independent preprocessing and adapters for local inference."""

from .pipeline import SkinAssessmentPipeline

__all__ = ["SkinAssessmentPipeline"]

# File summary: Exposes the pipeline coordinator while keeping model loading explicit.

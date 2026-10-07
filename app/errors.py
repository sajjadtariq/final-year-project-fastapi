"""Typed service errors that API routes can convert into stable error responses."""


class ImageValidationError(ValueError):
    """Raised when an upload fails a hard input requirement."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ModelExecutionError(RuntimeError):
    """Raised when a model cannot load or complete inference."""


# File summary: Defines safe validation and model-execution exceptions so routes do
# not expose internal stack traces or inconsistent error strings.

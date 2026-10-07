"""Deterministic upload validation and EfficientNet evaluation preprocessing."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from torchvision import transforms
from torchvision.transforms import functional as TF

from ..errors import ImageValidationError

JPEG_SIGNATURE = b"\xff\xd8\xff"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
WEBP_RIFF_SIGNATURE = b"RIFF"
WEBP_FORMAT_SIGNATURE = b"WEBP"
TIFF_SIGNATURES = (b"II*\x00", b"MM\x00*")
AVIF_BRANDS = {b"avif", b"avis"}
HEIF_BRANDS = {
    b"heic",
    b"heix",
    b"hevc",
    b"hevx",
    b"heim",
    b"heis",
    b"hevm",
    b"hevs",
    b"heif",
    b"mif1",
    b"msf1",
}
FORMAT_BY_SUFFIX = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".heic": "HEIF",
    ".heif": "HEIF",
    ".webp": "WEBP",
    ".avif": "AVIF",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}
FORMAT_BY_MIME = {
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG",
    "image/png": "PNG",
    "image/heic": "HEIF",
    "image/heif": "HEIF",
    "image/heic-sequence": "HEIF",
    "image/heif-sequence": "HEIF",
    "image/webp": "WEBP",
    "image/avif": "AVIF",
    "image/avif-sequence": "AVIF",
    "image/tiff": "TIFF",
    "image/x-tiff": "TIFF",
}
MIME_BY_FORMAT = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "HEIF": "image/heif",
    "WEBP": "image/webp",
    "AVIF": "image/avif",
    "TIFF": "image/tiff",
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

register_heif_opener(thumbnails=False)


def detect_header_format(data: bytes) -> str | None:
    if data.startswith(JPEG_SIGNATURE):
        return "JPEG"
    if data.startswith(PNG_SIGNATURE):
        return "PNG"
    if (
        data.startswith(WEBP_RIFF_SIGNATURE)
        and len(data) >= 12
        and data[8:12] == WEBP_FORMAT_SIGNATURE
    ):
        return "WEBP"
    if data.startswith(TIFF_SIGNATURES):
        return "TIFF"
    if len(data) >= 16 and data[4:8] == b"ftyp":
        box_size = int.from_bytes(data[:4], "big")
        box_end = min(len(data), box_size if box_size >= 16 else len(data))
        brands = {data[8:12]}
        brands.update(
            data[offset : offset + 4]
            for offset in range(16, box_end - 3, 4)
        )
        if brands & AVIF_BRANDS:
            return "AVIF"
        if brands & HEIF_BRANDS:
            return "HEIF"
    return None


def _normalize_decoded_format(value: str | None) -> str | None:
    if value in {"HEIC", "HEIF"}:
        return "HEIF"
    return value


def _apply_clahe(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.merge((clahe.apply(lightness), channel_a, channel_b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


def validate_and_prepare_image(
    data: bytes,
    filename: str,
    declared_mime_type: str | None,
    maximum_file_size_mb: float = 15,
) -> tuple[Image.Image, dict[str, Any]]:
    """Validate one supported in-memory image and apply low-light CLAHE."""
    if not data:
        raise ImageValidationError("EMPTY_FILE", "The uploaded image is empty")
    maximum_bytes = round(maximum_file_size_mb * 1024 * 1024)
    if len(data) > maximum_bytes:
        raise ImageValidationError(
            "FILE_TOO_LARGE",
            f"The image exceeds the {maximum_file_size_mb:g} MB limit",
        )

    suffix_format = FORMAT_BY_SUFFIX.get(Path(filename).suffix.lower())
    if suffix_format is None:
        raise ImageValidationError(
            "INVALID_FILE_TYPE",
            "Only JPG, PNG, HEIC/HEIF, WebP, AVIF, and TIFF images are accepted",
        )
    header_format = detect_header_format(data[:256])
    if header_format is None:
        raise ImageValidationError(
            "INVALID_FILE_HEADER",
            "The file header is not a recognized supported image signature",
        )
    if suffix_format != header_format:
        raise ImageValidationError(
            "MIME_MISMATCH",
            "The filename extension does not match the image header",
        )
    if declared_mime_type:
        normalized_mime = declared_mime_type.partition(";")[0].strip().lower()
        if normalized_mime != "application/octet-stream":
            declared_format = FORMAT_BY_MIME.get(normalized_mime)
            if declared_format != header_format:
                raise ImageValidationError(
                    "MIME_MISMATCH",
                    "The declared MIME type does not match the image header",
                )

    try:
        with Image.open(BytesIO(data)) as source:
            source.verify()
        with Image.open(BytesIO(data)) as source:
            detected_format = _normalize_decoded_format(source.format)
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
    except Exception as exc:
        raise ImageValidationError(
            "IMAGE_DECODE_FAILED",
            "The uploaded image could not be decoded",
        ) from exc

    if detected_format != header_format:
        image.close()
        raise ImageValidationError(
            "MIME_MISMATCH",
            "The decoded image format does not match its header",
        )

    width, height = image.size
    short_edge = min(width, height)
    if short_edge < 192:
        image.close()
        raise ImageValidationError(
            "IMAGE_TOO_SMALL",
            f"Image resolution {width}x{height} is below the 192px short-edge minimum",
        )

    rgb = np.asarray(image, dtype=np.uint8)
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(grayscale.mean())
    blur_score = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
    low_resolution = short_edge < 224
    blurry = blur_score < 50.0
    enhancement = "none"

    if brightness < 50.0:
        enhanced = _apply_clahe(rgb)
        image.close()
        image = Image.fromarray(enhanced)
        enhancement = "clahe"

    warnings: list[str] = []
    if low_resolution:
        warnings.append("short_edge_below_preferred")
    if blurry:
        warnings.append("blur_below_threshold")

    quality = {
        "accepted": True,
        "original_width": width,
        "original_height": height,
        "file_size_bytes": len(data),
        "detected_mime_type": MIME_BY_FORMAT[header_format],
        "brightness": brightness,
        "blur_score": blur_score,
        "focus_status": "warning" if blurry else "acceptable",
        "lighting_status": "enhanced" if enhancement == "clahe" else "acceptable",
        "resolution_status": "warning" if low_resolution else "acceptable",
        "enhancement_applied": enhancement,
        "warnings": warnings,
    }
    return image, quality


class ResizeAndPad:
    """Resize to fit in a square and pad without changing image proportions."""

    def __init__(self, size: int) -> None:
        self.size = int(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = self.size / max(width, height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        resized = TF.resize(image, [new_height, new_width], antialias=True)
        left = (self.size - new_width) // 2
        top = (self.size - new_height) // 2
        return TF.pad(
            resized,
            (
                left,
                top,
                self.size - new_width - left,
                self.size - new_height - top,
            ),
            fill=0,
        )


def build_efficientnet_transform(image_size: int) -> transforms.Compose:
    """Reproduce the selected checkpoint's deterministic evaluation transform."""
    return transforms.Compose(
        [
            ResizeAndPad(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# File summary:
# - validate_and_prepare_image accepts JPG, PNG, HEIC/HEIF, WebP, AVIF, and TIFF
#   while enforcing matching filename, declared MIME, header, and decoded format.
# - It reuses the trained pipeline's resolution, blur, brightness, EXIF, RGB, and
#   CLAHE behavior for every supported input format.
# - ResizeAndPad/build_efficientnet_transform reproduce 384px model preprocessing.

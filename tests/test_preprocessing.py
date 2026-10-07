"""Tests for the standalone backend's in-memory image quality contract."""

from __future__ import annotations

import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from app.errors import ImageValidationError
from app.inference.preprocessing import validate_and_prepare_image


def make_jpeg(width: int = 320, height: int = 240, value: int = 140) -> bytes:
    return make_image("JPEG", width, height, value)


def make_image(
    image_format: str,
    width: int = 320,
    height: int = 240,
    value: int = 140,
) -> bytes:
    pixels = np.full((height, width, 3), value, dtype=np.uint8)
    output = BytesIO()
    Image.fromarray(pixels).save(output, format=image_format)
    return output.getvalue()


class PreprocessingTests(unittest.TestCase):
    def test_all_documented_image_formats_are_accepted(self) -> None:
        cases = [
            ("JPEG", "sample.jpg", "image/jpeg", "image/jpeg"),
            ("JPEG", "sample.jpeg", "image/jpeg", "image/jpeg"),
            ("PNG", "sample.png", "image/png", "image/png"),
            ("HEIF", "sample.heic", "image/heic", "image/heif"),
            ("HEIF", "sample.heif", "image/heif", "image/heif"),
            ("WEBP", "sample.webp", "image/webp", "image/webp"),
            ("AVIF", "sample.avif", "image/avif", "image/avif"),
            ("TIFF", "sample.tif", "image/tiff", "image/tiff"),
            ("TIFF", "sample.tiff", "image/tiff", "image/tiff"),
        ]
        for image_format, filename, declared_mime, detected_mime in cases:
            with self.subTest(filename=filename):
                image, quality = validate_and_prepare_image(
                    make_image(image_format),
                    filename,
                    declared_mime,
                )
                try:
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(quality["detected_mime_type"], detected_mime)
                finally:
                    image.close()

    def test_generic_binary_mime_is_accepted_when_header_and_extension_match(
        self,
    ) -> None:
        image, quality = validate_and_prepare_image(
            make_image("HEIF"),
            "camera.heic",
            "application/octet-stream",
        )
        try:
            self.assertEqual(quality["detected_mime_type"], "image/heif")
        finally:
            image.close()

    def test_valid_jpeg_returns_frontend_quality_fields(self) -> None:
        image, quality = validate_and_prepare_image(
            make_jpeg(),
            "sample.jpg",
            "image/jpeg",
        )
        try:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(quality["original_width"], 320)
            self.assertEqual(quality["resolution_status"], "acceptable")
            self.assertIn(quality["focus_status"], {"acceptable", "warning"})
        finally:
            image.close()

    def test_short_edge_below_192_is_rejected(self) -> None:
        with self.assertRaises(ImageValidationError) as context:
            validate_and_prepare_image(
                make_jpeg(width=400, height=191),
                "small.jpg",
                "image/jpeg",
            )
        self.assertEqual(context.exception.code, "IMAGE_TOO_SMALL")

    def test_mime_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ImageValidationError) as context:
            validate_and_prepare_image(
                make_jpeg(),
                "sample.png",
                "image/png",
            )
        self.assertEqual(context.exception.code, "MIME_MISMATCH")

    def test_unsupported_image_extension_is_rejected(self) -> None:
        with self.assertRaises(ImageValidationError) as context:
            validate_and_prepare_image(
                make_image("PNG"),
                "sample.gif",
                "image/gif",
            )
        self.assertEqual(context.exception.code, "INVALID_FILE_TYPE")

    def test_corrupt_supported_image_is_rejected(self) -> None:
        with self.assertRaises(ImageValidationError) as context:
            validate_and_prepare_image(
                b"\x89PNG\r\n\x1a\nnot-a-real-image",
                "broken.png",
                "image/png",
            )
        self.assertEqual(context.exception.code, "IMAGE_DECODE_FAILED")


# File summary: Generated in-memory images verify every supported camera format,
# portable quality fields, hard resolution rejection, corruption handling, and
# MIME/header enforcement without real patient data.

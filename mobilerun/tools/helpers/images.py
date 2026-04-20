"""Small image helpers used by screenshot-only device backends."""

from __future__ import annotations

import struct
from io import BytesIO

from PIL import Image

MODEL_SCREENSHOT_MAX_SIDE = 2048


def image_dimensions(image: bytes) -> tuple[int, int]:
    """Return ``(width, height)`` for PNG or JPEG bytes."""
    if image.startswith(b"\x89PNG\r\n\x1a\n") and len(image) >= 24:
        width, height = struct.unpack(">II", image[16:24])
        return int(width), int(height)

    if image.startswith(b"\xff\xd8"):
        return _jpeg_dimensions(image)

    raise ValueError("Unsupported screenshot image format. Expected PNG or JPEG.")


def fit_dimensions_to_max_side(
    width: int, height: int, max_side: int = MODEL_SCREENSHOT_MAX_SIDE
) -> tuple[int, int]:
    """Return dimensions scaled down so the longest side is at most ``max_side``."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if max(width, height) <= max_side:
        return width, height

    scale = max_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def resize_image_to_max_side(
    image: bytes, max_side: int = MODEL_SCREENSHOT_MAX_SIDE
) -> bytes:
    """Resize image bytes to the same coordinate space exposed to vision agents."""
    width, height = image_dimensions(image)
    target_width, target_height = fit_dimensions_to_max_side(width, height, max_side)
    if (target_width, target_height) == (width, height):
        return image

    with Image.open(BytesIO(image)) as source:
        resized = source.convert("RGBA").resize(
            (target_width, target_height),
            Image.Resampling.LANCZOS,
        )
        output = BytesIO()
        resized.save(output, format="PNG")
        return output.getvalue()


def _jpeg_dimensions(image: bytes) -> tuple[int, int]:
    offset = 2
    length = len(image)
    while offset + 9 < length:
        if image[offset] != 0xFF:
            offset += 1
            continue

        while offset < length and image[offset] == 0xFF:
            offset += 1
        if offset >= length:
            break

        marker = image[offset]
        offset += 1

        if marker in {0xD8, 0xD9}:
            continue
        if marker == 0xDA:
            break
        if offset + 2 > length:
            break

        segment_length = int.from_bytes(image[offset : offset + 2], "big")
        if segment_length < 2:
            raise ValueError("Invalid JPEG segment length.")

        if _is_start_of_frame(marker):
            if offset + 7 > length:
                break
            height = int.from_bytes(image[offset + 3 : offset + 5], "big")
            width = int.from_bytes(image[offset + 5 : offset + 7], "big")
            return int(width), int(height)

        offset += segment_length

    raise ValueError("Could not read JPEG dimensions.")


def _is_start_of_frame(marker: int) -> bool:
    return marker in {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }

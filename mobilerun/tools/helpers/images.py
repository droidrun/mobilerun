"""Small image helpers used by screenshot-only device backends."""

from __future__ import annotations

import math
import struct
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

MODEL_SCREENSHOT_MAX_SIDE = 2048


def anthropic_resized_size(
    width: int, height: int, max_edge: int, max_tokens: int
) -> tuple[int, int]:
    """Dimensions Anthropic will resize an image to before the model sees it.

    Mirrors Anthropic's documented rule: images are split into 28x28 visual
    tokens; the image is shrunk (preserving aspect) until every edge is
    ``<= max_edge`` AND the token count ``<= max_tokens``. ``(max_edge,
    max_tokens)`` is ``(1568, 1568)`` for standard models and ``(2576, 4784)``
    for high-resolution models (Opus 4.7+, etc.).
    """
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")

    def tokens(w: int, h: int) -> int:
        return math.ceil(w / 28) * math.ceil(h / 28)

    def fits(w: int, h: int) -> bool:
        return (
            math.ceil(w / 28) * 28 <= max_edge
            and math.ceil(h / 28) * 28 <= max_edge
            and tokens(w, h) <= max_tokens
        )

    if fits(width, height):
        return width, height
    if height > width:
        rw, rh = anthropic_resized_size(height, width, max_edge, max_tokens)
        return rh, rw

    aspect = width / height
    lo, hi = 1, width
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid, max(round(mid / aspect), 1)):
            lo = mid
        else:
            hi = mid
    return lo, max(round(lo / aspect), 1)


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


def resize_image_to_max_side_with_grid(
    image: bytes, max_side: int = MODEL_SCREENSHOT_MAX_SIDE, divisions: int = 10
) -> bytes:
    """Resize image and overlay a model-only coordinate grid."""
    width, height = image_dimensions(image)
    target_width, target_height = fit_dimensions_to_max_side(width, height, max_side)
    return resize_image_to_dimensions_with_grid(
        image, target_width, target_height, divisions=divisions
    )


def resize_image_to_dimensions_with_grid(
    image: bytes, target_width: int, target_height: int, divisions: int = 10
) -> bytes:
    """Resize image to exact ``target_width``x``target_height`` and overlay the grid.

    Used by the vision coordinate contract so the image the model receives
    matches the declared coordinate space exactly.
    """
    width, height = image_dimensions(image)
    with Image.open(BytesIO(image)) as source:
        screenshot = source.convert("RGBA")
        if (target_width, target_height) != (width, height):
            screenshot = screenshot.resize(
                (target_width, target_height),
                Image.Resampling.LANCZOS,
            )

        _draw_coordinate_grid(screenshot, divisions=divisions)
        output = BytesIO()
        screenshot.save(output, format="PNG")
        return output.getvalue()


def _draw_coordinate_grid(image: Image.Image, divisions: int) -> None:
    width, height = image.size
    if divisions <= 0 or width <= 0 or height <= 0:
        return

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    line_color = (255, 255, 255, 68)
    major_line_color = (255, 230, 120, 110)
    label_fill = (255, 255, 255, 230)
    label_shadow = (0, 0, 0, 190)
    label_bg = (0, 0, 0, 115)

    for index in range(divisions + 1):
        x = round(index * (width - 1) / divisions)
        y = round(index * (height - 1) / divisions)
        color = (
            major_line_color if index in {0, divisions // 2, divisions} else line_color
        )
        draw.line([(x, 0), (x, height - 1)], fill=color, width=1)
        draw.line([(0, y), (width - 1, y)], fill=color, width=1)
        _draw_grid_label(
            draw,
            f"x={x}",
            (min(x + 3, width - 38), 4),
            font,
            label_fill,
            label_shadow,
            label_bg,
        )
        _draw_grid_label(
            draw,
            f"y={y}",
            (4, min(y + 3, height - 14)),
            font,
            label_fill,
            label_shadow,
            label_bg,
        )

    image.alpha_composite(overlay)


def _draw_grid_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    position: tuple[int, int],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    shadow: tuple[int, int, int, int],
    background: tuple[int, int, int, int],
) -> None:
    x, y = position
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle(
        (bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1),
        fill=background,
    )
    draw.text((x + 1, y + 1), text, font=font, fill=shadow)
    draw.text((x, y), text, font=font, fill=fill)


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

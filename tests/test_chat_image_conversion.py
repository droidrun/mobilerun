from io import BytesIO

import pytest
from PIL import Image

from mobilerun.agent.utils.chat_utils import _ensure_image_bytes


@pytest.mark.parametrize("from_path", [False, True])
def test_integer_images_keep_png_conversion_with_pillow_13(tmp_path, from_path):
    image = Image.new("I", (5, 1))
    image.putdata([-1, 0, 256, 65535, 65536])
    source = image
    if from_path:
        source = tmp_path / "integer.tiff"
        image.save(source)

    encoded = _ensure_image_bytes(source)

    with Image.open(BytesIO(encoded)) as decoded:
        assert decoded.format == "PNG"
        assert decoded.size == (5, 1)
        assert [decoded.getpixel((x, 0)) for x in range(5)] == [
            0,
            0,
            256,
            65535,
            65535,
        ]
    assert image.mode == "I"
    assert image.getpixel((4, 0)) == 65536

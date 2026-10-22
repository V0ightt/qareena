import io

import numpy as np
import pytest
from PIL import Image

from qareena.pipeline.intake import IntakeError, load_image_bytes, sniff


def encode(img: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def test_png_and_jpeg_are_accepted_and_returned_as_bgr():
    im = Image.new("RGB", (40, 20), (255, 0, 0))
    for fmt in ("PNG", "JPEG", "WEBP"):
        out = load_image_bytes(encode(im, fmt))
        assert out.format == fmt
        assert out.image.shape == (20, 40, 3)
        assert out.image[0, 0, 2] > 200  # red ends up in the last (R) channel of BGR


def test_format_comes_from_bytes_not_names():
    assert sniff(b"GIF89a....") is None
    with pytest.raises(IntakeError) as e:
        load_image_bytes(b"<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>")
    assert e.value.code == "UNSUPPORTED_FORMAT"


def test_exif_orientation_is_applied_and_metadata_dropped():
    im = Image.new("RGB", (40, 20), (0, 0, 255))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW
    exif[0x8825] = {2: (33, 18, 0)}  # GPS latitude
    out = load_image_bytes(encode(im, "JPEG", exif=exif.tobytes()))
    assert out.image.shape[:2] == (40, 20)  # upright after transpose
    assert isinstance(out.image, np.ndarray)  # plain pixels: no container, no EXIF, no GPS


def test_corrupt_and_oversized_files_are_rejected():
    png = encode(Image.new("RGB", (10, 10)), "PNG")
    with pytest.raises(IntakeError) as e:
        load_image_bytes(png[:30])
    assert e.value.code == "CORRUPT_IMAGE"
    with pytest.raises(IntakeError) as e:
        load_image_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * (16 * 1024 * 1024))
    assert e.value.code == "FILE_TOO_LARGE"


def test_pixel_bomb_is_rejected(monkeypatch):
    import qareena.pipeline.intake as intake

    monkeypatch.setattr(intake, "MAX_PIXELS", 1000)
    with pytest.raises(IntakeError) as e:
        load_image_bytes(encode(Image.new("RGB", (100, 100)), "PNG"))
    assert e.value.code == "TOO_MANY_PIXELS"

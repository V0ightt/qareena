"""Intake guard: the only way an uploaded file becomes an image the pipeline will look at.

* The format is decided from the bytes (magic numbers), never from the name or Content-Type.
* Only PNG, JPEG and WebP are decoded, through Pillow with an explicit format allowlist.
* Size and pixel caps; Pillow's decompression-bomb warning is promoted to an error.
* EXIF orientation is applied, then pixels are copied into a fresh array: all metadata,
  including GPS, is dropped and nothing from the original container survives.
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

MAX_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 40_000_000
_MAGIC = {
    b"\x89PNG\r\n\x1a\n": "PNG",
    b"\xff\xd8\xff": "JPEG",
}


class IntakeError(ValueError):
    """Raised with a stable code: FILE_TOO_LARGE, UNSUPPORTED_FORMAT, TOO_MANY_PIXELS, CORRUPT_IMAGE."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class Intake:
    image: np.ndarray  # BGR uint8, upright, metadata-free
    format: str
    width: int
    height: int


def sniff(data: bytes) -> str | None:
    for magic, fmt in _MAGIC.items():
        if data.startswith(magic):
            return fmt
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None


def load_image_bytes(data: bytes) -> Intake:
    if len(data) > MAX_BYTES:
        raise IntakeError("FILE_TOO_LARGE", f"{len(data)} bytes")
    fmt = sniff(data)
    if fmt is None:
        raise IntakeError("UNSUPPORTED_FORMAT", "not a PNG, JPEG or WebP image")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            with Image.open(io.BytesIO(data), formats=[fmt]) as im:
                if im.width * im.height > MAX_PIXELS:
                    raise IntakeError("TOO_MANY_PIXELS", f"{im.width}x{im.height}")
                im = ImageOps.exif_transpose(im)
                rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        except IntakeError:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise IntakeError("TOO_MANY_PIXELS", str(exc)) from None
        except Exception as exc:
            raise IntakeError("CORRUPT_IMAGE", type(exc).__name__) from None
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    return Intake(bgr, fmt, bgr.shape[1], bgr.shape[0])

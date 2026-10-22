"""Seeded photo degradations for synthetic documents (OpenCV, uint8 BGR images).

Each function returns a new image and never mutates its input. ``glare_over`` returns the
coverage mask too, so labels can be computed by construction: a field whose box is mostly
covered is labelled unreadable (expected output: null).
"""

from __future__ import annotations

import cv2
import numpy as np


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)


def motion_blur(img: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    return cv2.filter2D(img, -1, k / max(k.sum(), 1e-6))


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def downscale(img: np.ndarray, factor: float) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def perspective(
    img: np.ndarray,
    rng: np.random.Generator,
    max_shift: float = 0.08,
    margin: float = 0.12,
    background: tuple[int, int, int] = (90, 95, 100),
) -> np.ndarray:
    """Place the document on a canvas with a random projective tilt, like a hand-held photo."""
    h, w = img.shape[:2]
    mh, mw = int(h * margin), int(w * margin)
    canvas = np.full((h + 2 * mh, w + 2 * mw, 3), background, np.uint8)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-max_shift, max_shift, (4, 2)) * [w, h]
    dst = ([*src, mw, mh, *jitter]).astype(np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        img, m, (canvas.shape[1], canvas.shape[0]), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT
    )


def low_light(img: np.ndarray, gain: float, noise_sigma: float, rng: np.random.Generator) -> np.ndarray:
    dark = img.astype(np.float32) * gain
    dark += rng.normal(0, noise_sigma, img.shape)
    return np.clip(dark, 0, 255).astype(np.uint8)


def glare_over(
    img: np.ndarray, box: tuple[float, float, float, float], strength: float = 1.0, pad: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    """Bright elliptical hotspot covering ``box`` (pixels). Returns (image, coverage mask 0..1)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = (x1 - x0) / 2 * (1 + pad), (y1 - y0) / 2 * (1 + 2 * pad)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2
    mask = np.clip(1.25 - d, 0, 1) ** 0.6 * strength
    out = img.astype(np.float32) * (1 - mask[..., None]) + 255 * mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8), mask


def coverage(mask: np.ndarray, box: tuple[float, float, float, float], threshold: float = 0.6) -> float:
    """Fraction of the box whose mask value exceeds ``threshold`` (i.e. text is washed out)."""
    x0, y0, x1, y1 = (round(v) for v in box)
    region = mask[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
    return float((region > threshold).mean()) if region.size else 0.0

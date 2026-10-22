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
    dst = (src + np.float32([mw, mh]) + jitter).astype(np.float32)  # array arithmetic, not list concat
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
    x0, y0, x1, y1 = (int(round(float(v))) for v in box)  # noqa: RUF046 - numpy floats must become int
    region = mask[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
    return float((region > threshold).mean()) if region.size else 0.0


# --- phone-photo simulation ---------------------------------------------------------------
# Labels must survive every transform, so geometric steps warp the ink map (nearest neighbour)
# with exactly the matrix used for the image.

BACKGROUND_KINDS = ("wood", "fabric", "desk", "stone")


def background(rng: np.random.Generator, shape: tuple[int, int], kind: str | None = None) -> np.ndarray:
    """A procedural table surface (no photos, so no licences and no real documents in shot)."""
    h, w = shape
    kind = kind or str(rng.choice(BACKGROUND_KINDS))
    base = rng.uniform(40, 200, 3)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if kind == "wood":
        freq, warp = rng.uniform(0.015, 0.05), rng.uniform(5, 25)
        grain = np.sin((yy + warp * np.sin(xx * rng.uniform(0.002, 0.01))) * freq) * 0.5 + 0.5
        tex = grain[..., None] * rng.uniform(15, 40)
    elif kind == "fabric":
        tex = ((np.sin(xx * rng.uniform(0.3, 0.8)) * np.sin(yy * rng.uniform(0.3, 0.8))) * 6.0)[..., None]
    elif kind == "stone":
        small = rng.normal(0, 1, (max(2, h // 24), max(2, w // 24))).astype(np.float32)
        tex = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)[..., None] * rng.uniform(8, 18)
    else:  # desk: smooth gradient
        tex = ((xx / w) * rng.uniform(-30, 30) + (yy / h) * rng.uniform(-30, 30))[..., None]
    noise = rng.normal(0, rng.uniform(2, 6), (h, w, 1))
    return np.clip(base + tex + noise, 0, 255).astype(np.uint8)


def place_on_background(
    img: np.ndarray,
    alpha: np.ndarray,
    ink: np.ndarray,
    bg: np.ndarray,
    rng: np.random.Generator,
    *,
    fill: float = 0.75,
    max_shift: float = 0.06,
    offset: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lay the document on ``bg`` with a projective tilt, as a hand-held photo would see it.

    ``fill`` is the document width as a share of the frame; ``offset`` (fractions of the frame,
    may push the document past the edge for a cut-corner case) defaults to a random position
    that keeps it inside. Returns (photo, warped ink map, the four document corners in px).
    """
    h, w = img.shape[:2]
    H, W = bg.shape[:2]
    dw = fill * W
    dh = dw * h / w
    if offset is None:
        ox = rng.uniform(0.04 * W, max(0.04 * W, W - dw - 0.04 * W))
        oy = rng.uniform(0.04 * H, max(0.04 * H, H - dh - 0.04 * H))
    else:
        ox, oy = offset[0] * W, offset[1] * H
    dst = np.float32([[ox, oy], [ox + dw, oy], [ox + dw, oy + dh], [ox, oy + dh]])
    dst = (dst + rng.uniform(-max_shift, max_shift, (4, 2)) * np.float32([dw, dh])).astype(np.float32)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, m, (W, H), flags=cv2.INTER_LINEAR)
    a = cv2.warpPerspective(alpha, m, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255.0
    photo = (warped.astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a)).round().astype(np.uint8)
    ink_w = cv2.warpPerspective(ink, m, (W, H), flags=cv2.INTER_NEAREST)
    return photo, ink_w, dst


def camera(img: np.ndarray, rng: np.random.Generator, *, max_blur: float = 1.0) -> np.ndarray:
    """Camera-like changes that keep text legible (IDSPACE-style ranges): contrast ×0.9–1.1,
    exposure ×0.75–1.0 (auto-exposure keeps plain paper below clipping; clipping is glare's
    job), a colour cast, sensor noise and a little defocus. JPEG comes last, separately."""
    out = img.astype(np.float32)
    mean = float(out.mean())
    out = (out - mean) * rng.uniform(0.9, 1.1) + mean
    out *= rng.uniform(0.75, 1.0)
    out *= rng.uniform(0.95, 1.03, 3)  # white balance
    out += rng.normal(0, rng.uniform(0, 8), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    sigma = rng.uniform(0.0, max_blur)
    return gaussian_blur(out, sigma) if sigma > 0.3 else out


def occlude(
    img: np.ndarray, box: tuple[float, float, float, float], rng: np.random.Generator, cover: float = 0.6
) -> tuple[np.ndarray, np.ndarray]:
    """A fingertip over part of a field (pixels). Returns (image, mask 0..1 where it covers)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    bh = y1 - y0
    rx = max(4.0, (x1 - x0) * cover / 2)
    ry = max(4.0, bh * 1.6)
    cx = rng.uniform(x0 + rx * 0.5, max(x0 + rx * 0.5, x1 - rx * 0.5))
    cy = (y0 + y1) / 2 + rng.uniform(-0.2, 0.2) * bh
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(
        mask, (round(cx), round(cy)), (round(rx), round(ry)), float(rng.uniform(-20, 20)), 0, 360, 1.0, -1
    )
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, bh * 0.08))
    skin = np.float32(rng.choice([(110, 140, 190), (90, 120, 170), (70, 95, 140)]))  # BGR tones
    shade = 0.75 + 0.25 * cv2.GaussianBlur(
        np.random.default_rng(int(rng.integers(1 << 31))).random((h, w)).astype(np.float32), (0, 0), 9
    )
    finger = skin[None, None, :] * shade[..., None]
    out = img.astype(np.float32) * (1 - mask[..., None]) + finger * mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8), mask

import cv2
import numpy as np
import pytest

from qareena.pipeline.quality import assess, find_document, glare_fraction
from qareena.synth import degrade


def card(w=856, h=540, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (236, 240, 244), np.uint8)
    for y in range(80, h - 40, 42):  # text-like strokes: high-frequency detail
        x = 40
        while x < w - 80:
            ln = int(rng.integers(20, 70))
            cv2.rectangle(img, (x, y), (x + ln, y + 18), (30, 40, 60), -1)
            for k in range(x, x + ln, 6):
                cv2.line(img, (k, y), (k, y + 18), (236, 240, 244), 1)
            x += ln + int(rng.integers(10, 30))
    return img


def on_table(c, canvas=(1200, 1700), offset=(220, 300), bg=(70, 85, 95), scale=1.6):
    """Place a card on a plain table; ``scale`` enlarges it like a phone photo, clipped at the frame."""
    if scale != 1:
        size = (round(c.shape[1] * scale), round(c.shape[0] * scale))
        c = cv2.resize(c, size, interpolation=cv2.INTER_CUBIC)
        canvas = (round(canvas[0] * scale), round(canvas[1] * scale))
        offset = (round(offset[0] * scale), round(offset[1] * scale))
    out = np.full((*canvas, 3), bg, np.uint8)
    y, x = offset
    h, w = min(c.shape[0], canvas[0] - y), min(c.shape[1], canvas[1] - x)
    out[y : y + h, x : x + w] = c[:h, :w]
    return out


def test_clean_photo_passes_and_document_is_found():
    r = assess(on_table(card()))
    assert r.ok, r
    assert r.quad is not None
    assert r.rectified is not None
    assert 0.2 < r.metrics["area_frac"] < 0.4


def test_cut_corner():
    assert assess(on_table(card(), offset=(220, 1000))).reason == "CUT_CORNER"  # card runs off the right edge


def test_too_far():
    small = cv2.resize(card(), (300, 190))
    r = assess(on_table(small, offset=(500, 700), scale=1))
    assert r.reason == "TOO_FAR"
    assert r.metrics["doc_long_side_px"] < 900


def test_blur_is_caught():
    assert assess(degrade.gaussian_blur(on_table(card()), 6.0)).reason == "BLUR"


def test_glare_is_caught_but_white_paper_is_not():
    img = on_table(card())
    glared, _ = degrade.glare_over(img, (700, 520, 1450, 840))
    assert assess(glared).reason == "GLARE"
    assert glare_fraction(card()) == pytest.approx(0.0, abs=1e-3)


def test_dark_photo():
    dark = degrade.low_light(on_table(card()), 0.2, 2.0, np.random.default_rng(0))
    assert assess(dark).reason == "TOO_DARK"


def test_tiny_image_is_rejected_before_anything_else():
    assert assess(np.zeros((300, 400, 3), np.uint8)).reason == "TOO_SMALL"


def test_no_outline_falls_back_to_whole_image():
    plain = np.full((1000, 1400, 3), 128, np.uint8)
    assert find_document(plain) is None
    r = assess(plain)
    assert not r.ok
    assert r.notes

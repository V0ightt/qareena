"""Ink-mask classification (no browser needed): hue picks the value, its DOM box bounds it."""

import colorsys

import cv2
import numpy as np
import pytest

pytest.importorskip("jinja2")  # the renderer lives in the optional `synth` group

from qareena.synth.render import _ink_masks, _ink_palette, _tight_box


def _png(img):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _bgr(hue_deg):
    r, g, b = colorsys.hls_to_rgb(hue_deg / 360.0, 0.45, 1.0)
    return (round(b * 255), round(g * 255), round(r * 255))


def test_each_value_gets_its_own_ink_and_a_tight_box():
    hues = _ink_palette(3)
    img = np.full((200, 400, 3), 255, np.uint8)
    cv2.rectangle(img, (20, 40), (120, 60), _bgr(hues[0]), -1)  # "glyphs" of value 0
    cv2.rectangle(img, (200, 40), (300, 60), _bgr(hues[1]), -1)
    # value 1's hue also appears far away (an anti-aliasing artefact): outside its DOM box, ignored
    cv2.rectangle(img, (20, 150), (40, 170), _bgr(hues[1]), -1)
    dom = [[40, 150, 325, 350], [480, 150, 775, 350], None]  # 0-1000, loose like font metrics
    ink = _ink_masks(_png(img), hues, dom)
    assert _tight_box(ink == 1) == [50.0, 200.0, 302.5, 305.0]
    assert _tight_box(ink == 2) == [500.0, 200.0, 752.5, 305.0]
    assert not (ink[150:171, 20:41] == 2).any()
    assert _tight_box(ink == 3) is None

"""Render SPECIMEN document templates to images with exact ground-truth boxes.

Rendering goes through a real browser (Playwright driving the locally installed Microsoft
Edge, so no extra browser download), because browsers shape Arabic correctly. Each value in
a template carries ``data-field="<name>"``; after layout we read the tight text box of every
field from the DOM (``Range.getClientRects``) and normalise it to 0-1000 on the card.

Fonts are embedded as data URIs and checked with ``document.fonts.check`` before anything is
measured: a silently substituted font would make the boxes describe the wrong glyphs.

DOM boxes follow the font's line metrics, which for Arabic fonts reach well above and below the
ink (into the label above a value). So a second pass hides everything but the values, paints
each value in its own hue and screenshots again: classifying pixels by hue gives every field's
exact **ink mask**. Its tight box is the evidence-box ground truth, and the mask is what glare
and occlusion coverage are measured against when a field is labelled unreadable.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "templates"
FONTS = HERE / "fonts"
_ARABIC_DIGITS = str.maketrans("0123456789", "".join(chr(0x0660 + i) for i in range(10)))
_REQUIRED_FONTS = ("Naskh", "Amiri", "PlexAr", "Mono")

STYLES: dict[str, dict[str, str]] = {
    "ink_blue": {
        "font": "Naskh",
        "ink": "#16324f",
        "paper": "#f6f3ea",
        "tint": "#d9e6ef",
        "digits": "arabic",
        "num_font": "Naskh",
        "num_dir": "rtl",
    },
    "ink_green": {
        "font": "Amiri",
        "ink": "#1d3b2f",
        "paper": "#f3f5ee",
        "tint": "#dbe9dc",
        "digits": "latin",
        "num_font": "PlexAr",
        "num_dir": "ltr",
    },
}


@dataclass(frozen=True)
class Rendered:
    image: Path
    width: int
    height: int
    boxes: dict[str, list[float] | None]  # field -> [x0, y0, x1, y1] in 0-1000, None if empty
    template: str
    style: str
    values: dict[str, Any] = field(repr=False, default_factory=dict)
    ink_boxes: dict[str, list[float] | None] = field(default_factory=dict)  # tight ink box, 0-1000
    ink_fields: tuple[str, ...] = ()  # field of each ink-map index (index k + 1)
    ink_map: np.ndarray | None = field(repr=False, default=None)  # uint8, 0 = no value ink
    alpha: np.ndarray | None = field(repr=False, default=None)  # uint8, the card's shape (rounded corners)

    def ground_truth(self) -> dict[str, Any]:
        return {
            "image": self.image.name,
            "template": self.template,
            "style": self.style,
            "size": [self.width, self.height],
            "boxes": self.boxes,
            "ink_boxes": self.ink_boxes,
            "ink_fields": list(self.ink_fields),
            "values": self.values,
        }


@cache
def _font_uri(name: str) -> str:
    data = (FONTS / name).read_bytes()
    return "data:font/ttf;base64," + base64.b64encode(data).decode("ascii")


def _digits(value: Any, style: dict[str, str]) -> str:
    text = str(value)
    return text.translate(_ARABIC_DIGITS) if style["digits"] == "arabic" else text


def _date_fmt(value: Any, fmt: str, style: dict[str, str]) -> str:
    d = value if isinstance(value, dt.date) else dt.date.fromisoformat(str(value))
    return _digits(d.strftime(fmt), style)


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        autoescape=select_autoescape(["html", "j2"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.globals["font"] = _font_uri
    env.filters["digits"] = _digits
    env.filters["date_fmt"] = _date_fmt
    return env


_LOAD_FONTS_JS = """
async () => Promise.all([...document.fonts].map(f =>
  f.load().then(() => [f.family.replace(/"/g, ''), true], () => [f.family.replace(/"/g, ''), false])))
"""

_BOXES_JS = """
() => {
  const card = document.querySelector('.card').getBoundingClientRect();
  const out = {};
  for (const el of document.querySelectorAll('[data-field]')) {
    const r = document.createRange(); r.selectNodeContents(el);
    const rects = Array.from(r.getClientRects()).filter(b => b.width > 0 && b.height > 0);
    if (!rects.length || !el.textContent.trim()) { out[el.dataset.field] = null; continue; }
    const x0 = Math.min(...rects.map(b => b.left)), y0 = Math.min(...rects.map(b => b.top));
    const x1 = Math.max(...rects.map(b => b.right)), y1 = Math.max(...rects.map(b => b.bottom));
    const norm = (v, i) => i % 2 === 0 ? (v - card.left) / card.width : (v - card.top) / card.height;
    out[el.dataset.field] = [x0, y0, x1, y1].map((v, i) => Math.round(10000 * norm(v, i)) / 10);
  }
  return out;
}
"""


_INK_JS = """
(colors) => {
  const style = document.createElement('style');
  style.textContent = `
    .card, .card * { color: transparent !important; background: transparent !important;
      border-color: transparent !important; box-shadow: none !important; text-shadow: none !important; }
    .card { background: #ffffff !important; }
    .card::before, .card::after, .watermark, svg { display: none !important; }`;
  document.head.appendChild(style);
  const els = [...document.querySelectorAll('[data-field]')];
  els.forEach((el, i) => {
    el.style.setProperty('color', colors[i], 'important');
    el.style.setProperty('opacity', '1', 'important');
  });
  return els.map(el => el.dataset.field);
}
"""
_INK_SATURATION = 90  # of 255: faint anti-aliased edges have unreliable hue; their core does not
_INK_PAD = 0.25  # ink must lie within its DOM box grown by this share of the box height


def _ink_palette(n: int) -> list[float]:
    """n hues (degrees), spread around the colour wheel."""
    return [360.0 * k / max(n, 1) for k in range(n)]


def _ink_masks(png: bytes, hues: list[float], boxes: list[list[float] | None]) -> np.ndarray:
    """uint8 map: k + 1 where a pixel carries value k's ink, 0 elsewhere.

    A pixel is value k's ink if its hue is nearest to hue k *and* it lies near value k's DOM
    box (0-1000 units): the DOM box is loose but always holds the ink, while hue alone is
    unreliable on faint anti-aliased edges.
    """
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].astype(np.float32) * 2.0  # OpenCV stores hue / 2
    inked = hsv[..., 1] >= _INK_SATURATION
    h, w = hue.shape
    out = np.zeros((h, w), np.uint8)
    if not hues:
        return out
    tol = 180.0 / len(hues)
    for k, (hk, box) in enumerate(zip(hues, boxes, strict=True)):
        if box is None:
            continue
        x0, y0, x1, y1 = box[0] * w / 1000, box[1] * h / 1000, box[2] * w / 1000, box[3] * h / 1000
        pad = _INK_PAD * (y1 - y0)
        r0, r1 = max(0, int(y0 - pad)), min(h, int(y1 + pad) + 1)
        c0, c1 = max(0, int(x0 - pad)), min(w, int(x1 + pad) + 1)
        d = np.abs((hue[r0:r1, c0:c1] - hk + 180.0) % 360.0 - 180.0)
        region = out[r0:r1, c0:c1]
        region[inked[r0:r1, c0:c1] & (d < tol)] = k + 1
    return out


def _tight_box(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    h, w = mask.shape
    return [
        round(1000 * float(xs.min()) / w, 1),
        round(1000 * float(ys.min()) / h, 1),
        round(1000 * float(xs.max() + 1) / w, 1),
        round(1000 * float(ys.max() + 1) / h, 1),
    ]


class Renderer:
    """Context manager holding one browser; render many documents with it."""

    def __init__(self, scale: float = 2.0, channel: str = "msedge") -> None:
        self.scale, self.channel = scale, channel
        self._env = _env()

    def __enter__(self) -> Renderer:
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(channel=self.channel)
        # large enough for every template (cards and paper documents); the card element is cropped
        self._ctx = self._browser.new_context(
            device_scale_factor=self.scale, viewport={"width": 1400, "height": 1000}
        )
        self._page = self._ctx.new_page()
        return self

    def __exit__(self, *exc: object) -> None:
        self._ctx.close()
        self._browser.close()
        self._pw.stop()

    def render(self, template: str, values: dict[str, Any], out: Path, style: str = "ink_blue") -> Rendered:
        st = STYLES[style]
        html = self._env.get_template(f"{template}.html.j2").render(v=values, style=st)
        page = self._page
        page.set_content(html, wait_until="load")
        # Browsers load fonts lazily; force every declared face to load and fail loudly if one
        # cannot be decoded, instead of silently measuring a fallback font.
        status = page.evaluate(_LOAD_FONTS_JS)
        failed = sorted({fam for fam, ok in status if not ok})
        loaded = {fam for fam, ok in status if ok}
        if failed or not set(_REQUIRED_FONTS) <= loaded:
            raise RuntimeError(f"fonts failed to load: {failed or sorted(set(_REQUIRED_FONTS) - loaded)}")
        boxes = page.evaluate(_BOXES_JS)
        out.parent.mkdir(parents=True, exist_ok=True)
        card = page.locator(".card")
        rgba = cv2.imdecode(
            np.frombuffer(
                card.screenshot(scale="device", animations="disabled", omit_background=True), np.uint8
            ),
            cv2.IMREAD_UNCHANGED,
        )
        alpha = rgba[..., 3]
        a = alpha[..., None].astype(np.float32) / 255.0  # saved on white, like a flatbed scan
        cv2.imwrite(
            str(out), (rgba[..., :3].astype(np.float32) * a + 255.0 * (1.0 - a)).round().astype(np.uint8)
        )
        n = page.evaluate("() => document.querySelectorAll('[data-field]').length")
        hues = _ink_palette(n)
        fields = page.evaluate(_INK_JS, [f"hsl({h:.1f}, 100%, 45%)" for h in hues])
        png = card.screenshot(scale="device", animations="disabled")
        ink_map = _ink_masks(png, hues, [boxes.get(name) for name in fields])
        h, w = ink_map.shape
        ink_boxes = {name: _tight_box(ink_map == k + 1) for k, name in enumerate(fields)}
        return Rendered(out, w, h, boxes, template, style, values, ink_boxes, tuple(fields), ink_map, alpha)


def write_ground_truth(rendered: Rendered) -> Path:
    if rendered.ink_map is not None:
        cv2.imwrite(str(rendered.image.with_suffix(".ink.png")), rendered.ink_map)
    path = rendered.image.with_suffix(".json")
    path.write_text(
        json.dumps(rendered.ground_truth(), ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    return path

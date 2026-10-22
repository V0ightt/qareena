"""Render SPECIMEN document templates to images with exact ground-truth boxes.

Rendering goes through a real browser (Playwright driving the locally installed Microsoft
Edge, so no extra browser download), because browsers shape Arabic correctly. Each value in
a template carries ``data-field="<name>"``; after layout we read the tight text box of every
field from the DOM (``Range.getClientRects``) and normalise it to 0-1000 on the card.

Fonts are embedded as data URIs and checked with ``document.fonts.check`` before anything is
measured: a silently substituted font would make the boxes describe the wrong glyphs.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

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

    def ground_truth(self) -> dict[str, Any]:
        return {
            "image": self.image.name,
            "template": self.template,
            "style": self.style,
            "size": [self.width, self.height],
            "boxes": self.boxes,
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


class Renderer:
    """Context manager holding one browser; render many documents with it."""

    def __init__(self, scale: float = 2.0, channel: str = "msedge") -> None:
        self.scale, self.channel = scale, channel
        self._env = _env()

    def __enter__(self) -> Renderer:
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(channel=self.channel)
        self._ctx = self._browser.new_context(
            device_scale_factor=self.scale, viewport={"width": 960, "height": 640}
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
        page.locator(".card").screenshot(path=str(out), scale="device", animations="disabled")
        from PIL import Image

        with Image.open(out) as im:
            w, h = im.size
        return Rendered(out, w, h, boxes, template, style, values)


def write_ground_truth(rendered: Rendered) -> Path:
    path = rendered.image.with_suffix(".json")
    path.write_text(
        json.dumps(rendered.ground_truth(), ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    return path

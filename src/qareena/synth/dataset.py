"""Build the synthetic evaluation splits: packets, images, labels by construction, manifests.

DESIGN §9.1 and §11. Every packet is a new fictional person and business, rendered from our
SPECIMEN templates and then degraded like a scan or a phone photo. Labels come from the
construction, never from a model:

- each field's **ink mask** is rendered (``render.Rendered.ink_map``) and transformed with the
  image, so its evidence box is exact after perspective;
- a field is **unreadable** when a glare or finger mask hides any character-sized slice of its
  ink by more than half, or more than 20% of its ink overall; **legible** below 5% (and the
  photo is not blurred beyond σ 1); anything in between is **ambiguous** and is reported, not
  scored;
- documents made to fail the quality gate (heavy blur, darkness, a cut corner, too far) carry
  the retake reason they should get.

Leakage rules: values are randomised per packet (years included) and every person is unique
across all splits; each split has its own background stream; layout C appears only in Test-B.
Images are written under ``var/data`` (git-ignored). A manifest with the SHA-256 of every
file goes next to them, and its hash plus the generator commit is the split's **seal**
(``seal`` writes it to ``docs/data/seals/`` to be committed).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import random
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from qareena.config import REPO_ROOT, load_catalog
from qareena.synth import degrade
from qareena.synth.packets import EXPECTED, SCENARIOS, Packet, make_packet
from qareena.synth.render import STYLES, Rendered, Renderer

GENERATOR_VERSION = "1"


@dataclass(frozen=True)
class SplitSpec:
    name: str
    packets: int
    layouts: tuple[str, ...]  # layouts drawn for each document (C is the held-out one)
    stream: int  # seeds this split's own random streams (people, backgrounds, degradations)
    sealed: bool


SPLITS: dict[str, SplitSpec] = {
    "dev": SplitSpec("dev", 80, ("a", "b"), 1, sealed=False),
    "calib": SplitSpec("calib", 150, ("a", "b"), 2, sealed=False),
    "test_a": SplitSpec("test_a", 30, ("a", "b"), 3, sealed=True),
    "test_b": SplitSpec("test_b", 100, ("a", "b", "c"), 4, sealed=True),
}

# Roughly 40% consistent packets; every planted defect gets enough cases per split to measure.
SCENARIO_WEIGHTS: dict[str, float] = {
    "clean": 0.40,
    "name_variant": 0.08,
    "missing_laqab": 0.06,
    "grandfather_conflict": 0.07,
    "sound_alike": 0.06,
    "expired_id": 0.06,
    "expired_chamber": 0.05,
    "trade_conflict": 0.06,
    "mrz_mismatch": 0.06,
    "injection": 0.10,
}
# How each document was "captured".
CONDITION_WEIGHTS: dict[str, float] = {
    "scan": 0.30,  # the render itself, like a flatbed scan
    "photo": 0.42,  # on a table, tilted, camera noise: everything stays legible
    "glare": 0.08,  # photo + a hotspot over one field
    "finger": 0.08,  # photo + a fingertip over one field
    "retake": 0.12,  # photo made to fail the quality gate
}
RETAKE_KINDS = ("BLUR", "TOO_DARK", "CUT_CORNER", "TOO_FAR")
LEGIBLE_MAX = 0.05  # ink share covered below which a field must be read
UNREADABLE_SLICE = 0.5  # a character-sized slice covered beyond this is unreadable
UNREADABLE_TOTAL = 0.2  # ...as is a field with this share of its ink covered
COVERED = 0.6  # a mask value above this hides the ink under it
GLARE_RETAKE = 0.02  # washed-out share of the document beyond which the right answer is a retake


# --- labels -------------------------------------------------------------------------------


def visibility(ink: np.ndarray, cover: np.ndarray | None) -> tuple[str, float, float]:
    """(state, covered share of the ink, worst character-sized slice) for one field's ink mask."""
    if not ink.any():
        return "absent", 0.0, 0.0
    if cover is None:
        return "legible", 0.0, 0.0
    hidden = (cover > COVERED) & ink
    total = float(hidden.sum()) / float(ink.sum())
    ys, xs = np.nonzero(ink)
    height = int(ys.max() - ys.min() + 1)
    worst = 0.0
    for x in range(int(xs.min()), int(xs.max()) + 1, max(2, height // 2)):  # ~ one glyph wide
        col = ink[:, x : x + height]
        if col.sum() > 0:
            worst = max(worst, float(hidden[:, x : x + height].sum()) / float(col.sum()))
    if worst > UNREADABLE_SLICE or total > UNREADABLE_TOTAL:
        return "unreadable", total, worst
    if total < LEGIBLE_MAX:
        return "legible", total, worst
    return "ambiguous", total, worst


def _box_px(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


# --- one document ---------------------------------------------------------------------------


@dataclass
class Capture:
    image: np.ndarray
    ink: np.ndarray  # ink map in the final image's pixels
    condition: str
    expected_quality: str | None  # the retake reason the gate should give, if any
    cover: np.ndarray | None  # glare/finger mask, final image pixels
    target: str | None  # the field the glare or finger was aimed at
    quad: list[list[float]] | None
    params: dict[str, Any]


def capture(r: Rendered, condition: str, rng: np.random.Generator, bg_rng: np.random.Generator) -> Capture:
    """Turn a render into a scan or a phone photo, in physical order: the document on the table,
    a fingertip in front of the lens, the camera (exposure, noise, focus), glare clipping at the
    sensor, JPEG. Every mask is in the final image's pixels."""
    img = cv2.imread(str(r.image))
    ink, alpha = r.ink_map, r.alpha
    if ink is None or alpha is None:
        raise ValueError("render has no ink map or alpha (use Renderer.render)")
    if condition == "scan":
        return Capture(img, ink, condition, None, None, None, None, {})
    h, w = img.shape[:2]
    bg = degrade.background(bg_rng, (round(h * 1.45), round(w * 1.45)))
    expected, params = None, {}
    fill, offset = float(rng.uniform(0.62, 0.85)), None
    if condition == "retake":
        expected = str(rng.choice(RETAKE_KINDS))
        params["retake"] = expected
        if expected == "TOO_FAR":
            fill = float(rng.uniform(0.22, 0.32))
        elif expected == "CUT_CORNER":
            fill = float(rng.uniform(0.7, 0.85))
            offset = (float(rng.uniform(0.25, 0.4)), float(rng.uniform(-0.12, -0.05)))  # past the top edge
    photo, ink_w, quad = degrade.place_on_background(img, alpha, ink, bg, rng, fill=fill, offset=offset)
    params["fill"] = round(fill, 3)
    cover, target = None, None
    if condition in ("glare", "finger"):
        candidates = [
            name
            for k, name in enumerate(r.ink_fields)
            if not name.startswith(("mrz", "_")) and (ink_w == k + 1).any()
        ]
        target = str(rng.choice(candidates))
        box = _box_px(ink_w == r.ink_fields.index(target) + 1)
        if box is None:
            raise RuntimeError(f"no ink for {target}")
        box_f = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    if condition == "finger":
        photo, cover = degrade.occlude(photo, box_f, rng, cover=float(rng.uniform(0.35, 0.9)))
    photo = degrade.camera(photo, rng)
    if condition == "glare":
        photo, cover = degrade.glare_over(photo, box_f, strength=float(rng.uniform(0.9, 1.0)))
        doc_area = cv2.contourArea(quad.astype(np.float32))
        hot = np.zeros(cover.shape, np.uint8)
        cv2.fillConvexPoly(hot, quad.astype(np.int32), 1)
        washed = float(((cover > 0.9) & hot.astype(bool)).sum()) / max(doc_area, 1.0)
        params["glare_share"] = round(washed, 4)
        if washed > GLARE_RETAKE:
            expected = "GLARE"  # big enough that the gate should ask for another photo
    if expected == "BLUR":
        sigma = float(rng.uniform(3.5, 6.0))
        photo, params["blur_sigma"] = degrade.gaussian_blur(photo, sigma), round(sigma, 2)
    elif expected == "TOO_DARK":
        photo = degrade.low_light(photo, float(rng.uniform(0.12, 0.2)), 4.0, rng)
    photo = degrade.jpeg(photo, int(rng.integers(60, 96)))
    return Capture(photo, ink_w, condition, expected, cover, target, quad.round(1).tolist(), params)


# --- packets --------------------------------------------------------------------------------


def _weighted(rng: random.Random, weights: dict[str, float]) -> str:
    return rng.choices(list(weights), weights=list(weights.values()), k=1)[0]


def unique_packets(spec: SplitSpec, seed: int, today: dt.date, seen: set[str]) -> Iterator[Packet]:
    """Packets for one split; every fictional person is new (``seen`` spans all splits)."""
    rng = random.Random(seed * 1000 + spec.stream)
    for i in range(spec.packets):
        scenario = _weighted(rng, SCENARIO_WEIGHTS)
        for _ in range(100):
            packet = make_packet(f"{spec.name}-{i:04d}", scenario, rng, today=today)
            front = packet.values["iq_national_card_front"]
            keys = {f"nid:{front['national_id_number']}", f"name:{packet.meta['owner']}"}
            if seen.isdisjoint(keys):
                seen.update(keys)
                yield packet
                break
        else:
            raise RuntimeError("could not draw a new fictional person in 100 tries")


def build_split(
    spec: SplitSpec, out: Path, *, seed: int, today: dt.date, seen: set[str], renderer: Renderer
) -> dict[str, Any]:
    """Render, degrade and label one split under ``out/<split>``. Returns the split record."""
    root = out / spec.name
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed * 7919 + spec.stream)
    nrng = np.random.default_rng([seed, spec.stream, 1])
    bg_rng = np.random.default_rng([seed, spec.stream, 2])  # this split's own background stream
    catalog = load_catalog(REPO_ROOT / "config" / "documents.yaml")
    manifest = []
    for packet in unique_packets(spec, seed, today, seen):
        pdir = root / packet.packet_id
        pdir.mkdir(exist_ok=True)
        docs: dict[str, Any] = {}
        unreadable_required = False
        for doc_type, values in packet.values.items():
            layout = rng.choice(spec.layouts)
            style = rng.choice(sorted(STYLES))
            condition = _weighted(rng, CONDITION_WEIGHTS)
            rendered = renderer.render(
                f"{doc_type}_{layout}", values, pdir / f".{doc_type}.render.png", style=style
            )
            cap = capture(rendered, condition, nrng, bg_rng)
            image_path = pdir / f"{doc_type}.png"
            cv2.imwrite(str(image_path), cap.image)
            rendered.image.unlink()
            fields = {}
            for k, name in enumerate(rendered.ink_fields):
                if name.startswith("_"):
                    continue
                state, total, worst = visibility(cap.ink == k + 1, cap.cover)
                if cap.expected_quality is not None and state == "legible":
                    state = "ambiguous"  # the gate should stop this photo before anyone reads it
                fields[name] = {
                    "state": state,
                    "covered": round(total, 3),
                    "worst_slice": round(worst, 3),
                    "box": _box_px(cap.ink == k + 1),
                }
                if state == "unreadable" and catalog.documents[doc_type].fields[name].required:
                    unreadable_required = True  # an optional field hidden is fine: null is right
            docs[doc_type] = {
                "image": image_path.name,
                "layout": layout,
                "held_out_layout": layout == "c",
                "style": style,
                "condition": cap.condition,
                "expected_quality": cap.expected_quality,
                "target_field": cap.target,
                "quad": cap.quad,
                "params": cap.params,
                "size": [int(cap.image.shape[1]), int(cap.image.shape[0])],
                "fields": fields,
            }
        retakes = {d: v["expected_quality"] for d, v in docs.items() if v["expected_quality"]}
        if retakes:
            expected = "AWAITING_RETAKE"
        elif unreadable_required or EXPECTED[packet.scenario] == "NEEDS_REVIEW":
            expected = "NEEDS_REVIEW"
        else:
            expected = "PRE_CHECKED"
        truth = {
            "packet_id": packet.packet_id,
            "split": spec.name,
            "scenario": packet.scenario,
            "defects": list(packet.defects),
            "expected_outcome": expected,
            "expected_retakes": retakes,
            "today": today.isoformat(),
            "values": packet.values,
            "documents": docs,
        }
        truth_path = pdir / "truth.json"
        truth_path.write_text(
            json.dumps(truth, ensure_ascii=False, indent=1, default=_json_default),
            encoding="utf-8",
            newline="\n",
        )
        files = [*sorted(pdir.glob("*.png")), truth_path]
        manifest.append({"packet_id": packet.packet_id, "files": {f.name: sha256(f) for f in files}})
    manifest_path = root / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(m, sort_keys=True) + "\n" for m in manifest), encoding="utf-8", newline="\n"
    )
    record = {
        "split": spec.name,
        "packets": len(manifest),
        "seed": seed,
        "today": today.isoformat(),
        "generator_version": GENERATOR_VERSION,
        "generator_commit": _git_commit(),
        "spec": dataclasses.asdict(spec),
        "scenario_weights": SCENARIO_WEIGHTS,
        "condition_weights": CONDITION_WEIGHTS,
        "manifest_sha256": sha256(manifest_path),
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    (root / "split.json").write_text(json.dumps(record, indent=1), encoding="utf-8", newline="\n")
    return record


def seal(split_dir: Path, seals_dir: Path = REPO_ROOT / "docs" / "data" / "seals") -> Path:
    """Write the tamper-evident seal of a built split (commit it; the images stay private)."""
    record = json.loads((split_dir / "split.json").read_text(encoding="utf-8"))
    if sha256(split_dir / "manifest.jsonl") != record["manifest_sha256"]:
        raise RuntimeError(f"{split_dir.name}: manifest changed since it was built")
    seals_dir.mkdir(parents=True, exist_ok=True)
    path = seals_dir / f"{record['split']}.json"
    keep = (
        "split",
        "packets",
        "seed",
        "today",
        "generator_version",
        "generator_commit",
        "manifest_sha256",
        "created",
    )
    path.write_text(json.dumps({k: record[k] for k in keep}, indent=1) + "\n", encoding="utf-8", newline="\n")
    return path


def verify(split_dir: Path) -> list[str]:
    """Problems found re-hashing a split against its manifest (empty when intact)."""
    problems = []
    for line in (split_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        for name, digest in entry["files"].items():
            path = split_dir / entry["packet_id"] / name
            if not path.exists():
                problems.append(f"missing {entry['packet_id']}/{name}")
            elif sha256(path) != digest:
                problems.append(f"changed {entry['packet_id']}/{name}")
    return problems


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "config"],  # noqa: S607
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _json_default(o: object) -> object:
    if isinstance(o, dt.date):
        return o.isoformat()
    raise TypeError(type(o).__name__)


assert set(SCENARIO_WEIGHTS) == set(SCENARIOS)  # noqa: S101 - every scenario has a weight

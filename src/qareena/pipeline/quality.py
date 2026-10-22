"""Capture-quality gate: find the document, measure the photo, decide whether to ask for a retake.

Runs before any model, in well under a second, so the applicant gets guidance immediately.
Checks run in a fixed order and the first failure is the one reported, because one clear
instruction ("move closer") beats a list of problems.

All measurements are taken on the *document region* (or the whole image when no outline is
found): glare on the table around a card must not count against the card. Thresholds are
starting points to be fitted on labelled development photos, not universal constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

CARD_ASPECT = 85.60 / 53.98  # ISO/IEC 7810 ID-1


@dataclass(frozen=True)
class QualityThresholds:
    min_area_frac: float = 0.10  # framing floor: the document must not be a speck in the frame
    min_doc_long_side_px: int = 900  # legibility: pixels across the document's long side
    corner_margin_frac: float = 0.01  # a corner closer than this to the edge counts as cut
    max_tilt_deg: float = 25.0
    min_sharpness: float = 60.0  # variance of Laplacian on the rectified card at 1000 px width
    max_glare_frac: float = 0.02  # share of the document that is blown-out highlight
    min_brightness: float = 55.0  # mean grey level 0..255
    max_brightness: float = 240.0
    min_ink_contrast: float = 170.0  # TOO_BRIGHT only if even the darkest 1% (the ink) is paler than this
    min_long_side_px: int = 600  # below this, small print is not legible


@dataclass(frozen=True)
class QualityReport:
    ok: bool
    reason: str | None  # first failing check: CUT_CORNER, TOO_FAR, TILTED, BLUR, GLARE, TOO_DARK, ...
    metrics: dict[str, float]
    quad: np.ndarray | None = field(default=None, repr=False)  # 4x2 float32, TL TR BR BL
    rectified: np.ndarray | None = field(default=None, repr=False)
    notes: tuple[str, ...] = ()


def assess(img: np.ndarray, t: QualityThresholds | None = None) -> QualityReport:
    t = t or QualityThresholds()
    h, w = img.shape[:2]
    metrics: dict[str, float] = {"width": float(w), "height": float(h)}
    if max(h, w) < t.min_long_side_px:
        return QualityReport(False, "TOO_SMALL", metrics)

    quad = find_document(img)
    notes: list[str] = []
    if quad is not None and _is_internal_element(img, quad):
        notes.append("document fills the frame (scan or tight crop); found box is an internal element")
        quad = None
    if quad is not None and int(_near_edge(quad, w, h, t).sum()) == 4:
        # A cut-off document is closed by the frame at one to three corners. A box with all four
        # corners on the frame spans it: the document fills the frame, or this is a full-width
        # band inside such a document (a header strip, the MRZ zone). Read the whole image; a
        # field that is really cut off then comes back missing and goes to review.
        notes.append("outline spans the frame (scan or tight crop); measuring the whole image")
        quad = None
    if quad is None:
        notes.append("no document outline found; measuring the whole image")
        region = img
    else:
        area = cv2.contourArea(quad.astype(np.float32)) / float(w * h)
        metrics["area_frac"] = area
        near_edge = _near_edge(quad, w, h, t)
        metrics["corners_at_edge"] = float(near_edge.sum())
        metrics["tilt_deg"] = _tilt_deg(quad)
        metrics["doc_long_side_px"] = _long_side(quad)
        if near_edge.any():
            return QualityReport(False, "CUT_CORNER", metrics, quad, None, tuple(notes))
        if area < t.min_area_frac or metrics["doc_long_side_px"] < t.min_doc_long_side_px:
            return QualityReport(False, "TOO_FAR", metrics, quad, None, tuple(notes))
        if metrics["tilt_deg"] > t.max_tilt_deg:
            return QualityReport(False, "TILTED", metrics, quad, None, tuple(notes))
        region = rectify(img, quad)

    grey = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(
        grey, (1000, max(1, round(1000 * grey.shape[0] / grey.shape[1]))), interpolation=cv2.INTER_AREA
    )
    metrics["sharpness"] = float(cv2.Laplacian(scaled, cv2.CV_64F).var())
    metrics["brightness"] = float(scaled.mean())
    metrics["ink_p1"] = float(np.percentile(scaled, 1))  # how dark the darkest print still is
    metrics["glare_frac"] = glare_fraction(region)
    rect = region if quad is not None else None
    for reason, failed in (
        ("TOO_DARK", metrics["brightness"] < t.min_brightness),
        # A well-lit white document is bright; it is overexposed only when the ink washes out.
        ("TOO_BRIGHT", metrics["brightness"] > t.max_brightness and metrics["ink_p1"] > t.min_ink_contrast),
        ("BLUR", metrics["sharpness"] < t.min_sharpness),
        ("GLARE", metrics["glare_frac"] > t.max_glare_frac),
    ):
        if failed:
            return QualityReport(False, reason, metrics, quad, rect, tuple(notes))
    return QualityReport(True, None, metrics, quad, rect, tuple(notes))


def find_document(img: np.ndarray) -> np.ndarray | None:
    """Largest convex quadrilateral that plausibly is the document, ordered TL, TR, BR, BL."""
    h, w = img.shape[:2]
    scale = 800 / max(h, w)
    small = (
        cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        if scale < 1
        else img
    )
    s = scale if scale < 1 else 1.0
    grey = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(grey, 40, 120), np.ones((3, 3), np.uint8))
    frame_area = small.shape[0] * small.shape[1]

    quad = _largest_quad(cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], frame_area)
    if quad is None:
        # A document that runs off the frame has no closed outline: the frame edge is its
        # missing side. Close the outline against the border and look again; a quad found
        # this way touches the border, which ``assess`` reports as CUT_CORNER.
        closed = edges.copy()
        closed[[0, -1], :] = 255
        closed[:, [0, -1]] = 255
        quad = _largest_quad(cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0], frame_area)
    return None if quad is None else order_corners(quad / s)


def _largest_quad(contours, frame_area: float, max_frac: float = 0.90) -> np.ndarray | None:
    """Largest region whose convex hull simplifies to a quadrilateral.

    The hull removes dents where text touches the outline (e.g. text cut off by the frame).
    Hulls covering most of the frame are the frame or the table, never the document.
    """
    hulls = [cv2.convexHull(c) for c in contours]
    for hull in sorted(hulls, key=cv2.contourArea, reverse=True)[:12]:
        area = cv2.contourArea(hull)
        if area > max_frac * frame_area:
            continue
        if area < 0.02 * frame_area:
            break
        approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None


def order_corners(pts: np.ndarray) -> np.ndarray:
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)


def rectify(img: np.ndarray, quad: np.ndarray, width: int = 1712) -> np.ndarray:
    """Warp the document to a flat, upright card (ID-1 aspect when the outline is card-like)."""
    tl, tr, br, bl = quad
    measured = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / max(
        np.linalg.norm(bl - tl) + np.linalg.norm(br - tr), 1e-6
    )
    aspect = CARD_ASPECT if abs(measured - CARD_ASPECT) < 0.35 else measured
    height = round(width / aspect)
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(quad, dst), (width, height))


GLARE_INK_FRAC = 0.02  # dark print enclosed by a clipped blob: printed white space, not glare


def glare_fraction(region: np.ndarray, border_frac: float = 0.03) -> float:
    """Share of the document that is *clipped* highlight (every channel >= 253), in blobs.

    Specular glare saturates the sensor and washes out the ink under it. White paper and
    white printed strips can clip too, but the dark text printed on them survives: a blob that
    encloses dark ink is print, not glare. A thin border band is ignored (rounded corners and
    the table edge), as are specks smaller than 0.05% of the region.
    """
    blurred = cv2.GaussianBlur(region, (5, 5), 0)
    clipped = (blurred.min(axis=2) >= 253).astype(np.uint8)
    dark = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) < 110
    bh, bw = round(border_frac * clipped.shape[0]), round(border_frac * clipped.shape[1])
    mask = np.zeros_like(clipped)
    mask[bh : clipped.shape[0] - bh, bw : clipped.shape[1] - bw] = clipped[
        bh : clipped.shape[0] - bh, bw : clipped.shape[1] - bw
    ]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_blob = 0.0005 * mask.size
    blown = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_blob:
            continue
        x, y, bw_, bh_ = (int(v) for v in stats[i, :4])
        blob = (labels[y : y + bh_, x : x + bw_] == i).astype(np.uint8)
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(blob)
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
        enclosed = filled.astype(bool) & ~blob.astype(bool)
        ink = float((dark[y : y + bh_, x : x + bw_] & enclosed).sum()) / max(float(filled.sum()), 1.0)
        if ink < GLARE_INK_FRAC:
            blown += area
    return blown / float(mask.size)


def _near_edge(quad: np.ndarray, w: int, h: int, t: QualityThresholds) -> np.ndarray:
    margin = t.corner_margin_frac * max(w, h)
    return (
        (quad[:, 0] < margin)
        | (quad[:, 0] > w - 1 - margin)
        | (quad[:, 1] < margin)
        | (quad[:, 1] > h - 1 - margin)
    )


def _is_internal_element(img: np.ndarray, quad: np.ndarray) -> bool:
    """True when the found box is a part of a document that fills the frame.

    A real document lying on a table has little texture around it; a photo box or MRZ band
    inside a full-frame scan is surrounded by print. Compare edge density outside the box
    with inside: much busier outside means the box is not the document.
    """
    h, w = img.shape[:2]
    if cv2.contourArea(quad) / float(w * h) >= 0.5:
        return False
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(grey, (5, 5), 0), 40, 120) > 0
    inside = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(inside, quad.astype(np.int32), 1)
    inside_b = inside.astype(bool)
    d_in = edges[inside_b].mean() if inside_b.any() else 0.0
    d_out = edges[~inside_b].mean() if (~inside_b).any() else 0.0
    return d_out > 0.5 * d_in and d_out > 0.01


def _tilt_deg(quad: np.ndarray) -> float:
    """How far the outline is from a frontal rectangle: max deviation of corner angles from 90°."""
    worst = 0.0
    for i in range(4):
        a, b, c = quad[i - 1], quad[i], quad[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cos = float(np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-6))
        worst = max(worst, abs(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))) - 90.0))
    return worst


def _long_side(quad: np.ndarray) -> float:
    tl, tr, br, bl = quad
    return float(
        max(
            np.linalg.norm(tr - tl), np.linalg.norm(br - bl), np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
        )
    )

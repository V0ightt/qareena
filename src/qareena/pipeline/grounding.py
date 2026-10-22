"""Ground a VLM-extracted value in the independent OCR reader's lines, *at the field's label*.

A value is *grounded* when the OCR engine, reading the same photo on its own, also produced
it **where that field's value is printed**. Location matters: in the day-1 spike the VLM
sometimes put one field's value under another key (the grandfather's name returned as the
father's). That value does occur on the page, so matching it anywhere would "confirm" a wrong
answer and quietly break the two-source rule (ADR-004).

Grounding therefore has two steps:

1. ``map_labels`` finds the OCR lines that carry each field's printed label (Arabic or an
   alias, matched as whole tokens, longest label first) and lets each label claim the text
   that follows it in Arabic reading order: the lines just to its left on the same row,
   otherwise the first line directly below it.
2. ``ground_field`` matches the value only against lines claimed by that field (or its own
   label line, for "label: value" printed as one line). A value found only at *another*
   field's label is ``foreign``: a likely field shift. When the field's own label is not
   found (an unfamiliar layout), any line not claimed by another field may support it, and
   the result is marked not ``anchored`` so the scorer can discount it.

Matching happens at the spelling layer (digits-only for numeric kinds), so the readers may
differ in invisible marks or alef forms but not in letters. An OCR line shorter than the value
is compared whole (``ratio``), never as a fragment, so half a name cannot ground a full name.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from qareena.config import DocumentSpec, FieldKind
from qareena.readers.ocr import OcrLine
from qareena.text import arabic

MATCH = 0.90  # similarity (0..1) at which an OCR line counts as containing the value
SAME_ROW = 0.4  # vertical overlap, as a share of the shorter box, for two boxes to share a row
FIRST_GAP = 12.0  # widest label-to-value gap along a row, in label heights (label columns are wide)
LABEL_SIZED = 1.2  # text no taller than this many label heights is label-sized (values print larger)
NEXT_GAP = 1.5  # widest gap between pieces of one value along a row, in line heights
BELOW_GAP = 2.5  # deepest label-to-value gap below a label, in label heights
FUZZY_LETTERS = 5  # labels this long are recognised despite an OCR misread («الاسه» for «الاسم»)
FUZZY_EDITS = 0.15  # ... with at most this share of letters edited (and at least one): a misread, not
# a different label («اسم المحل» is not «اسم المكلف», though their similarity ratio is 0.82)
NOISE_SCORE = 0.5  # OCR lines below this confidence are background noise
NOISE_SKEW = (
    0.35  # rise over run of a line's top edge (~19 degrees; short words wobble to ~0.25): a rotated watermark
)

Box = tuple[float, float, float, float]
_PUNCT = re.compile(
    "[" + re.escape(":/|()[],.-_" + chr(0x060C) + chr(0x061B) + chr(0x2013) + chr(0x2014)) + "]+"
)


@dataclass(frozen=True)
class Grounding:
    matches: int  # eligible OCR lines containing the value at >= MATCH similarity
    similarity: float  # best similarity over eligible lines, 0..1
    line_index: int | None
    box: Box | None  # pixels, from the best line
    ocr_score: float | None  # the OCR engine's own confidence for that line
    exact: bool  # the value occurs verbatim (after folding) in the best line
    anchored: bool = False  # the supporting lines sit at this field's own label
    foreign: bool = False  # the value was found only at another field's label


@dataclass(frozen=True)
class LabelMap:
    n_lines: int
    label_lines: dict[int, frozenset[str]]  # OCR line index -> fields whose label it carries
    label_box: dict[str, Box]  # field -> union box of its label lines
    claims: dict[str, tuple[int, ...]]  # field -> its value lines, in reading order
    owner: dict[int, str]  # claimed line index -> field

    def eligible(self, name: str) -> tuple[list[int], bool]:
        """(lines that may support field ``name``, whether they are anchored at its own label)."""
        if name in self.label_box:
            inline = [i for i, fs in self.label_lines.items() if name in fs]
            return [*self.claims.get(name, ()), *inline], True
        return [i for i in range(self.n_lines) if i not in self.owner and i not in self.label_lines], False


def _tokens(text: str) -> list[str]:
    return _PUNCT.sub(" ", arabic.spelling(text)).split()


def _fold(text: str, kind: FieldKind) -> str:
    if kind in (FieldKind.DIGITS, FieldKind.DATE):
        return "".join(ch for ch in arabic.normalize_digits(text) if ch.isdigit())
    if kind in (FieldKind.CODE, FieldKind.MRZ_LINE):
        return "".join(arabic.strict(text).split()).upper()
    return arabic.spelling(text).replace(" ", "")


def _height(b: Box) -> float:
    return max(b[3] - b[1], 1.0)


def _union(boxes: Sequence[Box]) -> Box:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _polygon(b: Box) -> tuple[tuple[float, float], ...]:
    return ((b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3]))


def _label_variants(doc: DocumentSpec) -> list[tuple[str, list[str]]]:
    """(field, label tokens) for every printed label, plus Arabic+Kurdish glued by the OCR detector."""
    out = []
    for name, spec in doc.fields.items():
        labs = [lab for lab in (spec.label_ar, *spec.aliases) if _tokens(lab)]
        out += [(name, _tokens(lab)) for lab in labs]
        out += [(name, ["".join(_tokens(a)) + "".join(_tokens(b))]) for a in labs for b in labs if a != b]
    return out


def _match_label(toks: list[str], variants: list[tuple[str, list[str]]]) -> tuple[set[str], bool]:
    """(fields whose printed label is in ``toks``, whether other tokens remain on the line)."""
    spans = [
        (sum(map(len, lt)), s, s + len(lt), name)
        for name, lt in variants
        for s in range(len(toks) - len(lt) + 1)
        if toks[s : s + len(lt)] == lt
    ]
    used: set[int] = set()
    names: set[str] = set()
    for _, s, e, name in sorted(spans, reverse=True):  # longest first: «أب الأم» beats «الأم»
        if used.isdisjoint(range(s, e)):
            used.update(range(s, e))
            names.add(name)
    if len(used) == len(toks):
        return names, False
    # a misread label («الاسه», «باوى دايك» for «باوکی دایک»): the whole line, fuzzy, long labels only
    text = arabic.sound("".join(toks))
    longest = max((len("".join(toks[s:e])) for _, s, e, _ in spans), default=0)
    fuzzy = [
        (Levenshtein.distance(text, lab), -len(lab), name)
        for name, lt in variants
        if len(lab := arabic.sound("".join(lt))) >= max(FUZZY_LETTERS, longest + 1)
        and -1 <= len(text) - len(lab) <= 2  # a dropped letter, not a dropped article: «غرفة» ≠ «الغرفة»
    ]
    for dist, neg_len, name in sorted(fuzzy):
        if dist <= max(1, int(FUZZY_EDITS * -neg_len)):
            return {name}, False
        break
    return names, bool(names)


def _find_labels(doc: DocumentSpec, lines: Sequence[OcrLine]) -> dict[int, frozenset[str]]:
    """OCR line index -> fields whose printed label it carries."""
    variants = _label_variants(doc)
    found: dict[int, frozenset[str]] = {}
    toks = [_tokens(ln.text) for ln in lines]
    for i in range(len(lines)):
        names, _ = _match_label(toks[i], variants)
        if names:
            found[i] = frozenset(names)
    boxes = [ln.bbox for ln in lines]
    for i in range(len(lines)):  # a label the detector split in two («فصيلة» | «الدم»)
        if i in found:
            continue
        left = [
            j
            for j in range(len(lines))
            if j != i and j not in found and _same_row(boxes[j], boxes[i]) and _left_of(boxes[j], boxes[i])
        ]
        if not left:
            continue
        j = max(left, key=lambda j: boxes[j][2])
        if boxes[i][0] - boxes[j][2] > _height(boxes[i]):
            continue
        names, rest = _match_label(toks[i] + toks[j], [(n, lt) for n, lt in variants if len(lt) > 1])
        if names and not rest:
            found[i] = found[j] = frozenset(names)
    return found


def _same_row(a: Box, b: Box) -> bool:
    return min(a[3], b[3]) - max(a[1], b[1]) >= SAME_ROW * min(_height(a), _height(b))


def _left_of(b: Box, cur: Box) -> bool:
    """``b`` lies left of ``cur``; neighbouring Arabic word boxes may overlap a little."""
    return (b[0] + b[2]) / 2 < cur[0] and b[2] <= cur[0] + 0.6 * _height(cur)


def _noise(line: OcrLine) -> bool:
    """Watermark fragments, rotated stamps and stray glyphs: transparent when walking a row."""
    text = line.text.strip()
    if line.score < NOISE_SCORE or len(text) < 2 or not any(ch.isalnum() for ch in text):
        return True
    (x0, y0), (x1, y1) = line.polygon[0], line.polygon[1]
    return abs(x1 - x0) > 0 and abs(y1 - y0) / abs(x1 - x0) > NOISE_SKEW


def _left_chain(
    start: Box, first_gap: float, boxes: Sequence[Box], skip: Collection[int], stop: Collection[int]
) -> list[int]:
    """Lines left of ``start`` on its row, nearest first, ending at a label line or a wide gap."""
    chain: list[int] = []
    cur, limit = start, first_gap
    while True:
        cands = [
            j
            for j, b in enumerate(boxes)
            if j not in chain and j not in skip and b != cur and _same_row(b, cur) and _left_of(b, cur)
        ]
        if not cands:
            return chain
        j = max(cands, key=lambda j: boxes[j][2])  # the nearest box to the left
        if j in stop or cur[0] - boxes[j][2] > limit:
            return chain
        chain.append(j)
        cur, limit = boxes[j], NEXT_GAP * _height(boxes[j])


def _gap(b: Box, lab: Box) -> float:
    return max(0.0, lab[0] - b[2]) + max(0.0, b[1] - lab[3])


def map_labels(doc: DocumentSpec, lines: Sequence[OcrLine]) -> LabelMap:
    boxes = [ln.bbox for ln in lines]
    label_lines = _find_labels(doc, lines)
    noise = {j for j, ln in enumerate(lines) if j not in label_lines and _noise(ln)}
    by_field: dict[str, list[int]] = {}
    for i, names in label_lines.items():
        for name in names:
            by_field.setdefault(name, []).append(i)
    label_box: dict[str, Box] = {}
    label_h: dict[str, float] = {}
    for name, idx in by_field.items():
        a = boxes[max(idx, key=lambda i: len(lines[i].text))]  # the Arabic label, usually
        near = [  # the label with its Kurdish translation, not a repeat elsewhere on the page
            i
            for i in idx
            if abs(boxes[i][1] - a[1]) <= 2.5 * _height(a) and abs(boxes[i][2] - a[2]) <= 15 * _height(a)
        ]
        label_box[name] = _union([boxes[i] for i in near])
        label_h[name] = sorted(_height(boxes[i]) for i in near)[len(near) // 2]
    claims: dict[str, list[int]] = {}
    best: dict[int, tuple[float, str]] = {}
    for name, lab in label_box.items():
        lh = label_h[name]
        own = set(by_field[name])
        skip, stop = noise | own, set(label_lines) - own
        chain = _left_chain(lab, FIRST_GAP * lh, boxes, skip, stop)
        # Nothing, or only label-sized text (an unrecognised translation), follows on the row:
        # the value is the first row below the label block.
        if all(_height(boxes[j]) <= LABEL_SIZED * lh for j in chain):
            below = [
                j
                for j, b in enumerate(boxes)
                if j not in skip and b[1] >= lab[3] - 0.5 * lh and min(b[2], lab[2]) > max(b[0], lab[0])
            ]
            if below:
                top = min(below, key=lambda j: boxes[j][1])
                row = [j for j in below if _same_row(boxes[j], boxes[top])]
                first = max(row, key=lambda j: boxes[j][2])  # Arabic reads from the right
                if first not in stop and boxes[first][1] - lab[3] <= BELOW_GAP * lh:
                    rest = _left_chain(boxes[first], NEXT_GAP * _height(boxes[first]), boxes, skip, stop)
                    chain += [first, *rest]
        claims[name] = chain
        for j in chain:  # a line claimed by two labels belongs to the nearer one
            d = _gap(boxes[j], lab)
            if j not in best or d < best[j][0]:
                best[j] = (d, name)
    owner = {j: name for j, (_, name) in best.items()}
    kept = {name: tuple(j for j in chain if owner[j] == name) for name, chain in claims.items()}
    return LabelMap(len(lines), label_lines, label_box, kept, owner)


def ground(
    value: str | None, kind: FieldKind, lines: Sequence[OcrLine], only: Collection[int] | None = None
) -> Grounding:
    """Best match of ``value`` among ``lines`` (or only the line indices in ``only``)."""
    if not value:
        return Grounding(0, 0.0, None, None, None, False)
    target = _fold(value, kind)
    if not target:
        return Grounding(0, 0.0, None, None, None, False)
    best_i, best_sim, n = None, 0.0, 0
    for i, line in enumerate(lines):
        if only is not None and i not in only:
            continue
        hay = _fold(line.text, kind)
        if not hay:
            continue
        if target in hay:
            sim = 1.0
        elif len(hay) < len(target):
            sim = fuzz.ratio(target, hay) / 100.0  # a shorter line must match the whole value
        else:
            sim = fuzz.partial_ratio(target, hay) / 100.0
        if sim >= MATCH:
            n += 1
        if sim > best_sim:
            best_i, best_sim = i, sim
    if best_i is None:
        return Grounding(0, 0.0, None, None, None, False)
    line = lines[best_i]
    return Grounding(n, best_sim, best_i, line.bbox, line.score, target in _fold(line.text, kind))


def ground_field(
    name: str, value: str | None, kind: FieldKind, lines: Sequence[OcrLine], labels: LabelMap
) -> Grounding:
    """Ground ``value`` at the label of field ``name`` (see the module docstring)."""
    allowed, anchored = labels.eligible(name)
    candidates = list(lines)
    claimed = labels.claims.get(name, ())
    if len(claimed) > 1:  # one value that the OCR detector split into pieces along a row
        candidates.append(
            OcrLine(
                " ".join(lines[j].text for j in claimed),
                min(lines[j].score for j in claimed),
                _polygon(_union([lines[j].bbox for j in claimed])),
            )
        )
        allowed = [*allowed, len(candidates) - 1]
    g = ground(value, kind, candidates, only=set(allowed))
    if g.matches:
        return Grounding(g.matches, g.similarity, g.line_index, g.box, g.ocr_score, g.exact, anchored, False)
    anywhere = ground(value, kind, lines)
    foreign = anywhere.matches > 0 and (
        anywhere.line_index in labels.owner or anywhere.line_index in labels.label_lines
    )
    return Grounding(0, g.similarity, None, None, None, False, anchored, foreign)


REVERSED_MATCH = 0.8  # a backwards reading also drops letters («عر» for «رعد»), so match more loosely
_TEXT_KINDS = (FieldKind.NAME_PART, FieldKind.FULL_NAME, FieldKind.TEXT, FieldKind.ENUM)


def reads_reversed(
    name: str, value: str | None, kind: FieldKind, lines: Sequence[OcrLine], labels: LabelMap
) -> bool:
    """The OCR at this field's own label matches the value read *backwards*, not forwards.

    VLMs sometimes read Arabic left to right: «داجس» for «سجاد», word order kept or not. Only
    an anchored match counts, so a reversed reading is recognised where it was printed.
    """
    if kind not in _TEXT_KINDS or not value or not arabic.is_arabic_letters(value):
        return False
    if ground_field(name, value, kind, lines, labels).matches:
        return False
    for backwards in (value[::-1], " ".join(w[::-1] for w in value.split())):
        g = ground_field(name, backwards, kind, lines, labels)
        if g.anchored and g.similarity >= REVERSED_MATCH:
            return True
    return False

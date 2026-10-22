import json

from qareena.config import FieldKind
from qareena.pipeline.grounding import ground
from qareena.pipeline.logprobs import field_logprob, token_offsets
from qareena.readers.ocr import OcrLine
from qareena.readers.ollama import TokenLogprob


def line(text, score=0.95, x=0):
    return OcrLine(text, score, ((x, 0), (x + 100, 0), (x + 100, 20), (x, 20)))


LINES = [
    line("الاسم ناو"),
    line("مهدي", 0.97, 10),
    line("الأب باوك"),
    line("يوسف", 0.93, 200),
    line("٩٩٨٢٠٣٣٤٩١٣٢"),
]


def test_exact_value_is_grounded_with_its_box():
    g = ground("مهدي", FieldKind.NAME_PART, LINES)
    assert g.matches == 1
    assert g.exact
    assert g.box == (10, 0, 110, 20)
    assert g.ocr_score == 0.97


def test_digits_match_across_scripts():
    g = ground("998203349132", FieldKind.DIGITS, LINES)
    assert g.exact
    assert g.matches == 1


def test_hallucinated_value_is_not_grounded():
    g = ground("عباس", FieldKind.NAME_PART, LINES)
    assert g.matches == 0
    assert g.similarity < 0.9


def test_ambiguous_value_counts_every_matching_line():
    g = ground("علي", FieldKind.NAME_PART, [line("علي"), line("حسين"), line("علي")])
    assert g.matches == 2


def test_spelling_variants_still_ground():
    assert ground("أحمد", FieldKind.NAME_PART, [line("احمد")]).exact


def test_empty_value_is_not_grounded():
    assert ground(None, FieldKind.TEXT, LINES).matches == 0
    assert ground("", FieldKind.TEXT, LINES).matches == 0


def _tok(*pairs):
    return [TokenLogprob(t, lp) for t, lp in pairs]


def test_value_span_logprobs_exclude_json_scaffolding():
    content = json.dumps({"given_name": "مهدي", "sex": None}, ensure_ascii=False)
    pieces = ['{"', "given_name", '":', ' "', "مه", "دي", '",', ' "', "sex", '":', " null", "}"]
    assert "".join(pieces) == content
    lps = [-0.01, -0.01, -0.01, -0.01, -0.5, -0.2, -3.0, -0.01, -0.01, -0.01, -0.01, -0.01]
    span = field_logprob(content, _tok(*zip(pieces, lps, strict=True)), "given_name", "مهدي")
    assert span.n_tokens == 2
    assert span.min_lp == -0.5
    assert span.first_lp == -0.5


def test_replacement_characters_are_aligned_as_wildcards():
    content = '{"a": "مهدي"}'
    rep = chr(0xFFFD)
    pieces = ['{"a": "', "م", rep, rep + "دي", '"}']  # 'ه' split across two byte-level tokens
    offsets = token_offsets(content, _tok(*[(p, -0.1) for p in pieces]))
    assert offsets is not None
    assert offsets[-1][1] == len(content)


def test_unalignable_tokens_return_none():
    content = '{"a": "x"}'
    assert field_logprob(content, _tok(('{"a": "y"}', -0.1)), "a", "x") is None

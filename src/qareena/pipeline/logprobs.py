"""Token log-probabilities for one field's *value*, never for the JSON scaffolding around it.

Ollama returns llama-server's pre-constraint probabilities, so schema-forced tokens (braces,
keys, quotes) can look improbable and must be excluded. We locate the value's characters in
the raw JSON answer and keep the tokens that overlap them.

Known issue (Ollama #13497): a token that splits a multi-byte UTF-8 character comes back as
U+FFFD, so the token strings may not rebuild the answer. Each U+FFFD run is treated as a
wildcard when aligning; if alignment still fails we return None rather than guess.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from qareena.readers.ollama import TokenLogprob

_REPLACEMENT = chr(0xFFFD)


@dataclass(frozen=True)
class SpanLogprob:
    n_tokens: int
    min_lp: float
    mean_lp: float
    first_lp: float

    @property
    def min_p(self) -> float:
        return math.exp(self.min_lp)


_MAX_WILDCARDS = 64  # beyond this, give up rather than risk slow backtracking


def token_offsets(content: str, tokens: Sequence[TokenLogprob]) -> list[tuple[int, int]] | None:
    """Character span of each token in ``content``, or None if they cannot be aligned.

    A multi-byte character split across byte-level tokens appears as U+FFFD in each piece,
    and together the pieces cover exactly one character, so each individual wildcard may be
    empty. The token sequence is compiled to one regex (literals escaped, U+FFFD runs as lazy
    ``.*?``, one group per token) and matched against the whole answer.
    """
    if not any(_REPLACEMENT in t.token for t in tokens):
        spans, pos = [], 0
        for t in tokens:
            if not content.startswith(t.token, pos):
                return None
            spans.append((pos, pos + len(t.token)))
            pos += len(t.token)
        return spans if pos == len(content) else None

    parts, wildcards = [], 0
    for t in tokens:
        pieces = re.split(f"({_REPLACEMENT}+)", t.token)
        body = []
        for p in pieces:
            if not p:
                continue
            if p[0] == _REPLACEMENT:
                body.append(".*?")
                wildcards += 1
            else:
                body.append(re.escape(p))
        parts.append("(" + "".join(body) + ")")
    if wildcards > _MAX_WILDCARDS:
        return None
    m = re.fullmatch("".join(parts), content, flags=re.DOTALL)
    if m is None:
        return None
    return [m.span(i + 1) for i in range(len(tokens))]


def value_span(content: str, key: str, value: str) -> tuple[int, int] | None:
    """Character span of ``value`` as it appears after ``"key":`` in the raw JSON answer."""
    encoded = json.dumps(value, ensure_ascii=False)[1:-1]
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*"', content)
    if not m:
        return None
    start = m.end()
    if not content.startswith(encoded, start):
        return None
    return start, start + len(encoded)


def field_logprob(
    content: str, tokens: Sequence[TokenLogprob], key: str, value: str | None
) -> SpanLogprob | None:
    if not value or not tokens:
        return None
    span = value_span(content, key, value)
    offsets = token_offsets(content, tokens)
    if span is None or offsets is None:
        return None
    s, e = span
    lps = [t.logprob for t, (a, b) in zip(tokens, offsets, strict=True) if a < e and b > s]
    if not lps:
        return None
    return SpanLogprob(len(lps), min(lps), sum(lps) / len(lps), lps[0])

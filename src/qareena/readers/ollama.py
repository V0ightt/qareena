"""Minimal, typed client for Ollama's native chat API (JSON schema output + token logprobs).

The native ``/api/chat`` is used rather than the OpenAI-compatible endpoint because it
documents ``format`` (JSON schema), ``think`` and ``logprobs`` together. Images are sent as
base64. Every call returns the raw content, the parsed JSON (or the parse error), per-token
logprobs and Ollama's timing counters, so callers can measure without re-instrumenting.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


class OllamaError(RuntimeError):
    pass


@dataclass(frozen=True)
class TokenLogprob:
    token: str
    logprob: float
    top: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class ChatResult:
    model: str
    content: str
    data: dict[str, Any] | None
    parse_error: str | None
    logprobs: tuple[TokenLogprob, ...]
    metrics: dict[str, float] = field(default_factory=dict)  # seconds and token counts

    @property
    def ok(self) -> bool:
        return self.data is not None


class OllamaClient:
    def __init__(
        self, base_url: str = "http://127.0.0.1:11434", timeout_s: float = 180.0, retries: int = 2
    ) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=httpx.Timeout(timeout_s, connect=5.0))
        self._retries = retries

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def version(self) -> str:
        return str(self._get("/api/version")["version"])

    def running(self) -> list[dict[str, Any]]:
        """Loaded models with their total and VRAM-resident size (``ollama ps``)."""
        return list(self._get("/api/ps").get("models", []))

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        images: list[bytes] | None = None,
        num_ctx: int = 8192,
        logprobs: bool = True,
        top_logprobs: int = 3,
        keep_alive: str | int = -1,
        num_predict: int = 1024,
        options: dict[str, Any] | None = None,
    ) -> ChatResult:
        user_msg: dict[str, Any] = {"role": "user", "content": user}
        if images:
            user_msg["images"] = [base64.b64encode(b).decode("ascii") for b in images]
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, user_msg],
            "format": schema,
            "stream": False,
            "think": False,
            "keep_alive": keep_alive,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs if logprobs else 0,
            "options": {
                "temperature": 0,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "presence_penalty": 0,
                **(options or {}),
            },
        }
        t0 = time.perf_counter()
        resp = self._post("/api/chat", body)
        wall = time.perf_counter() - t0
        content = str(resp.get("message", {}).get("content", ""))
        data, err = None, None
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                data = parsed
            else:
                err = "top-level JSON is not an object"
        except json.JSONDecodeError as exc:
            err = f"invalid JSON: {exc.msg} at {exc.pos}"
        raw_lp = resp.get("logprobs") or resp.get("message", {}).get("logprobs") or []
        lps = tuple(
            TokenLogprob(
                str(t.get("token", "")),
                float(t.get("logprob", 0.0)),
                tuple(
                    (str(x.get("token", "")), float(x.get("logprob", 0.0)))
                    for x in t.get("top_logprobs") or []
                ),
            )
            for t in raw_lp
        )
        ns = 1e9
        metrics = {
            "wall_s": wall,
            "load_s": resp.get("load_duration", 0) / ns,
            "prompt_tokens": float(resp.get("prompt_eval_count", 0)),
            "prompt_s": resp.get("prompt_eval_duration", 0) / ns,
            "output_tokens": float(resp.get("eval_count", 0)),
            "output_s": resp.get("eval_duration", 0) / ns,
            "total_s": resp.get("total_duration", 0) / ns,
        }
        return ChatResult(model, content, data, err, lps, metrics)

    # --- transport --------------------------------------------------------------------

    def _get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, json=body)

    def _request(self, method: str, path: str, **kw: Any) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                r = self._http.request(method, path, **kw)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                last = exc
                time.sleep(0.5 * (2**attempt))
                continue
            if r.status_code >= 400:  # client/model errors are not retried
                raise OllamaError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
            return r.json()
        raise OllamaError(f"{method} {path} failed after {self._retries + 1} attempts: {last}")

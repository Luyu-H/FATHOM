"""Unified LLM client supporting OpenAI / Anthropic / Gemini / DeepSeek / Qwen.

Usage:
    from src.runner.llm_client import build_llm_client, Message
    llm = build_llm_client(cfg.llm)
    [resp] = llm.chat(messages=[Message("user", "hi")], system="be helpful")
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# =============================================================
# Common types
# =============================================================

@dataclass
class Message:
    role: str        # "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    finish_reason: Optional[str] = None
    raw: Any = None


# =============================================================
# Base interface
# =============================================================

class LLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: Sequence[Message],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
    ) -> List[LLMResponse]:
        ...


def _retry(label: str, max_retries: int, interval: float):
    """Decorator-like helper used inline in concrete clients."""
    # Simple loop helper – kept inline for clarity of error context per provider.
    def runner(fn):
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                return fn()
            except Exception as e:                      # noqa: BLE001
                last_err = e
                logger.warning("%s attempt %d/%d failed: %s",
                               label, attempt, max_retries, e)
                if attempt < max_retries:
                    time.sleep(interval)
        raise RuntimeError(f"{label} failed after {max_retries} attempts") from last_err
    return runner


# =============================================================
# OpenAI
# =============================================================

class OpenAIClient(LLMClient):
    # OpenAI reasoning families (o1/o3/o4/...) reject any user-supplied
    # `temperature` — the API only accepts the server default (1). We
    # detect them by model-id prefix and omit the parameter entirely
    # rather than sending the default, since the rejection is on the
    # *presence* of the field, not just non-default values.
    _REASONING_PREFIXES = ("o1", "o3", "o4", "o5")

    def __init__(self, model: str, api_key_env: str, max_tokens: int,
                 temperature: float, max_retries: int = 3,
                 request_interval: float = 1.0,
                 base_url: Optional[str] = None, **_: Any) -> None:
        import openai                                   # lazy import
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(f"Missing env var: {api_key_env}")
        # Optional `base_url` override lets this client talk to any
        # OpenAI-compatible endpoint (Azure-style gateways, OpenAI-compatible
        # proxies, self-hosted servers). `None` falls back to the SDK default
        # (https://api.openai.com/v1).
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_retries = max_retries
        self._interval = request_interval
        self._is_reasoning = self._model_is_reasoning(model)

    @classmethod
    def _model_is_reasoning(cls, model: str) -> bool:
        m = (model or "").lower().lstrip()
        return any(m == p or m.startswith(p + "-") for p in cls._REASONING_PREFIXES)

    def chat(self, messages, system=None, temperature=None,
             max_tokens=None, n=1):
        payload = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend({"role": m.role, "content": m.content} for m in messages)

        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=payload,
            max_completion_tokens=max_tokens or self._max_tokens,
            n=n,
        )
        if not self._is_reasoning:
            kwargs["temperature"] = (
                self._temperature if temperature is None else temperature
            )

        def _call():
            resp = self._client.chat.completions.create(**kwargs)
            # Defensive: a 200 OK with `choices=None` (upstream filter,
            # internal error, or reasoning timeout) would crash later
            # with `TypeError: 'NoneType' is not iterable`. Raise here
            # so _retry retries instead.
            if not resp.choices:
                raise RuntimeError(
                    f"openai returned no choices (resp_id={getattr(resp, 'id', '?')})"
                )
            return resp

        resp = _retry("openai", self._max_retries, self._interval)(_call)
        return [
            LLMResponse(text=ch.message.content or "",
                        finish_reason=ch.finish_reason, raw=ch)
            for ch in resp.choices
        ]


# =============================================================
# Anthropic
# =============================================================

class AnthropicClient(LLMClient):
    def __init__(self, model: str, api_key_env: str, max_tokens: int,
                 temperature: float, max_retries: int = 3,
                 request_interval: float = 1.0,
                 base_url: Optional[str] = None, **_: Any) -> None:
        import anthropic                                # lazy import
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(f"Missing env var: {api_key_env}")
        # Optional `base_url` override points the Anthropic SDK at a gateway
        # or relay (e.g. an Anthropic-protocol proxy). `None` falls back to
        # the SDK default (https://api.anthropic.com).
        self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        self.model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_retries = max_retries
        self._interval = request_interval

    def chat(self, messages, system=None, temperature=None,
             max_tokens=None, n=1):
        anth_msgs = [{"role": m.role, "content": m.content} for m in messages]
        results: List[LLMResponse] = []
        # Anthropic API has no native n>1 — call sequentially
        for _ in range(n):
            def _call():
                # Stream, then collapse back to a full Message. The SDK
                # refuses *non-streaming* requests whose estimated runtime
                # (driven by `max_tokens`) may exceed 10 min — with our
                # large `max_tokens` that check trips. Streaming sidesteps
                # it; `get_final_message()` returns the same Message shape
                # `messages.create` would, so downstream code is unchanged.
                with self._client.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens or self._max_tokens,
                    temperature=self._temperature if temperature is None else temperature,
                    system=system or "",
                    messages=anth_msgs,
                ) as stream:
                    resp = stream.get_final_message()
                # Defensive: missing `content` would crash the join below.
                if resp.content is None:
                    raise RuntimeError(
                        f"anthropic returned no content (resp_id={getattr(resp, 'id', '?')})"
                    )
                return resp
            resp = _retry("anthropic", self._max_retries, self._interval)(_call)
            text = "".join(b.text for b in resp.content if hasattr(b, "text"))
            results.append(LLMResponse(text=text,
                                       finish_reason=resp.stop_reason, raw=resp))
        return results


# =============================================================
# DeepSeek (OpenAI-compatible /chat/completions API)
# =============================================================

class DeepSeekClient(LLMClient):
    """DeepSeek's REST API is OpenAI-compatible — same schema, different
    base URL. We reuse the `openai` SDK and just point it at DeepSeek's
    endpoint.

    Notes on `n`: DeepSeek currently rejects `n > 1` on chat/completions.
    Issue multiple sequential calls instead, mirroring AnthropicClient.

    Reasoning mode is toggled via the ``reasoning_effort`` constructor
    arg (forwarded from ``cfg.llm.reasoning_effort`` in the YAML). When
    set to a real effort level (``low`` / ``medium`` / ``high``) the
    client sends both ``reasoning_effort=<level>`` and
    ``extra_body={"thinking": {"type": "enabled"}}`` on every chat call.
    Any falsey value or the string ``off`` / ``none`` disables reasoning
    entirely (neither field is sent). This makes flipping between
    reasoning vs. non-reasoning runs a single-line YAML edit.
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com"
    # Strings the YAML may set to mean "no reasoning". Case-insensitive.
    _REASONING_OFF = frozenset({"", "off", "none", "false", "no", "disabled"})

    def __init__(self, model: str, api_key_env: str, max_tokens: int,
                 temperature: float, max_retries: int = 3,
                 request_interval: float = 1.0,
                 base_url: Optional[str] = None,
                 reasoning_effort: Optional[str] = None,
                 **_: Any) -> None:
        import openai                                   # lazy import
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(f"Missing env var: {api_key_env}")
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
        )
        self.model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_retries = max_retries
        self._interval = request_interval
        self._reasoning_effort: Optional[str] = self._normalize_effort(
            reasoning_effort,
        )

    @classmethod
    def _normalize_effort(cls, value) -> Optional[str]:
        if value is None or value is False:
            return None
        s = str(value).strip().lower()
        if s in cls._REASONING_OFF:
            return None
        return s

    def chat(self, messages, system=None, temperature=None,
             max_tokens=None, n=1):
        payload = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend({"role": m.role, "content": m.content} for m in messages)

        extra: Dict[str, Any] = {}
        if self._reasoning_effort:
            extra["reasoning_effort"] = self._reasoning_effort
            extra["extra_body"] = {"thinking": {"type": "enabled"}}

        results: List[LLMResponse] = []
        for _ in range(n):
            def _call():
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=payload,
                    max_tokens=max_tokens or self._max_tokens,
                    temperature=self._temperature if temperature is None else temperature,
                    **extra,
                )
                # DeepSeek occasionally returns HTTP 200 with `choices=None`
                # (reasoning timeout, upstream filter, or empty completion).
                # Raise so _retry actually retries instead of crashing later
                # with a TypeError on `for ch in None`.
                if not resp.choices:
                    raise RuntimeError(
                        f"deepseek returned no choices (resp_id={getattr(resp, 'id', '?')})"
                    )
                return resp
            resp = _retry("deepseek", self._max_retries, self._interval)(_call)
            for ch in resp.choices:
                results.append(LLMResponse(
                    text=ch.message.content or "",
                    finish_reason=ch.finish_reason,
                    raw=ch,
                ))
        return results


# =============================================================
# Qwen (DashScope OpenAI-compatible API)
# =============================================================

class QwenClient(LLMClient):
    """Qwen via Alibaba DashScope's OpenAI-compatible endpoint.

    DashScope exposes Qwen models behind an OpenAI-shaped /chat/completions
    API, so we reuse the `openai` SDK and just swap the base URL + API key.

    For self-hosted Qwen (vLLM, Ollama, SGLang, etc.) point `base_url`
    at the local server — same client works as long as the server speaks
    the OpenAI chat-completions schema.

    Notes on `n`: DashScope rejects `n > 1` on chat/completions for
    several Qwen models. Issue sequential calls instead (mirrors
    DeepSeekClient / AnthropicClient).
    """

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(self, model: str, api_key_env: str, max_tokens: int,
                 temperature: float, max_retries: int = 3,
                 request_interval: float = 1.0,
                 base_url: Optional[str] = None, **_: Any) -> None:
        import openai                                   # lazy import
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(f"Missing env var: {api_key_env}")
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
        )
        self.model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_retries = max_retries
        self._interval = request_interval

    def chat(self, messages, system=None, temperature=None,
             max_tokens=None, n=1):
        payload = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend({"role": m.role, "content": m.content} for m in messages)

        results: List[LLMResponse] = []
        for _ in range(n):
            def _call():
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=payload,
                    max_tokens=max_tokens or self._max_tokens,
                    temperature=self._temperature if temperature is None else temperature,
                    extra_body={"enable_thinking": True},
                )
                # Defensive: same `choices=None` pattern as DeepSeek/OpenAI.
                if not resp.choices:
                    raise RuntimeError(
                        f"qwen returned no choices (resp_id={getattr(resp, 'id', '?')})"
                    )
                return resp
            resp = _retry("qwen", self._max_retries, self._interval)(_call)
            for ch in resp.choices:
                results.append(LLMResponse(
                    text=ch.message.content or "",
                    finish_reason=ch.finish_reason,
                    raw=ch,
                ))
        return results


# =============================================================
# Gemini
# =============================================================

class GeminiClient(LLMClient):
    def __init__(self, model: str, api_key_env: str, max_tokens: int,
                 temperature: float, max_retries: int = 3,
                 request_interval: float = 1.0,
                 base_url: Optional[str] = None, **_: Any) -> None:
        try:
            from google import genai
            from google.genai import types as gtypes
        except ImportError as e:
            raise ImportError(
                "Install google-genai: `pip install google-genai`") from e
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(f"Missing env var: {api_key_env}")
        self._gtypes = gtypes
        # The google-genai SDK doesn't accept `base_url` directly — it's set
        # via `http_options`. Only override when configured, otherwise let the
        # SDK use its default endpoint.
        http_options = (
            gtypes.HttpOptions(base_url=base_url) if base_url else None
        )
        self._client = genai.Client(api_key=api_key, http_options=http_options)
        self.model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_retries = max_retries
        self._interval = request_interval

    def chat(self, messages, system=None, temperature=None,
             max_tokens=None, n=1):
        types = self._gtypes
        contents = []
        for m in messages:
            role = "user" if m.role == "user" else "model"
            contents.append(types.Content(role=role,
                                          parts=[types.Part(text=m.content)]))

        def _call():
            resp = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=self._temperature if temperature is None else temperature,
                    max_output_tokens=max_tokens or self._max_tokens,
                    system_instruction=system,
                    candidate_count=n,
                ),
            )
            # Defensive: Gemini returns `candidates=None` when the prompt
            # is blocked by safety filters or when the response is empty.
            # Raise so _retry retries instead of crashing later.
            if not resp.candidates:
                block = getattr(getattr(resp, "prompt_feedback", None),
                                "block_reason", None)
                raise RuntimeError(
                    f"gemini returned no candidates (block_reason={block})"
                )
            return resp

        resp = _retry("gemini", self._max_retries, self._interval)(_call)
        out = []
        for c in resp.candidates:
            text = "".join(p.text for p in c.content.parts if hasattr(p, "text"))
            out.append(LLMResponse(text=text,
                                   finish_reason=str(c.finish_reason)
                                                  if c.finish_reason else None,
                                   raw=c))
        return out


# =============================================================
# Factory
# =============================================================

def build_llm_client(cfg) -> LLMClient:
    """`cfg` is the OmegaConf llm subtree."""
    provider = str(cfg.provider).lower()
    kwargs = dict(
        model=cfg.model,
        api_key_env=cfg.api_key_env,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        max_retries=cfg.get("max_retries", 3),
        request_interval=cfg.get("request_interval", 1.0),
    )
    if provider == "openai":
        # Optional `base_url` override — defaults to the OpenAI SDK endpoint.
        # Point at any OpenAI-compatible gateway/proxy or self-hosted server.
        return OpenAIClient(base_url=cfg.get("base_url", None), **kwargs)
    if provider == "anthropic":
        # Optional `base_url` override — defaults to the Anthropic SDK endpoint.
        # Point at an Anthropic-protocol gateway/relay if needed.
        return AnthropicClient(base_url=cfg.get("base_url", None), **kwargs)
    if provider == "gemini":
        # Optional `base_url` override (e.g. a proxy or OpenAI-compatible
        # gateway). Defaults to the SDK's standard Gemini endpoint.
        return GeminiClient(base_url=cfg.get("base_url", None), **kwargs)
    if provider == "deepseek":
        # Optional `base_url` override (defaults to https://api.deepseek.com).
        # `reasoning_effort` toggles thinking mode — null / off / none means
        # disabled; low / medium / high enables it at that effort level.
        return DeepSeekClient(
            base_url=cfg.get("base_url", None),
            reasoning_effort=cfg.get("reasoning_effort", None),
            **kwargs,
        )
    if provider == "qwen":
        # Optional `base_url` override — defaults to DashScope's
        # OpenAI-compatible endpoint. Point at a local server for
        # self-hosted Qwen (vLLM/Ollama/SGLang).
        return QwenClient(base_url=cfg.get("base_url", None), **kwargs)
    raise ValueError(f"Unknown LLM provider: {provider}")


# =============================================================
# Robust JSON extraction from LLM output
# =============================================================

def parse_json_block(raw: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction tolerating markdown fences and prose preamble."""
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text).strip()
    text = re.sub(r"\n?```\s*$", "", text).strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Locate the first balanced {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None
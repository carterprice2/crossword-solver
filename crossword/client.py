"""Model clients: one that talks to Nebius, and several that do not.

Nebius Token Factory serves an OpenAI-compatible REST API, so the live client
is a thin ``urllib`` wrapper over ``POST /v1/chat/completions``. Using the
stdlib rather than the ``openai`` SDK is a deliberate trade -- see README --
and it costs about 200 lines while making ``git clone && make test`` work with
no install step and no network.

Everything else in the project depends only on the :class:`ModelClient`
protocol, which is what lets the whole solver be exercised offline:

    NebiusClient     the real thing
    RecordingClient  wraps a client and writes a replayable trace
    ReplayClient     serves a recorded trace, no network
    ScriptedClient   canned responses for prompt-shape tests
    OracleClient     synthesizes candidates from a known solution, with
                     controllable error, recall and calibration noise
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
API_KEY_ENV = "NEBIUS_API_KEY"


def load_env_file(*paths: str) -> list[str]:
    """Load KEY=VAL lines into ``os.environ`` without overriding existing keys.

    Looks at ``./.env`` and the package root ``.env`` when no paths are given.
    Returns the files that were read. Never prints values.
    """
    if paths:
        candidates = list(paths)
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        package_root = os.path.dirname(here)
        candidates = [
            os.path.join(os.getcwd(), ".env"),
            os.path.join(package_root, ".env"),
        ]
    loaded: list[str] = []
    seen: set[str] = set()
    for raw_path in candidates:
        path = os.path.abspath(raw_path)
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'").strip('"')
                if key and key not in os.environ:
                    os.environ[key] = value
        loaded.append(path)
    return loaded

#: Verified against the Nebius Token Factory cookbook. Override per run with
#: --model; `crossword models ping` lists what the account can actually reach.
KNOWN_MODELS = (
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "meta-llama/Llama-3.3-70B-Instruct",
    "openai/gpt-oss-120b",
    "deepseek-ai/DeepSeek-V4-Pro",
    "moonshotai/Kimi-K2.6",
    "zai-org/GLM-5.2",
)

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_REASONING_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"


class ModelError(RuntimeError):
    """A request failed in a way retrying will not fix."""


class SchemaRejected(ModelError):
    """The model refused the structured-output schema (HTTP 400)."""


class CacheMiss(ModelError):
    """A replay client was asked for a request it has no recording of."""


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    #: Which rung of the structured-output ladder actually worked.
    rung: str = "strict_schema"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ModelClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        response_format: dict | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        seed: int | None = None,
    ) -> Completion: ...


def request_key(
    model: str,
    messages: list[dict],
    response_format: dict | None,
    temperature: float,
    seed: int | None,
) -> str:
    """Stable hash of a request, used as the record/replay cache key."""
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "response_format": response_format,
            "temperature": round(temperature, 4),
            "seed": seed,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class NebiusClient:
    """Nebius Token Factory over its OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        max_retries: int = 4,
        opener: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.api_key = api_key or os.environ.get(API_KEY_ENV, "")
        if not self.api_key:
            raise ModelError(
                f"{API_KEY_ENV} is not set. Get a key at "
                "https://tokenfactory.nebius.com/ and export it, or run with "
                "--backend oracle to work offline."
            )
        self.base_url = base_url if base_url.endswith("/") else base_url + "/"
        self.timeout = timeout
        self.max_retries = max_retries
        self._sleep = sleep
        # One shared context: building an SSL context per request is slow and
        # this client is called from a thread pool.
        self._opener = opener or urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ssl.create_default_context())
        )
        self._lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    # -- transport ---------------------------------------------------------

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        delay = 1.0
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8")[:400]
                except Exception:  # pragma: no cover - best effort
                    pass
                if exc.code == 400:
                    # Almost always guided-decoding rejecting the schema; the
                    # caller walks the degradation ladder rather than failing.
                    raise SchemaRejected(f"400 from {path}: {detail}") from exc
                if exc.code in (401, 403):
                    raise ModelError(f"{exc.code} from {path}: check {API_KEY_ENV}")
                if exc.code not in (408, 409, 429) and exc.code < 500:
                    raise ModelError(f"{exc.code} from {path}: {detail}") from exc
                last = exc
                wait = delay
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if retry_after:
                    try:
                        wait = max(wait, float(retry_after))
                    except ValueError:
                        pass
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
                wait = delay
            if attempt == self.max_retries:
                break
            # Jitter so a pool of threads does not retry in lockstep.
            self._sleep(wait * random.uniform(0.7, 1.3))
            delay = min(delay * 2, 30.0)
        raise ModelError(f"{path} failed after {self.max_retries + 1} attempts: {last}")

    # -- API ---------------------------------------------------------------

    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        response_format: dict | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        seed: int | None = None,
    ) -> Completion:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            body["response_format"] = response_format
        if seed is not None:
            body["seed"] = seed

        started = time.monotonic()
        payload = self._post("chat/completions", body)
        latency = time.monotonic() - started

        choices = payload.get("choices") or []
        if not choices:
            raise ModelError(f"no choices in response: {str(payload)[:200]}")
        text = (choices[0].get("message") or {}).get("content") or ""
        usage = payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        with self._lock:
            self.calls += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
        return Completion(
            text=text,
            model=payload.get("model", model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency,
        )

    def list_models(self) -> list[str]:
        request = urllib.request.Request(
            self.base_url + "models",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        with self._opener.open(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return sorted(item.get("id", "") for item in payload.get("data", []))


class RecordingClient:
    """Pass requests through, writing each exchange to a JSONL trace."""

    def __init__(self, inner: ModelClient, path: str):
        self.inner = inner
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    def complete(self, **kwargs) -> Completion:
        completion = self.inner.complete(**kwargs)
        key = request_key(
            kwargs["model"],
            kwargs["messages"],
            kwargs.get("response_format"),
            kwargs.get("temperature", 0.3),
            kwargs.get("seed"),
        )
        record = {
            "key": key,
            "request": {
                "model": kwargs["model"],
                "messages": kwargs["messages"],
                "response_format": kwargs.get("response_format"),
                "temperature": kwargs.get("temperature", 0.3),
                "seed": kwargs.get("seed"),
            },
            "response": {
                "text": completion.text,
                "model": completion.model,
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "latency_s": completion.latency_s,
            },
        }
        with self._lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return completion


class ReplayClient:
    """Serve a recorded trace. No network, fully deterministic."""

    def __init__(self, path: str, *, strict: bool = True):
        self.path = path
        self.strict = strict
        self._by_key: dict[str, dict] = {}
        self._order: list[dict] = []
        self._cursor = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self._by_key[record["key"]] = record["response"]
                self._order.append(record["response"])

    def complete(self, **kwargs) -> Completion:
        key = request_key(
            kwargs["model"],
            kwargs["messages"],
            kwargs.get("response_format"),
            kwargs.get("temperature", 0.3),
            kwargs.get("seed"),
        )
        response = self._by_key.get(key)
        if response is None:
            if self.strict:
                raise CacheMiss(f"no recording for request {key[:12]}")
            if self._cursor >= len(self._order):
                raise CacheMiss("replay trace exhausted")
            response = self._order[self._cursor]
            self._cursor += 1
        return Completion(
            text=response["text"],
            model=response.get("model", kwargs["model"]),
            prompt_tokens=response.get("prompt_tokens", 0),
            completion_tokens=response.get("completion_tokens", 0),
            latency_s=response.get("latency_s", 0.0),
        )


class ScriptedClient:
    """Return canned text, chosen by a matcher over the prompt."""

    def __init__(self, responses: list[str] | None = None, *, match: dict | None = None):
        self.responses = list(responses or [])
        self.match = match or {}
        self.requests: list[dict] = []
        self._cursor = 0

    def complete(self, **kwargs) -> Completion:
        self.requests.append(kwargs)
        prompt = json.dumps(kwargs["messages"])
        for needle, reply in self.match.items():
            if needle in prompt:
                return Completion(text=reply, model=kwargs["model"])
        if self._cursor < len(self.responses):
            text = self.responses[self._cursor]
            self._cursor += 1
            return Completion(text=text, model=kwargs["model"])
        return Completion(text='{"items": []}', model=kwargs["model"])


@dataclass
class OracleConfig:
    """How wrong the synthetic solver-under-test should be.

    These knobs exist so the constraint and repair layers can be measured
    offline: sweep them and you get a real curve showing how much error the
    architecture absorbs, with no API key and no network.
    """

    #: Probability a slot's *correct* answer is missing entirely.
    recall: float = 0.85
    #: Probability the correct answer is not ranked first.
    top1_error: float = 0.3
    #: Spread of the noise added to reported confidence.
    conf_noise: float = 0.15
    #: How many candidates to offer per slot.
    width: int = 4
    seed: int = 0
    #: When True, confirmed letters in the prompt raise P(include gold) and
    #: distractors must match the pattern. Independent draws (False) are the
    #: original sweep; they cannot test the repair prompt's actual mechanism.
    pattern_aware: bool = True


def effective_recall(base: float, pattern: str | None, answer: str) -> float:
    """P(include gold) given a confirmed pattern.

    No pattern (or all ``?``) leaves ``base`` unchanged, so round-0 behaviour
    matches the independent oracle. Each confirmed letter interpolates toward
    1. A pattern that contradicts the gold answer yields 0.
    """
    if not pattern or not answer or len(pattern) != len(answer):
        return base
    known = 0
    for got, want in zip(answer, pattern):
        if want == "?":
            continue
        if want != got:
            return 0.0
        known += 1
    if known == 0:
        return base
    return base + (1.0 - base) * (known / len(pattern))


class OracleClient:
    """Synthesize candidate lists from a known solution.

    Distractors are built by mutating the true answer, which matters: random
    strings would be trivially eliminated by crossing constraints, so the
    synthetic task would be far easier than the real one.
    """

    def __init__(self, gold: dict[str, str], config: OracleConfig | None = None):
        self.gold = {k.upper(): v.upper() for k, v in gold.items()}
        self.config = config or OracleConfig()
        self._rnd = random.Random(self.config.seed)
        self._lock = threading.Lock()
        self.calls = 0

    def _distractor(self, answer: str, pattern: str | None = None) -> str:
        letters = list(answer)
        mutable = [
            i
            for i, ch in enumerate(letters)
            if not pattern or i >= len(pattern) or pattern[i] == "?"
        ]
        if not mutable:
            mutable = list(range(len(letters)))
        for _ in range(self._rnd.randint(1, max(1, len(mutable) // 2 or 1))):
            i = mutable[self._rnd.randrange(len(mutable))]
            letters[i] = chr(self._rnd.randrange(ord("A"), ord("Z") + 1))
        if pattern:
            for i, ch in enumerate(pattern):
                if ch != "?" and i < len(letters):
                    letters[i] = ch
        out = "".join(letters)
        if out == answer and mutable:
            i = mutable[0]
            letters[i] = "A" if answer[i] != "A" else "B"
            out = "".join(letters)
        return out

    def _slot_specs(self, messages: list[dict]) -> list[dict]:
        """Pull slot requests out of the last JSON-shaped user message."""
        contents = [
            str(m.get("content", "")) for m in messages if isinstance(m, dict)
        ]
        from .schemas import extract_json

        for content in reversed(contents):
            payload = extract_json(content)
            if not payload:
                continue
            slots = payload.get("slots")
            if isinstance(slots, list) and slots:
                return [s for s in slots if isinstance(s, dict) and s.get("id")]

        prompt = "\n".join(contents)
        items: list[dict] = []
        seen: set[str] = set()
        for slot_id, length in re.findall(
            r'"id":\s*"([AD]\d+)",\s*"clue":.*?"len":\s*(\d+)', prompt
        ):
            if slot_id not in seen:
                seen.add(slot_id)
                items.append({"id": slot_id, "len": int(length)})
        if items:
            return items
        for slot_id in re.findall(r"(?<![A-Za-z0-9])([AD]\d+)(?![0-9])", prompt):
            if slot_id in self.gold and slot_id not in seen:
                seen.add(slot_id)
                items.append({"id": slot_id, "len": len(self.gold[slot_id])})
        return items

    def complete(self, **kwargs) -> Completion:
        messages = kwargs.get("messages") or []
        specs = self._slot_specs(messages)
        prompt = "\n".join(
            str(m.get("content", "")) for m in messages if isinstance(m, dict)
        )
        cfg = self.config
        payload = []
        with self._lock:
            self.calls += 1
            for spec in specs:
                slot_id = str(spec.get("id", "")).strip().upper()
                answer = self.gold.get(slot_id)
                length = int(spec.get("len") or (len(answer) if answer else 0))
                if answer is None or len(answer) != length:
                    continue
                raw_pattern = spec.get("pattern") if cfg.pattern_aware else None
                pattern = str(raw_pattern) if raw_pattern else None
                rejected = {
                    str(x).upper()
                    for x in (spec.get("rejected") or [])
                    if isinstance(x, str)
                }
                recall = (
                    effective_recall(cfg.recall, pattern, answer)
                    if cfg.pattern_aware
                    else cfg.recall
                )
                include_truth = (
                    answer not in rejected and self._rnd.random() < recall
                )
                candidates: list[dict] = []
                want = max(0, cfg.width - (1 if include_truth else 0))
                seen_answers = {answer} if include_truth else set()
                for _ in range(want * 3):
                    if len(candidates) >= want:
                        break
                    distractor = self._distractor(answer, pattern)
                    if distractor in seen_answers or distractor in rejected:
                        continue
                    seen_answers.add(distractor)
                    candidates.append(
                        {
                            "answer": distractor,
                            "confidence": round(
                                min(
                                    0.95,
                                    max(0.05, self._rnd.gauss(0.45, cfg.conf_noise)),
                                ),
                                3,
                            ),
                        }
                    )
                if include_truth:
                    base = 0.55 if self._rnd.random() < cfg.top1_error else 0.9
                    truth = {
                        "answer": answer,
                        "confidence": round(
                            min(
                                0.97,
                                max(0.05, self._rnd.gauss(base, cfg.conf_noise)),
                            ),
                            3,
                        ),
                    }
                    position = (
                        self._rnd.randrange(len(candidates) + 1) if candidates else 0
                    )
                    candidates.insert(position, truth)
                payload.append({"id": slot_id, "candidates": candidates})

        return Completion(
            text=json.dumps({"items": payload}),
            model=kwargs.get("model", "oracle"),
            prompt_tokens=len(prompt) // 4,
            completion_tokens=sum(len(str(p)) for p in payload) // 4,
        )


@dataclass
class Usage:
    """Running token and call totals, safe to update from worker threads."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_model: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, completion: Completion) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += completion.prompt_tokens
            self.completion_tokens += completion.completion_tokens
            self.by_model[completion.model] = (
                self.by_model.get(completion.model, 0) + completion.total_tokens
            )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

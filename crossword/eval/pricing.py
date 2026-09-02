"""USD cost from recorded token usage.

Rates are $ / 1M tokens, copied from public Nebius Token Factory prices
observed 2026-08-28 (InferenceBench). Update this table when rates move;
do not fetch on every eval. Unknown models return None rather than a guess.
"""

from __future__ import annotations

from typing import Mapping, Protocol

# (prompt_rate, completion_rate) in USD per 1M tokens.
RATES: dict[str, tuple[float, float]] = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": (0.10, 0.30),
    "Qwen/Qwen3-235B-A22B-Instruct-2507": (0.20, 0.60),
    "meta-llama/Llama-3.3-70B-Instruct": (0.13, 0.40),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "deepseek-ai/DeepSeek-V4-Pro": (1.75, 3.50),
    "moonshotai/Kimi-K2.6": (0.95, 4.00),
    "zai-org/GLM-5.2": (1.40, 4.40),
}


class _HasByModel(Protocol):
    by_model: Mapping[str, Mapping[str, int]]


def cost_usd(usage: _HasByModel) -> float | None:
    """Prompt+completion cost across every model that ran in this solve."""
    total = 0.0
    for model, counts in usage.by_model.items():
        rates = RATES.get(model)
        if rates is None:
            return None
        prompt_rate, completion_rate = rates
        total += counts.get("prompt", 0) * prompt_rate / 1_000_000
        total += counts.get("completion", 0) * completion_rate / 1_000_000
    return total

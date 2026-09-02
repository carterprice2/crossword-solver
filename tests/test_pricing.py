import unittest

from crossword.client import Completion, Usage
from crossword.eval.pricing import RATES, cost_usd

QWEN = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LLAMA = "meta-llama/Llama-3.3-70B-Instruct"


class TestCostUsd(unittest.TestCase):
    def test_known_model_one_million_each(self):
        usage = Usage()
        usage.record(
            Completion(
                text="",
                model=QWEN,
                prompt_tokens=1_000_000,
                completion_tokens=1_000_000,
            )
        )
        prompt_rate, completion_rate = RATES[QWEN]
        self.assertAlmostEqual(cost_usd(usage), prompt_rate + completion_rate)

    def test_unknown_model_is_none(self):
        usage = Usage()
        usage.record(
            Completion(
                text="",
                model="no-such/model",
                prompt_tokens=10,
                completion_tokens=10,
            )
        )
        self.assertIsNone(cost_usd(usage))

    def test_two_models_sum(self):
        usage = Usage()
        usage.record(
            Completion(text="", model=QWEN, prompt_tokens=1_000_000, completion_tokens=0)
        )
        usage.record(
            Completion(text="", model=LLAMA, prompt_tokens=0, completion_tokens=1_000_000)
        )
        self.assertAlmostEqual(cost_usd(usage), RATES[QWEN][0] + RATES[LLAMA][1])

    def test_empty_usage_is_zero(self):
        self.assertEqual(cost_usd(Usage()), 0.0)

    def test_flagship_models_have_rates(self):
        for model in (
            "Qwen/Qwen3.5-397B-A17B",
            "zai-org/GLM-5.2",
            "MiniMaxAI/MiniMax-M3",
        ):
            self.assertIn(model, RATES)

from dataclasses import dataclass

PRICE_PER_1M_INPUT = 0.15
PRICE_PER_1M_OUTPUT = 0.60

@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self):
        return (self.input_tokens / 1_000_000 * PRICE_PER_1M_INPUT
                + self.output_tokens / 1_000_000 * PRICE_PER_1M_OUTPUT)

    def as_dict(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.calls,
            "cost_usd": round(self.cost_usd, 6),
        }


_usage = Usage()


def reset_usage():
    global _usage
    _usage = Usage()


def record_usage(input_tokens, output_tokens):
    _usage.input_tokens += int(input_tokens or 0)
    _usage.output_tokens += int(output_tokens or 0)
    _usage.calls += 1


def get_usage():
    return _usage
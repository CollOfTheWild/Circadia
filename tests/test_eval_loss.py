import math

import mlx.core as mx
import pytest

import eval_loss


class UniformModel:
    def __init__(self, vocab_size: int = 5):
        self.vocab_size = vocab_size

    def __call__(self, input_ids):
        batch, seqlen = input_ids.shape
        return mx.zeros((batch, seqlen, self.vocab_size))


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return " ".join(m["content"] for m in messages)

    def encode(self, text):
        return [0, 1, 2, 3]


def test_loss_for_tokens_uniform() -> None:
    model = UniformModel(vocab_size=5)
    tokens = [0, 1, 2]
    loss = eval_loss.loss_for_tokens(model, tokens)
    assert loss == pytest.approx(math.log(5), rel=1e-3)


def test_evaluate_records_uses_tokens() -> None:
    model = UniformModel(vocab_size=4)
    tokenizer = DummyTokenizer()
    records = [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bye"}]},
    ]
    loss = eval_loss.evaluate_records(
        model=model,
        tokenizer=tokenizer,
        records=records,
        max_seq_len=16,
        max_samples=None,
    )
    assert loss > 0


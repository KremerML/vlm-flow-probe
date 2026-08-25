"""Shared CPU test doubles.

The tiny models mimic just enough of a causal-LM surface for the ablation and
caching machinery; ``adapt`` wraps one in a :class:`StubAdapter`, which is how
every model-facing test now drives the adapter interface.
"""

from types import SimpleNamespace

import torch
from torch import nn

from vlmflowprobe.adapters.stub import StubAdapter, StubTokenizer

TokenizerStub = StubTokenizer


class DummyModel(nn.Module):
    def __init__(self, d_model=4):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model)])

    def forward(self, input_ids=None, use_cache=False, **kwargs):
        x = torch.randn(1, 2, self.layers[0].in_features)
        return self.layers[0](x)


class CountingModel(nn.Module):
    def __init__(self, d_model=4, vocab=16):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model)])
        self.generate_calls = 0
        self.forward_calls = 0
        self.vocab = vocab

    def forward(self, input_ids=None, use_cache=False, **kwargs):
        self.forward_calls += 1
        seq_len = input_ids.shape[1]
        logits = torch.zeros(1, seq_len, self.vocab, device=input_ids.device)
        return SimpleNamespace(logits=logits)

    def generate(self, **kwargs):
        self.generate_calls += 1
        scores = torch.zeros(1, self.vocab)
        scores[0, 1] = 1.0
        return {
            "sequences": torch.tensor([[1]], dtype=torch.long),
            "scores": [scores],
        }


class DatasetStub:
    def __init__(self, num_samples=2):
        self.tokenizer = TokenizerStub()
        self.questions = []
        self.dataset_dict = {}
        for idx in range(num_samples):
            qid = f"q{idx}"
            self.questions.append({"q_id": qid, "attribute_tokens": []})
            self.dataset_dict[qid] = {
                "question": "what color is it",
                "answer": "yes",
                "true option": "yes",
                "false option": "no",
            }


def adapt(model, **kwargs) -> StubAdapter:
    """Wrap a stub model in a StubAdapter with the default test geometry."""
    kwargs.setdefault("tokenizer", TokenizerStub())
    return StubAdapter(model=model, **kwargs)

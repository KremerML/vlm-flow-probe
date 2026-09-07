"""Adapter probes for the contract suite, shared across test modules.

The LLaVA probe loads a 7B model. Two test modules need it and the GPU has 24 GB,
so the factory is memoized here rather than in either module — one load per
session, shared by ``test_adapter_contract`` and ``test_hf_llava_adapter``.

Adding a model means adding a factory here and one entry to ``ADAPTER_PROBES``.
"""

import os
from pathlib import Path

import pytest
import torch

from tests.stubs import CountingModel, TokenizerStub
from vlmflowprobe.adapters.contract import AdapterProbe, degenerate_by_stripping_image_tokens
from vlmflowprobe.adapters.stub import StubAdapter

ARCHIVE = Path(
    os.environ.get(
        "VFP_ARCHIVE_ROOT",
        Path.home() / "Documents/Github/cross-modal-information-flow-in-MLLM",
    )
)
# The archive checkout's copy locally; the repo-relative copy on the cluster.
_CLEVR_CANDIDATES = [ARCHIVE / "datasets/clevr_lite", Path("datasets/clevr_lite")]
CLEVR = next((p for p in _CLEVR_CANDIDATES if p.exists()), _CLEVR_CANDIDATES[0])

# The stub's sequence, laid out the way a real VLM's is: a leading token, an
# image block, question tokens, a trailing token. 9 stands in for the placeholder.
STUB_IMAGE_TOKEN = 9
STUB_SEQ = (1, STUB_IMAGE_TOKEN, STUB_IMAGE_TOKEN, 5, 6, 7)
STUB_QUESTION_SPAN = (3, 4)


def stub_probe() -> AdapterProbe:
    """The CPU double. Its logits are constant, so the end-to-end group is skipped."""
    adapter = StubAdapter(
        model=CountingModel(d_model=4, vocab=16),
        tokenizer=TokenizerStub(),
        seq_tokens=STUB_SEQ,
        question_span=STUB_QUESTION_SPAN,
        image_token_id=STUB_IMAGE_TOKEN,
    )
    probe = AdapterProbe(
        adapter=adapter,
        question="what color is it",
        image=object(),  # any non-None value; the stub only records that one is present
        installs_module_hooks=False,
        scores_are_meaningful=False,
    )
    probe.degenerate_image_batch = lambda: degenerate_by_stripping_image_tokens(probe)
    return probe


def clevr_dataset():
    """The CLEVR-Lite validation split from the archive checkout, loaded once."""
    if not CLEVR.exists():
        pytest.skip(f"CLEVR-Lite not found at {CLEVR}")
    if "clevr" not in _CACHE:
        from vlmflowprobe.data.datasets import CLEVRLiteVQADataset

        _CACHE["clevr"] = CLEVRLiteVQADataset(data_dir=str(CLEVR), split="val")
    return _CACHE["clevr"]


def hf_llava_probe() -> AdapterProbe:
    """The real LLaVA-1.5 adapter on the first CLEVR-Lite validation sample."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if not CLEVR.exists():
        pytest.skip(f"CLEVR-Lite not found at {CLEVR}")

    from vlmflowprobe.adapters.registry import create_adapter

    adapter = create_adapter({"adapter": "hf-llava", "name": "llava-hf/llava-1.5-7b-hf"})
    adapter.load()
    dataset = clevr_dataset()
    line = dataset.questions[0]
    detail = dataset.dataset_dict[line["q_id"]]
    probe = AdapterProbe(
        adapter=adapter,
        question=detail["question"],
        image=dataset.load_image(line),
        true_answer=detail["true option"].strip(),
        false_answer=detail["false option"].strip(),
    )
    probe.degenerate_image_batch = lambda: degenerate_by_stripping_image_tokens(probe)
    return probe


def hf_gemma3_probe() -> AdapterProbe:
    """The real Gemma 3 4B adapter on the first CLEVR-Lite validation sample.

    Knockout checks intervene at layer 5, the first global-attention layer, so
    the mask edit is exercised on a full-attention layer; layer 0 (sliding
    window) is covered by ``vfp-verify-adapter --layer 0``.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if not CLEVR.exists():
        pytest.skip(f"CLEVR-Lite not found at {CLEVR}")

    from vlmflowprobe.adapters.registry import create_adapter

    adapter = create_adapter({"adapter": "hf-gemma3", "name": "google/gemma-3-4b-it", "dtype": "bfloat16"})
    adapter.load()
    dataset = clevr_dataset()
    line = dataset.questions[0]
    detail = dataset.dataset_dict[line["q_id"]]
    probe = AdapterProbe(
        adapter=adapter,
        question=detail["question"],
        image=dataset.load_image(line),
        true_answer=detail["true option"].strip(),
        false_answer=detail["false option"].strip(),
        knockout_layer=5,
    )
    probe.degenerate_image_batch = lambda: degenerate_by_stripping_image_tokens(probe)
    return probe


_CACHE = {}

ADAPTER_PROBES = [
    pytest.param(stub_probe, id="stub"),
    pytest.param(hf_llava_probe, id="hf-llava", marks=pytest.mark.gpu),
    pytest.param(hf_gemma3_probe, id="hf-gemma3", marks=pytest.mark.gpu),
]


def cached(factory) -> AdapterProbe:
    """One probe per factory per session; a 7B load must not repeat per check."""
    if factory not in _CACHE:
        _CACHE[factory] = factory()
    return _CACHE[factory]

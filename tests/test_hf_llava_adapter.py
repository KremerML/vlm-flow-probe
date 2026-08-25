"""GPU checks for the HF-LLaVA adapter (pytest -m gpu).

Loads the real `llava-hf/llava-1.5-7b-hf` and a CLEVR-Lite sample from the
archive repo. Deselected by default; needs CUDA, the cached weights, and the
archive checkout (VFP_ARCHIVE_ROOT, defaulting to the sibling clone).
"""

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ARCHIVE = Path(
    os.environ.get(
        "VFP_ARCHIVE_ROOT",
        Path.home() / "Documents/Github/cross-modal-information-flow-in-MLLM",
    )
)
CLEVR = ARCHIVE / "datasets/clevr_lite"
SAMPLE_CACHE = (
    ARCHIVE
    / "output/sae_experiments/multilayer_clevr_lite_l10-14_attn_out_question/sample_cache.json"
)

pytestmark = pytest.mark.gpu

EXPECTED_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "USER: <image>\n{question} \nAnswer the question using a single word or phrase. ASSISTANT:"
)


@pytest.fixture(scope="module")
def adapter():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from vlmflowprobe.adapters.registry import create_adapter

    adapter = create_adapter({"adapter": "hf-llava", "name": "llava-hf/llava-1.5-7b-hf"})
    adapter.load()
    return adapter


@pytest.fixture(scope="module")
def dataset():
    if not CLEVR.exists():
        pytest.skip(f"CLEVR-Lite not found at {CLEVR}")
    from vlmflowprobe.data.datasets import CLEVRLiteVQADataset

    return CLEVRLiteVQADataset(data_dir=str(CLEVR), split="val")


@pytest.fixture(scope="module")
def batch(adapter, dataset):
    line = dataset.questions[0]
    detail = dataset.dataset_dict[line["q_id"]]
    return adapter.build_inputs(detail["question"], dataset.load_image(line)), line, detail


def test_static_properties(adapter):
    assert adapter.d_model == 4096
    assert adapter.n_layers == 32


def test_prompt_and_expansion(batch, adapter):
    b, _, detail = batch
    assert b.prompt == EXPECTED_PROMPT.format(question=detail["question"])
    assert adapter.n_image_tokens(b) == 576
    span = adapter.image_token_span(b)
    assert len(span) == 576
    assert b.seq_len == b.input_ids.shape[1]


def test_question_span_is_the_archive_fallback(batch, adapter):
    # Discovered during the port: the archive's sublist match NEVER hit for
    # CLEVR-Lite — in context the question follows "<image>\n", and
    # SentencePiece tokenizes "\nwhat" differently from bare "what", so every
    # published "question-position" run actually used the fallback span: from
    # the end of the image block through the final token (suffix + "ASSISTANT:"
    # included). The adapter must reproduce exactly that.
    b, _, detail = batch
    span = adapter.question_token_span(b)
    image_span = adapter.image_token_span(b)
    assert span == list(range(image_span.stop, b.seq_len))


def test_question_span_matches_archived_positions(adapter, dataset):
    if not SAMPLE_CACHE.exists():
        pytest.skip("archived sample_cache.json not available")
    records = json.loads(SAMPLE_CACHE.read_text())["records"]
    by_qid = {r["question_id"]: r for r in records}
    checked = 0
    for line in dataset.questions[:8]:
        rec = by_qid.get(str(line["q_id"]))
        if rec is None:
            continue
        detail = dataset.dataset_dict[line["q_id"]]
        b = adapter.build_inputs(detail["question"], dataset.load_image(line))
        assert adapter.question_token_span(b) == rec["positions"], line["q_id"]
        checked += 1
    assert checked > 0, "no overlapping question ids with the archived cache"


def test_knockout_changes_margin(batch, adapter):
    from vlmflowprobe.knockout.block_config import build_block_config, flow_block_pairs
    from vlmflowprobe.knockout.scoring import sequence_logprob

    b, _, detail = batch
    true_opt = detail["true option"].strip()

    base = sequence_logprob(adapter, b, true_opt)
    pairs = flow_block_pairs("Image->Question", b, adapter)
    assert pairs, "no block pairs resolved"
    blocked = sequence_logprob(
        adapter, b, true_opt,
        block_config=build_block_config(0, adapter.n_layers, 1, pairs),
        flow_target="Question",
    )
    assert base is not None and blocked is not None
    assert abs(base - blocked) > 1e-4, "knockout at layer 0 had no measurable effect"
    # cleanup left no hooks behind
    for layer in adapter._decoder_layers():
        assert len(layer._forward_pre_hooks) == 0

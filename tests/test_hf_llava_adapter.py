"""LLaVA-specific GPU checks (pytest -m gpu).

The model-agnostic invariants are in test_adapter_contract.py, which runs the
same suite against this adapter under the same mark. What is left here is what
only LLaVA-1.5 can be checked for: the byte-exact archive prompt, 576 image
tokens, and question spans equal to the archived positions.

Shares one loaded model with the contract suite via ``tests.probes`` — two
independent 7B loads do not fit in 24 GB. Needs CUDA, the cached weights, and
the archive checkout (VFP_ARCHIVE_ROOT, defaulting to the sibling clone).
"""

import json

import pytest

from tests.probes import ARCHIVE, cached, clevr_dataset, hf_llava_probe

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
def probe():
    return cached(hf_llava_probe)


@pytest.fixture(scope="module")
def adapter(probe):
    return probe.adapter


@pytest.fixture(scope="module")
def dataset():
    return clevr_dataset()


@pytest.fixture(scope="module")
def batch(probe, dataset):
    line = dataset.questions[0]
    detail = dataset.dataset_dict[line["q_id"]]
    return probe.batch(), line, detail


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

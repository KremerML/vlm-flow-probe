"""The ``ModelAdapter`` contract, written once as checks any adapter can run.

A new model's adapter has no archived results, so the LLaVA equivalence gate
(``gate/README.md``) cannot apply to it. What *can* apply is the contract every
adapter must satisfy regardless of model: post-expansion coordinates, geometry
that raises instead of guessing, execution conventions, and knockout hooks that
are always removed. Those checks live here so there is exactly one definition of
them, consumed by two callers:

* ``tests/test_adapter_contract.py`` -- parametrized over adapter factories, so
  the stub runs on CPU by default and real adapters run under ``-m gpu``;
* ``vfp-verify-adapter`` -- the same checks against a real model loaded from a
  config, plus the end-to-end group, printed as a pass/fail table.

Checks are grouped. ``modules``/``geometry``/``execution``/``knockout`` need
only a working adapter; ``end_to_end`` additionally needs a model whose logits
mean something, so it is skipped for the stub via
:attr:`AdapterProbe.scores_are_meaningful`.

Each check raises :class:`ContractViolation` (an ``AssertionError``, so pytest
reports it natively) and otherwise returns a one-line detail string for the
table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch

from vlmflowprobe.adapters.base import AdapterContractError, ModelAdapter, ModelBatch


class ContractViolation(AssertionError):
    """An adapter violated the contract."""


# --------------------------------------------------------------------------- probe


@dataclass
class AdapterProbe:
    """One adapter plus the inputs the checks need to exercise it.

    ``image`` is whatever the adapter's ``build_inputs`` accepts (a PIL image or
    a path); ``true_answer``/``false_answer`` are the forced-choice options the
    end-to-end group scores.
    """

    adapter: ModelAdapter
    question: str = "what color is the object"
    image: Any = None
    true_answer: str = "red"
    false_answer: str = "blue"
    #: Layer the knockout checks intervene at.
    knockout_layer: int = 0
    #: True when ``install_attention_knockout`` registers real module hooks. The
    #: stub records installs instead, so the hook-leak check has nothing to count.
    installs_module_hooks: bool = True
    #: True when the model's logits carry signal, gating the end_to_end group.
    scores_are_meaningful: bool = True
    #: Builds a batch whose image geometry is unresolvable -- pixel values present,
    #: no usable image tokens. ``n_image_tokens`` must raise on it rather than
    #: return 0. ``degenerate_by_stripping_image_tokens`` is the usual choice.
    degenerate_image_batch: Optional[Callable[[], ModelBatch]] = None
    #: False for text-only adapters, which legitimately have no image tokens.
    expects_image_tokens: bool = True
    _batch: Optional[ModelBatch] = field(default=None, repr=False)

    def batch(self) -> ModelBatch:
        """The sample every check runs against, built once."""
        if self._batch is None:
            self._batch = self.adapter.build_inputs(self.question, self.image)
        return self._batch


def degenerate_by_stripping_image_tokens(probe: "AdapterProbe") -> ModelBatch:
    """A batch with pixel values but the image tokens deleted from ``input_ids``.

    The default degenerate batch for any adapter that derives geometry from
    ``input_ids``: the sequence still claims an image (pixel values are present)
    but carries no placeholder tokens, which is exactly the state the archive
    reported as ``0`` image tokens and corrupted every downstream position with.
    """
    batch = probe.batch()
    span = probe.adapter.image_token_span(batch)
    keep = [i for i in range(batch.seq_len) if i not in set(span)]
    input_ids = batch.input_ids[:, keep]
    pixel_values = batch.pixel_values
    if pixel_values is None:
        pixel_values = torch.zeros(1, 1)
    return ModelBatch(
        input_ids=input_ids,
        attention_mask=None if batch.attention_mask is None else batch.attention_mask[:, keep],
        pixel_values=pixel_values,
        prompt=batch.prompt,
        question=batch.question,
        extra=dict(batch.extra),
    )


# --------------------------------------------------------------------------- helpers


def _hook_count(adapter: ModelAdapter) -> int:
    """Forward and forward-pre hooks currently registered on the decoder layers."""
    return sum(
        len(layer._forward_pre_hooks) + len(layer._forward_hooks)
        for layer in adapter._decoder_layers()
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractViolation(message)


def _expect_raises(fn: Callable[[], Any], what: str) -> None:
    try:
        result = fn()
    except AdapterContractError:
        return
    raise ContractViolation(f"{what} returned {result!r} instead of raising AdapterContractError")


# --------------------------------------------------------------------------- checks
# modules


def check_residual_site_is_the_decoder_layer(probe: AdapterProbe) -> str:
    adapter = probe.adapter
    layers = adapter._decoder_layers()
    _require(
        adapter.layer_module(0, "residual") is layers[0],
        "layer_module(0, 'residual') is not the decoder layer itself",
    )
    return f"{len(layers)} decoder layers"


def check_unknown_site_raises(probe: AdapterProbe) -> str:
    # The archive's get_target_module silently fell back to the residual stream
    # on a typo'd site, so an experiment could hook the wrong tensor and never say so.
    _expect_raises(lambda: probe.adapter.layer_module(0, "attn_output"), "layer_module(0, 'attn_output')")
    return "unknown site rejected"


def check_out_of_range_layer_raises(probe: AdapterProbe) -> str:
    n = probe.adapter.n_layers
    _expect_raises(lambda: probe.adapter.layer_module(n, "residual"), f"layer_module({n}, 'residual')")
    _expect_raises(lambda: probe.adapter.layer_module(-1, "residual"), "layer_module(-1, 'residual')")
    return f"rejects layers outside [0, {n})"


# geometry


def check_seq_len_matches_input_ids(probe: AdapterProbe) -> str:
    batch = probe.batch()
    _require(
        batch.seq_len == batch.input_ids.shape[1],
        f"seq_len {batch.seq_len} != input_ids.shape[1] {batch.input_ids.shape[1]}",
    )
    _require(batch.input_ids.shape[0] == 1, "adapters build one sample per batch")
    return f"seq_len {batch.seq_len}"


def check_image_span_is_post_expansion(probe: AdapterProbe) -> str:
    batch = probe.batch()
    adapter = probe.adapter
    count = adapter.n_image_tokens(batch)
    if not probe.expects_image_tokens:
        return "text-only adapter, no image tokens expected"
    _require(count > 0, "n_image_tokens returned 0 for a batch built with an image")
    span = adapter.image_token_span(batch)
    _require(
        len(span) == count,
        f"image_token_span has {len(span)} positions but n_image_tokens says {count}",
    )
    _require(
        all(0 <= pos < batch.seq_len for pos in span),
        f"image span {span} escapes [0, {batch.seq_len}) -- indices are not post-expansion",
    )
    _require(list(span) == list(range(span.start, span.stop)), "image span is not contiguous")
    return f"{count} image tokens at [{span.start}, {span.stop})"


def check_question_span_is_in_bounds(probe: AdapterProbe) -> str:
    batch = probe.batch()
    span = probe.adapter.question_token_span(batch)
    _require(span, "question_token_span is empty; every intervention position would be dropped")
    _require(
        all(0 <= pos < batch.seq_len for pos in span),
        f"question span escapes [0, {batch.seq_len}) -- indices are not post-expansion",
    )
    _require(span == sorted(span), "question span is not ascending")
    return f"{len(span)} positions at [{span[0]}, {span[-1]}]"


def check_last_token_index(probe: AdapterProbe) -> str:
    batch = probe.batch()
    index = probe.adapter.last_token_index(batch)
    _require(index == batch.seq_len - 1, f"last_token_index {index} != seq_len - 1 {batch.seq_len - 1}")
    return f"index {index}"


def check_answer_start_index(probe: AdapterProbe) -> str:
    # sequence_logprob scores answer token i at answer_start + i - 1; an adapter
    # that reported anything but seq_len here would score the wrong positions.
    batch = probe.batch()
    start = probe.adapter.answer_start_index(batch)
    _require(start == batch.seq_len, f"answer_start_index {start} != seq_len {batch.seq_len}")
    return f"index {start}"


def check_n_image_tokens_never_silently_zero(probe: AdapterProbe) -> str:
    if probe.degenerate_image_batch is None:
        raise ContractViolation(
            "probe supplies no degenerate_image_batch, so the 'never return a guessed 0' "
            "rule is untested; pass degenerate_by_stripping_image_tokens or an adapter-specific one"
        )
    degenerate = probe.degenerate_image_batch()
    _expect_raises(lambda: probe.adapter.n_image_tokens(degenerate), "n_image_tokens on an unresolvable batch")
    return "raises on a batch with pixel values but no image tokens"


# execution


def check_forward_appends_extra_input_ids(probe: AdapterProbe) -> str:
    batch = probe.batch()
    extra = torch.ones((1, 2), dtype=batch.input_ids.dtype, device=batch.input_ids.device)
    with torch.inference_mode():
        logits = probe.adapter.forward(batch, extra_input_ids=extra)
    _require(logits.ndim == 3, f"forward returned a {logits.ndim}-d tensor, expected [1, S, V]")
    _require(
        logits.shape[1] == batch.seq_len + 2,
        f"forward with 2 extra ids returned {logits.shape[1]} positions, expected {batch.seq_len + 2}",
    )
    return f"logits {tuple(logits.shape)}"


def check_generate_returns_new_tokens_only(probe: AdapterProbe) -> str:
    batch = probe.batch()
    with torch.inference_mode():
        result = probe.adapter.generate(batch, max_new_tokens=1)
    _require(result.sequences.ndim == 2, "GenResult.sequences must be [1, T]")
    length = int(result.sequences.shape[1])
    _require(length >= 1, "generate returned no tokens")
    _require(
        length <= batch.seq_len,
        f"generate returned {length} tokens for a {batch.seq_len}-token prompt -- "
        "sequences must be the new tokens only, not the prompt-inclusive HF sequence",
    )
    _require(
        result.first_token_scores.ndim == 2 and result.first_token_scores.shape[0] == 1,
        f"first_token_scores must be [1, V], got {tuple(result.first_token_scores.shape)}",
    )
    return f"{length} new token(s), scores {tuple(result.first_token_scores.shape)}"


# knockout


def _one_layer_block_config(probe: AdapterProbe) -> dict:
    """Block the question's first position from attending to the image's first."""
    batch = probe.batch()
    adapter = probe.adapter
    question = adapter.question_token_span(batch)
    if probe.expects_image_tokens:
        source = adapter.image_token_span(batch)[0]
    else:
        source = 0
    return {probe.knockout_layer: [(question[0], source)]}


def check_knockout_install_remove_balanced(probe: AdapterProbe) -> str:
    adapter = probe.adapter
    block_config = _one_layer_block_config(probe)
    before = _hook_count(adapter)
    with adapter.attention_knockout(block_config, flow_target="Question"):
        during = _hook_count(adapter)
    after = _hook_count(adapter)
    _require(after == before, f"knockout leaked hooks: {before} before, {after} after")
    if probe.installs_module_hooks:
        _require(during > before, "attention_knockout registered no hooks on the decoder layers")
    # Adapters that record installs rather than applying them (the stub) check
    # the pairing on their own ledger, which hook counting cannot see.
    if hasattr(adapter, "assert_knockouts_balanced"):
        adapter.assert_knockouts_balanced()
    return f"hooks {before} -> {during} -> {after}"


def check_knockout_removed_on_exception(probe: AdapterProbe) -> str:
    adapter = probe.adapter
    block_config = _one_layer_block_config(probe)
    before = _hook_count(adapter)
    try:
        with adapter.attention_knockout(block_config, flow_target="Question"):
            raise RuntimeError("deliberate failure inside the knockout context")
    except RuntimeError:
        pass
    after = _hook_count(adapter)
    _require(
        after == before,
        f"knockout hooks survived an exception: {before} before, {after} after",
    )
    return "context manager removed hooks on the error path"


# end to end


def check_sequence_logprob_scores_both_options(probe: AdapterProbe) -> str:
    from vlmflowprobe.knockout.scoring import sequence_logprob

    batch = probe.batch()
    true_lp = sequence_logprob(probe.adapter, batch, probe.true_answer)
    false_lp = sequence_logprob(probe.adapter, batch, probe.false_answer)
    _require(true_lp is not None and false_lp is not None, "sequence_logprob returned None for an option")
    _require(
        all(v <= 1e-6 for v in (true_lp, false_lp)),
        f"log-probabilities must be <= 0, got {true_lp:.4f} / {false_lp:.4f}",
    )
    return f"margin {true_lp - false_lp:+.4f} ({probe.true_answer!r} vs {probe.false_answer!r})"


def check_knockout_moves_the_margin(probe: AdapterProbe) -> str:
    from vlmflowprobe.knockout.block_config import build_block_config, flow_block_pairs
    from vlmflowprobe.knockout.scoring import sequence_logprob

    adapter, batch = probe.adapter, probe.batch()
    pairs = flow_block_pairs("Image->Question", batch, adapter)
    _require(bool(pairs), "Image->Question resolved to no block pairs")
    block_config = build_block_config(probe.knockout_layer, adapter.n_layers, 1, pairs)

    # The margin, not the true option alone: a confident model puts the true
    # option at log-prob ~0, where no single-layer edit registers, while the
    # false option's log-prob is far from saturation and moves.
    def margin(**kw):
        true_lp = sequence_logprob(adapter, batch, probe.true_answer, **kw)
        false_lp = sequence_logprob(adapter, batch, probe.false_answer, **kw)
        _require(true_lp is not None and false_lp is not None, "sequence_logprob returned None")
        return true_lp - false_lp

    base = margin()
    blocked = margin(block_config=block_config, flow_target="Question")
    _require(
        abs(base - blocked) > 1e-4,
        f"knockout at layer {probe.knockout_layer} left the margin unchanged "
        f"({base:.6f}); the mask edit is not reaching the attention computation",
    )
    _require(_hook_count(adapter) == 0, "knockout hooks survived sequence_logprob")
    return f"margin {base:+.4f} -> {blocked:+.4f} at layer {probe.knockout_layer}"


# --------------------------------------------------------------------------- registry


@dataclass(frozen=True)
class Check:
    name: str
    group: str
    fn: Callable[[AdapterProbe], str]
    description: str = ""

    def run(self, probe: AdapterProbe) -> str:
        return self.fn(probe)


CHECKS: Tuple[Check, ...] = (
    Check("residual_site_is_the_decoder_layer", "modules", check_residual_site_is_the_decoder_layer),
    Check("unknown_site_raises", "modules", check_unknown_site_raises),
    Check("out_of_range_layer_raises", "modules", check_out_of_range_layer_raises),
    Check("seq_len_matches_input_ids", "geometry", check_seq_len_matches_input_ids),
    Check("image_span_is_post_expansion", "geometry", check_image_span_is_post_expansion),
    Check("question_span_is_in_bounds", "geometry", check_question_span_is_in_bounds),
    Check("last_token_index", "geometry", check_last_token_index),
    Check("answer_start_index", "geometry", check_answer_start_index),
    Check("n_image_tokens_never_silently_zero", "geometry", check_n_image_tokens_never_silently_zero),
    Check("forward_appends_extra_input_ids", "execution", check_forward_appends_extra_input_ids),
    Check("generate_returns_new_tokens_only", "execution", check_generate_returns_new_tokens_only),
    Check("knockout_install_remove_balanced", "knockout", check_knockout_install_remove_balanced),
    Check("knockout_removed_on_exception", "knockout", check_knockout_removed_on_exception),
    Check("sequence_logprob_scores_both_options", "end_to_end", check_sequence_logprob_scores_both_options),
    Check("knockout_moves_the_margin", "end_to_end", check_knockout_moves_the_margin),
)

#: Groups that need only a working adapter, not a model whose logits mean anything.
STRUCTURAL_GROUPS = ("modules", "geometry", "execution", "knockout")


def checks_for(probe: AdapterProbe, groups: Optional[Sequence[str]] = None) -> List[Check]:
    """The checks that apply to ``probe``, filtered by group."""
    allowed = set(groups) if groups else None
    selected = []
    for check in CHECKS:
        if allowed is not None and check.group not in allowed:
            continue
        if check.group == "end_to_end" and not probe.scores_are_meaningful:
            continue
        selected.append(check)
    return selected


@dataclass
class CheckResult:
    name: str
    group: str
    passed: bool
    detail: str


def run_checks(probe: AdapterProbe, groups: Optional[Sequence[str]] = None) -> List[CheckResult]:
    """Run the applicable checks, collecting failures instead of raising.

    ``vfp-verify-adapter`` uses this; pytest runs one check per test instead so a
    failure points at the check that caused it.
    """
    results = []
    for check in checks_for(probe, groups):
        try:
            detail = check.run(probe)
            results.append(CheckResult(check.name, check.group, True, detail))
        except Exception as error:  # a violation, or the adapter blowing up
            results.append(CheckResult(check.name, check.group, False, f"{type(error).__name__}: {error}"))
    return results

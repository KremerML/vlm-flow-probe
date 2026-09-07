"""The adapter contract, run against every adapter the repo ships.

The checks themselves live in ``vlmflowprobe.adapters.contract`` so that
``vfp-verify-adapter`` runs the identical suite against a real model. This file
only parametrizes over the probes in ``tests.probes``: the stub runs on CPU by
default, real adapters carry ``pytest.mark.gpu`` on their param and are
deselected unless ``-m gpu`` is given.
"""

import pytest

from tests.probes import ADAPTER_PROBES, cached
from tests.stubs import CountingModel
from vlmflowprobe.adapters.contract import CHECKS
from vlmflowprobe.adapters.registry import adapter_class, create_adapter
from vlmflowprobe.adapters.stub import StubAdapter


@pytest.fixture
def probe(request):
    return cached(request.param)


@pytest.mark.parametrize("probe", ADAPTER_PROBES, indirect=True)
@pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.name)
def test_adapter_satisfies_contract(check, probe):
    if check.group == "end_to_end" and not probe.scores_are_meaningful:
        pytest.skip("end-to-end checks need a model whose logits carry signal")
    check.run(probe)


class TestRegistry:
    def test_stub_is_registered(self):
        assert adapter_class("stub") is StubAdapter

    def test_unknown_adapter_raises_with_known_list(self):
        with pytest.raises(KeyError, match="known adapters"):
            adapter_class("no-such-adapter")

    def test_create_adapter_requires_key(self):
        with pytest.raises(KeyError):
            create_adapter({"name": "some/model"})


class TestStubSpecificFailures:
    """Paths a real adapter cannot exercise, because its layers have the submodules."""

    def test_missing_submodule_raises(self):
        # nn.Linear layers have no self_attn/mlp: asking for those sites must raise
        # rather than quietly returning the layer itself.
        from vlmflowprobe.adapters.base import AdapterContractError

        adapter = StubAdapter(model=CountingModel(d_model=4))
        for site in ("attn_out", "mlp_out"):
            with pytest.raises(AdapterContractError):
                adapter.layer_module(0, site)


class TestConfiguredGeometryStillWorks:
    """The pinned-position stub form the rest of the suite uses is unchanged."""

    def test_configured_span_is_returned_verbatim(self):
        adapter = StubAdapter(model=CountingModel(d_model=4), image_span=range(0, 2))
        batch = adapter.build_inputs("q")
        assert adapter.n_image_tokens(batch) == 2
        assert adapter.image_token_span(batch) == range(0, 2)

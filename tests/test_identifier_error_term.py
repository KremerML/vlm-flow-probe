"""The identifier's splice hook: reconstruction by default, exact with the error term."""

import torch

from tests.stubs import CountingModel, DatasetStub, adapt
from vlmflowprobe.core.sparse_autoencoder import JumpReLUSAE
from vlmflowprobe.features.causal_feature_identifier import CausalFeatureIdentifier


def _identifier(error_term):
    sae = JumpReLUSAE(d_model=4, n_features=6)
    with torch.no_grad():
        sae.threshold.fill_(0.01)
    adapter = adapt(CountingModel(d_model=4))
    return CausalFeatureIdentifier(sae, adapter, DatasetStub(), 0, activation_site="residual",
                                   error_term=error_term), sae


def test_default_splices_the_reconstruction():
    identifier, sae = _identifier(False)
    acts = torch.randn(1, 3, 4)
    buffer = {}
    out = identifier.make_splice_hook(buffer)(None, None, acts)
    assert torch.allclose(out, sae.decode(sae.encode(acts), target_shape=acts.shape))
    assert buffer["z"].requires_grad and buffer["z"].shape == (3, 6)


def test_error_term_makes_the_forward_exact_but_keeps_the_gradient_path():
    identifier, sae = _identifier(True)
    acts = torch.randn(1, 3, 4)
    buffer = {}
    out = identifier.make_splice_hook(buffer)(None, None, acts)
    assert torch.allclose(out, acts, atol=1e-6)
    out.sum().backward()
    assert buffer["z"].grad is not None and torch.isfinite(buffer["z"].grad).all()


def test_tuple_outputs_are_rewritten_in_place():
    identifier, _ = _identifier(True)
    acts = torch.randn(1, 2, 4)
    out = identifier.make_splice_hook({})(None, None, (acts, "extra"))
    assert isinstance(out, tuple) and out[1] == "extra" and torch.allclose(out[0], acts, atol=1e-6)

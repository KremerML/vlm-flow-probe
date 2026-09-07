"""JumpReLU dictionaries: encode semantics, Gemma Scope conversion, loader dispatch."""

import torch

from vlmflowprobe.core.sparse_autoencoder import (
    JumpReLUSAE,
    SparseAutoencoder,
    build_sae_from_state,
    sae_state_from_checkpoint,
)


def _gemma_scope_tensors(d=6, f=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    w_dec = torch.randn(f, d, generator=g)
    w_dec = w_dec / w_dec.norm(dim=1, keepdim=True)
    return {
        "w_enc": torch.randn(d, f, generator=g),
        "w_dec": w_dec,
        "b_enc": torch.randn(f, generator=g) * 0.1,
        "b_dec": torch.randn(d, generator=g) * 0.1,
        "threshold": torch.rand(f, generator=g) * 0.5 + 0.1,
    }


def test_conversion_matches_reference_formula():
    t = _gemma_scope_tensors()
    sae = JumpReLUSAE.from_gemma_scope_tensors(t)
    x = torch.randn(7, 6)
    pre = x @ t["w_enc"] + t["b_enc"]
    z_ref = pre * (pre > t["threshold"]).float()
    recon_ref = z_ref @ t["w_dec"] + t["b_dec"]
    recon, z = sae(x)
    assert torch.allclose(z, z_ref, atol=1e-6)
    assert torch.allclose(recon, recon_ref, atol=1e-5)
    assert sae.encode(x).shape == (7, 10)


def test_gate_is_a_hard_threshold_not_a_relu():
    t = _gemma_scope_tensors()
    sae = JumpReLUSAE.from_gemma_scope_tensors(t)
    x = torch.randn(50, 6)
    pre = x @ t["w_enc"] + t["b_enc"]
    z = sae.encode(x)
    below = (pre > 0) & (pre <= t["threshold"])
    assert below.any(), "test needs some pre-activations between 0 and the threshold"
    assert (z[below] == 0).all()


def test_three_d_inputs_flatten_and_decode_back():
    sae = JumpReLUSAE.from_gemma_scope_tensors(_gemma_scope_tensors())
    x = torch.randn(1, 4, 6)
    recon, feats = sae(x)
    assert recon.shape == x.shape and feats.shape == (4, 10)
    assert sae.decode(feats, target_shape=x.shape).shape == x.shape


def test_missing_key_raises():
    t = _gemma_scope_tensors()
    del t["threshold"]
    try:
        JumpReLUSAE.from_gemma_scope_tensors(t)
    except KeyError as error:
        assert "threshold" in str(error)
    else:
        raise AssertionError("expected KeyError")


def test_loader_dispatches_on_state_dict():
    jump = JumpReLUSAE.from_gemma_scope_tensors(_gemma_scope_tensors())
    relu = SparseAutoencoder(d_model=6, n_features=10)
    payload = {"state": {"sae_state": jump.state_dict()}, "metadata": {}}
    rebuilt = build_sae_from_state(sae_state_from_checkpoint(payload))
    assert isinstance(rebuilt, JumpReLUSAE)
    x = torch.randn(3, 6)
    assert torch.allclose(rebuilt(x)[0], jump(x)[0])
    rebuilt_relu = build_sae_from_state(relu.state_dict(), l1_coeff=5e-4)
    assert isinstance(rebuilt_relu, SparseAutoencoder)
    assert rebuilt_relu.l1_coeff == 5e-4
    # Bare state dicts (the archive's older checkpoints) still resolve.
    assert isinstance(build_sae_from_state(sae_state_from_checkpoint(jump.state_dict())), JumpReLUSAE)


def test_ablation_hook_surface_is_shared():
    """The ablator only needs encode/decode/n_features/encoder.weight; both classes have them."""
    for sae in (JumpReLUSAE(6, 10), SparseAutoencoder(6, 10)):
        assert sae.n_features == 10 and sae.d_model == 6
        assert sae.encoder.weight.shape == (10, 6)
        z = sae.encode(torch.randn(2, 6))
        assert sae.decode(z, target_shape=(1, 2, 6)).shape == (1, 2, 6)

"""Sparse autoencoders: the L1/ReLU dictionary this repo trains, and the
JumpReLU form pre-trained dictionaries (Gemma Scope 2) ship in.

Both expose the same surface the rest of the package relies on --
``encode``, ``decode(z, target_shape)``, ``forward -> (recon, feats)``,
``n_features``, ``d_model``, and an ``encoder`` Linear whose weight gives the
device/dtype -- so the identifier, the ablators and the loaders never branch on
architecture. ``build_sae_from_state`` is the one place that does.
"""

from typing import Any, Dict, Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F


class SparseAutoencoder(nn.Module):
    """Basic sparse autoencoder with L1 penalty on activations."""

    def __init__(self, d_model: int, n_features: int, l1_coeff: float = 1e-3):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.l1_coeff = l1_coeff

        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model)
        self.activation = nn.ReLU()
        self.b_pre = nn.Parameter(torch.zeros(d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_flat, _ = self._flatten(x)
        feats = self.activation(self.encoder(x_flat - self.b_pre))
        return feats

    def decode(self, z: torch.Tensor, target_shape=None) -> torch.Tensor:
        recon = self.decoder(z)
        if target_shape is not None:
            recon = recon.view(target_shape)
        return recon

    def normalize_decoder(self) -> None:
        """Normalise decoder columns to unit norm after each optimiser step."""
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_flat, original_shape = self._flatten(x)
        feats = self.activation(self.encoder(x_flat - self.b_pre))
        recon = self.decoder(feats).view(original_shape)
        return recon, feats

    def get_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon, feats = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        l1_loss = feats.abs().mean()
        total = recon_loss + self.l1_coeff * l1_loss
        return total, recon_loss, l1_loss

    def _flatten(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
        if x.dim() <= 2:
            return x, x.shape
        original_shape = x.shape
        x_flat = x.view(-1, original_shape[-1])
        return x_flat, original_shape


class JumpReLUSAE(nn.Module):
    """JumpReLU sparse autoencoder (Rajamanoharan et al., 2024), Gemma Scope form.

    ``z = pre * H(pre - threshold)`` with ``pre = x W_enc + b_enc``; the
    decoder is affine, ``x_hat = z W_dec + b_dec``. Gemma Scope does not
    subtract ``b_dec`` before encoding, and its decoder rows are unit-norm.
    Thresholds are positive, so the gate also implies ``pre > 0``.

    Parameters are stored as ``encoder`` (``[F, d]`` weight = ``W_enc^T``),
    ``decoder`` (``[d, F]`` weight = ``W_dec^T``) and ``threshold`` (``[F]``),
    which is what :func:`build_sae_from_state` dispatches on: a state dict
    carrying ``threshold`` is a JumpReLU dictionary.
    """

    architecture = "jump_relu"

    def __init__(self, d_model: int, n_features: int):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model)
        self.threshold = nn.Parameter(torch.zeros(n_features))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_flat, _ = self._flatten(x)
        pre = self.encoder(x_flat)
        return F.relu(pre) * (pre > self.threshold).to(pre.dtype)

    def decode(self, z: torch.Tensor, target_shape=None) -> torch.Tensor:
        recon = self.decoder(z)
        if target_shape is not None:
            recon = recon.view(target_shape)
        return recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_flat, original_shape = self._flatten(x)
        feats = self.encode(x_flat)
        recon = self.decoder(feats).view(original_shape)
        return recon, feats

    @classmethod
    def from_gemma_scope_tensors(cls, tensors: Dict[str, torch.Tensor]) -> "JumpReLUSAE":
        """Build from the ``params.safetensors`` keys Gemma Scope 2 ships:
        ``w_enc [d, F]``, ``w_dec [F, d]``, ``b_enc [F]``, ``b_dec [d]``,
        ``threshold [F]``."""
        required = {"w_enc", "w_dec", "b_enc", "b_dec", "threshold"}
        missing = required - set(tensors)
        if missing:
            raise KeyError(f"Gemma Scope tensors missing {sorted(missing)}; have {sorted(tensors)}")
        w_enc = tensors["w_enc"].float()
        w_dec = tensors["w_dec"].float()
        d_model, n_features = w_enc.shape
        if tuple(w_dec.shape) != (n_features, d_model):
            raise ValueError(f"w_dec shape {tuple(w_dec.shape)} does not match w_enc {tuple(w_enc.shape)}")
        sae = cls(int(d_model), int(n_features))
        with torch.no_grad():
            sae.encoder.weight.copy_(w_enc.T)
            sae.encoder.bias.copy_(tensors["b_enc"].float())
            sae.decoder.weight.copy_(w_dec.T)
            sae.decoder.bias.copy_(tensors["b_dec"].float())
            sae.threshold.copy_(tensors["threshold"].float())
        return sae

    def _flatten(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
        if x.dim() <= 2:
            return x, x.shape
        original_shape = x.shape
        return x.view(-1, original_shape[-1]), original_shape


def sae_state_from_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """The SAE state dict inside a ``sae_checkpoint.pt`` payload (or a bare state dict)."""
    return checkpoint.get("state", {}).get("sae_state", checkpoint)


def build_sae_from_state(
    state: Dict[str, torch.Tensor], l1_coeff: Optional[float] = None
) -> nn.Module:
    """Instantiate the right dictionary class for a state dict and load it.

    Dimensions come from the state itself (``encoder.weight`` is ``[F, d]``),
    never from a config, so one config can drive several dictionaries.
    """
    n_features, d_model = state["encoder.weight"].shape
    if "threshold" in state:
        sae: nn.Module = JumpReLUSAE(d_model=int(d_model), n_features=int(n_features))
    else:
        sae = SparseAutoencoder(
            d_model=int(d_model),
            n_features=int(n_features),
            l1_coeff=1e-3 if l1_coeff is None else float(l1_coeff),
        )
    sae.load_state_dict(state)
    return sae

"""Sparse autoencoder implementation."""

from typing import Tuple
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

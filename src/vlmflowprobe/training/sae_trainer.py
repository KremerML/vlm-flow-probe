"""Training utilities for sparse autoencoders."""

from typing import Optional, Tuple
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from vlmflowprobe.data.collection import ActivationCollector
from vlmflowprobe.utils.checkpoint_utils import save_checkpoint, load_checkpoint
from vlmflowprobe.utils.config_utils import resolve_dtype


class SAETrainer:
    """Trainer for SAE models with activation collection support."""

    def __init__(
        self,
        sae: nn.Module,
        config,
        target_layer: int,
        activation_site: str = "residual",
        adapter=None,
    ):
        self.sae = sae
        self.config = config
        self.target_layer = target_layer
        self.activation_site = activation_site
        self.adapter = adapter

    def collect_activations(
        self,
        dataset,
        position_type: str = "question",
        max_samples: Optional[int] = None,
        show_progress: bool = False,
        checkpoint_dir: Optional[str] = None,
    ):
        if self.adapter is None:
            raise ValueError("an adapter is required for activation collection")
        collector = ActivationCollector(
            self.adapter,
            self.target_layer,
            activation_site=self.activation_site,
        )
        return collector.collect_from_dataset(
            dataset,
            position_type=position_type,
            max_samples=max_samples,
            show_progress=show_progress,
            checkpoint_dir=checkpoint_dir,
        )

    def train(
        self,
        activations: torch.Tensor,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        epochs: Optional[int] = None,
        device: Optional[str] = None,
        show_progress: bool = False,
        progress_desc: str = "SAE training",
    ) -> dict:
        train_cfg = self.config.get("training", {})
        batch_size = self._coerce_int(batch_size or train_cfg.get("batch_size", 32))
        learning_rate = self._coerce_float(learning_rate or train_cfg.get("learning_rate", 1e-4))
        epochs = self._coerce_int(epochs or train_cfg.get("epochs", 10))

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = resolve_dtype(train_cfg.get("dtype", "float32"))
        seed = self._coerce_int(train_cfg.get("seed", 42))

        n_rows, d = activations.shape
        act_mb = activations.element_size() * activations.nelement() / 1024 ** 2
        print(f"[SAETrainer] activations: {n_rows:,} rows × {d} dims, "
              f"device={activations.device}, dtype={activations.dtype}, size={act_mb:.0f} MB")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"[SAETrainer] GPU memory before training: "
                  f"{free/1024**3:.2f} GB free / {total/1024**3:.2f} GB total")

        self.sae.to(device=device, dtype=dtype)
        print(f"[SAETrainer] SAE model on {device}, training dtype={dtype}")
        # Keep activations in CPU RAM in their stored dtype (may be float16).
        # Each mini-batch is cast to the training dtype when moved to the device,
        # avoiding a full 2× RAM spike from an all-at-once upcast.
        print(f"[SAETrainer] activations kept on CPU in {activations.dtype}; "
              f"mini-batches cast to {dtype} on {device} per step")

        dataset = TensorDataset(activations)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.Adam(self.sae.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {
            "loss": [],
            "recon_loss": [],
            "l1_loss": [],
            "dead_feature_fraction": [],
            "seed": seed,
        }

        self.sae.train()
        pbar = None
        epoch_iter = range(epochs)
        if show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(epoch_iter, desc=progress_desc)
                epoch_iter = pbar
            except ImportError:
                pass

        for epoch_idx in epoch_iter:
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_l1 = 0.0
            feature_activation_counts = torch.zeros(self.sae.n_features, device=device)

            for (batch,) in loader:
                batch = batch.to(device=device, dtype=dtype)
                optimizer.zero_grad()
                recon, feats = self.sae.forward(batch)
                recon_loss = F.mse_loss(recon, batch)
                l1_loss = feats.abs().mean()
                total = recon_loss + self.sae.l1_coeff * l1_loss
                total.backward()
                optimizer.step()
                self.sae.normalize_decoder()

                with torch.no_grad():
                    feature_activation_counts += (feats > 0).float().sum(dim=0)

                epoch_loss += total.item()
                epoch_recon += recon_loss.item()
                epoch_l1 += l1_loss.item()

            scheduler.step()

            denom = max(1, len(loader))
            dead_fraction = (feature_activation_counts == 0).float().mean().item()
            avg_loss = epoch_loss / denom
            avg_recon = epoch_recon / denom
            avg_l1 = epoch_l1 / denom
            history["loss"].append(avg_loss)
            history["recon_loss"].append(avg_recon)
            history["l1_loss"].append(avg_l1)
            history["dead_feature_fraction"].append(dead_fraction)

            current_lr = scheduler.get_last_lr()[0]
            stats = {
                "loss": f"{avg_loss:.2e}",
                "recon": f"{avg_recon:.2e}",
                "l1": f"{avg_l1:.2e}",
                "dead": f"{dead_fraction:.2%}",
                "lr": f"{current_lr:.1e}",
            }
            if pbar is not None:
                pbar.set_postfix(stats)
            else:
                print(
                    f"[epoch {epoch_idx + 1:>{len(str(epochs))}}/{epochs}] "
                    + "  ".join(f"{k}={v}" for k, v in stats.items())
                )

        self.sae.eval()
        return history

    @staticmethod
    def _coerce_float(value) -> float:
        if isinstance(value, str):
            return float(value)
        return float(value)

    @staticmethod
    def _coerce_int(value) -> int:
        if isinstance(value, str):
            return int(float(value))
        return int(value)

    def save_checkpoint(self, path: str, metadata: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "sae_state": self.sae.state_dict(),
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else self.config,
            "target_layer": self.target_layer,
            "activation_site": self.activation_site,
        }
        save_checkpoint(state, path, metadata)

    def load_checkpoint(self, path: str) -> dict:
        state, metadata = load_checkpoint(path)
        self.sae.load_state_dict(state["sae_state"])
        return {"state": state, "metadata": metadata}

"""Shared setup helpers for experiment scripts."""

import torch

from vlmflowprobe.utils.checkpoint_utils import resolve_experiment_dir
from vlmflowprobe.utils.config_utils import resolve_dtype
from vlmflowprobe.utils.random_utils import resolve_seed, set_global_seed


def setup_experiment(args, config):
    """Resolve seed, set global seed, create experiment dir. Return (experiment_dir, seed)."""
    reproducibility_cfg = config.get("reproducibility", {})
    training_cfg = config.get("training", {})
    seed = resolve_seed(
        reproducibility_cfg.get("seed", training_cfg.get("seed")),
        fallback_seed=42,
    )
    set_global_seed(
        seed,
        deterministic=bool(reproducibility_cfg.get("deterministic", True)),
        benchmark=bool(reproducibility_cfg.get("benchmark", False)),
    )
    experiment_cfg = dict(config.get("experiment", {}))
    if getattr(args, "experiment_name", None):
        experiment_cfg["name"] = args.experiment_name
        experiment_cfg.pop("output_dir", None)
    experiment_dir = resolve_experiment_dir(
        experiment_cfg, getattr(args, "experiment_dir", None)
    )
    return experiment_dir, seed


def load_adapter(config):
    """Create and load the model adapter named by ``model.adapter``."""
    from vlmflowprobe.adapters.registry import create_adapter

    adapter = create_adapter(config.get("model", {}))
    adapter.load()
    return adapter


def load_sae(config, adapter, checkpoint_path):
    """Build SparseAutoencoder, load checkpoint, move to the adapter's device.

    ``d_model`` comes from the checkpoint itself (encoder weight shape) and is
    asserted against the adapter — a mismatched SAE fails here, not layers deep
    in a hook.
    """
    from vlmflowprobe.core.sparse_autoencoder import SparseAutoencoder

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state", {}).get("sae_state", ckpt)
    n_features, d_model = state["encoder.weight"].shape
    if d_model != adapter.d_model:
        raise ValueError(
            f"SAE checkpoint {checkpoint_path} has d_model={d_model} but the "
            f"adapter's model has d_model={adapter.d_model}"
        )
    sae = SparseAutoencoder(
        d_model=int(d_model),
        n_features=int(n_features),
        l1_coeff=config.get("sae", {}).get("l1_coeff", 1e-3),
    )
    sae.load_state_dict(state)
    train_cfg = config.get("training", {})
    sae.to(
        device=adapter.device,
        dtype=resolve_dtype(train_cfg.get("dtype", "float32")),
    )
    sae.eval()
    return sae

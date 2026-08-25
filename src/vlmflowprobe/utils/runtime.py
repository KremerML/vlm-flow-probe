"""Shared setup helpers for experiment scripts."""

import os

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


def load_llava_components(model_cfg, attn_implementation=None):
    """Load LLaVA model, tokenizer, image_processor. Return (tokenizer, model, image_processor)."""
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    model_path = os.path.expanduser(model_cfg.get("name", ""))
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        model_cfg.get("model_base"),
        model_name,
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    model.eval()
    return tokenizer, model, image_processor


def load_sae(config, model, checkpoint_path):
    """Build SparseAutoencoder, load checkpoint, move to device/dtype. Return sae."""
    from vlmflowprobe.core.sparse_autoencoder import SparseAutoencoder

    model_cfg = config.get("model", {})
    sae = SparseAutoencoder(
        d_model=model_cfg.get("d_model", 4096),
        n_features=config.get("sae", {}).get("n_features", 32768),
        l1_coeff=config.get("sae", {}).get("l1_coeff", 1e-3),
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sae.load_state_dict(ckpt.get("state", {}).get("sae_state", ckpt))
    train_cfg = config.get("training", {})
    sae.to(
        device=next(model.parameters()).device,
        dtype=resolve_dtype(train_cfg.get("dtype", "float32")),
    )
    sae.eval()
    return sae

"""Import a pre-trained dictionary as this pipeline's stage-01 artifact.

Stands in for ``vfp-train-sae`` when a model ships with public SAEs (Gemma 3:
Gemma Scope 2). Downloads the dictionary named by ``sae.pretrained`` for the
config's ``model.target_layer``, converts it to the repo's checkpoint format,
and writes the same artifacts training would -- ``sae_checkpoint.pt``,
``config.yaml``, ``provenance.json`` and, given ``--activations_path`` from
``vfp-collect``, ``reconstruction_eval.json`` measured on this task's
activations -- so ``vfp-identify``, ``vfp-ablate`` and ``vfp-multilayer`` run
unchanged.

The reconstruction evaluation is the part that matters scientifically: a
public dictionary was trained on text, and whether it reconstructs a VLM's
question-position activations is an empirical question the number answers.

    sae:
      pretrained:
        source: gemma-scope-2
        repo: google/gemma-scope-2-4b-it
        folder: attn_out_all
        width: 16k
        l0: big

    vfp-import-sae --config <cfg> --activations_path <dir from vfp-collect>
"""

import vlmflowprobe.cli._bootstrap  # noqa: F401

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch


def gemma_scope_subfolder(pretrained: Dict[str, Any], layer: int) -> str:
    folder = pretrained.get("folder", "attn_out_all")
    width = pretrained.get("width", "16k")
    l0 = pretrained.get("l0", "big")
    return f"{folder}/layer_{int(layer)}_width_{width}_l0_{l0}"


def fetch_gemma_scope(pretrained: Dict[str, Any], layer: int) -> Tuple[str, str]:
    """(params.safetensors path, config.json path) from the HF cache, downloading if needed."""
    from huggingface_hub import hf_hub_download

    repo = pretrained["repo"]
    sub = gemma_scope_subfolder(pretrained, layer)
    params = hf_hub_download(repo, f"{sub}/params.safetensors")
    config = hf_hub_download(repo, f"{sub}/config.json")
    return params, config


def convert_gemma_scope(params_path: str):
    """A :class:`JumpReLUSAE` from a Gemma Scope ``params.safetensors``."""
    from safetensors.torch import load_file

    from vlmflowprobe.core.sparse_autoencoder import JumpReLUSAE

    return JumpReLUSAE.from_gemma_scope_tensors(load_file(params_path))


def load_task_activations(activations_path: str, layer: int, max_rows: Optional[int]) -> torch.Tensor:
    acts_file = os.path.join(activations_path, f"layer_{layer}", "activations.pt")
    if not os.path.exists(acts_file):
        raise FileNotFoundError(f"no collected activations at {acts_file}")
    activations = torch.load(acts_file, weights_only=True, map_location="cpu")
    if max_rows is not None and activations.shape[0] > max_rows:
        activations = activations[:max_rows]
    return activations


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--target_layer", type=int, default=None)
    parser.add_argument("--activations_path", default=None,
                        help="vfp-collect output dir; enables reconstruction_eval.json on task activations")
    parser.add_argument("--eval_max_rows", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--experiment_dir", default=None)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    from vlmflowprobe.core.config import load_config, save_config
    from vlmflowprobe.training.sae_validation import compute_activation_stats, compute_reconstruction_metrics
    from vlmflowprobe.utils.checkpoint_utils import save_checkpoint
    from vlmflowprobe.utils.provenance import write_provenance
    from vlmflowprobe.utils.runtime import setup_experiment

    config = load_config(args.config, overrides=args.override)
    model_cfg = config.get("model", {})
    pretrained = (config.get("sae", {}) or {}).get("pretrained") or {}
    if not pretrained:
        raise SystemExit("config has no sae.pretrained section; nothing to import")
    source = pretrained.get("source", "gemma-scope-2")
    if source != "gemma-scope-2":
        raise SystemExit(f"unknown sae.pretrained.source {source!r}; only gemma-scope-2 is supported")

    experiment_dir, seed = setup_experiment(args, config)
    layer = args.target_layer if args.target_layer is not None else int(model_cfg.get("target_layer", 0))

    print(f"[import-sae] {pretrained['repo']} :: {gemma_scope_subfolder(pretrained, layer)}", flush=True)
    params_path, hf_config_path = fetch_gemma_scope(pretrained, layer)
    with open(hf_config_path) as handle:
        hf_config = json.load(handle)
    sae = convert_gemma_scope(params_path)
    print(f"[import-sae] JumpReLU {sae.n_features} x {sae.d_model}, target L0 {hf_config.get('l0')}, "
          f"hook {hf_config.get('hf_hook_point_in')}", flush=True)

    metadata: Dict[str, Any] = {
        "architecture": "jump_relu",
        "pretrained": {**pretrained, "layer": layer, "subfolder": gemma_scope_subfolder(pretrained, layer),
                       "hf_config": hf_config, "params_file": params_path},
        "seed": seed,
        "position_type": config.get("training", {}).get("position_type", "question"),
    }

    reconstruction_eval = None
    if args.activations_path:
        activations = load_task_activations(args.activations_path, layer, args.eval_max_rows)
        if activations.shape[1] != sae.d_model:
            raise SystemExit(
                f"collected activations are {activations.shape[1]} wide but the dictionary expects "
                f"{sae.d_model}; was vfp-collect run with model.activation_site matching the SAE's "
                "hook point?"
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sae.to(device=device, dtype=torch.float32)
        metrics = compute_reconstruction_metrics(sae, activations, batch_size=args.eval_batch_size)
        # Threshold-gated dictionaries can leave rows with no active feature at all;
        # that fraction is a health signal the L1 dictionaries never needed.
        with torch.no_grad():
            empty = 0
            for start in range(0, activations.shape[0], args.eval_batch_size):
                batch = activations[start:start + args.eval_batch_size]
                batch = batch.to(device=device, dtype=torch.float32)
                empty += int((sae.encode(batch) > 0).sum(dim=1).eq(0).sum().item())
        metrics["empty_row_fraction"] = empty / max(1, activations.shape[0])
        metrics["source"] = args.activations_path
        reconstruction_eval = {"task": metrics}
        metadata["activation_stats"] = compute_activation_stats(activations)
        metadata["reconstruction_loss"] = metrics["mse"]
        metadata["activation_samples"] = int(activations.shape[0])
        metadata["reconstruction_eval"] = reconstruction_eval
        print(f"[import-sae] task activations: {activations.shape[0]:,} rows | "
              f"explained var {metrics['explained_variance']:.5f} | norm. MSE {metrics['normalized_mse']:.2e} | "
              f"mean L0 {metrics['mean_l0']:.1f} | dead {metrics['dead_feature_fraction']:.4f} | "
              f"empty rows {metrics['empty_row_fraction']:.4f}", flush=True)
        sae.cpu()

    checkpoint_path = os.path.join(experiment_dir, "sae_checkpoint.pt")
    save_checkpoint({"sae_state": sae.state_dict()}, checkpoint_path, metadata=metadata)
    save_config(config, os.path.join(experiment_dir, "config.yaml"))
    write_provenance(experiment_dir, config=config.to_dict(), seed=seed, argv=sys.argv,
                     extra={"sae_import": metadata["pretrained"]})
    if reconstruction_eval is not None:
        with open(os.path.join(experiment_dir, "reconstruction_eval.json"), "w") as handle:
            json.dump(reconstruction_eval, handle, indent=2)
    print(f"[import-sae] wrote {checkpoint_path}")


if __name__ == "__main__":
    main()

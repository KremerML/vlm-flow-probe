"""Train an SAE on model activations (stage 01).

Two paths: live collection (loads the model, collects question-position
activations with disk-chunk checkpointing) or ``--activations_path`` (a
directory from ``vfp-collect`` — skips model loading entirely).

Memory-critical ordering, preserved from the archive: the model is deleted
and the CUDA cache emptied BEFORE chunk reassembly and training — that is
what makes 32768-feature SAEs fit beside a 7B model in 24 GB.

Writes ``{experiment_dir}/sae_checkpoint.pt``, ``reconstruction_eval.json``,
``holdout_split.json`` (if enabled), ``config.yaml``, ``provenance.json``.
"""

import vlmflowprobe.cli._bootstrap  # noqa: F401

import argparse
import json
import os
import random
import sys
import warnings
from typing import Any, Dict, List, Optional, Sequence

import torch


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _split_activation_rows_by_question(
    metadata: List[Dict[str, Any]],
    test_ratio: float,
    split_seed: int,
    min_test_samples: int,
) -> Dict[str, Any]:
    """Split activation row indices by question ID for holdout evaluation."""
    spans: List[Dict[str, Any]] = []
    for item in metadata:
        question_id = str(item.get("question_id", "")).strip()
        if not question_id:
            continue
        try:
            start_idx = int(item.get("start_idx", 0))
            count = int(item.get("count", 0))
        except (TypeError, ValueError):
            continue
        if count <= 0:
            continue
        spans.append({"question_id": question_id, "start_idx": start_idx, "count": count})

    if not spans:
        raise ValueError("No activation metadata spans available for holdout split.")

    question_ids = sorted({span["question_id"] for span in spans})
    n_questions = len(question_ids)
    if n_questions < 2:
        raise ValueError("Need at least 2 questions to create a holdout split.")

    ratio = float(max(0.0, min(1.0, test_ratio)))
    rng = random.Random(int(split_seed))
    rng.shuffle(question_ids)

    test_count = int(round(n_questions * ratio))
    if ratio > 0.0 and test_count == 0:
        test_count = 1
    test_count = max(test_count, int(min_test_samples))
    test_count = min(test_count, n_questions - 1)
    if test_count < int(min_test_samples):
        warnings.warn(
            "Could not satisfy holdout.min_test_samples with available questions. "
            f"Using {test_count} test questions out of {n_questions}.",
            RuntimeWarning,
        )
    if test_count <= 0:
        raise ValueError("Holdout split produced zero test questions.")

    test_question_ids = question_ids[:test_count]
    test_question_set = set(test_question_ids)
    train_question_ids = question_ids[test_count:]

    train_indices: List[int] = []
    test_indices: List[int] = []
    for span in spans:
        row_range = list(range(span["start_idx"], span["start_idx"] + span["count"]))
        if span["question_id"] in test_question_set:
            test_indices.extend(row_range)
        else:
            train_indices.extend(row_range)

    if not train_indices or not test_indices:
        raise ValueError(
            "Holdout split produced an empty train or test activation set. "
            f"train_rows={len(train_indices)}, test_rows={len(test_indices)}"
        )

    return {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "train_question_ids": train_question_ids,
        "test_question_ids": test_question_ids,
        "total_questions": n_questions,
    }


def _take_rows(activations: torch.Tensor, row_indices: Sequence[int]) -> torch.Tensor:
    if not row_indices:
        return activations[:0]
    max_index = max(int(idx) for idx in row_indices)
    if max_index >= int(activations.shape[0]):
        raise ValueError(
            f"Row index out of range for activations: max_index={max_index}, rows={activations.shape[0]}"
        )
    index_tensor = torch.tensor(list(row_indices), dtype=torch.long, device=activations.device)
    return activations.index_select(0, index_tensor)


from vlmflowprobe.training.sae_validation import (  # noqa: E402
    compute_reconstruction_metrics as _compute_reconstruction_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--target_layer", type=int, default=None)
    parser.add_argument("--position_type", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--show_progress", type=_parse_bool, default=True)
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--activations_path", type=str, default=None,
                        help="Directory of pre-collected activations (vfp-collect output). "
                             "Skips model loading.")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    from vlmflowprobe.core.config import load_config, save_config
    from vlmflowprobe.core.sparse_autoencoder import SparseAutoencoder
    from vlmflowprobe.data.datasets import build_dataset
    from vlmflowprobe.training.sae_trainer import SAETrainer
    from vlmflowprobe.training.sae_validation import compute_activation_stats
    from vlmflowprobe.utils.config_utils import resolve_training_position_type
    from vlmflowprobe.utils.provenance import write_provenance
    from vlmflowprobe.utils.runtime import load_adapter, setup_experiment

    config = load_config(args.config, overrides=args.override)
    model_cfg = config.get("model", {})
    holdout_cfg = config.get("holdout", {})
    feat_cfg = config.get("feature_identification", {})
    training_cfg = config.get("training", {})
    experiment_dir, seed = setup_experiment(args, config)

    checkpoint_path = args.checkpoint_path or os.path.join(experiment_dir, "sae_checkpoint.pt")
    target_layer = args.target_layer if args.target_layer is not None else model_cfg.get("target_layer", 12)
    train_position_type = resolve_training_position_type(
        args.position_type, training_cfg, feat_cfg, default="question",
    )

    def make_sae(d_model: int) -> SparseAutoencoder:
        return SparseAutoencoder(
            d_model=d_model,
            n_features=config.get("sae", {}).get("n_features", 32768),
            l1_coeff=config.get("sae", {}).get("l1_coeff", 1e-3),
        )

    adapter = None
    if args.activations_path:
        # Pre-collected activations: no model, d_model comes from the data.
        layer_dir = os.path.join(args.activations_path, f"layer_{target_layer}")
        acts_file = os.path.join(layer_dir, "activations.pt")
        meta_file = os.path.join(layer_dir, "metadata.json")
        if not os.path.exists(acts_file):
            raise FileNotFoundError(f"Pre-collected activations not found: {acts_file}")
        print(f"[01] Loading activations from {acts_file}")
        activations = torch.load(acts_file, weights_only=True, map_location="cpu")
        with open(meta_file) as _f:
            metadata = json.load(_f)
        if args.max_samples is not None and activations.shape[0] > args.max_samples:
            activations = activations[: args.max_samples]
            metadata = metadata[: args.max_samples]
            print(f"[01] Subsampled to {activations.shape[0]:,} rows (--max_samples)")
        print(f"[01] Loaded {activations.shape[0]:,} rows x {activations.shape[1]} dims")
        chunk_files = None
        sae = make_sae(int(activations.shape[1]))
        trainer = SAETrainer(
            sae=sae,
            config=config,
            target_layer=target_layer,
            activation_site=model_cfg.get("activation_site", "residual"),
            adapter=None,
        )
    else:
        adapter = load_adapter(config)
        dataset = build_dataset(config, tokenizer=adapter.tokenizer)
        sae = make_sae(adapter.d_model)
        trainer = SAETrainer(
            sae=sae,
            config=config,
            target_layer=target_layer,
            activation_site=model_cfg.get("activation_site", "residual"),
            adapter=adapter,
        )
        act_cache_dir = os.path.join(experiment_dir, "_activation_cache")
        result = trainer.collect_activations(
            dataset,
            position_type=train_position_type,
            max_samples=args.max_samples,
            show_progress=args.show_progress,
            checkpoint_dir=act_cache_dir,
        )
        if isinstance(result[0], list):
            metadata, chunk_files = result
            activations = None
        else:
            activations, metadata = result
            chunk_files = None

    if adapter is not None and torch.cuda.is_available():
        # Free the model BEFORE training/reassembly — the 24 GB constraint.
        trainer.adapter = None
        del adapter
        torch.cuda.empty_cache()
        import gc

        gc.collect()
        print("[01] Released the model to free memory for training")

    if chunk_files is not None:
        from vlmflowprobe.data.collection import reassemble_chunks

        print(f"[01] Reassembling {len(chunk_files)} chunks after model release...")
        activations = reassemble_chunks(chunk_files, cleanup=True)
        print(f"[01] Activations: {activations.shape[0]:,} rows x {activations.shape[1]} dims")

    holdout_enabled = bool(holdout_cfg.get("enabled", False))
    train_activations = activations
    test_activations: Optional[torch.Tensor] = None
    holdout_split: Optional[Dict[str, Any]] = None
    if holdout_enabled:
        split = _split_activation_rows_by_question(
            metadata=metadata,
            test_ratio=float(holdout_cfg.get("test_ratio", 0.2)),
            split_seed=int(holdout_cfg.get("split_seed", seed)),
            min_test_samples=int(holdout_cfg.get("min_test_samples", 50)),
        )
        train_activations = _take_rows(activations, split["train_indices"])
        test_activations = _take_rows(activations, split["test_indices"])
        holdout_split = {
            "enabled": True,
            "test_ratio": float(holdout_cfg.get("test_ratio", 0.2)),
            "split_seed": int(holdout_cfg.get("split_seed", seed)),
            "min_test_samples": int(holdout_cfg.get("min_test_samples", 50)),
            "total_questions": int(split["total_questions"]),
            "train_question_count": int(len(split["train_question_ids"])),
            "test_question_count": int(len(split["test_question_ids"])),
            "train_activation_rows": int(train_activations.shape[0]),
            "test_activation_rows": int(test_activations.shape[0]),
            "train_question_ids": split["train_question_ids"],
            "test_question_ids": split["test_question_ids"],
        }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    history = trainer.train(train_activations, show_progress=args.show_progress)
    activation_stats = compute_activation_stats(train_activations)
    eval_batch_size = int(holdout_cfg.get("eval_batch_size", 512))
    reconstruction_eval = {
        "train": _compute_reconstruction_metrics(trainer.sae, train_activations, batch_size=eval_batch_size),
    }
    if test_activations is not None:
        reconstruction_eval["test"] = _compute_reconstruction_metrics(
            trainer.sae, test_activations, batch_size=eval_batch_size,
        )
    recon_loss = float(reconstruction_eval["train"]["mse"])
    trainer.save_checkpoint(
        checkpoint_path,
        metadata={
            "history": history,
            "activation_stats": activation_stats,
            "reconstruction_loss": recon_loss,
            "activation_samples": int(train_activations.shape[0]),
            "activation_samples_total": int(activations.shape[0]),
            "seed": seed,
            "deterministic": bool(config.get("reproducibility", {}).get("deterministic", True)),
            "position_type": train_position_type,
            "holdout": holdout_split or {"enabled": False},
            "reconstruction_eval": reconstruction_eval,
        },
    )
    save_config(config, os.path.join(experiment_dir, "config.yaml"))
    write_provenance(experiment_dir, config=config.to_dict(), seed=seed, argv=sys.argv)

    if holdout_split is not None:
        with open(os.path.join(experiment_dir, "holdout_split.json"), "w", encoding="utf-8") as handle:
            json.dump(holdout_split, handle, indent=2)

    reconstruction_eval_path = os.path.join(experiment_dir, "reconstruction_eval.json")
    with open(reconstruction_eval_path, "w", encoding="utf-8") as handle:
        json.dump(reconstruction_eval, handle, indent=2)

    print(f"Saved SAE checkpoint to {checkpoint_path}")
    print(f"Saved reconstruction evaluation to {reconstruction_eval_path}")
    if holdout_split is not None:
        print(
            "Holdout split:",
            f"train_questions={holdout_split['train_question_count']},",
            f"test_questions={holdout_split['test_question_count']},",
            f"train_rows={holdout_split['train_activation_rows']},",
            f"test_rows={holdout_split['test_activation_rows']}",
        )
    print(f"Experiment directory: {experiment_dir}")


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    main()

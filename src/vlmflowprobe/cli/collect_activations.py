"""Collect activations for many layers in one forward pass per sample.

Training an SAE per layer by re-running the model per layer wastes a full
forward pass per layer; this hooks every requested layer at once and streams
position-selected activations to per-layer chunk files, reassembled into one
tensor per layer at the end (bounded RAM: one layer's tensor at a time).

Output layout (consumed by ``vfp-train-sae --activations_path``):
``{output_dir}/layer_{N}/activations.pt`` (fp16 [rows, d_model]),
``{output_dir}/layer_{N}/metadata.json`` (per-question spans with
``start_idx``/``count`` — the holdout split reads these), and
``{output_dir}/collection_info.json`` (provenance).

The archive tool hand-rolled an 8-way batched forward by splicing image
embeddings manually; that machinery died with the adapter port. This runs
sample-by-sample (batch=1) — slower per sample, model-agnostic, and with no
custom multimodal code to maintain.
"""

import vlmflowprobe.cli._bootstrap  # noqa: F401

import argparse
import json
import os
import sys
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True,
                        help="Any config with the right model/dataset sections.")
    parser.add_argument("--layers", type=str, default="0,10,11,12,13,14")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--position_type", type=str, default="question")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--flush_every", type=int, default=2000,
                        help="flush per-layer chunks to disk every N samples (bounds RAM)")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    from vlmflowprobe.core.config import load_config
    from vlmflowprobe.data.datasets import build_dataset
    from vlmflowprobe.data.loading import iter_batches
    from vlmflowprobe.positions import COLLECTION_POLICY, resolve_positions
    from vlmflowprobe.utils.provenance import write_provenance
    from vlmflowprobe.utils.runtime import load_adapter, setup_experiment

    config = load_config(args.config, overrides=args.override)
    _, seed = setup_experiment(args, config)
    layers = [int(l) for l in args.layers.split(",") if l.strip()]

    adapter = load_adapter(config)
    dataset = build_dataset(config, tokenizer=adapter.tokenizer)
    d_model = adapter.d_model

    site = config.get("model", {}).get("activation_site", "attn_out")
    os.makedirs(args.output_dir, exist_ok=True)

    # One capture hook per layer, all firing in the same forward pass.
    storage = {}
    handles = []

    def make_hook(layer):
        def hook(module, inputs, output):
            acts = output[0] if isinstance(output, (tuple, list)) else output
            storage[layer] = acts.detach()

        return hook

    for layer in layers:
        handles.append(adapter.layer_module(layer, site).register_forward_hook(make_hook(layer)))

    pending = {layer: [] for layer in layers}
    chunk_files = {layer: [] for layer in layers}
    metadata = {layer: [] for layer in layers}
    offsets = {layer: 0 for layer in layers}
    n_samples = 0
    started = time.time()

    def flush(layer):
        if not pending[layer]:
            return
        chunk_dir = os.path.join(args.output_dir, f"layer_{layer}", "_chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        path = os.path.join(chunk_dir, f"chunk_{len(chunk_files[layer]):05d}.pt")
        torch.save(torch.cat(pending[layer], dim=0), path)
        chunk_files[layer].append(path)
        pending[layer].clear()
    try:
        for batch, line in iter_batches(
            dataset, adapter,
            max_samples=args.max_samples,
            show_progress=not args.no_progress,
            progress_desc="Collecting activations",
        ):
            storage.clear()
            with torch.no_grad():
                adapter.forward(batch)
            positions = resolve_positions(
                args.position_type, batch, adapter, policy=COLLECTION_POLICY, line=line
            )
            if not positions:
                continue
            detail = dataset.dataset_dict[line["q_id"]]
            for layer in layers:
                acts = storage.get(layer)
                if acts is None:
                    continue
                if acts.dim() == 3:
                    acts = acts[0]
                selected = acts[positions].cpu().to(torch.float16)
                pending[layer].append(selected)
                metadata[layer].append({
                    "question_id": line["q_id"],
                    "question": detail.get("question", ""),
                    "answer": detail.get("answer", ""),
                    "positions": list(positions),
                    "attribute_tokens": line.get("attribute_tokens", []),
                    "start_idx": offsets[layer],
                    "count": len(positions),
                })
                offsets[layer] += len(positions)
            n_samples += 1
            if args.flush_every and n_samples % args.flush_every == 0:
                for layer in layers:
                    flush(layer)
    finally:
        for handle in handles:
            handle.remove()

    for layer in layers:
        layer_dir = os.path.join(args.output_dir, f"layer_{layer}")
        os.makedirs(layer_dir, exist_ok=True)
        flush(layer)
        # One layer's tensor in RAM at a time — same peak as vfp-train-sae.
        parts = [torch.load(path, weights_only=True) for path in chunk_files[layer]]
        stacked = torch.cat(parts, dim=0) if parts else torch.empty(0, d_model, dtype=torch.float16)
        torch.save(stacked, os.path.join(layer_dir, "activations.pt"))
        for path in chunk_files[layer]:
            os.remove(path)
        chunk_dir = os.path.join(layer_dir, "_chunks")
        if os.path.isdir(chunk_dir) and not os.listdir(chunk_dir):
            os.rmdir(chunk_dir)
        with open(os.path.join(layer_dir, "metadata.json"), "w") as handle:
            json.dump(metadata[layer], handle)
        print(f"layer {layer}: {stacked.shape[0]:,} rows -> {layer_dir}/activations.pt")

    info = {
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "layers": layers,
        "activation_site": site,
        "position_type": args.position_type,
        "n_samples": n_samples,
        "max_samples": args.max_samples,
        "rows_per_layer": {str(l): offsets[l] for l in layers},
        "runtime_seconds": round(time.time() - started, 1),
        "d_model": d_model,
    }
    with open(os.path.join(args.output_dir, "collection_info.json"), "w") as handle:
        json.dump(info, handle, indent=2)
    write_provenance(args.output_dir, config=config.to_dict(), seed=seed, argv=sys.argv, adapter=adapter)
    print(f"Collected {n_samples} samples across {len(layers)} layers in {info['runtime_seconds']}s")


if __name__ == "__main__":
    main()

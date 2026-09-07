"""Where the time goes in one Gemma 3 forward, under the adapter's options.

python scripts/gemma3_profile.py --config configs/experiments/gemma3_4b/knockout_clevr_lite.yaml
"""

import vlmflowprobe.cli._bootstrap  # noqa: F401

import argparse
import time

import torch


def timed(fn, n=10):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return 1000 * (time.time() - t0) / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    from vlmflowprobe.core.config import load_config
    from vlmflowprobe.data.datasets import build_dataset
    from vlmflowprobe.data.loading import iter_batches
    from vlmflowprobe.knockout.scoring import sequence_logprob
    from vlmflowprobe.utils.runtime import load_adapter
    from vlmflowprobe.utils.random_utils import set_global_seed

    config = load_config(args.config, overrides=args.override)
    adapter = load_adapter(config)
    dataset = build_dataset(config, tokenizer=adapter.tokenizer, split="val")
    batch, line = next(iter(iter_batches(dataset, adapter, max_samples=1)))
    detail = dataset.dataset_dict[line["q_id"]]
    print("options:", config.get("model", {}).get("adapter_options"))
    print("deterministic:", torch.are_deterministic_algorithms_enabled())

    with torch.inference_mode():
        print(f"forward (adapter, as configured): {timed(lambda: adapter.forward(batch)):.1f} ms")
        if batch.pixel_values is not None:
            print(
                f"vision tower + projector:          "
                f"{timed(lambda: adapter.model.get_image_features(batch.pixel_values)):.1f} ms"
            )
        from vlmflowprobe.adapters.base import ModelBatch

        pix = ModelBatch(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            pixel_values=batch.pixel_values,
            prompt=batch.prompt,
            question=batch.question,
            extra={"token_type_ids": batch.extra["token_type_ids"]},
        )
        print(f"forward via pixel_values:          {timed(lambda: adapter.forward(pix)):.1f} ms")
        print(
            f"sequence_logprob (one option):     "
            f"{timed(lambda: sequence_logprob(adapter, batch, detail['true option'])):.1f} ms"
        )
        print(f"generate(1):                       {timed(lambda: adapter.generate(batch, 1)):.1f} ms")
        set_global_seed(42, deterministic=False, benchmark=False)
        print(f"forward, deterministic off:        {timed(lambda: adapter.forward(batch)):.1f} ms")
        set_global_seed(42, deterministic=True, benchmark=False)
    print(f"gpu peak alloc {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()

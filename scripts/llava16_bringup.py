"""LLaVA-1.6 bring-up: the things vfp-verify-adapter does not print but the paper needs.

Run on a GPU node before any pipeline stage:

    python scripts/llava16_bringup.py --config configs/experiments/llava16/knockout_clevr_lite.yaml -n 32

Reports, on the first n validation questions: the decoded prompt layout (image
block, question span text), the AnyRes token count and the grid the processor
picked, what the model actually generates (casing, leading space), forced-choice
margins under the pipeline's answer convention and the two alternatives, forward
timing with and without the image-feature cache, and a one-layer
Image->Question knockout at every layer on the first sample. Writes a JSON
beside the log so the numbers are on record.

The comparison that matters is against LLaVA-1.5: same decoder, same prompt,
same data, 1176 image tokens instead of 576. Run the same script against the
llava15 config to get the other column.
"""

import vlmflowprobe.cli._bootstrap  # noqa: F401

import argparse
import json
import statistics
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("-n", type=int, default=32)
    parser.add_argument("--json", default="output/llava16_bringup.json")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    from vlmflowprobe.adapters.base import ModelBatch
    from vlmflowprobe.core.config import load_config
    from vlmflowprobe.data.datasets import build_dataset
    from vlmflowprobe.data.loading import iter_batches
    from vlmflowprobe.knockout.block_config import build_block_config, flow_block_pairs
    from vlmflowprobe.knockout.scoring import sequence_logprob
    from vlmflowprobe.utils.runtime import load_adapter

    config = load_config(args.config, overrides=args.override)
    adapter = load_adapter(config)
    dataset = build_dataset(config, tokenizer=adapter.tokenizer, split="val")
    tok = adapter.tokenizer
    vision = adapter.model.config
    report = {
        "model": config.get("model", {}).get("name"),
        "n_layers": adapter.n_layers,
        "d_model": adapter.d_model,
        "image_grid_pinpoints": getattr(vision, "image_grid_pinpoints", None),
        "vision_feature_layer": getattr(vision, "vision_feature_layer", None),
        "vision_feature_select_strategy": getattr(vision, "vision_feature_select_strategy", None),
        "samples": [],
    }
    print(f"layers={adapter.n_layers} d_model={adapter.d_model} "
          f"pinpoints={report['image_grid_pinpoints']}")

    fwd_times, gen_times = [], []
    for idx, (batch, line) in enumerate(iter_batches(dataset, adapter, max_samples=args.n)):
        detail = dataset.dataset_dict[line["q_id"]]
        true_opt, false_opt = detail["true option"], detail["false option"]
        ids = batch.input_ids[0]
        if idx == 0:
            span = adapter.image_token_span(batch)
            qspan = adapter.question_token_span(batch)
            print("prompt:", repr(batch.prompt))
            print(
                f"seq_len={batch.seq_len} image_tokens={adapter.n_image_tokens(batch)} "
                f"image=[{span.start},{span.stop}) question_span={qspan[0]}..{qspan[-1]} ({len(qspan)})"
            )
            print("pre-image tokens:", repr(tok.decode(ids[: span.start])))
            print("question span text:", repr(tok.decode(ids[qspan[0]: qspan[-1] + 1])))
            image_sizes = batch.extra.get("image_sizes")
            print("pixel_values:", tuple(batch.pixel_values.shape),
                  "image_sizes:", image_sizes.tolist() if image_sizes is not None else None)
            report["layout"] = {
                "seq_len": batch.seq_len,
                "n_image_tokens": adapter.n_image_tokens(batch),
                "image_span": [span.start, span.stop],
                "question_span": [qspan[0], qspan[-1]],
                "pixel_values_shape": list(batch.pixel_values.shape),
                "image_sizes": image_sizes.tolist() if image_sizes is not None else None,
                "cached_feature_rows": int(batch.extra["image_features"].shape[0])
                if "image_features" in batch.extra else None,
                "prompt": batch.prompt,
            }

        t0 = time.time()
        with torch.inference_mode():
            adapter.forward(batch)
        torch.cuda.synchronize()
        fwd_times.append(time.time() - t0)
        t0 = time.time()
        with torch.inference_mode():
            gen = adapter.generate(batch, max_new_tokens=4)
        torch.cuda.synchronize()
        gen_times.append(time.time() - t0)
        text = tok.decode(gen.sequences[0], skip_special_tokens=True)

        def margin_and_lp(prefix, fmt):
            saved_prefix, saved_fmt = adapter.answer_prefix, adapter.format_answer
            adapter.answer_prefix, adapter.format_answer = prefix, fmt
            try:
                lp_true = sequence_logprob(adapter, batch, true_opt)
                return lp_true - sequence_logprob(adapter, batch, false_opt), lp_true
            finally:
                adapter.answer_prefix, adapter.format_answer = saved_prefix, saved_fmt

        # The pipeline's own convention first (" " + lowercase, LLaVA-1.5's), then
        # the two alternatives: no leading space, and capitalized.
        m_fmt, lp_fmt = margin_and_lp(adapter.answer_prefix, adapter.format_answer)
        m_nospace, lp_nospace = margin_and_lp("", adapter.format_answer)
        m_upper, lp_upper = margin_and_lp(
            adapter.answer_prefix, lambda a: a.strip()[:1].upper() + a.strip()[1:]
        )
        row = {
            "q": detail["question"],
            "true": true_opt,
            "false": false_opt,
            "generated": text,
            "margin_formatted": m_fmt,
            "margin_nospace": m_nospace,
            "margin_upper": m_upper,
            "lp_true_formatted": lp_fmt,
            "lp_true_nospace": lp_nospace,
            "lp_true_upper": lp_upper,
        }
        print(
            f"[{idx}] {detail['question']!r} true={true_opt} gen={text!r} "
            f"margin fmt={m_fmt:+.3f} nospace={m_nospace:+.3f} upper={m_upper:+.3f} | lp(true) "
            f"fmt={lp_fmt:.3f} nospace={lp_nospace:.3f} upper={lp_upper:.3f}"
        )
        report["samples"].append(row)

    # The feature cache: same logits, and how much time it actually saves.
    batch, line = next(iter(iter_batches(dataset, adapter, max_samples=1)))
    detail = dataset.dataset_dict[line["q_id"]]
    true_opt, false_opt = detail["true option"], detail["false option"]
    uncached = ModelBatch(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        pixel_values=batch.pixel_values,
        prompt=batch.prompt,
        question=batch.question,
        extra={k: v for k, v in batch.extra.items() if k != "image_features"},
    )

    def median_forward_ms(target, repeats=5):
        times = []
        for _ in range(repeats):
            t0 = time.time()
            with torch.inference_mode():
                logits = adapter.forward(target)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        return 1000 * statistics.median(times), logits[0, -1].float()

    cached_ms, cached_logits = median_forward_ms(batch)
    pixel_ms, pixel_logits = median_forward_ms(uncached)
    report["feature_cache"] = {
        "cache_active": "image_features" in batch.extra,
        "cached_forward_ms": cached_ms,
        "pixel_forward_ms": pixel_ms,
        "speedup": pixel_ms / cached_ms if cached_ms else None,
        "max_abs_logit_diff": float((cached_logits - pixel_logits).abs().max()),
    }
    print("feature cache:", json.dumps(report["feature_cache"], indent=1))

    # One-layer Image->Question knockout at every layer, first sample.
    pairs = flow_block_pairs("Image->Question", batch, adapter)
    base = sequence_logprob(adapter, batch, true_opt) - sequence_logprob(adapter, batch, false_opt)
    report["knockout"] = {}
    for layer in range(adapter.n_layers):
        bc = build_block_config(layer, adapter.n_layers, 1, pairs)
        t = sequence_logprob(adapter, batch, true_opt, block_config=bc, flow_target="Question")
        f = sequence_logprob(adapter, batch, false_opt, block_config=bc, flow_target="Question")
        report["knockout"][layer] = base - (t - f)
    print("knockout margin drop, sample 0, per layer:",
          {k: round(v, 3) for k, v in report["knockout"].items()})

    def mean_of(key):
        return statistics.mean(s[key] for s in report["samples"])

    report["summary"] = {
        "forced_choice_acc_formatted": statistics.mean(
            1.0 if s["margin_formatted"] > 0 else 0.0 for s in report["samples"]
        ),
        "mean_margin_formatted": mean_of("margin_formatted"),
        "mean_margin_nospace": mean_of("margin_nospace"),
        "mean_margin_upper": mean_of("margin_upper"),
        "mean_lp_true_formatted": mean_of("lp_true_formatted"),
        "mean_lp_true_nospace": mean_of("lp_true_nospace"),
        "mean_lp_true_upper": mean_of("lp_true_upper"),
        # Case-insensitive, as the pipeline's own generation accuracy is
        # (feature_ablator lowercases the decoded prediction); reported beside
        # the raw generated string so a casing difference stays visible.
        "gen_matches_true_answer": statistics.mean(
            1.0 if s["generated"].strip().lower().startswith(s["true"].strip().lower()) else 0.0
            for s in report["samples"]
        ),
        "gen_is_capitalized": statistics.mean(
            1.0 if s["generated"].strip()[:1].isupper() else 0.0 for s in report["samples"]
        ),
        "forward_ms": 1000 * statistics.median(fwd_times),
        "generate4_ms": 1000 * statistics.median(gen_times),
        "gpu": torch.cuda.get_device_name(0),
    }
    print(json.dumps(report["summary"], indent=1))
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=1)
    print("wrote", args.json)


if __name__ == "__main__":
    main()

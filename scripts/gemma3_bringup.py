"""Gemma 3 bring-up: the things vfp-verify-adapter does not print but the paper needs.

Run on a GPU node before any pipeline stage:

    python scripts/gemma3_bringup.py --config configs/experiments/gemma3_4b/knockout_clevr_lite.yaml -n 32

Reports, on the first n validation questions: the decoded prompt layout
(bos count, image block, question span text), what the model actually
generates (casing, leading space), forced-choice margins with and without a
leading-space answer prefix, forward-pass timing, the width the attn_z tap
emits, and a one-layer knockout on a sliding and a global layer. Writes a JSON
beside the log so the numbers are on record.
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
    parser.add_argument("--json", default="gemma3_bringup.json")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

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
    report = {
        "n_layers": adapter.n_layers,
        "d_model": adapter.d_model,
        "attn_z_dim": adapter.site_dim(0, "attn_z"),
        "samples": [],
    }
    layer_types = [layer.attention_type for layer in adapter._decoder_layers()]
    report["layer_types"] = layer_types
    print(f"layers={adapter.n_layers} d_model={adapter.d_model} attn_z={report['attn_z_dim']}")
    print("global layers:", [i for i, t in enumerate(layer_types) if t == "full_attention"])

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
                f"seq_len={batch.seq_len} bos_count={(ids == tok.bos_token_id).sum().item()} "
                f"image=[{span.start},{span.stop}) question_span={qspan[0]}..{qspan[-1]} ({len(qspan)})"
            )
            print("pre-image tokens:", repr(tok.decode(ids[: span.start])))
            print("question span text:", repr(tok.decode(ids[qspan[0] : qspan[-1] + 1])))
            print("token_type_ids sum:", int(batch.extra["token_type_ids"].sum()))
            report["layout"] = {
                "seq_len": batch.seq_len,
                "image_span": [span.start, span.stop],
                "question_span": [qspan[0], qspan[-1]],
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

        # The pipeline's own convention first, then the two alternatives it rejected.
        m_fmt, lp_fmt = margin_and_lp(adapter.answer_prefix, adapter.format_answer)
        m_lower, lp_lower = margin_and_lp("", str.strip)
        m_space, lp_space = margin_and_lp(" ", adapter.format_answer)
        row = {
            "q": detail["question"],
            "true": true_opt,
            "false": false_opt,
            "generated": text,
            "margin_formatted": m_fmt,
            "margin_lower": m_lower,
            "margin_space": m_space,
            "lp_true_formatted": lp_fmt,
            "lp_true_lower": lp_lower,
            "lp_true_space": lp_space,
        }
        print(
            f"[{idx}] {detail['question']!r} true={true_opt} gen={text!r} "
            f"margin fmt={m_fmt:+.3f} lower={m_lower:+.3f} space={m_space:+.3f} | lp(true) "
            f"fmt={lp_fmt:.3f} lower={lp_lower:.3f} space={lp_space:.3f}"
        )
        report["samples"].append(row)

    # knockout on one sliding + one global layer, first sample
    batch, line = next(iter(iter_batches(dataset, adapter, max_samples=1)))
    detail = dataset.dataset_dict[line["q_id"]]
    pairs = flow_block_pairs("Image->Question", batch, adapter)
    true_opt, false_opt = detail["true option"], detail["false option"]
    base = sequence_logprob(adapter, batch, true_opt) - sequence_logprob(adapter, batch, false_opt)
    report["knockout"] = {}
    for layer in range(adapter.n_layers):
        bc = build_block_config(layer, adapter.n_layers, 1, pairs)
        t = sequence_logprob(adapter, batch, true_opt, block_config=bc, flow_target="Question")
        f = sequence_logprob(adapter, batch, false_opt, block_config=bc, flow_target="Question")
        report["knockout"][layer] = base - (t - f)
    print(
        "knockout margin drop, sample 0, per layer:", {k: round(v, 3) for k, v in report["knockout"].items()}
    )

    def mean_of(key):
        return statistics.mean(s[key] for s in report["samples"])

    report["summary"] = {
        "forced_choice_acc_formatted": statistics.mean(
            1.0 if s["margin_formatted"] > 0 else 0.0 for s in report["samples"]
        ),
        "mean_margin_formatted": mean_of("margin_formatted"),
        "mean_margin_lower": mean_of("margin_lower"),
        "mean_margin_space": mean_of("margin_space"),
        "mean_lp_true_formatted": mean_of("lp_true_formatted"),
        "mean_lp_true_lower": mean_of("lp_true_lower"),
        "mean_lp_true_space": mean_of("lp_true_space"),
        "gen_matches_formatted": statistics.mean(
            1.0 if s["generated"].strip().startswith(adapter.format_answer(s["true"])) else 0.0
            for s in report["samples"]
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

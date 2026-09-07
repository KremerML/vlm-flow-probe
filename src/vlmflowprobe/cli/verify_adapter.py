"""Verify a model adapter against the contract, on the real model.

What a new model gets in place of the LLaVA equivalence gate. The gate compares
against archived numbers; a model with no archived results has nothing to
compare to, so what can still be established is that the adapter honours the
contract every downstream stage assumes:

* the structural checks in ``adapters.contract`` -- module resolution that
  raises instead of falling back, post-expansion coordinates, execution
  conventions, knockout hooks that always come off;
* the end-to-end path those checks cannot cover on a stub -- build_inputs ->
  forward -> generate -> sequence_logprob on both options -> knockout at one
  layer moving the margin -> no hooks left behind.

Prints a pass/fail table and exits nonzero on the first failure. The sample
comes from the configured dataset when it is available, so the options scored
are real ones; ``--question``/``--image`` override that for a model whose
dataset is not set up yet.

    vfp-verify-adapter --config configs/experiments/llava15/sae_layer11_attn_out_question.yaml
    vfp-verify-adapter --config <cfg> --layer 5 --json report.json
"""

import vlmflowprobe.cli._bootstrap  # noqa: F401  (env defaults before torch)

import argparse
import json
import os
import sys

from vlmflowprobe.adapters.contract import (
    AdapterProbe,
    degenerate_by_stripping_image_tokens,
    run_checks,
)


def _dataset_sample(config, adapter, index):
    """(question, image, true option, false option) from the configured dataset.

    Returns ``None`` when the dataset is not present -- a model being brought up
    usually has no data wired yet, and that must not block adapter verification.
    """
    try:
        from vlmflowprobe.data.datasets import build_dataset

        dataset = build_dataset(config, tokenizer=adapter.tokenizer, split="val")
        line = dataset.questions[index]
        detail = dataset.dataset_dict[line["q_id"]]
        return (
            detail["question"],
            dataset.load_image(line),
            str(detail.get("true option", "")).strip(),
            str(detail.get("false option", "")).strip(),
        )
    except Exception as error:
        print(f"[verify-adapter] no dataset sample ({type(error).__name__}: {error}); "
              "falling back to --question/--image", file=sys.stderr)
        return None


def build_probe(config, adapter, args) -> AdapterProbe:
    question, image = args.question, args.image
    true_answer, false_answer = args.true_answer, args.false_answer

    if question is None:
        sample = _dataset_sample(config, adapter, args.sample)
        if sample is not None:
            question, image, sampled_true, sampled_false = sample
            true_answer = true_answer or sampled_true
            false_answer = false_answer or sampled_false

    if question is None:
        question = "what color is the object"
    if image is not None and isinstance(image, str):
        from PIL import Image

        image = Image.open(image).convert("RGB")

    probe = AdapterProbe(
        adapter=adapter,
        question=question,
        image=image,
        true_answer=true_answer or "red",
        false_answer=false_answer or "blue",
        knockout_layer=args.layer,
        expects_image_tokens=not args.text_only,
    )
    probe.degenerate_image_batch = lambda: degenerate_by_stripping_image_tokens(probe)
    return probe


def render_table(results, adapter_name) -> str:
    width = max(len(r.name) for r in results) + 2
    lines = [
        f"adapter contract: {adapter_name}",
        "",
        f"{'':<6}{'check':<{width}}{'group':<12}detail",
        "-" * (width + 72),
    ]
    for result in results:
        mark = "PASS" if result.passed else "FAIL"
        lines.append(f"{mark:<6}{result.name:<{width}}{result.group:<12}{result.detail}")
    failed = [r for r in results if not r.passed]
    lines.append("")
    lines.append(
        f"{len(results) - len(failed)}/{len(results)} checks passed"
        + (f"; FAILED: {', '.join(r.name for r in failed)}" if failed else "")
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[],
                        help="config override, section.key=value (repeatable)")
    parser.add_argument("--layer", type=int, default=0,
                        help="layer the knockout checks intervene at (default 0, "
                             "where Image->Question knockout has its largest effect)")
    parser.add_argument("--question", default=None,
                        help="bypass the dataset and probe with this question")
    parser.add_argument("--image", default=None, help="image path for --question")
    parser.add_argument("--true_answer", default=None)
    parser.add_argument("--false_answer", default=None)
    parser.add_argument("--sample", type=int, default=0,
                        help="index of the dataset sample to probe with")
    parser.add_argument("--groups", default=None,
                        help="comma-separated subset of check groups to run "
                             "(modules, geometry, execution, knockout, end_to_end)")
    parser.add_argument("--text_only", action="store_true",
                        help="adapter has no image tokens; skip the image-geometry checks")
    parser.add_argument("--json", dest="json_path", default=None,
                        help="also write the results as JSON here")
    args = parser.parse_args()

    from vlmflowprobe.core.config import load_config
    from vlmflowprobe.utils.provenance import provenance_dict
    from vlmflowprobe.utils.runtime import load_adapter

    config = load_config(args.config, overrides=args.override)
    adapter_name = config.get("model", {}).get("adapter", "?")
    model_name = config.get("model", {}).get("name", "?")
    print(f"[verify-adapter] loading {adapter_name} ({model_name})...", flush=True)
    adapter = load_adapter(config)
    print(f"[verify-adapter] {adapter.n_layers} layers, d_model {adapter.d_model}", flush=True)

    probe = build_probe(config, adapter, args)
    groups = [g.strip() for g in args.groups.split(",")] if args.groups else None
    results = run_checks(probe, groups=groups)

    table = render_table(results, f"{adapter_name} ({model_name})")
    print()
    print(table)

    if args.json_path:
        payload = {
            "adapter": adapter_name,
            "model": model_name,
            "n_layers": adapter.n_layers,
            "d_model": adapter.d_model,
            "knockout_layer": args.layer,
            "question": probe.question,
            "checks": [vars(r) for r in results],
            "passed": all(r.passed for r in results),
            "provenance": provenance_dict(argv=sys.argv, config=config.to_dict()),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json_path)), exist_ok=True)
        with open(args.json_path, "w") as handle:
            json.dump(payload, handle, indent=1)
        print(f"\nwrote {args.json_path}")

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

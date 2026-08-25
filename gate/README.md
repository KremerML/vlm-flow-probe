# The equivalence gate

`vfp-gate` establishes that this repo's HF-native harness reproduces the
archived results (produced by the LLaVA-NeXT-repo harness in
`cross-modal-information-flow-in-MLLM`). Reference artifacts are vendored in
`references/` (provenance: `references/SOURCES.md`); bulk inputs (the layer-11
SAE checkpoint, CLEVR-Lite) are read from `--archive-root`.

| stage | what | pass criteria |
|---|---|---|
| 0 geometry | all 256 archived samples rebuilt through the adapter | HARD EQUALITY, never waivable: sample order, byte-exact prompts, 576 image tokens, question spans == archived positions |
| 1 baselines | 256 per-sample baselines recomputed | margin r >= 0.999; mean abs delta <= 0.05; p99 <= 0.25; pred agreement >= 254/256 (flips only where cached prob < 0.5); accuracy within 1/256; self-jitter < 1e-3 |
| 2 knockout | 10-sample sweep, both flows, all 32 layers | Image->Question vs archived n=10 run: Spearman >= 0.95, identical top-3 layers, top-5 drops within max(0.05, 15%); Image->Last vs the n=7084 profile: Spearman >= 0.7; structural: last-layer Image->Question drop exactly 0 |
| 3 A0 regression | archived layer-11 SAE + catalog, replace-mode 200-feature ablation + the 15 archived control feature sets rerun verbatim | mean drop within 0.03 of 0.21307; per-sample drop r >= 0.98; mean control drop < 50% of binding |

Any relaxation of a criterion requires an entry in
`docs/deviation_ledger.md`. The gate report lands in
`output/gate/<timestamp>/gate_report.json` and includes provenance (git SHA,
package versions, GPU).

Run: `scripts/run_gate.sh` (or `vfp-gate --stage N` for one stage).

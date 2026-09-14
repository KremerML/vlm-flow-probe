# Multi-layer ablation summary

Span: [10, 11, 12, 13, 14]. 53 conditions.

## Redundancy index R(S) = A(S) / K(S)

**R(S) = 0.936** (95% CI 0.890 - 0.985), from A = 3.6422 and K = 3.8899 over n = 256.

Single layers, for comparison (same samples for numerator and denominator):

| layer | depth | ablation | knockout | ratio | 95% CI |
|---|---|---|---|---|---|
| 10 | 0.31 | — | — | undefined | missing_condition |
| 11 | 0.34 | — | — | undefined | missing_condition |
| 12 | 0.38 | — | — | undefined | missing_condition |
| 13 | 0.41 | — | — | undefined | missing_condition |
| 14 | 0.44 | 1.7021 | 1.2329 | 1.381 | 1.313 - 1.454 |

## Redundancy by span

| kind | span | size | A | K | R | 95% CI |
|---|---|---|---|---|---|---|
| nested | `{14}` | 1 | 1.7021 | 1.2329 | 138.1% | 131.3% - 145.4% |
| nested | `{13,14}` | 2 | 2.1778 | 1.1197 | 194.5% | 185.1% - 205.4% |
| nested | `{12,13,14}` | 3 | 2.8235 | 1.5376 | 183.6% | 175.6% - 192.4% |
| nested | `{11-14}` | 4 | 3.3639 | 3.5895 | 93.7% | 89.1% - 98.7% |
| nested | `{10-14}` | 5 | 3.6422 | 3.8899 | 93.6% | 89.0% - 98.5% |
| non-nested | `{10,11,12}` | 3 | 1.9796 | 3.4513 | 57.4% | 53.5% - 61.5% |
| non-nested | `{10,12,14}` | 3 | 2.4192 | 2.0060 | 120.6% | 116.1% - 125.3% |
| sensitivity | `{10,11,12,14}` | 4 | 3.1682 | 3.6939 | 85.8% | 81.2% - 90.6% |

Pooled recovery over the nested spans is 120.6% (95% CI 115.7% - 125.7%). The per-span ratios run 93.6% to 194.5% with NO value inside every interval, so R is not constant across spans and must not be reported as such. R trends with span size at -0.1896 per layer (95% CI -0.2120 - -0.1693), which excludes zero.

Observed spread across the nested spans is 100.9% (95% CI 92.5% - 110.9%); highest lower bound 185.1% against lowest upper bound 98.5%. 10000 draws, seed 42, n = 256.

## Comparative saturation

NEITHER curve saturates over this range -- both fits collapsed to straight lines (ablation 1.766/layer, knockout 0.762/layer), so b and d are not identified and b - d carries no information. What is identified is the slope ratio, 2.318: over spans of 1-5 layers the two curves grow in a fixed proportion, with no sign of the ablation curve turning over sooner. Wider spans would be needed to see saturation at all. This is a statement about the two slopes and NOT about R span by span -- see redundancy_by_span, where the ratios disperse beyond their own intervals.

## Interaction index

Not computed: this run contains no standalone ablation for layer(s) 10, 12, 13. Summing only the layers present would understate the denominator and inflate rho_A, so it is withheld. Supply the missing single-layer conditions, or compute rho_A against the published single-layer drops by hand -- the binding arm does not depend on the control regime, so those remain comparable.

## Leave-one-out

| layer | joint | without | marginal | standalone | marginal/standalone |
|---|---|---|---|---|---|
| 10 | 3.6422 | 3.3639 | +0.2783 | — | — |
| 11 | 3.6422 | 3.0463 | +0.5959 | 0.6949 | 0.858 |
| 12 | 3.6422 | 3.1644 | +0.4778 | — | — |
| 13 | 3.6422 | 3.1682 | +0.4740 | — | — |
| 14 | 3.6422 | 2.6982 | +0.9440 | 1.7021 | 0.555 |

A marginal contribution far below the standalone effect at every layer is the strong redundancy signature: the rest of the span already carries what that layer contributes.

## Significance

| condition | binding | z | z SE | control sets | p floor | Wilcoxon p | control regime |
|---|---|---|---|---|---|---|---|
| budget_spread40x5 | 1.3590 | -- | -- | 1 | 0.5000 | 9.62e-32 | {'10': 'matched_activation_mean', '11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean'} |
| joint_L10-14 | 3.6422 | 75 | 14 | 15 | 0.0625 | 1.03e-43 | {'10': 'matched_activation_mean', '11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean'} |
| nested_L11-14 | 3.3639 | -- | -- | 1 | 0.5000 | 9.64e-44 | {'10': 'matched_activation_mean', '11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean'} |
| nested_L12-14 | 2.8235 | -- | -- | 1 | 0.5000 | 1.42e-43 | {'10': 'matched_activation_mean', '11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean'} |
| nested_L13-14 | 2.1778 | -- | -- | 1 | 0.5000 | 1.11e-43 | {'10': 'matched_activation_mean', '11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean'} |
| nested_L14 | 1.7021 | -- | -- | 1 | 0.5000 | 9.86e-44 | {'10': 'matched_activation_mean', '11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean'} |
| nonnested_L10,12,14 | 2.4192 | -- | -- | 1 | 0.5000 | 1.86e-43 | {'10': 'matched_activation_mean', '11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean'} |
| nonnested_L10-12 | 1.9796 | -- | -- | 1 | 0.5000 | 1.32e-43 | {'10': 'matched_activation_mean', '11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean'} |

z is reported to one significant figure: it is estimated from a handful of control draws and its own standard error is large. The empirical p cannot go below its floor, so the Wilcoxon test over questions is the one with power.

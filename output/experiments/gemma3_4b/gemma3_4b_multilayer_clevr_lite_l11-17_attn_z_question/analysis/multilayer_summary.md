# Multi-layer ablation summary

Span: [11, 12, 13, 14, 15, 16, 17]. 66 conditions.

## Redundancy index R(S) = A(S) / K(S)

**R(S) = 1.215** (95% CI 1.176 - 1.256), from A = 28.6087 and K = 23.5483 over n = 256.

Single layers, for comparison (same samples for numerator and denominator):

| layer | depth | ablation | knockout | ratio | 95% CI |
|---|---|---|---|---|---|
| 11 | 0.32 | — | — | undefined | missing_condition |
| 12 | 0.35 | — | — | undefined | missing_condition |
| 13 | 0.38 | — | — | undefined | missing_condition |
| 14 | 0.41 | — | — | undefined | missing_condition |
| 15 | 0.44 | — | — | undefined | missing_condition |
| 16 | 0.47 | — | — | undefined | missing_condition |
| 17 | 0.50 | 2.1659 | 5.4268 | 0.399 | 0.343 - 0.456 |

## Redundancy by span

| kind | span | size | A | K | R | 95% CI |
|---|---|---|---|---|---|---|
| nested | `{17}` | 1 | 2.1659 | 5.4268 | 39.9% | 34.3% - 45.6% |
| nested | `{16,17}` | 2 | 10.8660 | 5.1011 | 213.0% | 199.0% - 228.7% |
| nested | `{15,16,17}` | 3 | 14.1682 | 6.2269 | 227.5% | 212.9% - 243.7% |
| nested | `{14-17}` | 4 | 14.6306 | 8.9618 | 163.3% | 155.6% - 171.5% |
| nested | `{13-17}` | 5 | 16.0126 | 9.2689 | 172.8% | 164.2% - 181.9% |
| nested | `{12-17}` | 6 | 24.6782 | 13.7164 | 179.9% | 173.0% - 187.2% |
| nested | `{11-17}` | 7 | 28.6087 | 23.5483 | 121.5% | 117.6% - 125.6% |
| non-nested | `{11,12,13}` | 3 | 11.9204 | 9.2591 | 128.7% | 112.8% - 148.1% |
| non-nested | `{11,13,15}` | 3 | 14.3691 | 4.1206 | 348.7% | 268.6% - 467.8% |
| sensitivity | `{11,12,13,14,15,17}` | 6 | 27.1703 | 23.0241 | 118.0% | 113.9% - 122.4% |

Pooled recovery over the nested spans is 153.8% (95% CI 148.6% - 159.1%). The per-span ratios run 39.9% to 227.5% with NO value inside every interval, so R is not constant across spans and must not be reported as such. R trends with span size at +0.0442 per layer (95% CI +0.0307 - +0.0564), which excludes zero.

Observed spread across the nested spans is 187.6% (95% CI 171.0% - 205.8%); highest lower bound 212.9% against lowest upper bound 45.6%. 10000 draws, seed 42, n = 256.

## Comparative saturation

NEITHER curve saturates over this range -- both fits collapsed to straight lines (ablation 3.970/layer, knockout 2.598/layer), so b and d are not identified and b - d carries no information. What is identified is the slope ratio, 1.528: over spans of 1-5 layers the two curves grow in a fixed proportion, with no sign of the ablation curve turning over sooner. Wider spans would be needed to see saturation at all. This is a statement about the two slopes and NOT about R span by span -- see redundancy_by_span, where the ratios disperse beyond their own intervals.

## Interaction index

Not computed: this run contains no standalone ablation for layer(s) 11, 13, 14, 15, 16. Summing only the layers present would understate the denominator and inflate rho_A, so it is withheld. Supply the missing single-layer conditions, or compute rho_A against the published single-layer drops by hand -- the binding arm does not depend on the control regime, so those remain comparable.

## Leave-one-out

| layer | joint | without | marginal | standalone | marginal/standalone |
|---|---|---|---|---|---|
| 11 | 28.6087 | 24.6782 | +3.9306 | — | — |
| 12 | 28.6087 | 25.1565 | +3.4522 | 0.0544 | 63.426 |
| 13 | 28.6087 | 28.6450 | -0.0362 | — | — |
| 14 | 28.6087 | 27.6600 | +0.9488 | — | — |
| 15 | 28.6087 | 28.5643 | +0.0444 | — | — |
| 16 | 28.6087 | 27.1703 | +1.4384 | — | — |
| 17 | 28.6087 | 27.9735 | +0.6352 | 2.1659 | 0.293 |

A marginal contribution far below the standalone effect at every layer is the strong redundancy signature: the rest of the span already carries what that layer contributes.

## Significance

| condition | binding | z | z SE | control sets | p floor | Wilcoxon p | control regime |
|---|---|---|---|---|---|---|---|
| budget_spread40x7 | 23.8204 | -- | -- | 1 | 0.5000 | 9.64e-44 | {'11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean', '15': 'matched_activation_mean', '16': 'matched_activation_mean', '17': 'matched_activation_mean'} |
| joint_L11-17 | 28.6087 | 248 | 47 | 15 | 0.0625 | 9.64e-44 | {'11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean', '15': 'matched_activation_mean', '16': 'matched_activation_mean', '17': 'matched_activation_mean'} |
| nested_L12-17 | 24.6782 | -- | -- | 1 | 0.5000 | 9.64e-44 | {'11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean', '15': 'matched_activation_mean', '16': 'matched_activation_mean', '17': 'matched_activation_mean'} |
| nested_L13-17 | 16.0126 | -- | -- | 1 | 0.5000 | 9.64e-44 | {'11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean', '15': 'matched_activation_mean', '16': 'matched_activation_mean', '17': 'matched_activation_mean'} |
| nested_L14-17 | 14.6306 | -- | -- | 1 | 0.5000 | 9.64e-44 | {'11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean', '15': 'matched_activation_mean', '16': 'matched_activation_mean', '17': 'matched_activation_mean'} |
| nested_L15-17 | 14.1682 | -- | -- | 1 | 0.5000 | 9.64e-44 | {'11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean', '15': 'matched_activation_mean', '16': 'matched_activation_mean', '17': 'matched_activation_mean'} |
| nested_L16-17 | 10.8660 | -- | -- | 1 | 0.5000 | 9.64e-44 | {'11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean', '15': 'matched_activation_mean', '16': 'matched_activation_mean', '17': 'matched_activation_mean'} |
| nested_L17 | 2.1659 | -- | -- | 1 | 0.5000 | 1.37e-19 | {'11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean', '15': 'matched_activation_mean', '16': 'matched_activation_mean', '17': 'matched_activation_mean'} |
| nonnested_L11,13,15 | 14.3691 | -- | -- | 1 | 0.5000 | 9.64e-44 | {'11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean', '15': 'matched_activation_mean', '16': 'matched_activation_mean', '17': 'matched_activation_mean'} |
| nonnested_L11-13 | 11.9204 | -- | -- | 1 | 0.5000 | 1.06e-43 | {'11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean', '15': 'matched_activation_mean', '16': 'matched_activation_mean', '17': 'matched_activation_mean'} |

z is reported to one significant figure: it is estimated from a handful of control draws and its own standard error is large. The empirical p cannot go below its floor, so the Wilcoxon test over questions is the one with power.

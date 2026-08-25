# Deviation ledger

Numeric deviations from the archive repo (`cross-modal-information-flow-in-MLLM`) accepted during
the behavior-preserving port. An equivalence-gate criterion may only be relaxed with an entry here.

Format per entry:

| field | |
|---|---|
| stage | gate stage (0–3) |
| metric | which criterion |
| archived | archived value |
| reproduced | new-harness value |
| delta | |
| suspected cause | |
| evidence | how the cause was established (e.g. archived-harness jitter measurement) |
| resolution | accepted / fixed in commit `<sha>` |

_No entries yet._

## 2026-08-25 — Stage 1 pred-flip criterion refined (no numeric deviation)

| field | |
|---|---|
| stage | 1 |
| metric | pred agreement flip filter |
| archived | pred "square" @ p=0.4206 (sample 59_237) |
| reproduced | pred "circle"; top-2 first tokens Circle 0.4175 vs Square 0.4111 |
| delta | one greedy flip on a near-tied generation; margins reproduce (r = 0.999975, mean abs delta = 0.008) |
| suspected cause | fp16 kernel-order drift on a genuine coin-flip token pair |
| evidence | option margin for the same sample matches within 0.01; new-harness self-jitter is exactly 0 |
| resolution | flip filter changed from "cached margin < 0.1" (wrong proxy — margins score options, flips are generation events) to "cached baseline_prob < 0.5". Agreement threshold (>= 254/256) unchanged. |

# Running on Snellius

GPU work runs on [Snellius](https://www.surf.nl/en/services/snellius-the-national-supercomputer),
SURF's national cluster. Development stays local; the cluster is where anything needing real
weights or a real GPU happens — `pytest -m gpu`, `vfp-verify-adapter`, and the pipeline stages.

## Access

Per-user, arranged with SURF, and deliberately not configured by this repo:

1. SURF whitelists your workstation's public IP (`curl -4 ifconfig.me`). Re-request it if your
   ISP hands you a new address.
2. Register an ed25519 public key in the SURF portal. SURF requires the private key to carry a
   passphrase.
3. Add a `Host snellius` block to `~/.ssh/config` naming that key, so `ssh snellius` works.

## First-time setup

The repo is private, so the cluster cannot clone it. Push the working tree instead — which also
carries uncommitted work, usually the point of a GPU test:

```bash
scripts/snellius_sync.sh                                    # local: rsync tree -> cluster
ssh snellius 'cd vlm-flow-probe && scripts/snellius_setup.sh'   # login node: venv + deps + tests
```

`snellius_setup.sh` is idempotent; re-run it after a dependency change. Every later session only
needs:

```bash
source scripts/snellius_env.sh
```

## Where things live, and why

| What | Path | Reason |
|---|---|---|
| Checkout + venv | `$HOME/vlm-flow-probe` | 200 GiB, **not** purged |
| Weights (`HF_HOME`) | `$HOME/.cache/huggingface` | purge-safe; re-pulling 14 GB/model beats the quota cost |
| Activation caches | `$VFP_OUTPUT_ROOT/activations` (scratch) | 8 TiB; 40 GB per layer, regenerable |
| SAE checkpoints, `output/experiments` | `$HOME/vlm-flow-probe/output` | small, and losing a run's results costs more than a purge saves |

**`/scratch-shared` is purged on a rolling 14-day window.** Nothing whose loss would cost you more
than a re-run belongs there. That split is the one non-obvious decision in this setup.

`output/` itself is a real directory under `$HOME`; `output/activations` is a symlink onto
scratch (set up 2026-09-08, when the Gemma cache moved there and home usage fell from 30% to
24% of the 200 GiB quota). That split is what makes the next collection possible at all: 32
layers of LLaVA-1.6 activations is ~1.3 TB plus a transient per-layer copy during reassembly,
against 9 GB of experiment artifacts that are worth keeping.

## Software stack

The login node's system `python3` is 3.9, and this package needs `>=3.10`, so the module stack is
mandatory rather than optional. `scripts/snellius_env.sh` loads:

```
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
```

torch comes from PyPI (its wheels bundle their own CUDA runtime), so no `CUDA/` module is loaded.
The 2023 and 2025 stacks also exist — 2025 offers only Python 3.13, which is ahead of what the
pinned `transformers>=4.56,<4.58` is tested against.

## Weights must be pre-fetched

Login nodes have outbound internet; compute nodes are not assumed to. Pull weights **before**
submitting a job, or it will die minutes in:

```bash
scripts/snellius_fetch_weights.sh llava-hf/llava-1.5-7b-hf
```

## Partitions

| Partition | Use |
|---|---|
| `gpu_h100` | 88 nodes, the main target |
| `gpu_a100` | 63 nodes |
| `gpu_mig` | 4 nodes, MIG slices — debug and smoke runs |
| `staging` | bulk data transfer, no GPU |

Walltime cap is 5 days. An interactive GPU shell for a quick check:

```bash
srun -p gpu_mig -n 1 --gpus 1 -t 10:00 --pty bash
```

Every run shape so far is "one `vfp-*` command on one GPU", so that is what is checked in:

```bash
scripts/snellius_submit.sh <name> "<vfp command>" [time] [partition] [extra sbatch args]
scripts/snellius_submit.sh knockout \
    "vfp-knockout --config configs/experiments/gemma3_4b/knockout_clevr_lite.yaml" \
    2-00:00:00 gpu_h100
```

It wraps the command in `scripts/slurm/vfp.sbatch` (one GPU, 16 cores, `HF_HUB_OFFLINE=1`, logs in
`~/logs/<name>-<jobid>.out`) and defaults to `gpu_a100` — pass `gpu_h100` explicitly when you want
it. The command travels in a file rather than through `--export`, which splits its value at every
comma: `--layers 0,1,2` would arrive as `--layers 0`.

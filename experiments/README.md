# Grid experiments

Scripts for the alpha and temperature grid in Section 11 of `distillation.ipynb`. The notebook
itself only *reads* the results, so it never has to hold a multi-hour training session.

## Files

| File | Purpose |
|---|---|
| `kd_core.py` | Data pipeline, models, loss, and training loop, with `alpha` and `temperature` as explicit arguments. Same logic as notebook sections 3-7. |
| `run_one.py` | Train one `(alpha, temperature)` configuration and write a JSON record. |
| `launch_grid.py` | Run the whole grid across both GPUs, one process per card. |
| `analyze_grid.py` | Build the summary table and figures from the JSON records. |
| `verify_core.py` | Check `kd_core` reproduces the notebook's loss exactly. |

## Reproducing the grid

```bash
# From the repo root, on a machine with two GPUs.
python experiments/verify_core.py        # confirm the refactor matches the notebook
python experiments/launch_grid.py --dry-run
python experiments/launch_grid.py        # ~4.4 h on two W7900s
python experiments/analyze_grid.py       # table + figures
```

Run the launcher inside `tmux` so it survives an SSH disconnect. It is **resumable**: `run_one.py`
skips any configuration whose result JSON already exists, so re-running after an interruption picks
up where it left off.

Single configuration:

```bash
HIP_VISIBLE_DEVICES=0 python experiments/run_one.py --alpha 0.25 --temperature 1.0
HIP_VISIBLE_DEVICES=1 python experiments/run_one.py --alpha 0.5 --temperature 0.5 --grad-match
```

## Conventions

`alpha` weights the **corpus** (cross-entropy) term:

```text
loss = alpha * CE + (1 - alpha) * kd_scale * T^2 * KL(teacher || student)
```

so `alpha=1` is corpus-only and `alpha=0` is teacher-only. Both endpoints are special-cased to be
exactly equivalent to the notebook's `corpus_only` and `teacher_only` methods; `alpha=1` also skips
the teacher forward pass entirely, which is why that run finishes in about a third of the time.

`verify_core.py` checks this equivalence numerically — the refactored loss matches the notebook's
to `0.00e+00` at `alpha` in {0, 0.5, 1} and `T` in {0.5, 1, 2, 4}.

## Why one process per GPU instead of DataParallel

Measured on this workload (batch 16, block 1024, 145 steps/epoch):

| Setup | sec/step | Peak VRAM |
|---|---|---|
| Single GPU | 0.89-0.93 | 32.1 GB |
| `DataParallel` over both cards | 0.83 (+8%) | 17.1 GB/GPU |
| Two independent processes | 0.930 / 0.943 (~2x throughput) | 32.1 GB each |

The 30M student is too small to amortize DataParallel's per-step replication cost, and the grid is
a set of *independent* runs, so pinning one run per GPU is both faster and keeps every run identical
to the original single-GPU recipe.

`kd_core.KDModule` still provides a DataParallel wrapper for single-run dual-GPU training. It
computes the loss **inside** the replicated module and returns scalars; wrapping only the student
instead would make DataParallel gather `(16, 1024, 50257)` logit tensors (~3.3 GB each) onto
`cuda:0`, which is what accounts for the 32 GB to 17 GB difference above.

## AMD/ROCm notes

- Set `HIP_VISIBLE_DEVICES` to mask the Ryzen integrated GPU, which ROCm enumerates as a third
  device that segfaults if compute lands on it.
- **Do not also set `ROCR_VISIBLE_DEVICES`.** The two masks compose: `ROCR_VISIBLE_DEVICES=1`
  narrows visibility to one device, after which `HIP_VISIBLE_DEVICES=1` selects index 1 of a
  one-element set and every GPU disappears. Training then silently falls back to CPU, roughly 100x
  slower. `run_one.py` hard-errors on a CPU fallback for this reason; override with `--allow-cpu`.

## Gradient-matched control

The Hinton `T^2` factor is derived in the large-`T` limit, so it is not guaranteed to hold below
`T = 1`. `--grad-match` rescales the KD term each epoch by

```text
c = ||grad KD(T=1)|| / ||grad KD(T)||
```

measured on a fixed probe batch, so the distillation gradient has the same magnitude as it would at
`T = 1`. Comparing a matched run against a standard one separates *sharpening the target* from
*taking larger steps*.

In our grid this produced a **null result**, which is the useful outcome: the two `T = 0.5` runs
differ by only 2.0 test perplexity against a 43-point penalty for using `T = 0.5` at all. The
required correction also averaged `c = 1.18` (above 1 in 78% of epochs) rather than the `c ~ 0.26`
predicted by probing a CE-trained student, so the `T^2` factor behaves better during actual KD
training than the asymptotic argument suggests. Every `c` is logged in the run JSON.

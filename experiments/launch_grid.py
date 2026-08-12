#!/usr/bin/env python
"""Run the alpha/temperature grid across both W7900 GPUs.

Two worker slots are pinned to HIP_VISIBLE_DEVICES=0 and =1 and pull from a shared
queue, so whichever card finishes first takes the next job. Measured on this box, two
independent processes show essentially zero contention (0.930 s/step solo vs
0.930/0.943 concurrent), which beats DataParallel (+8%) for a grid of independent runs.

Resumable: run_one.py skips any config whose result JSON already exists, so re-running
this script after an interruption picks up where it left off.

Usage (from the repo root, inside a tmux session):
    python experiments/launch_grid.py
    python experiments/launch_grid.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_ONE = Path(__file__).resolve().parent / "run_one.py"

# (alpha, temperature, grad_match, note)
#
# alpha weights the corpus/CE term, so alpha=1 is corpus-only and alpha=0 teacher-only.
# The (alpha=0.5, T=1) point is shared between the two axes and only trained once.
GRID = [
    # --- alpha axis at T=1 ---
    (0.00, 1.0, False, "teacher only (= published 'Teacher only')"),
    (0.25, 1.0, False, "new"),
    (0.50, 1.0, False, "combined (= published 'Teacher + corpus'); shared with T axis"),
    (0.75, 1.0, False, "new"),
    (1.00, 1.0, False, "corpus only (= published 'Corpus only'); skips teacher forward"),
    # --- temperature axis at alpha=0.5 ---
    (0.50, 0.5, False, "new: T < 1 (sharpened teacher)"),
    (0.50, 2.0, False, "new: T > 1 (softened teacher)"),
    # --- control for the T^2 confound ---
    (0.50, 0.5, True, "new: T=0.5 with KD gradient matched to T=1"),
]

# Longest-first: the teacher-guided runs train for more epochs before early stopping
# (published best epochs were 39 / 22 / 13), so starting them first keeps both GPUs
# busy instead of leaving one idle at the tail.
ORDER = [0, 5, 7, 1, 2, 6, 3, 4]


def tag_for(alpha: float, temperature: float, grad_match: bool) -> str:
    tag = f"a{alpha:.2f}_T{temperature:g}"
    if grad_match:
        tag += "_gm"
    return tag


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", type=str, default="0,1", help="Comma-separated GPU ids.")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    p.add_argument("--force", action="store_true", help="Re-run configs that already have results.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--limit-batches", type=int, default=None)
    p.add_argument("--log-dir", type=str, default=str(REPO_ROOT / "results" / "logs"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir = REPO_ROOT / "results" / "runs"

    jobs = [GRID[i] for i in ORDER]

    print(f"Grid: {len(jobs)} configs across {len(gpus)} GPU(s)\n")
    pending = []
    for alpha, temperature, grad_match, note in jobs:
        tag = tag_for(alpha, temperature, grad_match)
        done = (results_dir / f"{tag}.json").exists() and not args.force
        status = "DONE (skip)" if done else "queued"
        print(f"  {tag:<18} alpha={alpha:<5.2f} T={temperature:<4g} "
              f"gm={str(grad_match):<5} {status:<11} {note}")
        if not done:
            pending.append((alpha, temperature, grad_match, tag))

    print(f"\n{len(pending)} to run, {len(jobs) - len(pending)} already complete.")
    if args.dry_run:
        return 0
    if not pending:
        print("Nothing to do.")
        return 0

    work: queue.Queue = queue.Queue()
    for job in pending:
        work.put(job)

    failures: list[str] = []
    lock = threading.Lock()
    started = time.time()

    def worker(gpu: str, slot: int) -> None:
        # Stagger startup so the two processes don't hit the HF dataset cache at the
        # same instant on a cold cache.
        time.sleep(slot * 20)
        while True:
            try:
                alpha, temperature, grad_match, tag = work.get_nowait()
            except queue.Empty:
                return

            cmd = [sys.executable, str(RUN_ONE),
                   "--alpha", str(alpha), "--temperature", str(temperature), "--tag", tag]
            if grad_match:
                cmd.append("--grad-match")
            if args.force:
                cmd.append("--force")
            if args.epochs is not None:
                cmd += ["--epochs", str(args.epochs)]
            if args.limit_batches is not None:
                cmd += ["--limit-batches", str(args.limit_batches)]

            env = dict(os.environ)
            # Set HIP_VISIBLE_DEVICES *only*. The two masks compose: setting
            # ROCR_VISIBLE_DEVICES=1 first narrows visibility to a single device, after
            # which HIP_VISIBLE_DEVICES=1 asks for index 1 of a 1-element set and every
            # GPU disappears (device_count 0 -> silent CPU fallback).
            env["HIP_VISIBLE_DEVICES"] = gpu
            env.pop("ROCR_VISIBLE_DEVICES", None)

            log_path = log_dir / f"{tag}.log"
            elapsed = (time.time() - started) / 60
            with lock:
                print(f"[t+{elapsed:6.1f}m][gpu{gpu}] START {tag} -> {log_path}", flush=True)

            t0 = time.time()
            with open(log_path, "w") as fh:
                proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=fh,
                                      stderr=subprocess.STDOUT)
            mins = (time.time() - t0) / 60

            with lock:
                if proc.returncode == 0:
                    print(f"[t+{(time.time()-started)/60:6.1f}m][gpu{gpu}] DONE  {tag} "
                          f"({mins:.1f} min)", flush=True)
                else:
                    failures.append(tag)
                    print(f"[t+{(time.time()-started)/60:6.1f}m][gpu{gpu}] FAIL  {tag} "
                          f"(exit {proc.returncode}, see {log_path})", flush=True)
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu, i), daemon=True)
               for i, gpu in enumerate(gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = (time.time() - started) / 60
    print(f"\nGrid finished in {total:.1f} min ({total/60:.2f} h).")
    if failures:
        print(f"FAILURES: {', '.join(failures)}")
        return 1
    print("All runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Train one student for a single (alpha, temperature) configuration.

Runs on whichever GPU ``HIP_VISIBLE_DEVICES`` exposes, so the grid launcher can pin one
process per card. Idempotent: if the result JSON already exists the run is skipped,
which makes the whole grid resumable.

Examples
--------
    HIP_VISIBLE_DEVICES=0 python run_one.py --alpha 0.25 --temperature 1.0
    HIP_VISIBLE_DEVICES=1 python run_one.py --alpha 0.5 --temperature 0.5 --grad-match
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kd_core import (  # noqa: E402
    Config,
    build_dataloaders,
    build_seeded_student,
    build_teacher,
    build_tokenizer,
    count_parameters,
    env_info,
    evaluate_perplexity,
    get_device,
    method_for_alpha,
    train_model,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def default_tag(alpha: float, temperature: float, grad_match: bool) -> str:
    """Stable, filesystem-safe identifier, e.g. a0.50_T0.5_gm."""
    t = f"{temperature:g}"
    tag = f"a{alpha:.2f}_T{t}"
    if grad_match:
        tag += "_gm"
    return tag


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alpha", type=float, required=True,
                   help="Weight on the corpus (CE) term. 1=corpus only, 0=teacher only.")
    p.add_argument("--temperature", type=float, required=True,
                   help="Distillation temperature.")
    p.add_argument("--grad-match", action="store_true",
                   help="Rescale the KD term each epoch so its gradient norm matches T=1.")
    p.add_argument("--tag", type=str, default=None, help="Override the run identifier.")
    p.add_argument("--epochs", type=int, default=None, help="Override max epochs.")
    p.add_argument("--patience", type=int, default=None, help="Override early-stopping patience.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--limit-batches", type=int, default=None,
                   help="Cap training batches per epoch (smoke tests only).")
    p.add_argument("--results-dir", type=str, default=str(REPO_ROOT / "results" / "runs"))
    p.add_argument("--checkpoint-dir", type=str, default=str(REPO_ROOT / "checkpoints"))
    p.add_argument("--no-checkpoint", action="store_true", help="Skip saving model weights.")
    p.add_argument("--force", action="store_true", help="Re-run even if the result JSON exists.")
    p.add_argument("--allow-cpu", action="store_true",
                   help="Permit running without a GPU (otherwise a CPU fallback is a hard error).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    tag = args.tag or default_tag(args.alpha, args.temperature, args.grad_match)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{tag}.json"

    if result_path.exists() and not args.force:
        print(f"[{tag}] result already exists at {result_path}; skipping.", flush=True)
        return 0

    cfg = Config(alpha=args.alpha, temperature=args.temperature)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.patience is not None:
        cfg.early_stopping_patience = args.patience
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.limit_batches = args.limit_batches

    device = get_device()
    info = env_info()
    method = method_for_alpha(cfg.alpha)

    # A CPU fallback here is ~100x slower and would silently burn hours, so refuse it.
    # Most likely cause: HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES both set, which
    # compose and can hide every device.
    if device != "cuda" and not args.allow_cpu:
        print(f"[{tag}] FATAL: no GPU visible (device_count=0). "
              f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', 'unset')} "
              f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES', 'unset')}. "
              f"Refusing to train on CPU; pass --allow-cpu to override.", flush=True)
        return 2

    def log(msg: str) -> None:
        print(f"[{tag}] {msg}", flush=True)

    log(f"alpha={cfg.alpha} temperature={cfg.temperature} method={method} "
        f"grad_match={args.grad_match}")
    log(f"device={device} visible={os.environ.get('HIP_VISIBLE_DEVICES', 'unset')} env={info}")

    wall_start = time.time()

    tokenizer = build_tokenizer(cfg)
    train_loader, val_loader, test_loader = build_dataloaders(cfg, tokenizer)
    log(f"batches: train={len(train_loader)} val={len(val_loader)} test={len(test_loader)}")

    teacher = build_teacher(cfg, device)
    student = build_seeded_student(cfg, teacher, tokenizer, device)
    log(f"student params: {count_parameters(student):,}")

    history = train_model(
        student, teacher, train_loader, val_loader, cfg, device,
        name=tag, grad_match=args.grad_match, log=log,
    )

    test_ppl = evaluate_perplexity(student, test_loader, device)
    val_ppl = evaluate_perplexity(student, val_loader, device)
    log(f"final: val ppl {val_ppl:.2f}, test ppl {test_ppl:.2f}")

    checkpoint_path = None
    if not args.no_checkpoint:
        checkpoint_path = Path(args.checkpoint_dir) / tag
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        log(f"saved checkpoint -> {checkpoint_path}")

    record = {
        "tag": tag,
        "alpha": cfg.alpha,
        "temperature": cfg.temperature,
        "grad_match": args.grad_match,
        "method": method,
        "student_params": count_parameters(student),
        "val_ppl": val_ppl,
        "test_ppl": test_ppl,
        "peak_vram_gb": (
            torch.cuda.max_memory_allocated(0) / 1e9 if device == "cuda" else None
        ),
        "wall_seconds": time.time() - wall_start,
        "config": cfg.to_dict(),
        "env": info,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        **history,
    }

    # Write atomically so a reader never sees a half-written file.
    tmp_path = result_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(record, indent=2))
    tmp_path.replace(result_path)
    log(f"wrote {result_path} (best epoch {history['best_epoch']}, "
        f"best val ppl {history['best_val_ppl']:.2f}, test ppl {test_ppl:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

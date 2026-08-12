#!/usr/bin/env python
"""Check that the refactor did not change the science.

Compares kd_core's alpha-parameterized loss against the original notebook
implementation (inlined below verbatim) on a fixed batch:

  alpha = 1.0 must equal the notebook's method="corpus_only"
  alpha = 0.0 must equal the notebook's method="teacher_only"
  alpha = 0.5 must equal the notebook's method="combined"

Also reports KD gradient norms across temperatures, which is the measurement behind the
gradient-matched control run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kd_core import (  # noqa: E402
    Config, build_teacher, build_tokenizer, build_seeded_student,
    compute_training_loss, calibrate_kd_scale, get_device, set_seed,
)

# --- Original notebook implementation, copied verbatim from distillation.ipynb cell 12 ---
NOTEBOOK_CFG = {"temperature": 1.0, "alpha": 0.5}


def notebook_compute_training_loss(student_logits, teacher_logits, input_ids, method):
    vocab_size = student_logits.size(-1)
    student_next_token_scores = student_logits[:, :-1, :].contiguous()
    true_next_tokens = input_ids[:, 1:].contiguous()

    corpus_loss = F.cross_entropy(
        student_next_token_scores.view(-1, vocab_size),
        true_next_tokens.view(-1),
    )

    teacher_loss = torch.zeros((), device=student_logits.device)
    if teacher_logits is not None:
        teacher_next_token_scores = teacher_logits[:, :-1, :].contiguous()
        student_log_probs = F.log_softmax(student_next_token_scores / NOTEBOOK_CFG["temperature"], dim=-1)
        teacher_probs = F.softmax(teacher_next_token_scores / NOTEBOOK_CFG["temperature"], dim=-1)
        teacher_loss = F.kl_div(
            student_log_probs.view(-1, vocab_size),
            teacher_probs.view(-1, vocab_size),
            reduction="batchmean",
        ) * (NOTEBOOK_CFG["temperature"] ** 2)

    if method == "combined":
        total_loss = NOTEBOOK_CFG["alpha"] * corpus_loss + (1 - NOTEBOOK_CFG["alpha"]) * teacher_loss
    elif method == "corpus_only":
        total_loss = corpus_loss
    elif method == "teacher_only":
        total_loss = teacher_loss
    else:
        raise ValueError(method)
    return total_loss, corpus_loss, teacher_loss
# ---------------------------------------------------------------------------------------


def main() -> int:
    cfg = Config()
    device = get_device()
    tokenizer = build_tokenizer(cfg)
    teacher = build_teacher(cfg, device)
    student = build_seeded_student(cfg, teacher, tokenizer, device)

    # Deterministic fake batch: no dataset download needed for the equivalence check.
    set_seed(cfg.seed)
    input_ids = torch.randint(0, 50257, (2, 256), device=device)

    with torch.no_grad():
        student_logits = student(input_ids=input_ids).logits
        teacher_logits = teacher(input_ids=input_ids).logits

    cases = [
        (1.0, "corpus_only", None),
        (0.0, "teacher_only", teacher_logits),
        (0.5, "combined", teacher_logits),
    ]

    print(f"{'alpha':>6} {'method':>13} {'new total':>12} {'notebook':>12} {'abs diff':>11}  ok")
    all_ok = True
    for alpha, method, t_logits in cases:
        NOTEBOOK_CFG["alpha"] = alpha
        NOTEBOOK_CFG["temperature"] = 1.0

        new_total, _, _ = compute_training_loss(
            student_logits, t_logits, input_ids, alpha=alpha, temperature=1.0,
        )
        ref_total, _, _ = notebook_compute_training_loss(
            student_logits, t_logits, input_ids, method,
        )
        diff = abs(new_total.item() - ref_total.item())
        ok = diff < 1e-5
        all_ok &= ok
        print(f"{alpha:>6.2f} {method:>13} {new_total.item():>12.6f} "
              f"{ref_total.item():>12.6f} {diff:>11.2e}  {'PASS' if ok else 'FAIL'}")

    # Temperature equivalence for the combined objective.
    print("\nTemperature equivalence (alpha=0.5):")
    for T in [0.5, 1.0, 2.0, 4.0]:
        NOTEBOOK_CFG["alpha"] = 0.5
        NOTEBOOK_CFG["temperature"] = T
        new_total, _, _ = compute_training_loss(
            student_logits, teacher_logits, input_ids, alpha=0.5, temperature=T,
        )
        ref_total, _, _ = notebook_compute_training_loss(
            student_logits, teacher_logits, input_ids, "combined",
        )
        diff = abs(new_total.item() - ref_total.item())
        ok = diff < 1e-5
        all_ok &= ok
        print(f"  T={T:<4} new={new_total.item():>10.6f}  notebook={ref_total.item():>10.6f}  "
              f"diff={diff:.2e}  {'PASS' if ok else 'FAIL'}")

    # kd_scale=1.0 must be a no-op.
    a, _, _ = compute_training_loss(student_logits, teacher_logits, input_ids, 0.5, 2.0, kd_scale=1.0)
    b, _, _ = compute_training_loss(student_logits, teacher_logits, input_ids, 0.5, 2.0)
    ok = abs(a.item() - b.item()) < 1e-9
    all_ok &= ok
    print(f"\nkd_scale=1.0 is a no-op: {'PASS' if ok else 'FAIL'}")

    # Gradient-matching calibration: the measurement motivating the control run.
    print("\nKD gradient norms (fresh student, probe batch):")
    probe = input_ids[:2]
    for T in [0.5, 1.0, 2.0]:
        c = calibrate_kd_scale(student, teacher, probe, T)
        print(f"  T={T:<4} kd_scale to match T=1: {c:.4f}")

    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

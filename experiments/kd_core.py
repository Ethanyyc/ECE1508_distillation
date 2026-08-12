"""Core distillation logic, parameterized by alpha and temperature.

This module is the single source of truth for the experiment. It contains the same
data pipeline, model construction, loss, and training loop as ``distillation.ipynb``,
with two changes:

1. ``alpha`` and ``temperature`` are explicit arguments instead of globals read from
   ``CFG``. This lets us sweep them without mutating shared state.
2. A ``kd_scale`` hook multiplies the distillation term, used by the gradient-matched
   control run (see ``calibrate_kd_scale``).

Convention: ``alpha`` weights the *corpus* (cross-entropy) term.

    loss = alpha * CE + (1 - alpha) * kd_scale * T^2 * KL(teacher || student)

so ``alpha=1`` is corpus-only and ``alpha=0`` is teacher-only. Those two endpoints are
handled specially so they are exactly equivalent to the notebook's ``corpus_only`` and
``teacher_only`` methods (and ``alpha=1`` skips the teacher forward pass entirely,
roughly halving its runtime).
"""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

@dataclass
class Config:
    """Experiment settings. Defaults match CFG in distillation.ipynb."""

    # Data
    dataset_name: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    block_size: int = 1024
    batch_size: int = 16

    # Teacher and student model size
    teacher_name: str = "gpt2"
    student_n_layer: int = 6
    student_n_embd: int = 384
    student_n_head: int = 6

    # Training
    seed: int = 42
    lr: float = 5e-4
    epochs: int = 50
    early_stopping_patience: int = 2

    # Distillation
    temperature: float = 1.0
    alpha: float = 0.5

    # Debug / smoke-test knobs
    limit_batches: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    """Make results reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def env_info() -> dict:
    """Record the software/hardware stack so results stay auditable."""
    import transformers
    import datasets as _datasets

    info = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": _datasets.__version__,
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------

def build_tokenizer(cfg: Config) -> GPT2TokenizerFast:
    return GPT2TokenizerFast.from_pretrained(cfg.teacher_name)


def build_dataloaders(cfg: Config, tokenizer: GPT2TokenizerFast):
    """Tokenize WikiText-2, concatenate, and chop into fixed-size blocks.

    Identical in behaviour to notebook section 3.
    """
    raw_datasets = load_dataset(cfg.dataset_name, cfg.dataset_config)

    def tokenize_batch(batch):
        return tokenizer(batch["text"])

    tokenized = raw_datasets.map(tokenize_batch, batched=True, remove_columns=["text"])

    block_size = cfg.block_size

    def group_texts(batch):
        all_input_ids = []
        for token_list in batch["input_ids"]:
            all_input_ids.extend(token_list)

        total_length = (len(all_input_ids) // block_size) * block_size
        blocks = [all_input_ids[i:i + block_size] for i in range(0, total_length, block_size)]
        return {"input_ids": blocks}

    lm_datasets = tokenized.map(
        group_texts, batched=True, remove_columns=tokenized["train"].column_names
    )
    lm_datasets.set_format(type="torch", columns=["input_ids"])

    train_loader = DataLoader(lm_datasets["train"], batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(lm_datasets["validation"], batch_size=cfg.batch_size)
    test_loader = DataLoader(lm_datasets["test"], batch_size=cfg.batch_size)
    return train_loader, val_loader, test_loader


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_teacher(cfg: Config, device: str) -> GPT2LMHeadModel:
    teacher = GPT2LMHeadModel.from_pretrained(cfg.teacher_name).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def build_student(cfg: Config, teacher: GPT2LMHeadModel, tokenizer, device: str) -> GPT2LMHeadModel:
    student_config = GPT2Config(
        vocab_size=teacher.config.vocab_size,
        n_positions=teacher.config.n_positions,
        n_ctx=teacher.config.n_positions,
        n_embd=cfg.student_n_embd,
        n_layer=cfg.student_n_layer,
        n_head=cfg.student_n_head,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return GPT2LMHeadModel(student_config).to(device)


def build_seeded_student(cfg: Config, teacher, tokenizer, device: str) -> GPT2LMHeadModel:
    """Re-seed before construction so every run starts from identical weights."""
    set_seed(cfg.seed)
    return build_student(cfg, teacher, tokenizer, device)


# --------------------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------------------

def method_for_alpha(alpha: float) -> str:
    """Map alpha to the notebook's three named methods.

    alpha == 1 -> corpus_only (teacher forward can be skipped)
    alpha == 0 -> teacher_only
    otherwise  -> combined
    """
    if alpha >= 1.0:
        return "corpus_only"
    if alpha <= 0.0:
        return "teacher_only"
    return "combined"


def needs_teacher(alpha: float) -> bool:
    return method_for_alpha(alpha) != "corpus_only"


def compute_training_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    input_ids: torch.Tensor,
    alpha: float,
    temperature: float,
    kd_scale: float = 1.0,
):
    """Loss for one training batch.

    Returns ``(total_loss, corpus_loss, teacher_loss)``.

    GPT predicts the next token, so position t is compared against label t+1; both the
    logits and the labels are shifted by one accordingly.
    """
    vocab_size = student_logits.size(-1)

    student_next_token_scores = student_logits[:, :-1, :].contiguous()
    true_next_tokens = input_ids[:, 1:].contiguous()

    method = method_for_alpha(alpha)

    # Corpus loss: student vs. the real next token. Skipped for teacher-only, where it
    # would contribute nothing but cost a full softmax over the vocabulary.
    if method == "teacher_only":
        corpus_loss = torch.zeros((), device=student_logits.device)
    else:
        corpus_loss = F.cross_entropy(
            student_next_token_scores.view(-1, vocab_size),
            true_next_tokens.view(-1),
        )

    # Teacher loss: student vs. the teacher's full distribution.
    teacher_loss = torch.zeros((), device=student_logits.device)
    if teacher_logits is not None and method != "corpus_only":
        teacher_next_token_scores = teacher_logits[:, :-1, :].contiguous()

        student_log_probs = F.log_softmax(student_next_token_scores / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_next_token_scores / temperature, dim=-1)

        teacher_loss = F.kl_div(
            student_log_probs.view(-1, vocab_size),
            teacher_probs.view(-1, vocab_size),
            reduction="batchmean",
        ) * (temperature ** 2) * kd_scale

    if method == "combined":
        total_loss = alpha * corpus_loss + (1 - alpha) * teacher_loss
    elif method == "corpus_only":
        total_loss = corpus_loss
    else:
        total_loss = teacher_loss

    return total_loss, corpus_loss, teacher_loss


class KDModule(nn.Module):
    """Wrap student+teacher so the loss is computed *inside* the replicated module.

    Only needed for ``nn.DataParallel``. Computing the loss outside would make DP gather
    ``(batch, block, 50257)`` float logits onto cuda:0 -- about 3.3 GB per tensor at
    batch 16 -- which dominates memory. Returning scalars instead keeps peak VRAM at
    roughly 17 GB/GPU vs 32 GB single-GPU.

    Not used by the grid runner (independent per-GPU processes are ~2x faster than DP
    for this workload); provided so the notebook can demonstrate dual-GPU training.
    """

    def __init__(self, student, teacher, alpha: float, temperature: float, kd_scale: float = 1.0):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.alpha = alpha
        self.temperature = temperature
        self.kd_scale = kd_scale

    def forward(self, input_ids):
        student_logits = self.student(input_ids=input_ids).logits

        teacher_logits = None
        if needs_teacher(self.alpha):
            with torch.no_grad():
                teacher_logits = self.teacher(input_ids=input_ids).logits

        total, corpus, teacher_l = compute_training_loss(
            student_logits, teacher_logits, input_ids,
            self.alpha, self.temperature, self.kd_scale,
        )
        # DataParallel concatenates outputs along dim 0, so return 1-element tensors.
        return total.unsqueeze(0), corpus.detach().unsqueeze(0), teacher_l.detach().unsqueeze(0)


# --------------------------------------------------------------------------------------
# Gradient matching (control for the T^2 confound)
# --------------------------------------------------------------------------------------

def _kd_grad_norm(student, teacher, input_ids, temperature: float) -> float:
    """Backward the KD term alone at a given T and return the student grad norm."""
    student.zero_grad(set_to_none=True)
    student_logits = student(input_ids=input_ids).logits[:, :-1, :]
    with torch.no_grad():
        teacher_logits = teacher(input_ids=input_ids).logits[:, :-1, :]

    vocab_size = student_logits.size(-1)
    kd = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1).reshape(-1, vocab_size),
        F.softmax(teacher_logits / temperature, dim=-1).reshape(-1, vocab_size),
        reduction="batchmean",
    ) * (temperature ** 2)
    kd.backward()

    total = torch.zeros((), device=student_logits.device)
    for p in student.parameters():
        if p.grad is not None:
            total += (p.grad.detach() ** 2).sum()
    student.zero_grad(set_to_none=True)
    return math.sqrt(total.item())


def calibrate_kd_scale(student, teacher, probe_batch, temperature: float,
                       reference_temperature: float = 1.0) -> float:
    """Scalar making the KD gradient at ``temperature`` match that at T=1.

    The Hinton T^2 factor is derived in the large-T limit. At T < 1 it *over*-corrects:
    measured on a mid-training student, the KD gradient at T=0.5 is ~3.8x the T=1 value,
    so a naive T=0.5 run confounds target sharpening with a larger effective learning
    rate. This returns c = ||grad KD(T_ref)|| / ||grad KD(T)|| so that c * T^2 * KL has
    a gradient of comparable magnitude to the T=1 run.

    Recalibrated once per epoch, since the ratio drifts as the student trains (~0.81 at
    random init, ~3.77 after 50 steps).
    """
    was_training = student.training
    student.train()

    g_ref = _kd_grad_norm(student, teacher, probe_batch, reference_temperature)
    g_cur = _kd_grad_norm(student, teacher, probe_batch, temperature)

    if not was_training:
        student.eval()

    if g_cur <= 0 or not math.isfinite(g_cur) or not math.isfinite(g_ref):
        return 1.0
    return g_ref / g_cur


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate_perplexity(model, loader, device: str) -> float:
    """Token-weighted perplexity = exp(mean cross-entropy on the true next tokens)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        num_tokens = input_ids.size(0) * (input_ids.size(1) - 1)
        total_loss += outputs.loss.item() * num_tokens
        total_tokens += num_tokens

    return math.exp(total_loss / total_tokens)


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------

def train_model(
    model,
    teacher,
    train_loader,
    val_loader,
    cfg: Config,
    device: str,
    name: str = "student",
    grad_match: bool = False,
    log=print,
):
    """Train one student to early stopping, restoring the best-validation weights.

    Mirrors notebook section 7.2. Returns a dict of histories and best-epoch info.
    """
    set_seed(cfg.seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_history: list[float] = []
    validation_history: list[float] = []
    kd_scale_history: list[float] = []

    best_validation_ppl = None
    best_epoch = None
    best_state = None
    epochs_without_improvement = 0
    use_teacher = needs_teacher(cfg.alpha)

    # Fixed probe batch for gradient matching, held constant across epochs so the
    # calibration is comparable epoch to epoch.
    probe_batch = None
    if grad_match:
        first = next(iter(train_loader))
        probe_batch = first["input_ids"][: min(4, cfg.batch_size)].to(device)

    start_time = time.time()

    for epoch in range(cfg.epochs):
        kd_scale = 1.0
        if grad_match:
            kd_scale = calibrate_kd_scale(model, teacher, probe_batch, cfg.temperature)
            log(f"{name} epoch {epoch + 1}: kd_scale = {kd_scale:.4f}")
        kd_scale_history.append(kd_scale)

        model.train()
        total_epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)

            student_logits = model(input_ids=input_ids).logits

            teacher_logits = None
            if use_teacher:
                with torch.no_grad():
                    teacher_logits = teacher(input_ids=input_ids).logits

            loss, _corpus_loss, _teacher_loss = compute_training_loss(
                student_logits, teacher_logits, input_ids,
                cfg.alpha, cfg.temperature, kd_scale,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            num_batches += 1

            if cfg.limit_batches is not None and num_batches >= cfg.limit_batches:
                break

        average_epoch_loss = total_epoch_loss / max(num_batches, 1)
        loss_history.append(average_epoch_loss)

        validation_ppl = evaluate_perplexity(model, val_loader, device)
        validation_history.append(validation_ppl)
        log(
            f"{name} epoch {epoch + 1}/{cfg.epochs}: "
            f"train loss {average_epoch_loss:.4f}, val ppl {validation_ppl:.2f}"
        )

        if best_validation_ppl is None or validation_ppl < best_validation_ppl:
            best_validation_ppl = validation_ppl
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.early_stopping_patience:
            log(f"Early stopping {name}: validation perplexity did not improve.")
            break

    # Early stopping leaves the model 1-2 epochs past its best, so roll back.
    if best_state is not None:
        model.load_state_dict(best_state)
        log(f"Restored {name} to epoch {best_epoch} (val ppl {best_validation_ppl:.2f})")

    return {
        "loss_history": loss_history,
        "validation_history": validation_history,
        "kd_scale_history": kd_scale_history,
        "best_epoch": best_epoch,
        "best_val_ppl": best_validation_ppl,
        "epochs_run": len(loss_history),
        "train_seconds": time.time() - start_time,
    }

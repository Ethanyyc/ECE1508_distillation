#!/usr/bin/env python
"""Turn results/runs/*.json into the grid table and figures.

Importable from the notebook (Section 11) so the analysis lives in one place and the
notebook never needs to hold a multi-hour training session.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "results" / "runs"
RESULTS_DIR = REPO_ROOT / "results"

# Published Table 1 numbers, from the original Windows-ROCm run on different hardware.
# Kept for reference only -- never mixed into the new curves.
PUBLISHED = {
    "Teacher (GPT-2)": {"val_ppl": 30.6, "test_ppl": 29.4},
    "Teacher only (alpha=0)": {"val_ppl": 89.4, "test_ppl": 90.7, "best_epoch": 39},
    "Teacher + corpus (alpha=0.5)": {"val_ppl": 111.2, "test_ppl": 115.6, "best_epoch": 22},
    "Corpus only (alpha=1)": {"val_ppl": 214.5, "test_ppl": 227.2, "best_epoch": 13},
}


def load_runs(runs_dir: Path | str = RUNS_DIR) -> pd.DataFrame:
    """Load every run JSON into a tidy DataFrame, one row per run."""
    runs_dir = Path(runs_dir)
    records = []
    for path in sorted(runs_dir.glob("*.json")):
        if path.name.startswith("smoke"):
            continue
        with open(path) as fh:
            r = json.load(fh)

        val_hist = r.get("validation_history", []) or []
        records.append({
            "tag": r["tag"],
            "alpha": r["alpha"],
            "temperature": r["temperature"],
            "grad_match": r.get("grad_match", False),
            "method": r.get("method"),
            "best_epoch": r.get("best_epoch"),
            "epochs_run": r.get("epochs_run"),
            "best_val_ppl": r.get("best_val_ppl"),
            "val_ppl": r.get("val_ppl"),
            "test_ppl": r.get("test_ppl"),
            # Budget-controlled view: best val PPL within a fixed epoch budget, so the
            # comparison is not confounded by early stopping at different epochs.
            "val_ppl_at_15": min(val_hist[:15]) if val_hist else np.nan,
            "val_ppl_at_10": min(val_hist[:10]) if val_hist else np.nan,
            "wall_min": (r.get("wall_seconds") or 0) / 60,
            "validation_history": val_hist,
            "loss_history": r.get("loss_history", []),
            "kd_scale_history": r.get("kd_scale_history", []),
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(["grad_match", "temperature", "alpha"]).reset_index(drop=True)
    return df


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Human-readable table, also written to results/grid_summary.csv."""
    cols = ["tag", "alpha", "temperature", "grad_match", "best_epoch", "epochs_run",
            "best_val_ppl", "test_ppl", "val_ppl_at_15", "wall_min"]
    out = df[cols].copy()
    out.columns = ["Tag", "Alpha", "T", "Grad-matched", "Best ep.", "Epochs run",
                   "Best val PPL", "Test PPL", "Val PPL @15ep", "Wall (min)"]
    return out.round(2)


def plot_alpha_sweep(df: pd.DataFrame, save: bool = True):
    """Alpha curve at T=1: quality vs the corpus/teacher mix."""
    sub = df[(df.temperature == 1.0) & (~df.grad_match)].sort_values("alpha")
    if sub.empty:
        print("No T=1 runs yet.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    ax = axes[0]
    ax.plot(sub.alpha, sub.best_val_ppl, marker="o", label="Best val PPL")
    ax.plot(sub.alpha, sub.test_ppl, marker="s", label="Test PPL")
    ax.plot(sub.alpha, sub.val_ppl_at_15, marker="^", linestyle=":", alpha=0.75,
            label="Best val PPL within 15 epochs")
    ax.set_xlabel(r"$\alpha$  (weight on corpus / CE term)")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title(r"Effect of $\alpha$ at $T=1$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    # Nudge the end labels inward so they are not clipped by the axes.
    ax.margins(x=0.10)
    for _, r in sub.iterrows():
        ha = "left" if r.alpha == sub.alpha.min() else ("right" if r.alpha == sub.alpha.max() else "center")
        dx = 5 if ha == "left" else (-5 if ha == "right" else 0)
        ax.annotate(f"ep {r.best_epoch}", (r.alpha, r.best_val_ppl),
                    textcoords="offset points", xytext=(dx, -14), fontsize=7, ha=ha)
    ax.text(0.02, 0.97, r"$\alpha=0$: teacher only" "\n" r"$\alpha=1$: corpus only",
            transform=ax.transAxes, va="top", fontsize=7.5,
            bbox=dict(boxstyle="round", fc="white", alpha=0.75))

    ax = axes[1]
    for _, r in sub.iterrows():
        hist = r.validation_history
        ax.plot(range(1, len(hist) + 1), hist, marker="o", markersize=3,
                label=rf"$\alpha$={r.alpha:g}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation perplexity")
    ax.set_title(r"Validation curves by $\alpha$")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.bar([f"{a:g}" for a in sub.alpha], sub.best_epoch, color="tab:purple", alpha=0.8)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Best epoch")
    ax.set_title("Epochs before overfitting\n(higher = teacher signal regularizes longer)",
                 fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        plt.savefig(RESULTS_DIR / "alpha_sweep.png", dpi=150, bbox_inches="tight")
    return fig


def plot_temperature_grid(df: pd.DataFrame, save: bool = True):
    """Temperature curve at alpha=0.5, including the gradient-matched T=0.5 control."""
    sub = df[(df.alpha == 0.5) & (~df.grad_match)].sort_values("temperature")
    gm = df[(df.alpha == 0.5) & (df.grad_match)]
    if sub.empty:
        print("No alpha=0.5 runs yet.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    ax = axes[0]
    ax.plot(sub.temperature, sub.best_val_ppl, marker="o", label="Best val PPL")
    ax.plot(sub.temperature, sub.test_ppl, marker="s", label="Test PPL")
    if not gm.empty:
        g = gm.iloc[0]
        ax.scatter([g.temperature], [g.best_val_ppl], marker="*", s=260,
                   color="tab:red", zorder=5, label="T=0.5, grad-matched (val)")
        ax.scatter([g.temperature], [g.test_ppl], marker="X", s=130,
                   color="tab:orange", zorder=5, label="T=0.5, grad-matched (test)")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.5, 1.0, 2.0])
    ax.set_xticklabels(["0.5", "1", "2"])
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title(r"Effect of $T$ at $\alpha=0.5$ (full training)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for _, r in sub.iterrows():
        hist = r.validation_history
        ax.plot(range(1, len(hist) + 1), hist, marker="o", markersize=3, label=f"T={r.temperature:g}")
    if not gm.empty:
        g = gm.iloc[0]
        ax.plot(range(1, len(g.validation_history) + 1), g.validation_history,
                marker="*", markersize=5, linestyle="--", color="tab:red",
                label="T=0.5 grad-matched")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation perplexity")
    ax.set_title("Validation curves by $T$")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # The confound itself: how far the T^2 factor is from equalizing gradient scale.
    ax = axes[2]
    if not gm.empty:
        scales = gm.iloc[0].kd_scale_history
        if scales:
            ax.plot(range(1, len(scales) + 1), scales, marker="o", color="tab:red")
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1,
                       label=r"1.0 = $T^2$ alone suffices")
            mean_c = float(np.mean(scales))
            ax.axhline(mean_c, color="tab:red", linestyle=":", linewidth=1.2, alpha=0.8,
                       label=rf"mean $c$ = {mean_c:.2f}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(r"$c$ applied to $T^2\,\mathrm{KL}$")
            # c > 1 means the T^2-scaled KD gradient is *weaker* than at T=1 and needs
            # boosting -- the opposite of what the static CE-trained probe predicted.
            ax.set_title("Gradient-matching correction at $T=0.5$\n"
                         r"($c>1$ ⟹ $T^2$ under-corrects during KD training)", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "grad-matched run not available", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()

    plt.tight_layout()
    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        plt.savefig(RESULTS_DIR / "temperature_alpha_grid.png", dpi=150, bbox_inches="tight")
    return fig


def main() -> int:
    df = load_runs()
    if df.empty:
        print(f"No runs found in {RUNS_DIR}")
        return 1

    table = summary_table(df)
    print(table.to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "grid_summary.csv"
    table.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    plot_alpha_sweep(df)
    plot_temperature_grid(df)
    print(f"Wrote {RESULTS_DIR / 'alpha_sweep.png'}")
    print(f"Wrote {RESULTS_DIR / 'temperature_alpha_grid.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

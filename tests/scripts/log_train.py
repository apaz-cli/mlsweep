#!/usr/bin/env python3
"""Colorful fake training script for testing the mlsweep log viewer.

Outputs a mix of ANSI styles: 256-color gradients, bold/dim text, colored
log-level prefixes, per-batch progress bars, and a metrics table per epoch.
"""

import argparse
import math
import os
import time

from mlsweep.logger import MLSweepLogger

# ── ANSI helpers ───────────────────────────────────────────────────────────────

R   = "\033[0m"   # reset
B   = "\033[1m"   # bold
DIM = "\033[2m"   # dim

RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
GRAY    = "\033[90m"
BRED    = "\033[91m"
BGREEN  = "\033[92m"
BYELLOW = "\033[93m"
BCYAN   = "\033[96m"


def c256(n: int) -> str:
    return f"\033[38;5;{n}m"


def gradient_line(ch: str, width: int, lo: int = 39, hi: int = 82) -> str:
    """A horizontal rule whose color slides across the 256-color palette."""
    out = ""
    for i in range(width):
        out += c256(round(lo + (hi - lo) * i / max(width - 1, 1))) + ch
    return out + R


def bar(frac: float, width: int = 20) -> str:
    """Block progress bar, colored red → yellow → green by fill level."""
    frac   = max(0.0, min(1.0, frac))
    filled = round(frac * width)
    color  = BGREEN if frac > 0.66 else BYELLOW if frac > 0.33 else BRED
    return color + "█" * filled + DIM + "░" * (width - filled) + R


def metric_color(frac: float) -> str:
    """Color for a metric value that's better when higher (0–1 normalised)."""
    return BGREEN if frac > 0.66 else BYELLOW if frac > 0.33 else BRED


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--bs",     type=int,   default=64)
    parser.add_argument("--epochs", type=int,   default=18)
    parser.add_argument("--seed",   type=int,   default=42)
    args = parser.parse_args()

    run_id = os.environ.get("MLSWEEP_RUN_NAME", "local")
    logger = MLSweepLogger(hparams=vars(args))

    W = 58  # terminal width for decorations

    # ── Banner ──────────────────────────────────────────────────────────────────
    print(gradient_line("═", W))
    print(c256(39) + "║" + R
          + f"  {B}mlsweep{R} · log viewer test"
          + f"  {GRAY}run {run_id[:24]}{R}")
    print(c256(51) + "║" + R
          + f"  {CYAN}lr={args.lr:<10g}{R}"
          + f"  {MAGENTA}bs={args.bs:<5}{R}"
          + f"  {YELLOW}epochs={args.epochs:<4}{R}"
          + f"  {GRAY}seed={args.seed}{R}")
    print(gradient_line("═", W))
    print(flush=True)

    # ── Startup log ─────────────────────────────────────────────────────────────
    def info(msg: str) -> None:
        print(f"  {B}{BLUE}[INFO]{R}  {msg}", flush=True)

    def warn(msg: str) -> None:
        print(f"  {B}{YELLOW}[WARN]{R}  {msg}", flush=True)

    def good(msg: str) -> None:
        print(f"  {B}{BGREEN}[ OK ]{R}  {msg}", flush=True)

    def ckpt(msg: str) -> None:
        print(f"  {B}{MAGENTA}[CKPT]{R}  {msg}", flush=True)

    info(f"Optimizer  AdamW  lr={args.lr}  weight_decay=1e-4")
    info(f"Scheduler  CosineAnnealingLR  T_max={args.epochs}")
    info(f"Dataset    50 000 train / 10 000 val / 10 classes")
    info(f"Batch size {args.bs}  →  {50_000 // args.bs} steps / epoch")
    info(f"Seed {args.seed}")
    warn("CUDA not detected — using CPU")
    print(flush=True)

    # ── Training loop ────────────────────────────────────────────────────────────
    MAX_LOSS    = 2.4
    MINI_STEPS  = 14
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0   = time.time()
        frac = (epoch - 1) / max(args.epochs - 1, 1)

        # Fake metrics — smooth decay + per-seed noise
        noise      = 0.025 * math.sin(epoch * args.seed * 0.41)
        train_loss = round(MAX_LOSS * math.exp(-4.2 * frac) + 0.07 + noise, 4)
        val_loss   = round(MAX_LOSS * math.exp(-4.0 * frac) + 0.10 + noise * 0.6, 4)
        train_acc  = round(min(0.99, 1.0 - train_loss / (MAX_LOSS * 1.1)), 4)
        val_acc    = round(min(0.99, 1.0 - val_loss   / (MAX_LOSS * 1.1)), 4)
        lr_now     = args.lr * (1 + math.cos(math.pi * frac)) / 2

        # Epoch header — hue shifts from blue toward green as training progresses
        hue = c256(round(39 + frac * 83))
        print(f"\n  {hue}{'─' * 8}{R}  "
              f"{B}{hue}Epoch {epoch:>2} / {args.epochs}{R}"
              f"  {GRAY}lr={lr_now:.2e}{R}"
              f"  {hue}{'─' * 14}{R}",
              flush=True)

        # Mini-batch steps
        for step in range(1, MINI_STEPS + 1):
            # Batch loss starts above epoch loss and converges to it
            bl = train_loss * (1 + 0.6 * math.exp(-step / 5))
            print(f"  {GRAY}  step {step:>2}/{MINI_STEPS}"
                  f"  {bar(step / MINI_STEPS, 18)}"
                  f"  loss {BYELLOW}{bl:.4f}{R}"
                  f"  acc {metric_color(train_acc)}{train_acc:.4f}{R}{GRAY}{R}",
                  flush=True)
            time.sleep(0.05)

        elapsed = time.time() - t0

        # Per-epoch metrics table
        lf  = 1 - train_loss / MAX_LOSS
        vlf = 1 - val_loss   / MAX_LOSS
        print(f"\n  {'':4}{DIM}{'metric':<10}{'train':>10}{'val':>10}  {'':4}{R}")
        print(f"  {'':4}{GRAY}{'─' * 36}{R}")
        print(f"  {'':4}{'loss':<10}"
              f"{metric_color(lf)}{train_loss:>10.4f}{R}"
              f"{metric_color(vlf)}{val_loss:>10.4f}{R}"
              f"  {bar(lf, 14)}")
        print(f"  {'':4}{'acc':<10}"
              f"{metric_color(train_acc)}{train_acc:>10.4f}{R}"
              f"{metric_color(val_acc)}{val_acc:>10.4f}{R}"
              f"  {bar(train_acc, 14)}")
        print(f"  {GRAY}  elapsed: {elapsed:.2f}s{R}", flush=True)

        logger.log({
            "loss": train_loss, "val_loss": val_loss,
            "acc":  train_acc,  "val_acc":  val_acc,
        }, step=epoch)

        # Sporadic events
        if epoch == 3:
            warn(f"Gradient norm spike: 18.3  →  clipping to max_norm=1.0")
        if epoch == args.epochs // 2:
            info(f"Cosine decay past midpoint  lr={lr_now:.2e}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            good(f"New best  val_acc={B}{BGREEN}{val_acc:.4f}{R}"
                 f"  val_loss={B}{BGREEN}{val_loss:.4f}{R}")
        if epoch % 5 == 0 and epoch < args.epochs:
            ckpt(f"Saving  {GRAY}checkpoints/epoch_{epoch:03d}.pt{R}")

    # ── Footer ───────────────────────────────────────────────────────────────────
    print()
    print(gradient_line("━", W, lo=82, hi=39))
    print(f"  {B}{BGREEN}Training complete{R}"
          f"  best val_acc={B}{BGREEN}{best_val_acc:.4f}{R}"
          f"  final val_loss={B}{BGREEN}{val_loss:.4f}{R}")
    print(gradient_line("━", W, lo=82, hi=39))

    logger.close()


if __name__ == "__main__":
    main()

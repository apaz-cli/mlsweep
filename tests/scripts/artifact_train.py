#!/usr/bin/env python3
"""Test training script that generates artifact files (no GPU, no extra deps).

Writes to MLSWEEP_RUN_DIR (the artifacts dir) or ./artifacts/ as fallback:
  plot.png      — loss-curve image, color varies by lr/bs for visual comparison
  results.json  — final metric summary
  training.csv  — per-step loss and acc
"""

import argparse
import json
import math
import os
import struct
import zlib

from mlsweep.logger import MLSweepLogger


# ── PNG writer (stdlib only) ──────────────────────────────────────────────────


def _png_chunk(ctype: bytes, data: bytes) -> bytes:
    payload = ctype + data
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)


def write_loss_curve_png(path: str, losses: list[float], lr: float, bs: int) -> None:
    """Draw a simple 240×120 loss-curve PNG. Color encodes lr for easy comparison."""
    W, H = 240, 120
    MARGIN = 12

    # Pick a hue from lr (log scale maps to hue 0–240 degrees)
    lr_t = max(0.0, min(1.0, (math.log10(max(lr, 1e-9)) + 5) / 4))  # 1e-5..1e-1 → 0..1
    bs_t = max(0.0, min(1.0, (math.log2(max(bs, 1)) - 3) / 7))       # 8..1024 → 0..1

    def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
        h = h % 1.0
        i = int(h * 6)
        f = h * 6 - i
        p, q, t_ = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
        sectors = [(v, t_, p), (q, v, p), (p, v, t_), (p, q, v), (t_, p, v), (v, p, q)]
        r, g, b = sectors[i % 6]
        return int(r * 255), int(g * 255), int(b * 255)

    curve_color = hsv_to_rgb(lr_t * 0.7, 0.9, 1.0)
    bg = (28, 28, 28)
    grid = (50, 50, 50)

    # Pixel buffer: list-of-rows
    px: list[list[tuple[int, int, int]]] = [[bg] * W for _ in range(H)]

    def set_pixel(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < W and 0 <= y < H:
            px[y][x] = color

    # Draw light grid lines
    for gy in range(MARGIN, H - MARGIN, (H - 2 * MARGIN) // 4):
        for x in range(MARGIN, W - MARGIN):
            set_pixel(x, gy, grid)
    for gx in range(MARGIN, W - MARGIN, (W - 2 * MARGIN) // 6):
        for y in range(MARGIN, H - MARGIN):
            set_pixel(gx, y, grid)

    if losses:
        lo = min(losses)
        hi = max(losses)
        span = hi - lo if hi > lo else 1.0
        plot_w = W - 2 * MARGIN
        plot_h = H - 2 * MARGIN

        prev_x, prev_y = None, None
        for i, loss in enumerate(losses):
            x = MARGIN + int(i * plot_w / max(len(losses) - 1, 1))
            y_norm = 1.0 - (loss - lo) / span
            y = MARGIN + int(y_norm * plot_h)
            y = max(MARGIN, min(H - MARGIN - 1, y))

            # Draw anti-aliased-ish line from prev to current
            if prev_x is not None and prev_y is not None:
                dx, dy = x - prev_x, y - prev_y
                steps = max(abs(dx), abs(dy), 1)
                for s in range(steps + 1):
                    lx = prev_x + int(s * dx / steps)
                    ly = prev_y + int(s * dy / steps)
                    set_pixel(lx, ly, curve_color)
                    set_pixel(lx, ly - 1, curve_color)  # 2px thick

            prev_x, prev_y = x, y

        # Endpoint dot
        if prev_x is not None:
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dx * dx + dy * dy <= 5:
                        set_pixel(prev_x + dx, prev_y + dy, curve_color)

    # Encode PNG
    raw = b"".join(b"\x00" + bytes(c for rgb in row for c in rgb) for row in px)
    png_data = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, level=1))
        + _png_chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(png_data)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    artifacts_dir = os.environ.get("MLSWEEP_RUN_DIR") or os.path.join("artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    steps_data: list[dict] = []
    with MLSweepLogger() as logger:
        for step in range(1, args.steps + 1):
            # Simulated loss: decays with lr, offset by batch size
            loss = math.exp(-args.lr * 150 * step / args.steps) + 0.8 / math.log(args.bs + 1) * (1 - step / args.steps) + 0.05
            acc = max(0.0, min(1.0, 1.0 - loss * 0.45))
            loss = round(loss, 4)
            acc  = round(acc, 4)
            logger.log({"loss": loss, "acc": acc}, step=step)
            steps_data.append({"step": step, "loss": loss, "acc": acc})

    # artifacts/plot.png — loss curve
    write_loss_curve_png(
        os.path.join(artifacts_dir, "plot.png"),
        [d["loss"] for d in steps_data],
        args.lr, args.bs,
    )

    # artifacts/results.json
    final = steps_data[-1]
    with open(os.path.join(artifacts_dir, "results.json"), "w") as f:
        json.dump({
            "lr": args.lr,
            "bs": args.bs,
            "steps": args.steps,
            "final_loss": final["loss"],
            "final_acc": final["acc"],
            "min_loss": round(min(d["loss"] for d in steps_data), 4),
        }, f, indent=2)

    # artifacts/training.csv
    with open(os.path.join(artifacts_dir, "training.csv"), "w") as f:
        f.write("step,loss,acc\n")
        for d in steps_data:
            f.write(f"{d['step']},{d['loss']},{d['acc']}\n")

    print(f"[artifact_train] lr={args.lr}  bs={args.bs}  final_loss={final['loss']:.4f}  artifacts → {artifacts_dir}")


if __name__ == "__main__":
    main()

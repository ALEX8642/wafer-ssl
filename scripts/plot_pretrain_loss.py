#!/usr/bin/env python
"""
plot_pretrain_loss.py — Plot the SimCLR NT-Xent loss curve from a pretraining log.

`wafer_ssl.pretrain` prints a table to stdout (Epoch / NT-Xent loss / LR) every
10 epochs. This script parses that table from a saved log file and renders the
loss curve (with the LR schedule on a secondary axis) to a PNG for the README.

Capture the log first, then plot:
    python -m wafer_ssl.pretrain --config configs/pretrain.yaml | tee outputs/pretrain.log
    python scripts/plot_pretrain_loss.py --log outputs/pretrain.log

Regenerate after the FULL run (epoch 200) before committing — a partial-run
curve should not ship as the final artifact.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Matches table rows like:  "     1          3.8304    3.00e-04"
ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+e[+-]?\d+)\s*$")


def parse_log(log_path: Path) -> tuple[list[int], list[float], list[float]]:
    epochs, losses, lrs = [], [], []
    for line in log_path.read_text().splitlines():
        m = ROW.match(line)
        if m:
            epochs.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            lrs.append(float(m.group(3)))
    if not epochs:
        raise SystemExit(
            f"No 'Epoch / loss / LR' rows parsed from {log_path}. "
            "Pass the saved stdout of wafer_ssl.pretrain (see --help)."
        )
    return epochs, losses, lrs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", type=Path, required=True, help="Saved pretrain stdout log")
    p.add_argument("--out", type=Path, default=Path("assets/pretrain_loss.png"))
    args = p.parse_args()

    epochs, losses, lrs = parse_log(args.log)

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(epochs, losses, "o-", color="#1f77b4", lw=2, ms=4, label="NT-Xent loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("NT-Xent loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, lrs, "--", color="#999999", lw=1.2, label="learning rate")
    ax2.set_ylabel("learning rate", color="#999999")
    ax2.tick_params(axis="y", labelcolor="#999999")

    ax1.set_title(
        f"SimCLR pretraining — {losses[0]:.3f} → {losses[-1]:.3f} "
        f"over {epochs[-1]} epochs"
    )
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}  ({len(epochs)} points, "
          f"epochs {epochs[0]}–{epochs[-1]}, loss {losses[0]:.4f}→{losses[-1]:.4f})")


if __name__ == "__main__":
    main()

"""Sprint 4: plot loss and top-k accuracy curves from runs/<name>/metrics.csv -> curves.png"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="base_v1")
    args = p.parse_args()
    run = Path("runs") / args.name
    df = pd.read_csv(run / "metrics.csv")

    train = df[df.kind == "train"]
    quick = df[df.kind == "val_quick"]
    full = df[df.kind == "val_full"]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(train.step, train.loss, label="train (running avg)", alpha=0.8)
    if len(quick):
        ax[0].plot(quick.step, quick.loss, label="val (quick)", alpha=0.8)
    if len(full):
        ax[0].plot(full.step, full.loss, "o-", label="val (full, per epoch)")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("cross-entropy loss"); ax[0].set_title("Loss")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    src = full if len(full) else quick
    for col in ("top1", "top3", "top5"):
        if len(src):
            ax[1].plot(src.step, src[col] * 100, "o-", label=f"val {col}")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("accuracy (%)"); ax[1].set_title("Validation accuracy")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    fig.tight_layout()
    out = run / "curves.png"
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()

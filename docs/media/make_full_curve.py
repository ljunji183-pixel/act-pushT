"""Rebuild docs/media/loss_curves_full_106ep.png from the raw loss logs.

The 106 epochs were produced by one from-scratch run plus five resume runs.
Each run wrote its own loss_log.csv; they are kept verbatim in
docs/media/loss_logs/ and named by the epoch range they cover, so the figure
can be rebuilt from the repository alone:

    python docs/media/make_full_curve.py

Numbers in the annotations are closed-loop results measured separately with
evaluate_stable.py (see the README results table).
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LOGS = HERE / "loss_logs"
OUT = HERE / "loss_curves_full_106ep.png"

PLATEAU_EPOCH = 24            # val loss settles on ~0.24 here and never leaves
SUCCESS_EARLY = "~8% (2/25)"  # ep24-25 checkpoint, 25 episodes, seed 1000
SUCCESS_LATE = "~60%"         # ep96 checkpoint, 10 episodes, seed 1000


def read_logs():
    """Concatenate every loss_log.csv, keyed by epoch (later rows win)."""
    by_epoch = {}
    for path in sorted(LOGS.glob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                try:
                    epoch = int(float(row["epoch"]))
                    by_epoch[epoch] = (
                        epoch,
                        float(row["train_action"]),
                        float(row["train_kl"]),
                        float(row["val_action"]),
                    )
                except (KeyError, TypeError, ValueError):
                    continue  # partial row still being written by the trainer
    return [by_epoch[e] for e in sorted(by_epoch)]


rows = read_logs()
epochs = [r[0] for r in rows]
train_action = [r[1] for r in rows]
train_kl = [r[2] for r in rows]
val_action = [r[3] for r in rows]
print(f"epochs {epochs[0]}-{epochs[-1]} ({len(rows)} rows), val tail ~{val_action[-1]:.3f}")

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
band = dict(color="red", alpha=0.12, lw=0)

# --- action loss -----------------------------------------------------------
axes[0].plot(epochs, train_action, lw=1.3, color="tab:blue", label="train_action")
axes[0].plot(epochs, val_action, lw=1.3, color="tab:orange", label="val_action")
axes[0].axvspan(PLATEAU_EPOCH, PLATEAU_EPOCH + 2, **band,
                label=f"val loss plateau begins (~ep{PLATEAU_EPOCH})")
axes[0].set_ylabel("action L1")
axes[0].set_title(f"ACT on PushT: train / val action loss (epoch {epochs[0]}-{epochs[-1]})")
axes[0].legend(loc="upper right")
axes[0].grid(alpha=0.3)

# --- KL --------------------------------------------------------------------
axes[1].plot(epochs, train_kl, lw=1.3, color="tab:green", label="train KL")
axes[1].axvspan(PLATEAU_EPOCH, PLATEAU_EPOCH + 2, **band)
axes[1].set_ylabel("KL")
axes[1].legend(loc="upper right")
axes[1].grid(alpha=0.3)

# --- val loss, same curve zoomed, with the two readings --------------------
axes[2].plot(epochs, val_action, lw=1.5, color="tab:orange")
axes[2].axvspan(PLATEAU_EPOCH, PLATEAU_EPOCH + 2, **band)
axes[2].axhline(0.24, ls=":", lw=1.0, color="0.45")
axes[2].text(60, 0.2404, "0.24 floor", color="0.35", fontsize=8,
             va="bottom", ha="center")

axes[2].annotate(
    f"ep{PLATEAU_EPOCH}/25 checkpoint: val loss stuck ~0.24\n"
    f"closed-loop success only {SUCCESS_EARLY}, jerky",
    xy=(PLATEAU_EPOCH + 2, 0.2425), xytext=(33, 0.300),
    arrowprops=dict(arrowstyle="->", lw=1.0),
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.9),
)
axes[2].annotate(
    "train loss keeps dropping (overfit)\n"
    f"success {SUCCESS_EARLY} \u2192 {SUCCESS_LATE} by ep96",
    xy=(96, 0.2395), xytext=(50, 0.372),
    arrowprops=dict(arrowstyle="->", lw=1.0),
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.9),
)

axes[2].set_ylabel("val action L1")
axes[2].set_xlabel("epoch")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT, dpi=160)
print(OUT)

"""Plot the metrics saved in an ACT training run's loss_log.csv."""
import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt


def read_log(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No training rows found in {csv_path}")
    if "epoch" not in rows[0]:
        raise ValueError(f"Missing required 'epoch' column in {csv_path}")
    columns = {}
    for key in rows[0]:
        values = []
        for row in rows:
            raw = row.get(key, "")
            try:
                values.append(float(raw) if raw not in (None, "") else float("nan"))
            except ValueError:
                values.append(float("nan"))
        columns[key] = values
    return columns


def plot_log(csv_path: Path, output_path: Path, show: bool = False):
    data = read_log(csv_path)
    epochs = data["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(f"Training curves — {csv_path.parent.name}")

    ax = axes[0, 0]
    for key, label in (("train_action", "Train action"), ("val_action", "Validation action")):
        if key in data:
            ax.plot(epochs, data[key], label=label)
    ax.set(title="Action loss", xlabel="Epoch", ylabel="Loss")
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend()

    ax = axes[0, 1]
    for key, label in (("train_kl", "Train KL"), ("train_total", "Train total")):
        if key in data:
            ax.plot(epochs, data[key], label=label)
    ax.set(title="KL / total loss", xlabel="Epoch", ylabel="Loss")
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend()

    ax = axes[1, 0]
    if "val_raw_action" in data:
        ax.plot(epochs, data["val_raw_action"], label="Validation raw-action MAE", color="tab:green")
    ax.set(title="Validation error in original action units", xlabel="Epoch", ylabel="MAE")
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend()

    ax = axes[1, 1]
    for key, label in (("encoder_time_scale", "Encoder time scale"),
                       ("decoder_time_scale", "Decoder time scale")):
        if key in data:
            ax.plot(epochs, data[key], label=label)
    ax.set(title="Learned sin/cos time-encoding gates", xlabel="Epoch", ylabel="Scale")
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend()
    if not ax.lines:
        ax.text(0.5, 0.5, "No time-gate columns in this log", ha="center", va="center", transform=ax.transAxes)

    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="A run directory or its loss_log.csv")
    parser.add_argument("--output", type=Path, default=None,
                        help="PNG output path (default: loss_curves.png next to the CSV)")
    parser.add_argument("--show", action="store_true", help="Also open the plot in a window")
    parser.add_argument("--watch-seconds", type=float, default=0,
                        help="Refresh the PNG periodically while training (Ctrl+C to stop)")
    args = parser.parse_args()
    if args.watch_seconds < 0:
        parser.error("--watch-seconds must be nonnegative")

    csv_path = args.path / "loss_log.csv" if args.path.is_dir() else args.path
    if not csv_path.is_file():
        parser.error(f"Could not find training log: {csv_path}")
    output_path = args.output or (csv_path.parent / "loss_curves.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    watch_fig = watch_ax = watch_image = None
    try:
        while True:
            plot_log(csv_path, output_path, show=args.show and not args.watch_seconds)
            if args.watch_seconds and args.show:
                if watch_fig is None:
                    watch_fig, watch_ax = plt.subplots(figsize=(12, 8))
                    watch_image = watch_ax.imshow(plt.imread(output_path))
                    watch_ax.axis("off")
                    watch_fig.canvas.manager.set_window_title("ACT training curves (watch)")
                    plt.show(block=False)
                else:
                    watch_image.set_data(plt.imread(output_path))
                    watch_fig.canvas.draw_idle()
                plt.pause(0.1)
            print(f"Saved {output_path} ({len(read_log(csv_path)['epoch'])} epochs)", flush=True)
            if not args.watch_seconds:
                break
            time.sleep(max(0, args.watch_seconds - (0.1 if args.show else 0)))
    except KeyboardInterrupt:
        print("Stopped watching; the last saved plot is kept.")
    finally:
        if watch_fig is not None:
            plt.close(watch_fig)


if __name__ == "__main__":
    main()

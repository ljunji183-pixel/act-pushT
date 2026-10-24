"""读取 A1 的 CSV 日志，绘制并可实时刷新损失曲线。
Plot A1 CSV metrics, optionally refreshing while training continues.

python plot_act_loss.py --watch
python plot_act_loss.py --run-dir "D:\\path\\to\\run"
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


DEFAULT_ROOT = Path(r".\runs_act_a1")
FIELDS = ["epoch", "train_action", "train_kl", "train_total", "val_action", "val_raw_action"]


def configure_font():
    # 优先使用本机中文字体 / Prefer an installed Chinese font
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "SimSun"]:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def read_metrics(path):
    """跳过未写完的行；同一轮保留最后一条 / Ignore partial rows and deduplicate epochs."""
    if not path.exists():
        return []
    by_epoch = {}
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            for raw in csv.DictReader(f):
                try:
                    row = {field: float(raw[field]) for field in FIELDS}
                    if not all(math.isfinite(value) for value in row.values()):
                        continue
                    by_epoch[int(row["epoch"])] = row
                except (KeyError, TypeError, ValueError):
                    # 训练进程可能正在写入这一行 / The trainer may be writing this row
                    continue
    except (PermissionError, OSError):
        return None
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def draw_curves(fig, axes, rows, run_name):
    for ax in axes.flat:
        ax.clear()
        ax.set_xlabel("训练轮数 / Epoch")
        ax.grid(alpha=0.25)

    if not rows:
        fig.suptitle(f"{run_name}\n等待第一轮训练和验证完成 / Waiting for epoch 1")
        for ax in axes.flat:
            ax.text(0.5, 0.5, "暂无日志数据 / No data yet", ha="center", va="center",
                    transform=ax.transAxes)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        return

    epochs = [row["epoch"] for row in rows]

    # 两条曲线的目标相同，但训练与验证的 z 来源不同
    # Both use action L1, but training and validation use different z inputs
    ax = axes[0, 0]
    ax.plot(epochs, [r["train_action"] for r in rows], marker=".", label="Train (CVAE z)")
    ax.plot(epochs, [r["val_action"] for r in rows], marker=".", label="Val (z=0)")
    ax.set_title("归一化动作误差 / Action L1")
    ax.set_ylabel("Mean absolute error")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, [r["train_kl"] for r in rows], color="tab:orange", marker=".")
    ax.set_title("训练 KL 损失 / Train KL")
    ax.set_ylabel("KL")

    ax = axes[1, 0]
    ax.plot(epochs, [r["train_total"] for r in rows], color="tab:green", marker=".")
    ax.set_title("训练总损失 / Train L1 + beta * KL")
    ax.set_ylabel("Total loss")

    ax = axes[1, 1]
    ax.plot(epochs, [r["val_raw_action"] for r in rows], color="tab:red", marker=".")
    ax.set_title("验证动作误差：原始坐标单位 / Val raw-coordinate MAE")
    ax.set_ylabel("Mean absolute error")

    fig.suptitle(f"ACT A1 · {run_name} · Epoch {int(epochs[-1])}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="指定运行目录 / Run directory")
    parser.add_argument("--watch", action="store_true", help="每 5 秒检查日志 / Watch the CSV")
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        runs = sorted(path for path in DEFAULT_ROOT.glob("*") if path.is_dir())
        if not runs:
            parser.error("还没有训练运行目录，请先启动训练 / No training run found")
        run_dir = runs[-1]
    if not run_dir.is_dir():
        parser.error(f"运行目录不存在 / Run directory does not exist: {run_dir}")

    configure_font()
    log_path = run_dir / "loss_log.csv"
    image_path = run_dir / "loss_curves.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    last_signature = None

    def refresh():
        nonlocal last_signature
        rows = read_metrics(log_path)
        if rows is None:
            return
        signature = tuple(tuple(row[field] for field in FIELDS) for row in rows)
        if signature == last_signature:
            return
        last_signature = signature
        draw_curves(fig, axes, rows, run_dir.name)
        if rows:
            fig.savefig(image_path, dpi=160)
            print(f"已更新到第 {int(rows[-1]['epoch'])} 轮 / Saved: {image_path}", flush=True)
        fig.canvas.draw_idle()

    print("日志 / CSV:", log_path)
    if not log_path.exists():
        print("日志会在一轮训练和验证结束后生成 / Waiting for the first completed epoch")
    refresh()
    if args.watch:
        # 定时器属于画图窗口，不会干预训练进程
        # This GUI timer does not interact with the training process
        timer = fig.canvas.new_timer(interval=5000)
        timer.add_callback(refresh)
        timer.start()
        plt.show()
        timer.stop()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()

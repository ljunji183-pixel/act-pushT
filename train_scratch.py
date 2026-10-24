"""Train a sinusoidal-time ACT model from scratch in this workspace."""
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT = Path(r".")


def load_data_training_utils(project):
    source = project / "train_act_a1.py"
    if not source.is_file():
        raise FileNotFoundError(f"Training/data utilities not found: {source}")
    sys.path.insert(0, str(project))
    spec = importlib.util.spec_from_file_location("act_data_training_utils", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT,
                        help="Existing project containing data/pusht and train_act_a1.py")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta-kl", type=float, default=0.01)
    parser.add_argument("--free-bits", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "runs_sincos_scratch")
    parser.add_argument("--no-pretrained-backbone", action="store_true",
                        help="Do not use ImageNet weights for the ResNet visual backbone")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.chunk_size) < 1 or args.lr <= 0:
        parser.error("epochs, batch-size and chunk-size must be positive; lr must be positive")
    if args.beta_kl < 0 or args.free_bits < 0 or args.weight_decay < 0:
        parser.error("beta-kl, free-bits and weight-decay must be nonnegative")

    base = load_data_training_utils(args.project)
    base.DATA_ROOT = args.data_root or (args.project / "data" / "pusht")
    base.CHUNK_SIZE = args.chunk_size
    base.BATCH_SIZE = args.batch_size
    base.BETA_KL = args.beta_kl
    base.FREE_BITS = args.free_bits
    base.SEED = args.seed
    from act_policy_sincos import ACTPolicy

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(base.DATA_ROOT)
    info = json.loads((data_root / "meta" / "info.json").read_text(encoding="utf-8"))
    total_episodes = int(info["total_episodes"])
    if total_episodes < 2:
        raise ValueError("Need at least two episodes for a train/validation split")
    generator = torch.Generator().manual_seed(args.seed)
    ids = torch.randperm(total_episodes, generator=generator).tolist()
    split = min(max(int(total_episodes * 0.9), 1), total_episodes - 1)
    train_ids, val_ids = sorted(ids[:split]), sorted(ids[split:])

    train_source = base.make_dataset(train_ids, info["fps"])
    val_source = base.make_dataset(val_ids, info["fps"])
    stats_cpu = base.compute_train_stats(train_source)
    stats = {key: value.to(device) for key, value in stats_cpu.items()}
    cache_dir = ROOT / ".cache_sincos"
    digest = hashlib.sha1(json.dumps([train_ids, val_ids]).encode()).hexdigest()[:12]
    train_set = base.CachedDataset(train_source, cache_dir, f"train_{digest}")
    val_set = base.CachedDataset(val_source, cache_dir, f"val_{digest}")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    policy = ACTPolicy(
        chunk_size=args.chunk_size,
        pretrained_backbone=not args.no_pretrained_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    run_dir = args.out / datetime.now().strftime("%Y%m%d_%H%M%S_scratch")
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "model": "ACT with gated sinusoidal action-time encoding",
        "data_root": str(data_root.resolve()),
        "chunk_size": args.chunk_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "beta_kl": args.beta_kl,
        "free_bits": args.free_bits,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "fps": info["fps"],
        "latent_dim": 32,
        "model_dim": 256,
        "pretrained_backbone": not args.no_pretrained_backbone,
        "time_encoding": "sinusoidal; tanh-gated; initialized to preserve learned-position baseline",
        "train_episode_ids": train_ids,
        "val_episode_ids": val_ids,
        "rollout_default_exec_steps": 8,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log_path = run_dir / "loss_log.csv"
    best_val = float("inf")
    global_step = 0
    print(f"Device: {device}; episodes train/val={len(train_ids)}/{len(val_ids)}")
    print(f"Training 1-{args.epochs}; batch={args.batch_size}; lr={args.lr:g}; beta={args.beta_kl:g}; free_bits={args.free_bits:g}")
    print(f"Output: {run_dir}")

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_metrics = base.train_one_epoch(
            policy, train_loader, optimizer, stats, device, args.beta_kl,
            tag=f"Epoch {epoch}/{args.epochs} ",
        )
        val_metrics = base.validate(
            policy, val_loader, stats, device, tag=f"Epoch {epoch}/{args.epochs} "
        )
        global_step += train_metrics.pop("steps")
        row = {
            "epoch": epoch, "beta": args.beta_kl, **train_metrics, **val_metrics,
            "seconds": time.perf_counter() - started,
            "lr": optimizer.param_groups[0]["lr"],
            "encoder_time_scale": torch.tanh(policy.cvae_encoder.temporal_scale_raw).item(),
            "decoder_time_scale": torch.tanh(policy.decoder_temporal_scale_raw).item(),
        }
        with log_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        payload = {
            "model": policy.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch,
            "global_step": global_step,
            "best_val": min(best_val, val_metrics["val_action"]),
            "stats": stats_cpu, "config": config,
            "train_episode_ids": train_ids, "val_episode_ids": val_ids,
        }
        torch.save(payload, run_dir / "last.pt")
        if val_metrics["val_action"] < best_val:
            best_val = val_metrics["val_action"]
            payload["best_val"] = best_val
            torch.save(payload, run_dir / "best.pt")
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs}: train={row['train_action']:.4f}, "
              f"val={row['val_action']:.4f}, raw_MAE={row['val_raw_action']:.3f}, "
              f"sin/cos gates=({row['encoder_time_scale']:.4f}, {row['decoder_time_scale']:.4f})",
              flush=True)
    print(f"Done. Checkpoints: {run_dir}")


if __name__ == "__main__":
    main()

"""Fine-tune the current ACT checkpoint with gated sinusoidal time encoding."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT = Path(r".")
DEFAULT_CKPT = DEFAULT_PROJECT / "runs_act_a1/20261022_152925_686598_resume_e077/epoch_090.pt"


def load_base_training_utils(project):
    sys.path.insert(0, str(project))
    spec = importlib.util.spec_from_file_location("act_training_utils", project / "train_act_a1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT,
                        help="Existing project containing data/pusht and training data utilities")
    parser.add_argument("--additional-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--beta-kl", type=float, default=None,
                        help="Override KL weight (default: keep checkpoint value)")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Override AdamW weight decay (default: keep checkpoint value)")
    parser.add_argument("--result-json", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "runs_sincos")
    args = parser.parse_args()
    if args.additional_epochs < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("additional-epochs and lr must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("batch-size must be positive")
    for value in (args.beta_kl, args.weight_decay):
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error("beta-kl and weight-decay must be finite and nonnegative")

    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    chunk_size = int(cfg["chunk_size"])
    old_epoch = int(checkpoint["epoch"])
    batch_size = args.batch_size or int(cfg.get("batch_size", 8))
    base = load_base_training_utils(args.project)
    base.DATA_ROOT = Path(cfg["data_root"])
    base.CHUNK_SIZE = chunk_size
    base.BATCH_SIZE = batch_size
    base.BETA_KL = float(cfg.get("beta_kl", 0.1)) if args.beta_kl is None else args.beta_kl
    weight_decay = float(cfg.get("weight_decay", 1e-4)) if args.weight_decay is None else args.weight_decay
    base.FREE_BITS = float(cfg.get("free_bits", 0.0))
    base.SEED = int(cfg.get("seed", 42))

    from act_policy_sincos import ACTPolicy

    torch.manual_seed(base.SEED)
    random.seed(base.SEED)
    np.random.seed(base.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_info = json.loads((base.DATA_ROOT / "meta/info.json").read_text(encoding="utf-8"))
    train_ids = checkpoint["train_episode_ids"]
    val_ids = checkpoint["val_episode_ids"]
    train_source = base.make_dataset(train_ids, data_info["fps"])
    val_source = base.make_dataset(val_ids, data_info["fps"])
    stats_cpu = checkpoint["stats"]
    stats = {key: value.to(device) for key, value in stats_cpu.items()}

    cache_dir = ROOT / ".cache_sincos"
    digest = hashlib.sha1(json.dumps([train_ids, val_ids]).encode()).hexdigest()[:12]
    train_set = base.CachedDataset(train_source, cache_dir, f"train_{digest}")
    val_set = base.CachedDataset(val_source, cache_dir, f"val_{digest}")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    policy = ACTPolicy(chunk_size=chunk_size, pretrained_backbone=False).to(device)
    incompatible = policy.load_state_dict(checkpoint["model"], strict=False)
    expected_new = {"cvae_encoder.temporal_scale_raw", "decoder_temporal_scale_raw"}
    # Accept either a legacy ACT checkpoint (the two time gates are newly initialized)
    # or a checkpoint from this sinusoidal-time model (all parameters are present).
    if set(incompatible.missing_keys) not in (expected_new, set()) or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}")
    # Each fine-tuning run starts a fresh optimizer/schedule at the requested LR.
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.additional_epochs, eta_min=args.lr * 0.1
    )

    args.out.mkdir(parents=True, exist_ok=True)
    run_dir = args.out / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_from_e{old_epoch:03d}"
    run_dir.mkdir()
    config = {
        **cfg,
        "model": "ACT with gated sinusoidal action-time encoding",
        "input_checkpoint": str(args.resume.resolve()),
        "input_epoch": old_epoch,
        "start_epoch": old_epoch + 1,
        "target_epoch": old_epoch + args.additional_epochs,
        "additional_epochs": args.additional_epochs,
        "batch_size": batch_size,
        "learning_rate": args.lr,
        "beta_kl": base.BETA_KL,
        "weight_decay": weight_decay,
        "time_encoding": "sinusoidal; tanh-gated; existing gates preserved, missing gates initialized to zero",
        "rollout_default_exec_steps": 8,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log_path = run_dir / "loss_log.csv"
    best_val = float("inf")
    global_step = int(checkpoint.get("global_step", 0))
    print(f"Device: {device}; epoch {old_epoch + 1}-{old_epoch + args.additional_epochs}; batch={batch_size}; lr={args.lr:g}")
    print(f"Output: {run_dir}")

    for epoch in range(old_epoch + 1, old_epoch + args.additional_epochs + 1):
        train_metrics = base.train_one_epoch(
            policy, train_loader, optimizer, stats, device, base.BETA_KL,
            tag=f"Epoch {epoch} "
        )
        val_metrics = base.validate(policy, val_loader, stats, device, tag=f"Epoch {epoch} ")
        global_step += train_metrics.pop("steps")
        row = {"epoch": epoch, "beta": base.BETA_KL, **train_metrics, **val_metrics,
               "lr": optimizer.param_groups[0]["lr"],
               "encoder_time_scale": torch.tanh(policy.cvae_encoder.temporal_scale_raw).item(),
               "decoder_time_scale": torch.tanh(policy.decoder_temporal_scale_raw).item()}
        with log_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if epoch == old_epoch + 1:
                writer.writeheader()
            writer.writerow(row)
        payload = {
            "model": policy.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch,
            "global_step": global_step, "best_val": min(best_val, val_metrics["val_action"]),
            "stats": stats_cpu, "config": config,
            "train_episode_ids": train_ids, "val_episode_ids": val_ids,
        }
        torch.save(payload, run_dir / "last.pt")
        if val_metrics["val_action"] < best_val:
            best_val = val_metrics["val_action"]
            payload["best_val"] = best_val
            torch.save(payload, run_dir / "best.pt")
        scheduler.step()
        print(f"Epoch {epoch}: train={row['train_action']:.4f}, val={row['val_action']:.4f}, "
              f"raw_MAE={row['val_raw_action']:.3f}, sincos_scales="
              f"({row['encoder_time_scale']:.4f}, {row['decoder_time_scale']:.4f})")
    print(f"Done. Checkpoints: {run_dir}")
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps({"run_dir": str(run_dir.resolve()),
                                               "best_val": best_val}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

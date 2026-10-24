import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from act_policy import ACTPolicy

# ==================== 核心超参数 ====================
DATA_ROOT = Path(r".\data\pusht")
OUTPUT_ROOT = Path(r".\runs_act_a1")
EPOCHS = 80            # 延长训练轮数，让末期余弦退火的 last.pt 充分学透接触微操
BATCH_SIZE = 16        # 兼顾训练速度与每轮充足的梯度更新步数 (steps)
CHUNK_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BETA_KL = 0.01         # 固定小权重：既防止第 1 轮爆表，又不会把 z 压死或拉高总损失
FREE_BITS = 0.01       # 每维下限 0.01 (32 维总下限 0.32)，防止 KL 坍缩到 0
SEED = 42
SAVE_EVERY = 10


def compute_act_loss(
    predicted_actions,
    target_actions,
    action_is_pad,
    mu,
    logvar,
    beta=BETA_KL,
    free_bits=FREE_BITS,
):
    """ACT 损失：有效时间步的 L1 重建损失 + 带下限保护的 KL 散度"""
    l1_per_step = F.l1_loss(predicted_actions, target_actions, reduction="none").mean(dim=-1)
    valid_mask = ~action_is_pad
    action_loss = (l1_per_step * valid_mask).sum() / valid_mask.sum().clamp_min(1)

    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_per_dim.sum(dim=-1).mean()
    kl_for_grad = kl_per_dim.clamp_min(free_bits).sum(dim=-1).mean()

    total_loss = action_loss + beta * kl_for_grad
    return total_loss, action_loss, kl_loss


def make_dataset(episode_ids, fps):
    return LeRobotDataset(
        repo_id="lerobot/pusht",
        root=DATA_ROOT,
        episodes=episode_ids,
        delta_timestamps={"action": [i / fps for i in range(CHUNK_SIZE)]},
    )


class CachedDataset(torch.utils.data.Dataset):
    """将视频帧解码缓存到磁盘一次，后续每个 epoch 直接毫秒级读取"""

    def __init__(self, dataset, cache_dir, name):
        cache_dir.mkdir(parents=True, exist_ok=True)
        img_path = cache_dir / f"{name}_images.npy"
        meta_path = cache_dir / f"{name}_meta.pt"
        count = len(dataset)

        if img_path.exists() and meta_path.exists():
            meta = torch.load(meta_path, weights_only=True)
            if meta["count"] == count:
                self.images = np.load(img_path, mmap_mode="r")
                self.states = meta["states"]
                self.actions = meta["actions"]
                self.pads = meta["pads"]
                print(f"缓存命中: {name} ({count} 帧)")
                return

        print(f"首次构建缓存: {name} ({count} 帧)...")
        self.images = np.lib.format.open_memmap(
            img_path, mode="w+", dtype=np.uint8, shape=(count, 96, 96, 3)
        )
        self.states = torch.empty(count, 2, dtype=torch.float32)
        self.actions = torch.empty(count, CHUNK_SIZE, 2, dtype=torch.float32)
        self.pads = torch.empty(count, CHUNK_SIZE, dtype=torch.bool)

        for i in tqdm(range(count), desc=f"Cache {name}", leave=False):
            sample = dataset[i]
            self.images[i] = (
                sample["observation.image"].permute(1, 2, 0).numpy() * 255
            ).round().astype(np.uint8)
            self.states[i] = sample["observation.state"]
            self.actions[i] = sample["action"]
            self.pads[i] = sample["action_is_pad"]

        torch.save(
            {"count": count, "states": self.states, "actions": self.actions, "pads": self.pads},
            meta_path,
        )

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        image = torch.from_numpy(np.asarray(self.images[index])).float().permute(2, 0, 1) / 255.0
        return {
            "observation.image": image,
            "observation.state": self.states[index],
            "action": self.actions[index],
            "action_is_pad": self.pads[index],
        }


def compute_train_stats(dataset):
    table = dataset.hf_dataset or dataset.load_hf_dataset()
    states = torch.stack(list(table.select_columns(["observation.state"])["observation.state"])).float()
    actions = torch.stack(list(table.select_columns(["action"])["action"])).float()
    return {
        "state_mean": states.mean(dim=0),
        "state_std": states.std(dim=0, unbiased=False).clamp_min(1e-6),
        "action_mean": actions.mean(dim=0),
        "action_std": actions.std(dim=0, unbiased=False).clamp_min(1e-6),
        "image_mean": torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1),
        "image_std": torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1),
    }


def prepare_batch(batch, stats, device):
    images = (batch["observation.image"].to(device) - stats["image_mean"]) / stats["image_std"]
    states = (batch["observation.state"].to(device) - stats["state_mean"]) / stats["state_std"]
    actions = (batch["action"].to(device) - stats["action_mean"]) / stats["action_std"]
    padding = batch["action_is_pad"].to(device, dtype=torch.bool)
    return images, states, actions, padding


def train_one_epoch(policy, loader, optimizer, stats, device, beta=BETA_KL, tag=""):
    policy.train()
    error_sum, valid_count = 0.0, 0
    kl_sum, sample_count, steps = 0.0, 0, 0

    for batch in tqdm(loader, desc=f"{tag}Train", leave=False):
        images, states, actions, padding = prepare_batch(batch, stats, device)
        predicted, mu, logvar = policy(images, states, actions, padding)
        total_loss, action_loss, kl_loss = compute_act_loss(
            predicted, actions, padding, mu, logvar, beta=beta, free_bits=FREE_BITS
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        n_valid = (~padding).sum().item()
        batch_size = states.size(0)
        error_sum += action_loss.item() * n_valid
        valid_count += n_valid
        kl_sum += kl_loss.item() * batch_size
        sample_count += batch_size
        steps += 1

    action_mean = error_sum / max(valid_count, 1)
    kl_mean = kl_sum / max(sample_count, 1)
    return {
        "train_action": action_mean,
        "train_kl": kl_mean,
        "train_total": action_mean + beta * kl_mean,
        "steps": steps,
    }


@torch.no_grad()
def validate(policy, loader, stats, device, tag=""):
    policy.eval()
    norm_err_sum, raw_err_sum, valid_count = 0.0, 0.0, 0

    for batch in tqdm(loader, desc=f"{tag}Val", leave=False):
        images, states, actions, padding = prepare_batch(batch, stats, device)
        predicted, _, _ = policy(images, states)  # 推理模式：自动使用 z = 0

        coord_err = (predicted - actions).abs()
        norm_err = coord_err.mean(dim=-1)
        raw_err = (coord_err * stats["action_std"]).mean(dim=-1)

        norm_err_sum += norm_err.masked_fill(padding, 0.0).sum().item()
        raw_err_sum += raw_err.masked_fill(padding, 0.0).sum().item()
        valid_count += (~padding).sum().item()

    return {
        "val_action": norm_err_sum / max(valid_count, 1),
        "val_raw_action": raw_err_sum / max(valid_count, 1),
    }


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备:", device)

    info = json.loads((DATA_ROOT / "meta/info.json").read_text(encoding="utf-8"))
    generator = torch.Generator().manual_seed(SEED)
    ids = torch.randperm(info["total_episodes"], generator=generator).tolist()
    cut = int(len(ids) * 0.9)
    train_ids, val_ids = sorted(ids[:cut]), sorted(ids[cut:])

    run_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = make_dataset(train_ids, info["fps"])
    val_dataset = make_dataset(val_ids, info["fps"])
    stats_cpu = compute_train_stats(train_dataset)
    stats = {k: v.to(device) for k, v in stats_cpu.items()}

    cache_dir = DATA_ROOT / ".cache"
    train_key = f"train{len(train_ids)}_{abs(hash(tuple(train_ids))) % 10**8}"
    val_key = f"val{len(val_ids)}_{abs(hash(tuple(val_ids))) % 10**8}"
    train_dataset = CachedDataset(train_dataset, cache_dir, train_key)
    val_dataset = CachedDataset(val_dataset, cache_dir, val_key)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    policy = ACTPolicy(chunk_size=CHUNK_SIZE).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE * 0.05
    )

    config = {
        "data_root": str(DATA_ROOT.resolve()),
        "chunk_size": CHUNK_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "beta_kl": BETA_KL,
        "free_bits": FREE_BITS,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "fps": info["fps"],
        "latent_dim": 32,
        "model_dim": 256,
    }

    best_val = float("inf")
    global_step = 0
    log_path = run_dir / "loss_log.csv"

    for epoch in range(1, EPOCHS + 1):
        tag = f"Epoch {epoch}/{EPOCHS} "
        t0 = time.perf_counter()
        train_metrics = train_one_epoch(policy, train_loader, optimizer, stats, device, BETA_KL, tag)
        val_metrics = validate(policy, val_loader, stats, device, tag)
        scheduler.step()
        seconds = time.perf_counter() - t0
        global_step += train_metrics.pop("steps")

        row = {"epoch": epoch, "beta": BETA_KL, **train_metrics, **val_metrics, "seconds": seconds}
        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)

        checkpoint = {
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val": min(best_val, val_metrics["val_action"]),
            "stats": stats_cpu,
            "config": config,
            "train_episode_ids": train_ids,
            "val_episode_ids": val_ids,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if val_metrics["val_action"] < best_val:
            best_val = val_metrics["val_action"]
            torch.save(checkpoint, run_dir / "best.pt")
        if epoch % SAVE_EVERY == 0:
            torch.save(checkpoint, run_dir / f"epoch_{epoch:03d}.pt")

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] | "
            f"Train L1={row['train_action']:.4f} | KL={row['train_kl']:.3f} | "
            f"Total={row['train_total']:.4f} | Val(z=0)={row['val_action']:.4f} | "
            f"Val原始={row['val_raw_action']:.2f} | {seconds:.1f}s"
        )

    print("训练完成，输出目录:", run_dir)


if __name__ == "__main__":
    main()

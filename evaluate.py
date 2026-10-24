"""Evaluate a trained ACT checkpoint on PushT.

离线：验证集上用 z=0 计算 L1（与训练时的验证一致）。
Offline: validation-set L1 with z=0, same as during training.
闭环：在 gym-pusht 仿真中真正执行动作，统计成功率和最大覆盖率。
Closed loop: roll the policy out in gym-pusht, report success rate and coverage.

用法 / Usage:
    python evaluate.py                     # 默认评估 best.pt
    python evaluate.py --episodes 50 --exec-steps 8
"""

import argparse
import json
from pathlib import Path

import cv2
import gymnasium as gym
import gym_pusht  # noqa: F401  (registers the env)
import numpy as np
import torch
from torch.utils.data import DataLoader

from act_policy import ACTPolicy
from train_act_a1 import DATA_ROOT, make_dataset, validate

DEFAULT_CKPT = Path(
    r".\runs_act_a1"
    r"D:\20261021_210723_666657\last.pt"
)


def load_policy(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    policy = ACTPolicy(chunk_size=ckpt["config"]["chunk_size"]).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    stats = {k: v.to(device) for k, v in ckpt["stats"].items()}
    return policy, stats, ckpt


def offline_eval(policy, stats, ckpt, device):
    info = json.loads((DATA_ROOT / "meta/info.json").read_text(encoding="utf-8"))
    val_set = make_dataset(ckpt["val_episode_ids"], info["fps"])
    loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=0)
    return validate(policy, loader, stats, device)


@torch.no_grad()
def predict_chunk(policy, stats, pixels, agent_pos, device):
    """单个观测 -> 反归一化后的动作块 [chunk, 2] / One observation -> raw action chunk."""
    image = torch.from_numpy(pixels).to(device).permute(2, 0, 1).float() / 255.0
    image = (image.unsqueeze(0) - stats["image_mean"]) / stats["image_std"]
    state = torch.from_numpy(agent_pos).to(device).float().unsqueeze(0)
    state = (state - stats["state_mean"]) / stats["state_std"]
    actions, _, _ = policy(image, state)  # z = 0
    return (actions[0] * stats["action_std"] + stats["action_mean"]).cpu().numpy()


class ACTTemporalEnsembler:
    """Align overlapping chunks by target step; original ACT older-first weights."""

    def __init__(self, coefficient=0.01):
        self.coefficient = coefficient
        self.history = []

    def update(self, step, chunk):
        if not np.isfinite(chunk).all():
            raise ValueError("模型输出了非有限动作块")
        self.history.append((step, chunk))
        self.history = [(start, actions) for start, actions in self.history
                        if start + len(actions) > step]
        # Chronological order: the original ACT implementation gives older
        # predictions greater weight: exp(-k * arange(number_of_predictions)).
        candidates = np.stack([actions[step - start]
                               for start, actions in self.history])
        weights = np.exp(-self.coefficient * np.arange(len(candidates)))
        weights /= weights.sum()
        # Explicit validity via timestamps: zero-valued actions remain valid.
        return (candidates * weights[:, None]).sum(axis=0)


def rollout_eval(policy, stats, device, episodes, exec_steps, seed, video_dir=None,
                 temporal_agg=True, temporal_k=0.01):
    env = gym.make(
        "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300,
        render_mode="rgb_array",  # pixels 观测必须有渲染模式 / pixel obs requires a render mode
    )
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
    successes, best_cov, final_cov = [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done, cov_max, cov = False, 0.0, 0.0
        step = 0
        ensembler = ACTTemporalEnsembler(temporal_k)
        writer = None
        if video_dir is not None:
            # 渲染尺寸由环境决定（当前为 680x680），先取一帧确定宽高
            # Render size is env-defined (680x680 here); take one frame to size it
            frame = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
            writer = cv2.VideoWriter(
                str(video_dir / f"episode_{ep:03d}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), 30,
                (frame.shape[1], frame.shape[0]),
            )
            writer.write(frame)
        while not done:
            chunk = predict_chunk(
                policy, stats, obs["pixels"], obs["agent_pos"], device
            )
            if temporal_agg:
                actions_to_execute = [ensembler.update(step, chunk)]
            else:
                actions_to_execute = chunk[:exec_steps]
            for action in actions_to_execute:
                if not np.isfinite(action).all():
                    raise ValueError(f"模型输出了非法动作: {action}")
                action = np.clip(
                    action, env.action_space.low, env.action_space.high
                ).astype(np.float32)
                obs, _, terminated, truncated, info = env.step(action)
                step += 1
                if writer is not None:
                    frame = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
                    writer.write(cv2.resize(
                        frame, (512, 512), interpolation=cv2.INTER_AREA
                    ))
                cov = info["coverage"]
                cov_max = max(cov_max, cov)
                done = terminated or truncated
                if done:
                    break
        if writer is not None:
            writer.release()
        successes.append(bool(info["is_success"]))
        best_cov.append(cov_max)
        final_cov.append(cov)
        print(f"  episode {ep + 1}/{episodes}: success={successes[-1]} "
              f"max_cov={cov_max:.3f} final_cov={cov:.3f}")
    env.close()
    return {
        "success_rate": float(np.mean(successes)),
        "mean_max_coverage": float(np.mean(best_cov)),
        "mean_final_coverage": float(np.mean(final_cov)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--exec-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--temporal-agg", action=argparse.BooleanOptionalAction,
                        default=True, help="ACT时间集成（默认启用）；--no-temporal-agg恢复分块执行")
    parser.add_argument("--temporal-k", type=float, default=0.01,
                        help="原版ACT指数权重系数，默认0.01，较早预测权重更大")
    parser.add_argument("--skip-offline", action="store_true")
    parser.add_argument("--record", action="store_true",
                        help="把每个回合存成 mp4 / save each rollout as mp4")
    parser.add_argument("--video-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.episodes < 1 or args.exec_steps < 1:
        parser.error("episodes and exec-steps must be positive")
    if not np.isfinite(args.temporal_k) or args.temporal_k < 0:
        parser.error("temporal-k must be finite and nonnegative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, stats, ckpt = load_policy(args.ckpt, device)
    if not args.temporal_agg and args.exec_steps > ckpt["config"]["chunk_size"]:
        parser.error("exec-steps exceeds checkpoint chunk_size")
    print(f"检查点 / Checkpoint: {args.ckpt}  (epoch {ckpt['epoch']})")

    if not args.skip_offline:
        print("离线验证 / Offline:", offline_eval(policy, stats, ckpt, device))

    video_dir = None
    if args.record:
        video_dir = args.video_dir or (args.ckpt.parent / "videos")
        print("录像输出 / Videos:", video_dir)

    mode = (f"ACT时间集成，每步推理，k={args.temporal_k}（exec-steps不参与）"
            if args.temporal_agg else f"分块执行，exec_steps={args.exec_steps}")
    print(f"仿真闭环 / Rollout: {args.episodes} episodes; {mode}")
    print(rollout_eval(
        policy, stats, device, args.episodes, args.exec_steps, args.seed, video_dir,
        temporal_agg=args.temporal_agg, temporal_k=args.temporal_k
    ))


if __name__ == "__main__":
    main()

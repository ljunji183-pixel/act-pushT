"""Evaluate a sinusoidal-time ACT checkpoint on deterministic PushT seeds."""
import argparse
import json
from pathlib import Path

import cv2
import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
import torch

from act_policy_sincos import ACTPolicy


@torch.no_grad()
def predict(policy, stats, obs, device):
    image = torch.as_tensor(obs["pixels"], device=device).permute(2, 0, 1).float() / 255.0
    image = (image.unsqueeze(0) - stats["image_mean"]) / stats["image_std"]
    state = torch.as_tensor(obs["agent_pos"], device=device).float().unsqueeze(0)
    state = (state - stats["state_mean"]) / stats["state_std"]
    actions, _, _ = policy(image, state)
    return (actions[0] * stats["action_std"] + stats["action_mean"]).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--exec-steps", type=int, default=8)
    parser.add_argument("--record-dir", type=Path, default=None)
    parser.add_argument("--result-json", type=Path, default=None,
                        help="Save summary and per-episode metrics as JSON")
    args = parser.parse_args()
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    if args.episodes < 1 or args.exec_steps < 1 or args.exec_steps > checkpoint["config"]["chunk_size"]:
        parser.error("episodes must be positive and exec-steps must fit the action chunk")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ACTPolicy(chunk_size=checkpoint["config"]["chunk_size"], pretrained_backbone=False).to(device)
    policy.load_state_dict(checkpoint["model"], strict=True)
    policy.eval()
    stats = {key: value.to(device) for key, value in checkpoint["stats"].items()}
    if args.record_dir:
        args.record_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                   max_episode_steps=300, render_mode="rgb_array")
    outcomes = []
    try:
        for index in range(args.episodes):
            scene_seed = args.seed + index
            obs, info = env.reset(seed=scene_seed)
            done, coverage, max_coverage, steps = False, 0.0, 0.0, 0
            frames = [] if args.record_dir else None
            if frames is not None:
                frames.append(cv2.resize(env.render(), (512, 512)))
            while not done:
                chunk = predict(policy, stats, obs, device)
                for action in chunk[:args.exec_steps]:
                    if not np.isfinite(action).all():
                        raise ValueError(f"Non-finite model action at seed {scene_seed}")
                    action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
                    obs, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    steps += 1
                    coverage = float(info["coverage"])
                    max_coverage = max(max_coverage, coverage)
                    if frames is not None:
                        frames.append(cv2.resize(env.render(), (512, 512)))
                    if done:
                        break
            if frames is not None:
                path = args.record_dir / f"episode_{index:03d}_seed_{scene_seed}.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         float(env.metadata.get("render_fps", 10)), (512, 512))
                try:
                    for frame in frames:
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                finally:
                    writer.release()
            result = {"seed": scene_seed, "success": bool(info["is_success"]),
                      "max_cov": max_coverage, "final_cov": coverage, "steps": steps}
            outcomes.append(result)
            print(f"episode {index + 1}/{args.episodes}: seed={scene_seed} "
                  f"success={result['success']} max_cov={max_coverage:.3f} "
                  f"final_cov={coverage:.3f} steps={steps}", flush=True)
    finally:
        env.close()
    summary = {"epoch": checkpoint["epoch"],
           "success_rate": float(np.mean([row["success"] for row in outcomes])),
           "mean_max_coverage": float(np.mean([row["max_cov"] for row in outcomes])),
           "mean_final_coverage": float(np.mean([row["final_cov"] for row in outcomes]))}
    print(summary)
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps({"checkpoint": str(args.ckpt.resolve()),
            "seed": args.seed, "exec_steps": args.exec_steps,
            "summary": summary, "episodes": outcomes}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

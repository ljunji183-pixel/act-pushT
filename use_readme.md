# 使用指南（use_readme）

所有命令都在 `D:\PythonProject\act-pusht` 目录下用 PowerShell 执行，Python 统一用 lerobot 环境的解释器：

```powershell
& "C:\Users\ljj36\miniconda3\envs\lerobot\python.exe" .\xxx.py ...
```

数据已经链接好了：`data\pusht` 指向老项目里的同一份数据（junction，不占额外磁盘空间），所以下面的命令都不用再写 `--project`。

## 1. 从零开始训练（train_scratch.py）

```powershell
& "C:\Users\ljj36\miniconda3\envs\lerobot\python.exe" .\train_scratch.py --epochs 100
```

常用可选参数（不写就用默认值）：

- `--epochs 100`：总训练轮数，默认 200
- `--batch-size 8`：批大小，默认 8
- `--chunk-size 16`：每次预测的动作步数，默认 16
- `--lr 1e-4`：学习率，默认 1e-4
- `--beta-kl 0.01`：KL 损失权重，默认 0.01
- `--free-bits 0.01`：free bits 下限，默认 0.01
- `--seed 42`：随机种子，默认 42

输出会保存在 `runs_sincos_scratch/<时间戳>_scratch/`，里面有：

- `best.pt`：验证集 loss 最低的 checkpoint
- `last.pt`：最后一个 epoch 的 checkpoint
- `loss_log.csv`：每个 epoch 的训练/验证指标

第一次训练会先解码全部视频并构建图像缓存（`.cache_sincos_scratch/`），这一步比较慢，但只发生一次，之后所有 epoch 直接命中缓存，速度会快很多。如果以后换了数据或 episode 划分方式，缓存会自动重建。

## 2. 在已有 checkpoint 基础上继续训练（train_stable.py）

```powershell
& "C:\Users\ljj36\miniconda3\envs\lerobot\python.exe" .\train_stable.py --resume .\runs_sincos\<你的run目录>\last.pt --additional-epochs 10 --lr 1e-5
```

说明：

- `--resume` 接上一个 checkpoint 文件，可以用 `last.pt`（接着上次的优化器状态继续，最常见）或 `best.pt`（加载验证 loss 最低的那版权重重新开始）。第一次从零训练产生的是 `runs_sincos_scratch/` 下的路径；之后每次续训产生的输出在 `runs_sincos/<时间戳>_from_eXXX/` 下，XXX 表示从哪个 epoch 接上的。
- `--additional-epochs 10` 表示在这个基础上**再**训 10 轮，不是训到第 10 轮。
- `--lr 1e-5` 是本轮新创建的优化器使用的学习率。续训时通常要比从零训练时小一个量级，想更稳可以降到 `5e-6` 或 `1e-6`。
- 其他可选参数：`--batch-size`、`--beta-kl`、`--weight-decay`、`--result-json`。

续训输出目录同样包含 `best.pt` / `last.pt` / `loss_log.csv`，目录名里的 `from_eXXX` 就是从第 XXX epoch 接上的意思。

## 3. 评估（evaluate_stable.py）

```powershell
& "C:\Users\ljj36\miniconda3\envs\lerobot\python.exe" .\evaluate_stable.py --ckpt .\runs_sincos\<run目录>\best.pt --episodes 20 --seed 1000 --exec-steps 8 --result-json .\eval_result.json
```

参数说明：

- `--ckpt`：要评估的 checkpoint 路径，必填，`best.pt` 或 `last.pt` 都可以。
- `--episodes 20`：评测多少局，默认 50。调试阶段建议先用 10~20，最终报告再跑满。
- `--seed 1000`：rollout 的随机种子。**对比两个模型时务必用同一个 seed**，否则结果不可比。
- `--exec-steps 8`：每次预测 16 步动作块后，闭环执行其中前几步再重新观测。默认 8；这个值越小闭环纠错越强，太大模型会闭眼走很多步。
- `--record-dir .\videos_xxx`：可选，把每一局存成 mp4 方便人工检查失败原因。
- `--result-json .\eval_result.json`：可选，把逐局 success / max_coverage / final_coverage 写入 json。

运行结束后会在终端打印：

{'success_rate': 0.6, 'mean_max_coverage': 0.85, 'mean_final_coverage': 0.80}

- `success_rate`：成功率，覆盖率达到 0.95 的局数占比。
- `mean_max_coverage`：平均最高覆盖率。
- `mean_final_coverage`：平均最终覆盖率。

## 4. 训练时怎么看曲线（plot_curves.py）

训练过程中，另开一个 PowerShell 窗口，让曲线图实时刷新：

```powershell
& "C:\Users\ljj36\miniconda3\envs\lerobot\python.exe" .\plot_curves.py .\runs_sincos\<run目录> --watch-seconds 30
```

- 不加 `--watch-seconds 30` 就只画一次图，画完退出。
- 加 `--show` 会额外弹出窗口实时显示。
- 图会保存到 run 目录下的 `loss_curves.png`，训练结束后也可以直接打开看。

图里一共有 4 张子图：

| 子图 | 内容 | 怎么看 |
| --- | --- | --- |
| Action loss | train / val 动作 L1 | val 持续下降 = 还在学；val 上升 + train 下降 = 过拟合 |
| KL / total loss | train KL 与总损失 | KL 突然上涨通常是 z 开始携带信息；一直为 0 说明 z 塌缩 |
| Validation raw MAE | 原始坐标单位的验证误差 | 最直观的离真值多远，越小越好 |
| sin/cos time gates | encoder / decoder 时间编码门控 | 从 0 缓慢上升 = 模型在采纳正余弦时间信号；一直是 0 = 没用到 |

训练过程中重点看 `val_action` 和 `val_raw_action` 是否还在下降；门控曲线用来确认正余弦时间编码是不是真的在被模型使用。

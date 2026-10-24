# 下载 lerobot/pusht 数据集到 <repo>/data/pusht
# 国内网络可能需要代理：$env:HF_ENDPOINT = "https://hf-mirror.com"

from pathlib import Path

from huggingface_hub import snapshot_download

# 本文件在 scripts/ 下，数据要落到仓库根目录的 data/pusht
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data" / "pusht"
print("数据保存位置:", data_dir)

snapshot_download(repo_id="lerobot/pusht", repo_type="dataset", local_dir=data_dir)
print("数据集下载完成:", data_dir)

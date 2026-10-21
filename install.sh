#!/usr/bin/env bash
# 一键安装 / 修复训练环境：Python venv + PyTorch + sd-scripts + musubi-tuner + 反推依赖。
# 需要联网，PyTorch 约 2.7 GB。可以重复执行，装好的部分会跳过。
#   ./install.sh
#   ./install.sh --torch cu128
#   ./install.sh --repair           修复依赖
#   PYTHON=/root/miniconda3/bin/python ./install.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${PYTHON:-}" ]; then
  if [ -x python/bin/python ]; then PYTHON="$PWD/python/bin/python"; else PYTHON="python3"; fi
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "找不到 $PYTHON。先装 Python 3.10+，或用 PYTHON=/path/to/python3 ./install.sh 指定。"
  exit 1
fi

exec "$PYTHON" scripts/setup_env.py "$@"

#!/usr/bin/env bash
# 前台启动 LoRA 训练器（Linux / 云主机）。
#   ./run.sh                 默认监听 0.0.0.0:28100
#   PORT=8888 ./run.sh       换端口
#   LORA_TRAINER_HOST=127.0.0.1 ./run.sh   只听本机（配合 SSH 隧道）
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-28100}"
HOST="${LORA_TRAINER_HOST:-0.0.0.0}"
# 优先用项目自带的 Python（python/，部署时用 conda / venv 建的），没有就用系统的
if [ -z "${PYTHON:-}" ]; then
  if [ -x python/bin/python ]; then PYTHON="$PWD/python/bin/python"; else PYTHON="python3"; fi
fi

if [ ! -x env/bin/python ]; then
  echo "训练环境还没安装：先运行 ./install.sh（只装训练器界面本身的话也可以直接启动，训练前再装）"
fi

exec "$PYTHON" app/main.py --host "$HOST" --port "$PORT" --no-browser

#!/usr/bin/env bash
# 后台启动 LoRA 训练器，日志写 trainer.log，PID 写 trainer.pid。
#   ./start.sh
#   PORT=28100 HOST=0.0.0.0 ./start.sh
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-28100}"
HOST="${HOST:-0.0.0.0}"
if [ -z "${PYTHON:-}" ]; then
  if [ -x python/bin/python ]; then PYTHON="$PWD/python/bin/python"; else PYTHON="python3"; fi
fi
LOG="${LOG:-trainer.log}"
PIDFILE="trainer.pid"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
  echo "训练器已经在运行（PID $(cat "$PIDFILE")）：http://127.0.0.1:$PORT/"
  exit 0
fi

if [ ! -x env/bin/python ]; then
  echo "提示：训练环境还没装（训练前需要）。装：./install.sh"
fi

nohup env LORA_TRAINER_HOST="$HOST" PYTHONUTF8=1 PYTHONUNBUFFERED=1 \
  "$PYTHON" app/main.py --host "$HOST" --port "$PORT" --no-browser >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"

for _ in $(seq 1 30); do
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "启动失败，日志最后几行："
    tail -n 30 "$LOG"
    rm -f "$PIDFILE"
    exit 1
  fi
  if "$PYTHON" - "$PORT" <<'PY' 2>/dev/null
import sys, urllib.request
urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/index.html", timeout=1).read(100)
PY
  then
    break
  fi
  sleep 0.5
done

echo "LoRA 训练器已启动"
echo "  本机：http://127.0.0.1:$PORT/"
echo "  局域网 / 容器内：http://<本机IP>:$PORT/"
echo "  日志：$(pwd)/$LOG   停止：./stop.sh"
if [ "$HOST" = "0.0.0.0" ]; then
  echo "  云主机上从自己电脑访问：ssh -N -L $PORT:127.0.0.1:$PORT -p <SSH端口> root@<SSH地址>"
  echo "  然后浏览器打开 http://127.0.0.1:$PORT/（云平台支持自定义端口映射的话也可以直接映射 $PORT）"
fi

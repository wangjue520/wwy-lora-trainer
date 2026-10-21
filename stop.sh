#!/usr/bin/env bash
# 停止后台运行的训练器（正在训练的任务会被中断；想保住进度先在界面里点「冻结」）。
set -uo pipefail
cd "$(dirname "$0")"

PIDFILE="trainer.pid"
PID=""
if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE" 2>/dev/null || true)"
fi

if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
  kill -TERM "$PID" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 0.3
  done
  kill -9 "$PID" 2>/dev/null || true
  echo "已停止（PID $PID）"
  rm -f "$PIDFILE"
  exit 0
fi

# 没有 PID 文件（比如用 ./run.sh 前台启动的）：按命令行找
PIDS="$(pgrep -f "$(pwd)/app/main.py" || true)"
if [ -n "$PIDS" ]; then
  kill -TERM $PIDS 2>/dev/null || true
  sleep 1
  kill -9 $PIDS 2>/dev/null || true
  echo "已停止（PID $(echo $PIDS | tr '\n' ' ')）"
else
  echo "训练器没有在运行"
fi
rm -f "$PIDFILE"

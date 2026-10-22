#!/usr/bin/env bash
# 仙宫云上启动训练器：后台运行，监听 0.0.0.0 的 28100、7860 和 80（控制台「打开webui」按钮开的是 7860）。
cd "$(dirname "$0")/../.."
export LORA_TRAINER_EXTRA_PORTS="${LORA_TRAINER_EXTRA_PORTS:-7860 80}"
HOST=0.0.0.0 PORT="${PORT:-28100}" ./start.sh
ID="${XGC_INSTANCE_ID:-$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^XGC_INSTANCE_ID=//p')}"
if [ -n "$ID" ]; then
    echo "  在自己的浏览器里打开：https://$ID-28100.container.x-gpu.com/"
    echo "  （控制台「打开webui」开的是 https://$ID-7860.container.x-gpu.com/，同一个训练器）"
fi

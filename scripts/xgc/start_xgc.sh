#!/usr/bin/env bash
# 仙宫云上启动训练器：后台运行，监听 0.0.0.0 的 28100 和 80（控制台「打开webui」按钮开的是 80）。
cd "$(dirname "$0")/../.."
export LORA_TRAINER_EXTRA_PORTS="${LORA_TRAINER_EXTRA_PORTS:-80}"
HOST=0.0.0.0 PORT="${PORT:-28100}" ./start.sh
echo "  仙宫云外网地址：控制台的「打开webui」，或 OS 桌面的「LoRA训练器」"

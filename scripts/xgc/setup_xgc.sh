#!/usr/bin/env bash
# 仙宫云（xiangongyun.com）镜像用：仙宫云 OS 桌面图标 + 开机自动启动训练器。可以重复执行。
#   bash scripts/xgc/setup_xgc.sh
# 训练器同时监听 28100 和 80：控制台「打开webui」按钮固定开 80 端口。界面在自己的浏览器里开，不放进 OS 窗口（OS 里很卡）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ICONS="$ROOT/scripts/xgc/desktop"
DESK=/.xgcos/desktop
OLD=/root/.xgc_old    # 基础镜像自带、和训练器无关的启动项 / 图标挪到这里，不删

# ---------- 开机自启：仙宫云的 /scripts/start.sh 开机时会执行 /scripts/start.d/ 里的每个脚本 ----------
mkdir -p /scripts/start.d "$OLD"
for f in /scripts/start.d/start_app.sh; do   # 基础镜像留下的「开机跑 /root/run.sh」
    [ -f "$f" ] && mv -f "$f" "$OLD/"
done
cat > /scripts/start.d/wwy_lora_trainer.sh <<EOF
#!/bin/bash
bash "$ROOT/scripts/xgc/start_xgc.sh" >/dev/null 2>&1
EOF
chmod +x /scripts/start.d/wwy_lora_trainer.sh "$ROOT/scripts/xgc/start_xgc.sh" "$ROOT"/*.sh

# ---------- 仙宫云 OS 桌面 ----------
if [ -d "$DESK" ]; then
    # 基础镜像里给别的应用（7860 端口）用的图标
    for a in open.app restart.app check.app exitapp.app backend.app app-outputs.app; do
        [ -d "$DESK/$a" ] && rm -rf "$OLD/$a" && mv "$DESK/$a" "$OLD/"
    done
    if [ -f "$DESK/使用说明.txt" ] && ! grep -q "LoRA" "$DESK/使用说明.txt"; then
        mv -f "$DESK/使用说明.txt" "$OLD/"
    fi

    app() {   # app <目录名> <显示名> <图标> <类型> <props 内容>
        local d="$DESK/$1.app"
        rm -rf "$d" && mkdir -p "$d"
        cp "$ICONS/$3.png" "$d/$3.png"
        printf 'name: %s\ntitle: %s\nicon: %s.png\ntype: %s\nprops:\n%s\n' "$2" "$2" "$3" "$4" "$5" > "$d/info.yaml"
    }
    rm -rf "$DESK/wwy-lora.app"   # 界面不在 OS 里开（OS 里很卡），用自己浏览器开端口地址
    app wwy-lora-start "启动训练器" lora terminal '  exec: "bash main.sh"'
    printf 'bash "%s/scripts/xgc/start_xgc.sh"\n' "$ROOT" > "$DESK/wwy-lora-start.app/main.sh"
    app wwy-lora-stop "停止训练器" stop terminal '  exec: "bash main.sh"'
    printf 'bash "%s/stop.sh"\n' "$ROOT" > "$DESK/wwy-lora-stop.app/main.sh"
    app wwy-lora-log "训练器日志" log terminal '  exec: "bash main.sh"'
    printf 'tail -n 200 -f "%s/trainer.log"\n' "$ROOT" > "$DESK/wwy-lora-log.app/main.sh"
    app wwy-lora-datasets "训练集" datasets filemanager "    path: $ROOT/datasets"
    app wwy-lora-output "训练输出" output filemanager "    path: $ROOT/output"
    app wwy-lora-models "模型文件" models filemanager "    path: $ROOT/models"
    echo "仙宫云 OS 桌面图标已更新（桌面空白处右键 → 刷新）"
else
    echo "没有 $DESK（不是仙宫云 OS 镜像），跳过桌面图标"
fi

mkdir -p "$ROOT/datasets" "$ROOT/output" "$ROOT/models"
echo "开机自启已设置：/scripts/start.d/wwy_lora_trainer.sh"

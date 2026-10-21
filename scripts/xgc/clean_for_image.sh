#!/usr/bin/env bash
# 保存仙宫云镜像前清理：测试产物、队列历史、日志、缓存，模型只留 KEEP_MODELS 里的。
#   bash scripts/xgc/clean_for_image.sh          只列出会删什么
#   bash scripts/xgc/clean_for_image.sh --yes    真的删
# 默认保留 Anima 三件套（models/base 里）和全部打标模型（models/taggers）；
# 想多留别的：KEEP_MODELS="anima-base-v1.0.safetensors qwen_3_06b_base.safetensors ..." bash ...
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KEEP_MODELS="${KEEP_MODELS:-anima-base-v1.0.safetensors qwen_3_06b_base.safetensors qwen_image_vae.safetensors}"
DO=0; [ "${1:-}" = "--yes" ] && DO=1

del() {   # del <路径>...
    for p in "$@"; do
        [ -e "$p" ] || [ -L "$p" ] || continue
        echo "  $(du -sh "$p" 2>/dev/null | cut -f1)	$p"
        [ $DO = 1 ] && rm -rf "$p"
    done
}

echo "== 训练产物、任务、队列、日志"
del "$ROOT"/output/* "$ROOT"/data/tasks/* "$ROOT"/data/caption_backups/* \
    "$ROOT"/data/queue.json "$ROOT"/data/draft.json "$ROOT"/data/installs.json "$ROOT"/data/net_stats.json \
    "$ROOT"/trainer.log "$ROOT"/trainer.pid
echo "== 训练集（镜像里不带用户数据）"
del "$ROOT"/datasets/*
echo "== 模型（保留：$KEEP_MODELS）"
for f in "$ROOT"/models/base/*; do
    case " $KEEP_MODELS " in *" $(basename "$f") "*) ;; *) del "$f" ;; esac
done
del "$ROOT"/models/moss_native "$ROOT"/models/moss_tokenizer
echo "== 缓存"
del /root/.cache/pip /root/.cache/huggingface/hub /tmp/pip-* "$ROOT"/app/__pycache__ "$ROOT"/app/*/__pycache__
[ $DO = 1 ] && command -v conda >/dev/null && conda clean -y --all >/dev/null 2>&1

if [ $DO = 1 ]; then
    mkdir -p "$ROOT/output" "$ROOT/datasets" "$ROOT/data/tasks"
    echo "清理完成。"; df -h / | tail -1
else
    echo "以上是会删除的内容（没有删）。确认后加 --yes 再运行。"
fi

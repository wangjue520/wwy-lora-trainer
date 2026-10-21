# -*- coding: utf-8 -*-
"""标签翻译子进程（在训练环境里运行）：用反推用的 Qwen3-VL 做纯文字翻译，模型只加载一次。

用法：python translate_worker.py <config.json>；每翻完一条输出「@@{"src": 原文, "zh": 译文}」。"""
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tag_worker import VLM, apply_vram_limit, emit  # noqa: E402


def main():
    cfg = json.load(open(sys.argv[1], encoding="utf-8"))
    apply_vram_limit()
    vlm = VLM(cfg.get("vlm") or {}, cfg["models_dir"])
    torch, model, proc = vlm.torch, vlm.model, vlm.proc
    tok = getattr(proc, "tokenizer", proc)
    emit(text=f"模型已加载，开始翻译 {len(cfg['texts'])} 条")
    for src in cfg["texts"]:
        try:
            msgs = [{"role": "user", "content": [{"type": "text", "text": cfg["prompt"] + src}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tok([text], return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=min(2048, len(src) * 2 + 64), do_sample=False,
                                     repetition_penalty=1.05)
            zh = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            if "</think>" in zh:
                zh = zh.split("</think>", 1)[1]
            emit(src=src, zh=zh.strip())
            del inputs, out
        except Exception as e:
            print(traceback.format_exc(), flush=True)
            emit(text=f"[失败] {str(e)[:200]}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

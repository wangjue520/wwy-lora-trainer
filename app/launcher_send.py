# -*- coding: utf-8 -*-
"""把训练好的 LoRA 送到 WWY 启动器：共享模型库或某个实例（WebUI / ComfyUI）的 LoRA 文件夹。

和启动器的约定保持一致（启动器 app/model_library.py）：
- 共享模型库：<库>/loras/<底模>/<类型>（启动器「LoRA 自动整理」开着时按它的模板 base/type 放；关着就放 loras 根目录）
- 预览图：<名字>.preview.png（用训练中测试出的那一轮的示例图）
- 模型信息：<名字>.civitai.info（启动器读 trainedWords 当触发词、baseModel 当底模、tags 当类型）
- WebUI 的 LoRA 卡片：<名字>.json（activation text = 触发词）"""
import json
import os
import re
import shutil

import scanner
import tasks
import triggers

TYPES = [("角色", "character"), ("画风", "style"), ("服装", "clothing"), ("姿势动作", "poses"), ("概念", "concept"),
         ("场景", "background"), ("工具", "tool"), ("其他", "")]
_BAD = '<>:"/\\|?*'


def _clean(name):
    name = "".join("_" if c in _BAD else c for c in str(name)).strip(" .")
    return name[:40] or "未分类"


def _configs():
    """最近用过的启动器排前面，每个启动器的配置。"""
    out = []
    for root in scanner._launchers_by_recent():
        for p in (os.path.join(root, "launcher_config.json"), os.path.join(root, "launcher_data", "launcher_config.json")):
            try:
                with open(p, encoding="utf-8") as f:
                    out.append((root, json.load(f)))
                break
            except (OSError, ValueError):
                continue
    return out


def targets():
    """可以送去的位置：[{id, kind, label, dir, organize, template}]，同一个文件夹只出现一次。"""
    res, seen = [], set()

    def add(kind, label, d, **kw):
        k = os.path.normcase(os.path.abspath(d))
        if k in seen:
            return
        seen.add(k)
        res.append(dict(id=str(len(res)), kind=kind, label=label, dir=d, **kw))

    for root, cfg in _configs():
        lib = cfg.get("model_library_path")
        if lib and cfg.get("model_library_enabled", True) and os.path.isdir(lib):
            add("library", f"共享模型库 · {lib}", os.path.join(lib, "loras"),
                organize=bool(cfg.get("lora_organize_enabled", True)), template=cfg.get("lora_organize_template") or "base/type")
        for inst in cfg.get("instances") or []:
            r = inst.get("webui_root")
            if not r or not os.path.isdir(r):
                continue
            comfy = inst.get("webui_branch") == "comfyui" or os.path.isfile(os.path.join(r, "main.py"))
            d = os.path.join(r, "models", "loras" if comfy else "Lora")
            add("comfyui" if comfy else "webui", f"{inst.get('name') or ('ComfyUI' if comfy else 'WebUI')} · {r}", d,
                organize=False, template="", instance=inst.get("name") or "")
    return res


def _subdir(t, base, typ):
    if not t.get("organize"):
        return ""
    parts = []
    for key in (t.get("template") or "base/type").split("/"):
        if key == "base":
            parts.append(base or "未知底模")
        elif key == "type":
            parts.append(typ or "未分类")
        elif key == "instance":
            parts.append(t.get("instance") or "实例")
    return os.path.join(*[_clean(p) for p in parts]) if parts else ""


def _epoch_of(task, lora_name):
    """LoRA 文件对应第几轮：name-000004.safetensors → 4；最终文件 → 总轮数。"""
    m = re.search(r"-(\d{6})\.safetensors$", lora_name)
    if m:
        return int(m.group(1))
    try:
        return int(task.get("epochs_total") or str((task.get("progress") or {}).get("epoch") or "").split("/")[1])
    except (ValueError, IndexError):
        return int(task["params"].get("epochs") or 0)


def preview_for(task, lora_name):
    """这个 LoRA 那一轮的示例图（第一条提示词）；按步数出图时找最接近那一轮结束的一组。"""
    samples = tasks.list_samples(task)
    if not samples:
        return None
    ep = _epoch_of(task, lora_name)
    eps = [s for s in samples if s["epoch"] and s["num"] == ep]
    if not eps:
        spe = tasks._spe(task)
        steps = [s for s in samples if not s["epoch"] and s["num"] > 0]
        if spe and steps:
            want = ep * spe
            best = min(steps, key=lambda s: abs(s["num"] - want))
            eps = [s for s in steps if s["num"] == best["num"]]
    if not eps:
        return None
    eps.sort(key=lambda s: (s["prompt"], -int(s["time"])))   # 同一轮同一条提示词有好几张时用最新的
    return eps[0]["file"]


def info(task_id, lora_path):
    t = tasks.get(task_id)
    if not t:
        raise ValueError("找不到这个任务")
    tr = triggers.get(t["params"].get("dataset_dir")) or {}
    typ = ("角色" if tr.get("features") or tr.get("outfit") else "画风") if tr else ""
    return {"targets": targets(), "preview": preview_for(t, os.path.basename(lora_path)), "trigger": tr.get("prompt") or "",
            "type": typ, "types": [x[0] for x in TYPES], "base": _base_label(t)}


def _base_label(t):
    import models
    spec = models.get(t["model_type"])
    return spec.base_label(t) if spec else t["model_type"]


def send(task_id, lora_path, target_ids, typ="", trigger="", preview=None, overwrite=False):
    t = tasks.get(task_id)
    if not t:
        raise ValueError("找不到这个任务")
    if not os.path.isfile(lora_path):
        raise ValueError("LoRA 文件不存在：" + lora_path)
    all_t = {x["id"]: x for x in targets()}
    chosen = [all_t[i] for i in target_ids if i in all_t]
    if not chosen:
        raise ValueError("请选择要送到哪里")
    base = _base_label(t)
    name = os.path.splitext(os.path.basename(lora_path))[0]
    tag = dict(TYPES).get(typ, "")
    words = [w.strip() for w in [trigger] if w and w.strip()]
    done, exists = [], []
    for x in chosen:
        d = os.path.join(x["dir"], _subdir(x, base, typ))
        dst = os.path.join(d, name + ".safetensors")
        if os.path.exists(dst) and not overwrite:
            if os.path.getsize(dst) == os.path.getsize(lora_path):
                done.append({"target": x["label"], "path": dst, "skipped": True})
            else:
                exists.append(dst)
            continue
        os.makedirs(d, exist_ok=True)
        shutil.copy2(lora_path, dst + ".part")
        os.replace(dst + ".part", dst)
        stem = os.path.join(d, name)
        if preview and os.path.isfile(preview):
            shutil.copyfile(preview, stem + ".preview.png")
        civ = {"modelName": name, "modelType": "LORA", "baseModel": base, "fileName": name + ".safetensors",
               "trainedWords": words, "trainedWordsSource": "WWY LoRA 训练器", "tags": [tag] if tag else [],
               "description": f"WWY LoRA 训练器训练（{base}，任务 {t['name']}）"}
        with open(stem + ".civitai.info", "w", encoding="utf-8") as f:
            json.dump(civ, f, ensure_ascii=False, indent=2)
        if x["kind"] in ("webui", "library"):   # Forge / A1111 的 LoRA 卡片读这个
            card = {"description": civ["description"], "activation text": ", ".join(words), "preferred weight": 1, "notes": ""}
            with open(stem + ".json", "w", encoding="utf-8") as f:
                json.dump(card, f, ensure_ascii=False, indent=2)
        done.append({"target": x["label"], "path": dst})
    return {"done": done, "exists": exists}

# -*- coding: utf-8 -*-
"""导出当前配置（启动训练页「导出配置」）：一个 TOML 文件，两部分——

  - 上面：这份配置实际会传给 sd-scripts / musubi-tuner 的训练参数（参数名和 kohya、秋叶 lora-scripts 一样，别的训练器也看得懂）
  - 最后的 [wwy_lora_trainer]：本训练器的完整配置（底模、模型文件、全部参数、测试提示词），「导入外部配置」读回来不丢东西

sd-scripts 用 --config_file 读这个文件时会把各小节摊平，[wwy_lora_trainer] 里的两个键不是它的参数，会被忽略。"""
import json
import os
import re
import tempfile
import time

import models

# 不导出：本机临时文件（数据集配置、测试提示词、日志目录在任务目录里，换台电脑就没了）、密钥
_DROP = {"dataset_config", "sample_prompts", "logging_dir", "wandb_api_key", "huggingface_token"}
_LIST = {"network_args", "optimizer_args", "lr_scheduler_args"}
_STR = {"output_name", "log_prefix", "log_tracker_name", "wandb_run_name", "caption_extension"}


def _val(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_val(x) for x in v) + "]"
    return json.dumps(str(v), ensure_ascii=False)   # JSON 字符串的转义和 TOML 基本字符串兼容


def _num(k, s):
    if k in _STR or not isinstance(s, str):
        return s
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if re.fullmatch(r"-?(\d+\.\d*|\.\d+|\d+)([eE]-?\d+)?", s):
        return float(s)
    return s


def _args_kv(cmd):
    """训练命令 → {参数名: 值}。没有值的开关是 true；network_args 这类多值参数是列表。"""
    kv, key = {}, None
    for tok in cmd:
        if tok.startswith("--"):
            key = tok[2:]
            kv[key] = [] if key in _LIST else True
        elif key is None:
            continue   # 脚本名
        elif key in _LIST:
            kv[key].append(tok)
        elif kv[key] is True:
            kv[key] = _num(key, tok)
        else:
            kv[key] = (kv[key] if isinstance(kv[key], list) else [kv[key]]) + [tok]
    return {k: v for k, v in kv.items() if k not in _DROP}


def _dataset_kv(path, dataset_dir):
    """dataset.toml → 秋叶 / kohya 单文件写法的几项（train_data_dir、resolution、分桶……）。"""
    import cfgimport
    try:
        with open(path, encoding="utf-8") as f:
            d = cfgimport._toml(f.read())
    except (OSError, ValueError):
        return {}, []
    g = d.get("general") or {}
    dss = d.get("datasets") or [{}]
    # sd-scripts：设置在 [[datasets]]、每个文件夹一个 [[datasets.subsets]]；musubi-tuner：设置在 [general]、每个文件夹一个 [[datasets]]
    ds = dict(g, **dss[0]) if dss[0].get("subsets") else g
    subs = [dict(s, image_dir=s.get("image_dir") or s.get("image_directory"))
            for s in (dss[0].get("subsets") if dss[0].get("subsets") else dss) if not s.get("is_reg")]
    subs = [s for s in subs if s.get("image_dir")]
    out = {}
    if dataset_dir:
        out["train_data_dir"] = dataset_dir
    if ds.get("resolution"):
        out["resolution"] = ",".join(str(x) for x in ds["resolution"])
    for k in ("enable_bucket", "min_bucket_reso", "max_bucket_reso", "bucket_reso_steps", "bucket_no_upscale"):
        if k in ds:
            out[k] = ds[k]
    if "batch_size" in ds:
        out["train_batch_size"] = ds["batch_size"]
    for k in ("caption_extension", "shuffle_caption", "keep_tokens", "keep_tokens_separator"):
        if k in g:
            out[k] = g[k]
    return out, subs


def export(draft):
    """返回 {filename, text}。"""
    spec = models.get(draft.get("model_type"))
    if not spec:
        raise ValueError("请先在「模型设置」选择底模")
    p = dict(spec.defaults())
    p.update(draft.get("params") or {})
    name = p.get("output_name") or "lora"
    out_dir = draft.get("output_dir") or ""
    task = {"id": "export", "model_type": spec.id, "params": p, "models": dict(draft.get("models") or {}),
            "sample": draft.get("sample") or {}, "output_dir": os.path.join(out_dir, name) if draft.get("subdir_per_task", True) else out_dir}
    head = [f"# WWY LoRA 训练器导出的配置 · {spec.label}（{spec.engine}）· {time.strftime('%Y-%m-%d %H:%M')}",
            "# 上面是训练参数（sd-scripts / musubi-tuner 的参数名）；最后的 [wwy_lora_trainer] 是本训练器的完整配置，「导入外部配置」读回来不丢东西。"]
    body, err = [], ""
    if spec.engine in ("sd-scripts", "musubi-tuner"):
        try:
            with tempfile.TemporaryDirectory() as td:
                steps = spec.build_steps(task, td)
                train = next(s for s in steps if s.get("kind") == "train")
                kv = _args_kv(train["cmd"])
                ds, subs = _dataset_kv(os.path.join(td, "dataset.toml"), p.get("dataset_dir"))
        except Exception as e:   # 参数有问题（开训前检查会报的那种）也照样导出本训练器的配置
            err = str(e) or e.__class__.__name__
        else:
            reps = sorted({s.get("num_repeats") for s in subs if s.get("num_repeats")})
            if reps:
                body.append(f"# 每张图重复 {'/'.join(str(r) for r in reps)} 次：本训练器直接设；用 kohya / 秋叶训练时把图片放进「{reps[0]}_名字」这样的子文件夹")
            for k, v in list(ds.items()) + [(k, v) for k, v in kv.items() if k not in ds]:
                body.append(f"{k} = {_val(v)}")
    if err:
        body.append(f"# 训练参数没有导出：{err}")
    native = {k: draft.get(k) for k in ("model_type", "models", "params", "sample", "output_dir", "subdir_per_task")}
    native["params"] = {k: v for k, v in p.items() if k not in ("wandb_api_key",)}   # 密钥不跟着配置文件到处传
    tail = ["", "[wwy_lora_trainer]", f"model_type = {_val(spec.id)}", f"config = {_val(json.dumps(native, ensure_ascii=False))}"]
    safe = re.sub(r'[\\/:*?"<>|]+', "_", name)
    return {"filename": f"{safe}.toml", "text": "\n".join(head + [""] + body + tail) + "\n"}

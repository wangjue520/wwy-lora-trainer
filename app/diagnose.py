# -*- coding: utf-8 -*-
"""报错翻译 + 自动修复。

两个入口：
  preflight(task)        开训前体检：训练集（空标签、空图片、损坏图片）、模型文件完整性、显存占用、硬盘空间、环境
  diagnose(log, task)    训练失败后，按日志里的报错特征给出中文原因和可选的修复动作

修复动作都只是「建议」，前端会把要做的事（删哪些文件、结束哪些进程……）列给用户确认后，
才调用 apply_fix() 真正执行。"""
import os
import re
import shutil

import gpu
import models
import paths
import scanner
import settings
from models.base import IMAGE_EXTS

_MAGIC = {
    b"\x89PNG": "png", b"\xff\xd8\xff": "jpg", b"RIFF": "webp", b"BM": "bmp",
}


def _image_ok(p):
    try:
        if os.path.getsize(p) < 64:
            return False
        with open(p, "rb") as f:
            head = f.read(16)
        if head.startswith(b"RIFF") and head[8:12] != b"WEBP":
            return False
        return any(head.startswith(k) for k in _MAGIC)
    except OSError:
        return False


def fix(*args, danger=False, **data):
    """fix(动作id, 按钮文字, 说明, **附带数据)。用位置参数，避免和下载信息里的 label 等键重名。"""
    fid, label, desc = args
    return {"id": fid, "label": label, "desc": desc, "danger": danger, "data": data}


# ======================= 开训前体检 =======================
def check_dataset(dataset_dir, caption_ext=".txt"):
    """返回 (issues, stats)。"""
    issues = []
    if not dataset_dir:
        return [{"level": "error", "code": "no_dataset", "title": "没有设置训练集位置",
                 "reason": "请在「高级参数」里选择训练集文件夹。", "fixes": []}], {}
    if not os.path.isdir(dataset_dir):
        return [{"level": "error", "code": "no_dataset", "title": "训练集文件夹不存在",
                 "reason": f"找不到 {dataset_dir}，可能被移动或删除了。", "fixes": []}], {}
    imgs, empty_caps, missing_caps, bad_imgs, orphan_caps = [], [], [], [], []
    for root, dirs, files in os.walk(dataset_dir):
        dirs[:] = [d for d in dirs if not d.startswith(("_", "."))]
        names = set(files)
        for fn in files:
            p = os.path.join(root, fn)
            low = fn.lower()
            if low.endswith(IMAGE_EXTS):
                imgs.append(p)
                if not _image_ok(p):
                    bad_imgs.append(p)
                    continue
                cap = os.path.splitext(fn)[0] + caption_ext
                if cap not in names:
                    missing_caps.append(p)
                else:
                    cp = os.path.join(root, cap)
                    try:
                        if not open(cp, "r", encoding="utf-8", errors="ignore").read().strip():
                            empty_caps.append(cp)
                    except OSError:
                        empty_caps.append(cp)
            elif low.endswith(caption_ext):
                stem = os.path.splitext(fn)[0]
                if not any((stem + e) in names or (stem + e.upper()) in names for e in IMAGE_EXTS):
                    if os.path.getsize(p) == 0:
                        orphan_caps.append(p)
    stats = {"images": len(imgs), "empty_caps": len(empty_caps), "missing_caps": len(missing_caps), "bad_imgs": len(bad_imgs)}
    if not imgs:
        issues.append({"level": "error", "code": "empty_dataset", "title": "训练集里没有图片",
                       "reason": f"{dataset_dir} 里没有找到 png / jpg / webp / bmp 图片。", "fixes": []})
        return issues, stats
    if empty_caps or bad_imgs or orphan_caps:
        files = [{"path": x, "what": "空标签文件"} for x in empty_caps] + \
                [{"path": x, "what": "损坏或空的图片"} for x in bad_imgs] + \
                [{"path": x, "what": "没有对应图片的空标签"} for x in orphan_caps]
        parts = []
        if empty_caps:
            parts.append(f"{len(empty_caps)} 个标签文件是空的")
        if bad_imgs:
            parts.append(f"{len(bad_imgs)} 张图片损坏或是空文件")
        if orphan_caps:
            parts.append(f"{len(orphan_caps)} 个多余的空标签")
        issues.append({
            "level": "error" if bad_imgs else "warn", "code": "dirty_dataset", "title": "训练集里有空文件 / 损坏文件",
            "reason": "，".join(parts) + "。空标签会让这些图在训练时没有任何文字描述；损坏的图片会让训练直接报错。",
            "fixes": [fix("clean_dataset", "清理这些文件",
                          "删除空标签文件和空图片；损坏的图片移到训练集里的「_已移除」文件夹（可以找回）。",
                          danger=True, files=files),
                      fix("ignore", "忽略，继续训练", "不处理，按现状训练。")] if not bad_imgs else
                     [fix("clean_dataset", "清理这些文件",
                          "删除空标签文件和空图片；损坏的图片移到训练集里的「_已移除」文件夹（可以找回）。",
                          danger=True, files=files)]})
    if missing_caps:
        issues.append({
            "level": "warn", "code": "missing_caption", "title": f"{len(missing_caps)} 张图片没有标签文件",
            "reason": "这些图片会以空描述参与训练。建议先到「标签反推」给它们打标。",
            "fixes": [fix("ignore", "忽略，继续训练", "按空描述训练这些图片。")],
            "files": missing_caps[:50]})
    return issues, stats


def preflight(task):
    spec = models.get(task.get("model_type"))
    issues = []
    if not paths.env_python():
        issues.append({"level": "error", "code": "no_env", "title": "训练环境还没安装",
                       "reason": "需要先安装训练环境（PyTorch + 训练引擎），大约需要下载 3–4 GB。",
                       "fixes": [fix("install_env", "现在安装", "在「训练器设置」里自动安装训练环境，装完后继续。")]})
        return issues
    if spec is None:
        return [{"level": "error", "code": "bad_model_type", "title": "未知的底模类型", "reason": str(task.get("model_type")), "fixes": []}]
    p = task.get("params") or {}
    ds_issues, _ = check_dataset(p.get("dataset_dir"), p.get("caption_extension") or ".txt")
    issues += ds_issues
    m = task.get("models") or {}
    for s in spec.slots:
        path = m.get(s.key)
        if not path:
            if s.required:
                issues.append({"level": "error", "code": "no_model", "title": f"没有选择{s.label}",
                               "reason": "请到「模型设置」页选择，或一键下载官方模型。",
                               "fixes": [fix("download_model", f"下载 {d['label']}（{d['size_gb']} GB）",
                                             "从 HuggingFace 下载（国内网络自动走镜像），下好后自动填入并继续。",
                                             slot=s.key, **d) for d in s.downloads[:2]]})
            continue
        err = scanner.check_file(path)
        if not err and path.lower().endswith((".safetensors", ".sft")):
            import safetensors_meta as stm
            info = stm.classify(path)
            if not spec.accept(s, dict(info, name=os.path.basename(path), path=path)):
                issues.append({"level": "error", "code": "wrong_model", "title": f"{s.label}选错了文件",
                               "reason": f"{os.path.basename(path)} 识别为「{info['kind']}」{'（量化版）' if info.get('quant') else ''}，"
                                         f"不是可以用来训练的{s.label}。请到「模型设置」重新选择。",
                               "fixes": [fix("download_model", f"下载 {d['label']}（{d['size_gb']} GB）", "下载官方文件并自动替换。",
                                             slot=s.key, **d) for d in s.downloads[:1]]})
                continue
        if err:
            issues.append({"level": "error", "code": "broken_model", "title": f"{s.label}文件损坏",
                           "reason": f"{os.path.basename(path)}：{err}",
                           "fixes": [fix("download_model", f"重新下载 {d['label']}（{d['size_gb']} GB）",
                                         "下载到训练器的 models/base 目录，原文件不会被删除。", slot=s.key, **d)
                                     for d in s.downloads[:2]]})
    # 测试提示词要和训练集的标法对上：全标时只写触发词出不来人物特征，测试图看起来会像没练好
    smp = task.get("sample") or {}
    if smp.get("enabled") and p.get("dataset_dir"):
        import triggers
        synced, n = triggers.sync_prompts(p["dataset_dir"], smp.get("prompts") or [], only_with_main=True)
        if n:
            info = triggers.get(p["dataset_dir"]) or {}
            pr = triggers.norm_prune(info.get("prune"))
            how = "删标（只写触发词）" if pr["features"] and pr["outfit"] else ("全标" if not pr["features"] and not pr["outfit"] else "部分删标")
            issues.append({"level": "warn", "code": "sample_prompt_trigger", "title": f"{n} 条测试提示词和训练集的标法对不上",
                           "reason": f"这个训练集是{how}，用 LoRA 时应该写「{info.get('prompt')}」。测试提示词没写全的话，"
                                     "测试图会显得不像，没法判断练得怎么样。",
                           "fixes": [fix("sync_sample_prompts", "自动改好测试提示词后继续", "把每条测试提示词开头换成上面这一串，其余内容不变。"),
                                     fix("ignore", "不改，继续训练", "")]})
    # 多卡：选中的显卡都要存在
    g = gpu.info(max_age=0)
    want = task.get("gpus") or settings.gpu_list()
    have = {x["index"] for x in (g.get("all") or [])}
    missing = [x for x in want if have and x not in have]
    if missing:
        issues.append({"level": "error", "code": "gpu_missing", "title": f"找不到显卡 {', '.join(missing)}",
                       "reason": f"这个任务要用显卡 {', '.join(want)}，但电脑上只有 {', '.join(sorted(have))}。"
                                 "请到「训练器设置 → 训练 → 使用的显卡」重新选择，然后把任务删掉重新加入队列。", "fixes": []})
    # 显存：别的程序占着大量显存
    if g and g.get("total_gb"):
        if g["used_gb"] > max(2.0, g["total_gb"] * 0.25):
            procs = gpu.processes()
            issues.append({
                "level": "warn", "code": "vram_busy", "title": f"显存已被占用 {g['used_gb']} / {g['total_gb']} GB",
                "reason": "其他程序（WebUI、ComfyUI、游戏、浏览器等）正占着显存，训练可能会显存不足或变慢。",
                "fixes": ([fix("kill_gpu", "结束这些占用显存的程序", "强制结束下列进程，未保存的内容会丢失。", danger=True,
                               procs=procs)] if procs else []) + [fix("ignore", "忽略，继续训练", "")]})
    # 内存：块交换把模型块放在内存里、文本编码器在 CPU 上缓存，都要大内存
    p = task.get("params") or {}
    total_ram, avail_ram = ram_gb()
    heavy = int(p.get("blocks_to_swap") or 0) >= 20 or p.get("te_cpu")
    if total_ram and heavy and total_ram < 31:
        issues.append({"level": "warn", "code": "ram_small", "title": f"内存只有 {total_ram:.0f} GB",
                       "reason": "现在的设置（大量块交换 / 文本编码器在 CPU 上缓存）要把十几 GB 的模型放在内存里，"
                                 "建议 32 GB 以上内存，不够的话会非常慢甚至失败。可以把 Windows 虚拟内存调大到 64 GB 以上再试。",
                       "fixes": [fix("ignore", "忽略，继续训练", "")]})
    elif avail_ram and avail_ram < (12 if heavy else 4):
        issues.append({"level": "warn", "code": "ram_busy", "title": f"可用内存只剩 {avail_ram:.1f} GB",
                       "reason": "其他程序占了大量内存，训练可能变得很慢或失败。建议先关掉不用的程序（浏览器标签、游戏、另一个 WebUI 等）。",
                       "fixes": [fix("ignore", "忽略，继续训练", "")]})
    # 硬盘
    out = task.get("output_dir") or settings.load()["output_dir"]
    try:
        os.makedirs(out, exist_ok=True)
        free = shutil.disk_usage(out).free / 1e9
        if free < 2:
            issues.append({"level": "error", "code": "disk_full", "title": "保存位置所在硬盘空间不足",
                           "reason": f"只剩 {free:.1f} GB，至少需要 2 GB（开了保存训练状态需要更多）。请清理硬盘或换保存位置。",
                           "fixes": []})
    except OSError as e:
        issues.append({"level": "error", "code": "bad_output", "title": "保存位置无法写入", "reason": str(e), "fixes": []})
    return issues


# ======================= 训练报错诊断 =======================
_MODULE_PKG = {"cv2": "opencv-python==4.10.0.84", "PIL": "pillow", "bitsandbytes": "bitsandbytes", "toml": "toml",
               "voluptuous": "voluptuous", "einops": "einops", "safetensors": "safetensors", "triton": "triton-windows",
               "prodigyopt": "prodigyopt", "lion_pytorch": "lion-pytorch", "schedulefree": "schedulefree",
               "tensorboard": "tensorboard", "sentencepiece": "sentencepiece", "imagesize": "imagesize",
               "onnxruntime": "onnxruntime-gpu", "flash_attn": None, "sageattention": None,
               # xformers 锁定了具体的 PyTorch 版本：直接 pip install 可能把 PyTorch 换成 PyPI 上的 CPU 版，整个环境就废了
               "xformers": None}

RULES = [
    # (code, 正则, 标题, 原因, 修复生成函数)
    ("oom", r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation|out of memory",
     "显存不足", "训练需要的显存超过了显卡剩余的显存。", "oom"),
    ("pagefile", r"页面文件太小|paging file is too small|WinError 1455",
     "虚拟内存（页面文件）太小", "Windows 的虚拟内存不够，多线程加载数据或加载大模型时失败。", "pagefile"),
    ("ram", r"DefaultCPUAllocator: not enough memory|Unable to allocate .* for an array|MemoryError",
     "内存（RAM）不足", "电脑内存不够加载模型或数据。", "ram"),
    ("broken_model", r"SafetensorError|Error while deserializing header|MetadataIncompleteBuffer|incomplete metadata|HeaderTooLarge|invalid load key",
     "模型文件损坏", "模型文件没下载完整或已经损坏。", "broken_model"),
    ("wrong_model", r"Missing key\(s\) in state_dict|Unexpected key\(s\) in state_dict|size mismatch for",
     "模型和所选底模类型不匹配", "选的文件不是这个底模的对应组件（例如把别的模型当成了主模型 / 文本编码器）。", "wrong_model"),
    ("bad_image", r"UnidentifiedImageError|cannot identify image file|image file is truncated|broken data stream|Truncated File Read",
     "训练集里有损坏的图片", "有图片文件损坏或不是真正的图片格式。", "bad_image"),
    ("no_images", r"No data found|画像がありません|no images found|num_train_images\D+0\b|dataset is empty|No valid items",
     "训练集里没有可用的图片", "训练集路径不对，或者图片格式不受支持。", None),
    ("no_module", r"ModuleNotFoundError: No module named '([\w\.]+)'",
     "训练环境缺少组件", "训练环境里缺少 Python 包「{0}」。", "no_module"),
    ("no_cuda", r"Torch not compiled with CUDA enabled|CUDA driver version is insufficient|no kernel image is available|CUDA error: no CUDA-capable device|NVIDIA driver on your system is too old|driver version .* too old",
     "PyTorch 和显卡 / 驱动不匹配", "装的 PyTorch 不支持这张显卡，或者显卡驱动太旧。", "no_cuda"),
    ("bnb", r"bitsandbytes.*(error|Error)|libbitsandbytes|CUDA Setup failed",
     "8bit 优化器组件加载失败", "bitsandbytes 没装好或和显卡不兼容。", "bnb"),
    ("fp8", r"_scaled_mm|fp8.*not supported|Float8_e4m3fn.*not implemented|e4m3fn.*is not supported",
     "显卡不支持 fp8", "30 系及更早的显卡不支持 fp8 计算。", "fp8"),
    ("triton", r"No module named 'triton'|triton.*not (found|available)|TritonMissing|Cannot find a working triton installation",
     "缺少 Triton（int8 加速需要）", "int8 ConvRot 量化需要 triton 才能加速。", "triton"),
    ("xformers", r"xformers.*(not|error|Error)|memory_efficient_attention.*(not|error)",
     "xformers 不可用", "xformers 没装或版本和 PyTorch 不匹配。", "xformers"),
    ("flash", r"No module named 'flash_attn'|flash_attn.*(error|not)",
     "flash-attn 不可用", "没有安装 flash-attn。", "attn_torch"),
    ("sage", r"No module named 'sageattention'",
     "SageAttention 不可用", "没有安装 SageAttention。", "attn_torch"),
    ("nan", r"loss is nan|avr_loss=nan|NaN detected|loss=nan",
     "训练出现 NaN（数值爆炸）", "学习率太高，或者精度设置不合适，loss 变成了 NaN，继续训练没有意义。", "nan"),
    ("bf16", r"bf16.*(not supported|is only supported)|Bfloat16 is not supported",
     "显卡不支持 bf16", "20 系及更早的显卡不支持 bf16。", "bf16"),
    ("disk", r"No space left on device|Errno 28|磁盘空间不足|There is not enough space on the disk",
     "硬盘空间不足", "保存位置或缓存所在的硬盘满了。", None),
    ("file_lock", r"WinError 32|being used by another process|另一个程序正在使用此文件",
     "文件被占用", "有其他程序正在使用训练要写入的文件。", None),
    ("not_found", r"FileNotFoundError|No such file or directory|系统找不到指定的",
     "找不到文件", "模型、训练集或配置里的某个路径不存在。", None),
    ("network", r"ConnectionError|ReadTimeout|ConnectTimeout|Max retries exceeded|SSLError|getaddrinfo failed|LocalEntryNotFoundError",
     "网络连接失败", "下载模型 / 组件时连不上服务器。", "network"),
    ("cache_mismatch", r"cache.*(mismatch|does not match)|latents.*shape|Cached .* invalid",
     "缓存和当前设置不一致", "分辨率或训练集改动后，旧的缓存不能再用。", "clear_cache"),
    ("te_cache_conflict", r"shuffle_caption.*cache_text_encoder_outputs|cache_text_encoder_outputs.*shuffle|caption_dropout.*cache",
     "参数冲突：打乱标签 / 标签丢弃 与 文本编码缓存", "这两项不能同时开。", None),
    ("ddp_port", r"EADDRINUSE|address already in use|server socket has failed to listen|通常每个套接字地址.*只允许使用一次",
     "多卡通信端口被占用", "多卡训练用来通信的端口已经被别的程序（或上次没退干净的训练）占着。", "ddp_port"),
    ("crash", r"0xC0000005|-1073741819|3221225477|0xC0000409|-1073740791|Segmentation fault|access violation",
     "训练进程崩溃（系统层面的错误）", "训练进程被 Windows 强制结束了，常见原因：显卡驱动不稳定 / 超频、内存不足、和别的程序（录屏、超频软件）冲突。", "crash"),
    ("device_mismatch", r"Expected all tensors to be on the same device",
     "数据不在同一个设备上", "训练里有一部分数据在显卡、一部分在内存，通常出现在续训恢复状态或开了显存卸载的时候。", "device_mismatch"),
    ("bad_arg", r"error: unrecognized arguments: (\S+)|error: argument (--\S+)",
     "训练引擎不认识某个参数", "训练命令里的参数 {0} 训练引擎不认识——通常是训练引擎更新后改了参数名，或者装的引擎版本太旧。", "bad_arg"),
    ("blocks_swap_conflict", r"blocks_to_swap.*(cannot|not).*(combined|used)|turbo_dit.*blocks_to_swap",
     "参数冲突：块交换", "块交换不能和当前开着的某个选项一起用。", None),
]


def _tail(text, n=60):
    lines = [x for x in (text or "").splitlines() if x.strip()]
    return "\n".join(lines[-n:])


def diagnose(log_text, task):
    """返回 {code, title, reason, detail, fixes}；完全认不出时 code='unknown'。"""
    tail = _tail(log_text, 200)
    p = (task or {}).get("params") or {}
    spec = models.get((task or {}).get("model_type"))
    for code, pat, title, reason, fx in RULES:
        m = re.search(pat, tail, re.I)
        if not m:
            continue
        reason = reason.format(*(m.groups() or ("",)))
        fixes = _fixes_for(fx, m, task, spec, p, tail)
        hint = {
            "no_images": "检查「训练集位置」是否选对，图片是否是 png / jpg / webp。",
            "disk": "清理硬盘空间或者到「启动训练」页换一个保存位置，然后重试。",
            "file_lock": "关闭正在打开这些文件的程序（看图软件、WebUI 正在加载这个 LoRA 等）后重试。",
            "not_found": "看下方原始报错里的路径，到对应页面重新选择。",
            "te_cache_conflict": "到「高级参数」关闭「打乱标签顺序」或把标签丢弃率改成 0。",
            "blocks_swap_conflict": "到「高级参数」把块交换改成 0，或关闭冲突的选项。",
            "no_cuda": "如果提示驱动太旧，请到 NVIDIA 官网下载最新的 Game Ready / Studio 驱动安装后重启电脑。",
        }.get(code, "")
        return {"code": code, "title": title, "reason": reason + (" " + hint if hint else ""),
                "detail": _tail(log_text, 30), "fixes": fixes}
    return {"code": "unknown", "title": "训练意外中止",
            "reason": "没能自动识别这个报错。下面是日志最后几十行，最后出现的 Error 一行通常就是原因。",
            "detail": _tail(log_text, 40), "fixes": []}


def _fixes_for(kind, m, task, spec, p, tail):
    if kind == "oom":
        procs = gpu.processes()
        res = []
        if procs:
            res.append(fix("kill_gpu", "结束其他占用显存的程序后重试", "强制结束下列进程，未保存的内容会丢失。", danger=True, procs=procs))
        res.append(fix("lower_vram", "调低显存占用后重试", _lower_vram_desc(task, spec)))
        return res
    if kind == "pagefile":
        return [fix("workers_zero", "数据加载线程改为 0 后重试", "减少同时占用的内存。如果仍然失败，需要手动把 Windows 虚拟内存调大到 32 GB 以上（系统设置 → 高级系统设置 → 性能 → 虚拟内存）。")]
    if kind == "ram":
        return [fix("workers_zero", "减少内存占用后重试", "数据加载线程改为 0" + ("，并开启块交换" if spec and spec.id == "krea2" else "") + "。")]
    if kind in ("broken_model", "wrong_model"):
        res = []
        for s in (spec.slots if spec else []):
            path = (task.get("models") or {}).get(s.key)
            if not path:
                continue
            bad = scanner.check_file(path)
            if bad or kind == "wrong_model":
                for d in s.downloads[:1]:
                    res.append(fix("download_model", f"重新下载{s.label}：{d['label']}（{d['size_gb']} GB）",
                                   (f"{os.path.basename(path)}：{bad}。" if bad else "") + "下载官方文件后自动替换并重试。",
                                   slot=s.key, **d))
        return res
    if kind == "bad_image":
        issues, _ = check_dataset(p.get("dataset_dir"), p.get("caption_extension") or ".txt")
        for it in issues:
            if it["code"] == "dirty_dataset":
                return [f for f in it["fixes"] if f["id"] == "clean_dataset"]
        return []
    if kind == "no_module":
        mod = m.group(1).split(".")[0]
        pkg = _MODULE_PKG.get(mod, mod)
        if pkg is None:   # 可选的注意力加速组件：改回默认注意力，不去装它
            return _fixes_for("attn_torch", m, task, spec, p, tail)
        return [fix("pip_install", f"安装 {pkg} 后重试", f"在训练环境里执行 pip install {pkg}。", pkg=pkg),
                fix("repair_env", "修复整个训练环境", "重新安装全部依赖（不会重新下载 PyTorch）。")]
    if kind == "no_cuda":
        return [fix("reinstall_torch", "重新安装 PyTorch 后重试", "按「训练器设置」里选的 CUDA 版本重新安装 PyTorch（约 3 GB）。如果是驱动太旧，需要先到 NVIDIA 官网更新驱动。")]
    if kind == "bnb":
        return [fix("pip_install", "重装 bitsandbytes 后重试", "pip install -U bitsandbytes", pkg="-U bitsandbytes"),
                fix("set_params", "改用 AdamW 优化器后重试", "不再依赖 bitsandbytes，显存会多用一些。", values={"optimizer_type": "AdamW"})]
    if kind == "fp8":
        return [fix("set_params", "改用 int8 ConvRot 量化后重试", "30 系显卡用 int8 量化，速度更快。", values={"quant": "convrot"})]
    if kind == "bad_arg":
        return [fix("update_engine", "同步训练引擎和环境后重试", "把 sd-scripts / musubi-tuner 切到这个版本的训练器测试过的版本，并装好对应依赖。"
                                                          "还是不行的话到「训练器设置 → 一键更新」把训练器也更新到最新。"),
                fix("repair_env", "修复整个训练环境", "重新安装全部依赖（不会重新下载 PyTorch）。")]
    if kind == "ddp_port":
        port = int(p.get("main_process_port") or 29500)
        return [fix("set_params", f"换一个通信端口（{port} → {port + 17}）后重试", "多卡通信改用另一个空闲端口。", values={"main_process_port": port + 17})]
    if kind == "crash":
        return [fix("workers_zero", "数据加载线程改为 0 后重试", "减少同时运行的进程，排除内存不足导致的崩溃。如果还崩，请检查显卡驱动 / 关闭超频软件后再试。")]
    if kind == "device_mismatch":
        # 续训恢复状态时的这个问题训练器已经自动处理（lora_optim.Automagic.load_state_dict）
        if int(p.get("blocks_to_swap") or 0):
            return [fix("set_params", "关闭显存卸载（块交换改 0）后重试", "如果是开了块交换才出现的，关掉它通常就好了。", values={"blocks_to_swap": 0})]
        return []
    if kind == "triton":
        return [fix("pip_install", "安装 triton-windows 后重试", "pip install triton-windows", pkg="triton-windows"),
                fix("set_params", "改用 fp8 量化后重试", "不需要 triton，但 30 系显卡上会慢一些。", values={"quant": "fp8"})]
    if kind in ("xformers", "attn_torch"):
        return [fix("set_params", "改用默认注意力实现后重试", "注意力实现改回 torch SDPA。",
                    values={"attn_mode": "sdpa" if spec and spec.id in ("krea2", "sdxl") else "torch"})]
    if kind == "nan":
        lr = float(p.get("learning_rate") or 1e-4)
        return [fix("set_params", f"学习率减半（{lr:g} → {lr / 2:g}）后重试", "同时把混合精度设为 bf16、开启梯度裁剪 1.0。",
                    values={"learning_rate": lr / 2, "mixed_precision": "bf16", "max_grad_norm": 1.0})]
    if kind == "bf16":
        return [fix("set_params", "改用 fp16 后重试", "混合精度和保存精度改为 fp16。", values={"mixed_precision": "fp16", "save_precision": "fp16"})]
    if kind == "network":
        if not settings.is_cn():
            return [fix("switch_cn", "切换到国内网络（镜像）后重试", "HuggingFace 走 hf-mirror.com，pip 走清华源，GitHub 走加速代理。")]
        return []
    if kind == "clear_cache":
        return [fix("clear_cache", "清除缓存后重试", "删除这个任务的缓存和训练集里的 .npz 缓存文件，重新编码。", danger=False)]
    return []


def _lower_vram_desc(task, spec):
    new = lower_vram_values(task, spec)
    if not new:
        return "已经是最省显存的设置了。"
    names = {"blocks_to_swap": "块交换", "batch_size": "单批数量", "gradient_checkpointing": "梯度检查点", "resolution": "分辨率",
             "quant": "量化", "gc_cpu_offload": "梯度检查点卸载", "fp8_base": "大模型 fp8 加载", "convrot_int8": "DiT int8 量化", "te_fp8": "文本编码器 fp8", "optimizer_type": "优化器"}
    p = task.get("params") or {}
    return "；".join(f"{names.get(k, k)}：{p.get(k)} → {v}" for k, v in new.items())


def lower_vram_values(task, spec):
    p = dict(spec.defaults() if spec else {})
    p.update(task.get("params") or {})
    new = {}
    if not p.get("gradient_checkpointing"):
        new["gradient_checkpointing"] = True
    if int(p.get("batch_size") or 1) > 1:
        new["batch_size"] = max(1, int(p["batch_size"]) // 2)
        return new
    if spec and spec.id == "krea2" and p.get("quant") == "none":
        new["quant"] = "auto"
        return new
    has = spec.defaults() if spec else {}
    if "fp8_base" in has and not p.get("fp8_base"):   # SDXL：大模型改 fp8 加载
        new["fp8_base"] = True
        return new
    if "convrot_int8" in has and not p.get("convrot_int8"):   # MOSS-Image：DiT 改 int8
        new["convrot_int8"] = True
        return new
    swap = int(p.get("blocks_to_swap") or 0)
    top = getattr(spec, "MAX_SWAP", 26)   # 各底模能交换的块数上限（Anima / Krea 2 26，MOSS-Image 40）
    if "blocks_to_swap" in has and swap < top:
        new["blocks_to_swap"] = min(top, swap + 8)
        return new
    if spec and spec.id == "krea2" and not p.get("gc_cpu_offload"):
        new["gc_cpu_offload"] = True
        return new
    # 分辨率降一档：从这个底模自己的选项里挑（Krea 2 只有 512 / 1024，没有 768）
    cur = int(p.get("resolution") or 1024) if str(p.get("resolution") or "").isdigit() or isinstance(p.get("resolution"), int) else 1024
    opts = []
    for f in (spec.params() if spec else []):
        if f["key"] == "resolution":
            opts = [int(o[0]) for o in f.get("options") or [] if str(o[0]).isdigit()]
    lower = [r for r in (opts or [512, 768, 1024]) if 512 <= r < cur]
    if lower:
        new["resolution"] = max(lower)
        return new
    if p.get("optimizer_type") not in ("Adafactor", "AdaFactor"):
        new["optimizer_type"] = "AdaFactor" if spec and spec.id == "krea2" else "Adafactor"
    return new


def ram_gb():
    """(总内存, 可用内存) GB；拿不到返回 (0, 0)。"""
    if os.name != "nt":
        try:
            info = {}
            with open("/proc/meminfo", encoding="ascii", errors="ignore") as f:
                for line in f:
                    k, _, v = line.partition(":")
                    if k in ("MemTotal", "MemAvailable"):
                        info[k] = int(v.split()[0]) / 1024 / 1024
            return info.get("MemTotal", 0), info.get("MemAvailable", 0)
        except (OSError, ValueError, IndexError):
            return 0, 0
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        return 0, 0
    return m.ullTotalPhys / 2 ** 30, m.ullAvailPhys / 2 ** 30

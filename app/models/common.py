# -*- coding: utf-8 -*-
"""各底模共用的参数字段。具体底模可以改默认值、追加自己的字段。"""
from .base import F

SCHEDULERS = [["constant", "constant · 恒定"], ["constant_with_warmup", "constant_with_warmup · 预热后恒定"],
              ["cosine", "cosine · 余弦下降"], ["cosine_with_restarts", "cosine_with_restarts · 余弦重启"],
              ["cosine_with_min_lr", "cosine_with_min_lr · 余弦降到最低学习率"],
              ["warmup_stable_decay", "warmup_stable_decay · 预热-稳定-衰减"],
              ["linear", "linear · 线性下降"], ["polynomial", "polynomial · 多项式下降"],
              ["inverse_sqrt", "inverse_sqrt · 平方根倒数"], ["adafactor", "adafactor · 仅配合 Adafactor"],
              ["custom", "自定义（填写 模块.类名）"]]
WARMUP_SCHEDULERS = ["constant_with_warmup", "cosine", "cosine_with_restarts", "cosine_with_min_lr", "warmup_stable_decay",
                     "linear", "polynomial", "inverse_sqrt"]


def common_fields(d=None, optimizers=None, timesteps=None, schedulers=None):
    """d：默认值覆盖。"""
    d = d or {}

    def v(k, default):
        return d.get(k, default)

    return [
        # ---- 最上面 ----
        F("gradient_checkpointing", "梯度检查点", "bool", v("gradient_checkpointing", True), "top",
          help="用时间换显存：几乎不影响质量，显存占用大幅下降，速度慢约 20%。显存不富裕一定要开。"),

        # ---- 基础 ----
        F("dataset_dir", "训练集位置", "dir", v("dataset_dir", ""), "basic",
          help="图片和同名 .txt 标签放在一起。子文件夹会作为独立子集；文件夹名带「10_」前缀时可沿用该重复次数。"),
        F("output_name", "保存名称", "text", v("output_name", "mylora_v1"), "basic",
          help="LoRA 文件名（不含扩展名）。中间保存的文件会自动加上 -000001 这样的轮数后缀。"),
        F("repeats", "每张图重复次数", "int", v("repeats", 10), "basic", min=1, max=1000,
          help="每轮里每张图训练几次。不用再改文件夹名，这里直接设。"),
        F("use_folder_repeats", "沿用文件夹名里的重复次数", "bool", v("use_folder_repeats", True), "basic",
          help="子文件夹叫「5_角色A」这种格式时，按 5 次重复；关掉则全部用上面的值。"),
        F("steps_mode", "训练长度按", "select", v("steps_mode", "epochs"), "basic",
          options=[["epochs", "轮数"], ["steps", "总步数"]]),
        F("epochs", "训练轮数", "int", v("epochs", 10), "basic", min=1, max=10000, show_if={"steps_mode": "epochs"}),
        F("max_train_steps", "总步数", "int", v("max_train_steps", 2000), "basic", min=1, max=10000000,
          show_if={"steps_mode": "steps"}, help="直接指定总步数，轮数会自动换算。"),
        F("save_every_n_epochs", "每几轮保存一次", "int", v("save_every_n_epochs", 1), "basic", min=0, max=1000,
          help="0 = 只保存最终结果。"),
        F("batch_size", "单批训练数量", "int", v("batch_size", 2), "basic", min=1, max=64,
          help="一次喂给模型几张图。越大越快、越吃显存；改了它一般也要按比例调学习率。"),
        F("resolution", "训练分辨率", "select", v("resolution", 1024), "basic",
          options=[[512, "512"], [768, "768"], [1024, "1024"], [1280, "1280"], [1536, "1536"], ["custom", "自定义宽×高"]],
          help="按面积分桶，非方图会自动按比例裁到接近这个面积。"),
        F("resolution_wh", "自定义分辨率（宽,高）", "text", v("resolution_wh", "1024,1024"), "basic", show_if={"resolution": "custom"},
          help="例如 896,1152。必须是 64 的倍数。"),
        F("network_dim", "网络维度 dim", "int", v("network_dim", 32), "basic", min=1, max=512,
          help="LoRA 容量。画风 / 角色 16–32 足够，太大容易过拟合、文件也大。"),
        F("network_alpha", "网络 alpha", "float", v("network_alpha", 16), "basic", min=0, max=512, step=1,
          help="缩放系数，实际强度 = alpha / dim。alpha 越小，等效学习率越低。"),

        # ---- 学习率 ----
        F("learning_rate", "学习率", "float", v("learning_rate", 1e-4), "lr", step=1e-5, sci=True),
        F("optimizer_type", "优化器", "select", v("optimizer_type", "AdamW8bit"), "lr",
          options=optimizers or [["AdamW8bit", "AdamW8bit（推荐，省显存）"], ["AdamW", "AdamW"],
                                 ["Adafactor", "Adafactor（最省显存）"]]),
        F("optimizer_custom", "自定义优化器类", "text", v("optimizer_custom", ""), "lr", show_if={"optimizer_type": "custom"},
          help="模块.类名，例如 pytorch_optimizer.CAME、torch.optim.RAdam。训练环境里没有的包会在报错时提示安装。"),
        F("optimizer_args", "优化器参数", "text", v("optimizer_args", ""), "lr",
          help="空格分隔，例如 weight_decay=0.01 betas=0.9,0.99"),
        F("lr_scheduler", "学习率调度", "select", v("lr_scheduler", "constant_with_warmup"), "lr", options=schedulers or SCHEDULERS,
          help="ScheduleFree 系列优化器请配 constant。"),
        F("lr_warmup_steps", "预热步数", "int", v("lr_warmup_steps", 50), "lr", min=0,
          show_if={"lr_scheduler": WARMUP_SCHEDULERS + ["rex"]}),
        F("lr_scheduler_num_cycles", "余弦周期数", "int", v("lr_scheduler_num_cycles", 1), "lr", min=1,
          show_if={"lr_scheduler": "cosine_with_restarts"}, help="1 = 不重启，2 = 中间重启 1 次，以此类推。"),
        F("lr_scheduler_power", "多项式幂次", "float", v("lr_scheduler_power", 1.0), "lr", step=0.1,
          show_if={"lr_scheduler": "polynomial"}),
        F("lr_scheduler_min_lr_ratio", "最低学习率比例", "float", v("lr_scheduler_min_lr_ratio", 0.1), "lr", step=0.01, min=0, max=1,
          show_if={"lr_scheduler": ["cosine_with_min_lr", "warmup_stable_decay", "rex"]}, help="最低学习率 = 学习率 × 这个比例。"),
        F("lr_scheduler_type", "自定义调度器类", "text", v("lr_scheduler_type", ""), "lr", show_if={"lr_scheduler": "custom"},
          help="例如 torch.optim.lr_scheduler.CosineAnnealingLR，参数写在下面「调度器参数」里。"),
        F("lr_scheduler_args", "调度器参数", "text", v("lr_scheduler_args", ""), "lr",
          help="空格分隔的 key=value，例如 T_max=100。一般留空。"),
        F("lr_decay_steps", "衰减步数", "int", v("lr_decay_steps", 0), "lr", min=0,
          show_if={"lr_scheduler": "warmup_stable_decay"}, help="最后多少步开始衰减；0 = 自动。"),
        F("max_grad_norm", "梯度裁剪", "float", v("max_grad_norm", 1.0), "lr", step=0.1, help="0 = 不裁剪。"),

        # ---- 其他高级设置 ----
        F("seed", "随机种子", "int", v("seed", 42), "adv", "训练细节"),
        F("grad_accum", "梯度累积步数", "int", v("grad_accum", 1), "adv", "训练细节", min=1, max=64,
          help="显存不够又想要大批量时用：等效批量 = 单批数量 × 累积步数。"),
        F("caption_extension", "标签文件扩展名", "text", v("caption_extension", ".txt"), "adv", "训练细节"),
        F("enable_bucket", "分桶（保持原图比例）", "bool", v("enable_bucket", True), "adv", "训练细节"),
        F("min_bucket_reso", "最小桶边长", "int", v("min_bucket_reso", 256), "adv", "训练细节", show_if={"enable_bucket": True}),
        F("max_bucket_reso", "最大桶边长", "int", v("max_bucket_reso", 2048), "adv", "训练细节", show_if={"enable_bucket": True}),
        F("bucket_no_upscale", "不放大小图", "bool", v("bucket_no_upscale", True), "adv", "训练细节"),
        F("mixed_precision", "混合精度", "select", v("mixed_precision", "bf16"), "adv", "训练细节",
          options=[["bf16", "bf16（推荐）"], ["fp16", "fp16（老显卡）"]]),
        F("save_precision", "保存精度", "select", v("save_precision", "bf16"), "adv", "训练细节",
          options=[["bf16", "bf16"], ["fp16", "fp16"], ["float", "fp32"]]),
        F("save_state", "保存训练状态（可断点续训）", "bool", v("save_state", False), "adv", "训练细节",
          help="每次保存时额外存优化器状态，占空间较大。"),
        F("save_last_n", "只保留最近几个中间文件", "int", v("save_last_n", 0), "adv", "训练细节", min=0,
          help="0 = 全部保留。"),
        F("save_last_n_epochs_state", "只保留最近几个训练状态", "int", v("save_last_n_epochs_state", 0), "adv", "训练细节", min=0,
          show_if={"save_state": True}, help="训练状态很占空间（每个约几 GB）。0 = 全部保留。"),
        F("save_state_on_train_end", "训练结束时保存训练状态", "bool", v("save_state_on_train_end", False), "adv", "训练细节",
          help="只在最后存一次训练状态，方便以后接着训，又不会每轮都占空间。"),
        F("full_fp16", "LoRA 权重全程 fp16", "bool", v("full_fp16", False), "adv", "训练细节",
          help="只给不支持 bf16 的老显卡用，容易出现 NaN。"),
        F("resume", "从训练状态继续", "dir", v("resume", ""), "adv", "训练细节",
          help="选之前开了「保存训练状态」生成的 xxx-state 文件夹，从中断处接着训。"),

        F("timestep_sampling", "时间步采样", "select", v("timestep_sampling", "shift"), "adv", "时间步 / 噪声",
          options=timesteps or [["shift", "shift"], ["sigmoid", "sigmoid"], ["uniform", "uniform"]]),
        F("discrete_flow_shift", "flow shift", "float", v("discrete_flow_shift", 3.0), "adv", "时间步 / 噪声", step=0.1,
          show_if={"timestep_sampling": "shift"}),
        F("sigmoid_scale", "sigmoid scale", "float", v("sigmoid_scale", 1.0), "adv", "时间步 / 噪声", step=0.1,
          show_if={"timestep_sampling": ["sigmoid", "shift"]}),
        F("weighting_scheme", "时间步损失权重", "select", v("weighting_scheme", "none"), "adv", "时间步 / 噪声",
          options=[["none", "none · 不加权（默认）"], ["logit_normal", "logit_normal"], ["mode", "mode"],
                   ["cosmap", "cosmap"], ["sigma_sqrt", "sigma_sqrt"]]),
        F("logit_mean", "logit 均值", "float", v("logit_mean", 0.0), "adv", "时间步 / 噪声", step=0.1,
          show_if={"weighting_scheme": "logit_normal"}),
        F("logit_std", "logit 标准差", "float", v("logit_std", 1.0), "adv", "时间步 / 噪声", step=0.1,
          show_if={"weighting_scheme": "logit_normal"}),
        F("mode_scale", "mode 缩放", "float", v("mode_scale", 1.29), "adv", "时间步 / 噪声", step=0.01,
          show_if={"weighting_scheme": "mode"}),
        F("min_timestep", "最小时间步", "int", v("min_timestep", 0), "adv", "时间步 / 噪声", min=0, max=1000,
          help="0–1000，限制训练的噪声范围。默认 0。"),
        F("max_timestep", "最大时间步", "int", v("max_timestep", 1000), "adv", "时间步 / 噪声", min=0, max=1000,
          help="默认 1000。"),

        F("blocks_to_swap", "块交换数量", "int", v("blocks_to_swap", 0), "adv", "显存 / 速度", min=0, max=26,
          help="把部分模型块放到内存里轮换，显存不够时加大（最多 26），越大越慢。"),
        F("attn_mode", "注意力实现", "select", v("attn_mode", "torch"), "adv", "显存 / 速度",
          options=[["torch", "torch SDPA（默认，最稳）"], ["xformers", "xformers"], ["flash", "flash-attn"]]),
        F("num_workers", "数据加载线程", "int", v("num_workers", 2), "adv", "显存 / 速度", min=0, max=16,
          help="Windows 上报「页面文件太小」时改成 0。"),
        F("cache_latents_to_disk", "潜变量缓存写入硬盘", "bool", v("cache_latents_to_disk", True), "adv", "显存 / 速度",
          help="下次训练同一批图时不用重新编码。"),
        F("split_attn", "拆分注意力计算", "bool", v("split_attn", False), "adv", "显存 / 速度",
          help="进一步省显存但变慢；用 xformers 时会自动开启。"),
        F("full_bf16", "LoRA 权重全程 bf16", "bool", v("full_bf16", False), "adv", "显存 / 速度",
          help="省一点显存，但可能降低精度，一般不建议开。"),

        F("network_dropout", "网络 dropout", "float", v("network_dropout", 0), "adv", "LoRA 结构", step=0.05, min=0, max=1),
        F("rank_dropout", "rank dropout", "float", v("rank_dropout", 0), "adv", "LoRA 结构", step=0.05, min=0, max=1,
          help="随机丢弃部分 rank，抑制过拟合。0 = 关。"),
        F("module_dropout", "module dropout", "float", v("module_dropout", 0), "adv", "LoRA 结构", step=0.05, min=0, max=1),
        F("loraplus_lr_ratio", "LoRA+ 学习率倍数", "float", v("loraplus_lr_ratio", 0), "adv", "LoRA 结构", step=1, min=0,
          help="LoRA+：up 矩阵用更高学习率，收敛更快。推荐 16，0 = 关。开启后基础学习率建议降低。"),
        F("scale_weight_norms", "权重范数上限", "float", v("scale_weight_norms", 0), "adv", "LoRA 结构", step=0.1, min=0,
          help="限制 LoRA 权重大小防止炸模型，推荐 1。0 = 关。"),
        F("network_args", "网络参数 network_args", "text", v("network_args", ""), "adv", "LoRA 结构",
          help="空格分隔的 key=value，例如 exclude_patterns=[...]"),
        F("network_weights", "从已有 LoRA 继续训练", "file", v("network_weights", ""), "adv", "LoRA 结构",
          help="选一个之前训练的 LoRA 文件，在它的基础上接着训。"),
        F("dim_from_weights", "dim 跟随已有 LoRA", "bool", v("dim_from_weights", False), "adv", "LoRA 结构",
          help="继续训练时自动使用已有 LoRA 的 dim。"),
        F("base_weights", "先合并进底模的 LoRA", "file", v("base_weights", ""), "adv", "LoRA 结构",
          help="训练前把这个 LoRA 合并进底模（差异炼丹），新 LoRA 只学剩下的差异。"),
        F("base_weights_multiplier", "合并强度", "float", v("base_weights_multiplier", 1.0), "adv", "LoRA 结构", step=0.1),
        F("log_with", "训练日志", "select", v("log_with", "tensorboard"), "adv", "日志",
          options=[["tensorboard", "TensorBoard（训练页的 loss / 学习率图表用它）"], ["all", "TensorBoard + 同时上传 wandb"]],
          help="日志在任务目录的 logs 文件夹。训练页的四张图表就是从这里读的，所以一直开着。"),
        F("log_prefix", "日志名前缀", "text", v("log_prefix", ""), "adv", "日志", show_if={"log_with": ["tensorboard", "all"]}),
        F("log_tracker_name", "日志项目名", "text", v("log_tracker_name", ""), "adv", "日志", show_if={"log_with": ["tensorboard", "all"]}),
        F("wandb_api_key", "wandb API Key", "text", v("wandb_api_key", ""), "adv", "日志", show_if={"log_with": "all"}),
        F("wandb_run_name", "wandb 运行名", "text", v("wandb_run_name", ""), "adv", "日志", show_if={"log_with": "all"}),
        F("export_comfy", "训练完额外导出 ComfyUI 格式", "bool", v("export_comfy", False), "adv", "其他",
          help="另存一份 名称_comfy.safetensors，原文件保留。"),
        F("ddp_timeout", "进程通信超时（分钟）", "int", v("ddp_timeout", 0), "adv", "多卡训练", min=0,
          help="多张卡之间同步等待的最长时间。缓存大训练集时间很长时调大，0 = 默认（30 分钟）。只在「训练器设置」里选了多张显卡时生效。"),
        F("ddp_gradient_as_bucket_view", "梯度共享内存（省显存）", "bool", v("ddp_gradient_as_bucket_view", True), "adv", "多卡训练",
          help="梯度直接放在通信缓冲区里，少占一份显存。"),
        F("ddp_static_graph", "静态计算图（更快）", "bool", v("ddp_static_graph", False), "adv", "多卡训练",
          help="模型结构每步都一样时可开，通信更快；开梯度检查点或报错时关掉。"),
        F("main_process_port", "通信端口", "int", v("main_process_port", 29500), "adv", "多卡训练", min=1024, max=65535,
          help="同时跑两个多卡训练时要错开端口。"),
        F("num_cpu_threads_per_process", "每个进程 CPU 线程数", "int", v("num_cpu_threads_per_process", 1), "adv", "多卡训练", min=1, max=64),
        F("extra_args", "额外命令行参数", "textarea", v("extra_args", ""), "adv", "其他",
          help="原样追加到训练命令末尾，例如 --min_snr_gamma 5"),
    ]


def optimizer_name(p):
    return (p.get("optimizer_custom") or "").strip() if p.get("optimizer_type") == "custom" else (p.get("optimizer_type") or "AdamW8bit")


def common_args(p, task_dir):
    """两个 kohya 引擎共有的命令行参数（调度器附加项、时间步权重、LoRA 正则、日志……）。"""
    import os
    from .base import split_args
    a = []
    sch = p.get("lr_scheduler") or "constant"
    if sch in WARMUP_SCHEDULERS or sch == "rex":
        a += ["--lr_warmup_steps", str(int(p.get("lr_warmup_steps") or 0))]
    if sch == "cosine_with_restarts":
        a += ["--lr_scheduler_num_cycles", str(int(p.get("lr_scheduler_num_cycles") or 1))]
    if sch == "polynomial":
        a += ["--lr_scheduler_power", str(p.get("lr_scheduler_power") or 1.0)]
    if sch in ("cosine_with_min_lr", "warmup_stable_decay", "rex"):
        r = p.get("lr_scheduler_min_lr_ratio")
        a += ["--lr_scheduler_min_lr_ratio", str(r if r is not None else 0.1)]
    if sch == "warmup_stable_decay" and int(p.get("lr_decay_steps") or 0) > 0:
        a += ["--lr_decay_steps", str(int(p["lr_decay_steps"]))]
    ws = p.get("weighting_scheme") or "none"
    if ws != "none":
        a += ["--weighting_scheme", ws]
        if ws == "logit_normal":
            a += ["--logit_mean", str(p.get("logit_mean") or 0.0), "--logit_std", str(p.get("logit_std") or 1.0)]
        if ws == "mode":
            a += ["--mode_scale", str(p.get("mode_scale") or 1.29)]
    if int(p.get("min_timestep") or 0) > 0:
        a += ["--min_timestep", str(int(p["min_timestep"]))]
    if p.get("max_timestep") not in (None, "") and int(p["max_timestep"]) < 1000:
        a += ["--max_timestep", str(int(p["max_timestep"]))]
    if p.get("split_attn") or p.get("attn_mode") == "xformers":
        a.append("--split_attn")
    if p.get("full_bf16"):
        a.append("--full_bf16")
    if float(p.get("scale_weight_norms") or 0) > 0:
        a += ["--scale_weight_norms", str(p["scale_weight_norms"])]
    if p.get("network_weights") and p.get("dim_from_weights"):
        a.append("--dim_from_weights")
    if p.get("base_weights"):
        a += ["--base_weights", p["base_weights"], "--base_weights_multiplier", str(p.get("base_weights_multiplier") or 1.0)]
    if p.get("resume"):
        a += ["--resume", p["resume"]]
    log = p.get("log_with") if p.get("log_with") in ("tensorboard", "all", "wandb") else "tensorboard"   # 训练页图表需要 TensorBoard 日志
    if log == "wandb":
        log = "all"
    a += ["--log_with", log, "--logging_dir", os.path.join(task_dir, "logs")]
    for k in ("log_prefix", "log_tracker_name", "wandb_run_name"):
        if (p.get(k) or "").strip():
            a += ["--" + k, p[k].strip()]
    if log == "all" and (p.get("wandb_api_key") or "").strip():
        a += ["--wandb_api_key", p["wandb_api_key"].strip()]
    if sch == "custom" and (p.get("lr_scheduler_type") or "").strip():
        a += ["--lr_scheduler_type", p["lr_scheduler_type"].strip()]
    if (p.get("lr_scheduler_args") or "").strip():
        a += ["--lr_scheduler_args"] + split_args(p["lr_scheduler_args"])
    if p.get("save_state") and int(p.get("save_last_n_epochs_state") or 0) > 0:
        a += ["--save_last_n_epochs_state", str(int(p["save_last_n_epochs_state"]))]
    if p.get("save_state_on_train_end"):
        a.append("--save_state_on_train_end")
    if p.get("full_fp16"):
        a.append("--full_fp16")
    if int(p.get("_num_gpus") or 1) > 1:   # 多卡时才传 DDP 参数（_num_gpus 由任务队列按选中的显卡填入）
        if int(p.get("ddp_timeout") or 0) > 0:
            a += ["--ddp_timeout", str(int(p["ddp_timeout"]))]
        if p.get("ddp_gradient_as_bucket_view", True):
            a.append("--ddp_gradient_as_bucket_view")
        if p.get("ddp_static_graph"):
            a.append("--ddp_static_graph")
    return a


def scheduler_name(p):
    """自定义调度器时，主调度器名填 constant（真正的调度器由 --lr_scheduler_type 指定）。"""
    s = p.get("lr_scheduler") or "constant"
    return "constant" if s == "custom" else s


def resolution_wh(p):
    """训练分辨率 (宽, 高)。"""
    if p.get("resolution") == "custom":
        try:
            w, h = [int(x) for x in str(p.get("resolution_wh") or "1024,1024").replace("x", ",").replace("×", ",").split(",")[:2]]
            return max(64, w // 64 * 64), max(64, h // 64 * 64)
        except ValueError:
            return 1024, 1024
    r = int(p.get("resolution") or 1024)
    return r, r


def common_network_args(p):
    """network_args 里共有的部分。"""
    from .base import split_args
    n = split_args(p.get("network_args") or "")
    for k in ("rank_dropout", "module_dropout", "loraplus_lr_ratio"):
        if float(p.get(k) or 0) > 0:
            n.append(f"{k}={p[k]}")
    return n


def override(fields, key, **kw):
    for f in fields:
        if f["key"] == key:
            f.update(kw)
    return fields


def drop(fields, *keys):
    return [f for f in fields if f["key"] not in keys]


def insert_after(fields, after_key, new_fields):
    out = []
    for f in fields:
        out.append(f)
        if f["key"] == after_key:
            out.extend(new_fields)
    return out

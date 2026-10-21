# -*- coding: utf-8 -*-
"""MOSS-Image 的第二个训练引擎：waw1w1 专门为 MOSS 写的训练器 moss_trainer（项目根目录 moss_trainer/）。

和 sd-scripts 引擎（moss.py）是同一个底模（family = "moss"），界面上是一张卡片、在卡片下面切换引擎。
sd-scripts 引擎兼容 int8 / 换主模型出图 / 冻结续训；这个引擎只认 bf16 单文件 + 官方配置目录，
支持 LoRA / LoCon / LoHa / LoKr（LyCORIS）、packed batch、融合 SITU / RMSNorm、torch.compile、
checkpoint 卸载、logit-normal 时间步、caption dropout、按轮出预览图。
用主训练环境（env）跑，额外只要 lycoris-lora。

原始项目：https://github.com/waw1w1/moss_image_trainer（已获作者同意整合）。"""
import functools
import json
import math
import os
import re

import paths
import safetensors_meta
from .base import F, Slot, write_atomic
from .moss import Moss

REPO = "OpenMOSS-Team/MOSS-Image-Base"   # 只下配置 / pipeline / 分词器，不下权重


@functools.lru_cache(maxsize=64)
def _official_vae(path):
    """官方 MOSS VAE 的解码器是精简版（decoder.conv_in 384 通道）；ComfyUI / BFL 的 flux2-vae 是完整的 512 通道，
    编码器一样（sd-scripts 能用），但专用引擎按官方 vae/config.json 严格加载，只能用官方那个。"""
    try:
        h, _ = safetensors_meta.read_header(path)
        return int(h["decoder.conv_in.weight"]["shape"][0]) == 384
    except Exception:
        return False


class MossNative(Moss):
    id = "moss_native"
    label = "MOSS-Image"
    subtitle = Moss.subtitle
    engine = "moss-trainer"
    family = "moss"
    engine_label = "MOSS 专用引擎"
    engine_note = "只认 bf16 单文件和官方 VAE；不支持冻结 / 续训"
    resume = False          # v1：专用引擎的断点续训还没接进来（界面不显示冻结 / 续训）
    slots = [
        Slot(
            "dit", "主模型（MOSS-Image DiT，bf16 单文件）", ["moss_dit"],
            help="专用引擎只支持未量化的 bf16 / fp16 单文件（int8 ConvRot、GGUF、FP8 都不认）；"
                 "官方分片目录请换 sd-scripts 引擎。",
            downloads=[{"label": "MOSS-Image Base bf16（10.2 GB）", "repo": "Comfy-Org/MOSS-Image",
                        "file": "diffusion_models/moss_image_base_bf16.safetensors", "size_gb": 10.16},
                       {"label": "MOSS-Image SFT bf16（10.2 GB）", "repo": "Comfy-Org/MOSS-Image",
                        "file": "diffusion_models/moss_image_sft_bf16.safetensors", "size_gb": 10.16}]),
        Slot(
            "te", "文本编码器（Qwen3.5-9B，bf16）", ["qwen35_9b"],
            help="必须未量化：bf16 版 19 GB，24 GB 显卡缓存特征时比较紧（DiT 不会同时加载）。int8 请换 sd-scripts 引擎。",
            downloads=[{"label": "Qwen3.5-9B bf16（19.3 GB）", "repo": "Comfy-Org/MOSS-Image",
                        "file": "text_encoders/qwen3.5_9b_bf16.safetensors", "size_gb": 19.31}]),
        Slot(
            "vae", "VAE（官方 MOSS VAE）", ["flux2_vae"],
            help="专用引擎要按官方 vae/config.json 加载，必须用官方仓库的 vae/diffusion_pytorch_model.safetensors"
                 "（decoder 384 通道）。ComfyUI 的 flux2-vae 是 512 通道（sd-scripts 能用，专用引擎会报形状不符）。",
            downloads=[{"label": "官方 MOSS VAE（专用引擎用，0.32 GB）", "repo": "OpenMOSS-Team/MOSS-Image-Base",
                        "file": "vae/diffusion_pytorch_model.safetensors", "size_gb": 0.32},
                       {"label": "ComfyUI flux2-vae（sd-scripts 用，0.34 GB）", "repo": "Comfy-Org/MOSS-Image",
                        "file": "vae/flux2-vae.safetensors", "size_gb": 0.34}]),
    ]
    sample_defaults = dict(Moss.sample_defaults, steps=40)

    def accept(self, slot, item):
        """专用引擎只吃未量化的 bf16 / fp16 单文件；int8 ConvRot / int4 / GGUF 都不认。"""
        if re.search(r"int[48]|convrot|fp8|nvfp4|gguf|w4a8|gptq|nf4|q[2-8]_[0-9k]", item["name"].lower()):
            return False
        if slot.key == "vae" and item["kind"] in slot.kinds:
            return _official_vae(item.get("path") or "")
        return item["kind"] in slot.kinds

    def score(self, slot_key, item):
        n = item["name"].lower()
        s = 0
        if slot_key == "dit":
            s += 100 if self.is_official(n) else 0
            s += 20 if "base" in n else (15 if "sft" in n else 0)
            s -= 60 if "int" in n or "convrot" in n else 0
        elif slot_key == "te":
            s += 5 if "qwen3.5" in n or "qwen35" in n else 0
        elif slot_key == "vae":
            s += 10 if "flux2" in n or "moss" in n else 0
        return s

    def base_label(self, task):
        return "MOSS"

    # ---------------- 参数 ----------------
    def params(self):
        return [
            F("gradient_checkpointing", "梯度检查点", "bool", True, "top",
              help="专用引擎用 torch 的 checkpoint 包住每个主干预块，省显存、慢一点。"),

            F("dataset_dir", "训练集位置", "dir", "", "basic",
              help="图片和同名 .txt 放在一起，子文件夹也会递归扫描。专用引擎按 kohya 分桶，不放大图。"),
            F("output_name", "保存名称", "text", "mylora_moss_v1", "basic", help="导出的 LoRA 文件名（不含扩展名）。"),
            F("repeats", "每张图重复次数", "int", 1, "basic", min=1, max=1000),
            F("epochs", "训练轮数", "int", 10, "basic", min=1, max=10000),
            F("batch_size", "单批训练数量", "int", 1, "basic", min=1, max=64),
            F("resolution", "训练分辨率", "select", 1024, "basic",
              options=[[768, "768"], [1024, "1024"], [1280, "1280"], [1536, "1536"], ["custom", "自定义宽×高"]],
              help="按面积分桶，边长步长 64。"),
            F("resolution_wh", "自定义分辨率（宽,高）", "text", "1024,1024", "basic",
              show_if={"resolution": "custom"}),
            F("network_type", "网络类型", "select", "lora", "basic",
              options=[["lora", "LoRA"], ["locon", "LoCon"], ["loha", "LoHa"], ["lokr", "LoKr"]],
              help="用 LyCORIS 训练，导出 ComfyUI 能直接读的 LyCORIS 格式。"),
            F("network_dim", "网络维度 dim", "int", 32, "basic", min=1, max=512),
            F("network_alpha", "网络 alpha", "float", 32, "basic", min=0, max=512, step=1),
            F("lokr_factor", "LoKr 分解因子", "int", 8, "basic", min=-1, max=64, show_if={"network_type": "lokr"},
              help="-1 = 自动。"),

            F("learning_rate", "学习率", "float", 1e-4, "lr", step=1e-5, sci=True),
            F("optimizer_type", "优化器", "select", "adamw", "lr",
              options=[["adamw", "AdamW"], ["adam", "Adam"], ["sgd", "SGD（momentum 0.9）"], ["rmsprop", "RMSprop"]]),
            F("lr_scheduler", "学习率调度", "select", "cosine", "lr",
              options=[["constant", "constant · 恒定"], ["linear", "linear · 线性下降"],
                       ["cosine", "cosine · 余弦下降"], ["cosine_with_restarts", "cosine_with_restarts · 余弦重启"]]),
            F("lr_warmup_ratio", "预热比例", "float", 0.3, "lr", step=0.05, min=0, max=1,
              help="预热占 1 轮优化步数的比例，0 = 不预热。"),
            F("lr_warmup_steps", "预热步数（覆盖比例）", "int", -1, "lr", min=-1,
              help="-1 = 用上面的比例；填具体步数则按步数预热。"),
            F("weight_decay", "权重衰减", "float", 0.01, "lr", step=0.001, min=0),
            F("max_grad_norm", "梯度裁剪", "float", 1.0, "lr", step=0.1, min=0),
            F("grad_accum", "梯度累积步数", "int", 4, "lr", min=1, max=64,
              help="等效批量 = 单批数量 × 累积步数。"),

            F("timestep_sampling", "时间步采样", "select", "logit_normal", "adv", "时间步 / 噪声",
              options=[["logit_normal", "logit_normal（专用引擎默认，推荐）"], ["uniform", "uniform"]]),
            F("logit_mean", "logit 均值", "float", 0.0, "adv", "时间步 / 噪声", step=0.1,
              show_if={"timestep_sampling": "logit_normal"}),
            F("logit_std", "logit 标准差", "float", 1.0, "adv", "时间步 / 噪声", step=0.1,
              show_if={"timestep_sampling": "logit_normal"}),
            F("timestep_shift", "时间步偏移", "float", 1.0, "adv", "时间步 / 噪声", step=0.1, min=0.1,
              help="1.0 = 不偏移；官方推理 shift=5，训练默认 1.0。"),
            F("caption_dropout", "标签丢弃概率", "float", 0.1, "adv", "时间步 / 噪声", step=0.01, min=0, max=1,
              help="每步随机把一部分样本的标签换成空标签，减少过拟合、让 CFG 更好用。"),
            F("checkpoint_policy", "检查点策略", "select", "full", "adv", "时间步 / 噪声",
              options=[["full", "full（整块重算，最省显存）"], ["attention", "attention（保留注意力，快一点）"]]),
            F("save_every_n_epochs", "每几轮保存一次", "int", 1, "adv", "训练细节", min=1, max=100),
            F("seed", "随机种子", "int", 42, "adv", "训练细节"),
            F("num_workers", "数据加载线程", "int", 0, "adv", "训练细节", min=0, max=16,
              help="专用引擎默认 0（主进程读缓存）；调大可提速。"),
            F("moss_max_token_length", "标签最大长度（token）", "int", 4096, "adv", "训练细节", min=256, max=4096, step=256),
            F("enable_bucket", "分桶", "bool", True, "adv", "训练细节"),
            F("min_bucket_reso", "最小桶边长", "int", 256, "adv", "训练细节"),
            F("max_bucket_reso", "最大桶边长", "int", 2048, "adv", "训练细节"),
            F("bucket_no_upscale", "不放大小图", "bool", True, "adv", "训练细节"),
            F("alpha_mask", "透明通道当 loss 遮罩", "bool", False, "adv", "训练细节",
              help="和 sd-scripts 版一样：不开时训练前自动把透明底改成白底。"),

            F("train_text_projection", "同时训练文本投影层", "bool", False, "adv", "LoRA 结构",
              help="给 DiT 里接文本的投影层 / 细化层也加适配器（Qwen 仍冻结）。"),
            F("train_norm", "同时训练 RMSNorm", "bool", False, "adv", "LoRA 结构", show_if={"network_type": "lokr"}),
            F("network_dropout", "dropout", "float", 0.0, "adv", "LoRA 结构", step=0.05, min=0, max=1,
              show_if={"network_type": ["lora", "locon"]}),
            F("rank_dropout", "rank dropout", "float", 0.0, "adv", "LoRA 结构", step=0.05, min=0, max=1),
            F("module_dropout", "module dropout", "float", 0.0, "adv", "LoRA 结构", step=0.05, min=0, max=1),
            F("use_tucker", "LoKr 用 Tucker 分解", "bool", False, "adv", "LoRA 结构",
              show_if={"network_type": "lokr"}),
            F("full_matrix", "LoKr 用完整矩阵", "bool", False, "adv", "LoRA 结构",
              show_if={"network_type": "lokr"}),

            F("compile_blocks", "torch.compile 主干预块", "bool", False, "adv", "显存 / 速度",
              help="首次运行要编译，之后快一些；需要 triton。先跑通再开。"),
            F("checkpoint_offload", "检查点激活卸载到 CPU", "bool", False, "adv", "显存 / 速度",
              help="进一步省显存，少量传输开销；配合梯度检查点用。"),
            F("fused_situ", "融合 SITU 激活", "bool", False, "adv", "显存 / 速度",
              help="用 Triton 融合推理激活的前向 / 反向；需要 triton，出问题就关掉。"),
            F("fused_rmsnorm", "融合 RMSNorm", "bool", False, "adv", "显存 / 速度"),

            F("config_dir", "官方配置目录", "dir", "", "adv", "其他",
              help="含 pipeline_moss_image.py / transformer / text_encoder / vae 配置和 tokenizer 的目录。"
                   "留空 = 自动下载到 models/moss_official（只下配置，约几十 MB）。"),
            F("sample_every_epochs", "每几轮出一张预览图", "int", 1, "adv", "其他", min=0, max=100,
              help="0 = 不出。预览图用「训练中测试」第一条提示词，存放在输出目录的 sample 里。"),
        ]

    def vram_presets(self):
        return [
            {"id": "16g", "label": "16 GB", "min_gb": 15.5,
             "note": "bf16 DiT + 梯度检查点 + 检查点卸载；批次 1",
             "values": {"batch_size": 1, "gradient_checkpointing": True, "checkpoint_offload": True,
                        "grad_accum": 4, "num_workers": 0}},
            {"id": "24g", "label": "24 GB", "min_gb": 23,
             "note": "bf16 DiT（约 10 GB）+ 梯度检查点；批次 1，累积 4",
             "values": {"batch_size": 1, "gradient_checkpointing": True, "checkpoint_offload": False,
                        "grad_accum": 4, "num_workers": 0}},
            {"id": "32g", "label": "32 GB+", "min_gb": 31,
             "note": "批次 2，累积 2；可以试试 fused 内核",
             "values": {"batch_size": 2, "gradient_checkpointing": True, "checkpoint_offload": False,
                        "grad_accum": 2, "num_workers": 2, "fused_situ": True, "fused_rmsnorm": True}},
        ]

    # ---------------- 环境 / 路径 ----------------
    @staticmethod
    def env_python():
        return paths.env_python()

    def config_dir(self, p):
        d = (p.get("config_dir") or "").strip()
        return d or os.path.join(paths.ROOT, "models", "moss_official")

    # ---------------- 训练步骤 ----------------
    def build_steps(self, task, task_dir):
        p = dict(self.defaults())
        p.update(task.get("params") or {})
        m = task.get("models") or {}
        env_py = self.env_python()
        if not env_py:
            raise RuntimeError("训练环境还没安装")
        notes = []
        if len(task.get("gpus") or []) > 1:
            notes.append("专用引擎目前只支持单卡，多卡设置会被忽略。")
        cfg_dir = self.config_dir(p)
        steps = []
        if not os.path.isfile(os.path.join(cfg_dir, "pipeline_moss_image.py")):
            import settings
            steps.append({"name": "下载 MOSS 官方配置", "kind": "cache", "cwd": paths.ROOT, "python": env_py,
                          "cmd": [os.path.join(paths.APP, "workers", "fetch_moss_config.py"), REPO, cfg_dir],
                          "env": {"PYTHONPATH": paths.ROOT, "HF_ENDPOINT": settings.hf_endpoint()}})
        cfg = os.path.join(task_dir, "moss_native.toml")
        self._write_config(cfg, task, p, m, task_dir, cfg_dir)
        out_dir = os.path.join(task_dir, "native_out")
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
        steps.append({"name": "缓存文本 / VAE 特征", "kind": "cache", "cwd": task_dir, "python": env_py,
                      "cmd": ["-m", "moss_trainer.cli", "cache", "--config", cfg],
                      "env": {"PYTHONPATH": paths.ROOT}})
        steps.append({"name": "训练", "kind": "train", "cwd": task_dir, "python": env_py,
                      "cmd": ["-m", "moss_trainer.cli", "train", "--config", cfg],
                      "env": {"PYTHONPATH": paths.ROOT}, "preview_name": p.get("output_name") or "mylora_moss_v1"})
        steps.append({"name": "导出 LoRA", "kind": "post", "cwd": paths.ROOT, "python": env_py,
                      "cmd": [os.path.join(paths.APP, "workers", "moss_native_export.py"),
                              "--src", out_dir, "--dest", task.get("output_dir") or paths.OUTPUT,
                              "--name", p.get("output_name") or "mylora_moss_v1"],
                      "env": {"PYTHONPATH": paths.ROOT}})
        if notes:
            steps[0].setdefault("notes", []).extend(notes)
        return steps

    def write_sample_prompts(self, task, task_dir):
        return None   # 专用引擎的预览图直接用它自己的 [sample] 配置

    # ---------------- 配置 ----------------
    def _write_config(self, path, task, p, m, task_dir, cfg_dir):
        w, h = self._resolution(p)
        sample = task.get("sample") or {}
        smp = sample.get("prompts") or []
        prompt = (smp[0].get("prompt") if smp else None) or self.sample_defaults.get("prompt") or ""
        negative = (smp[0].get("negative") if smp else None) or ""
        cfg = (smp[0].get("cfg") if smp else None) or 3.0
        max_epochs = int(p.get("epochs") or 10)
        algo = p.get("network_type") or "lora"
        warmup_steps = int(p.get("lr_warmup_steps") if p.get("lr_warmup_steps") is not None else -1)
        # 专用引擎的配置里 warmup_ratio / warmup_steps、max_epochs / max_steps、
        # save_every_epochs / save_every 各只能出现一个，否则 load_config 直接报错
        if warmup_steps >= 0:
            warmup_line = f"warmup_steps = {warmup_steps}"
        else:
            warmup_line = f"warmup_ratio = {float(p.get('lr_warmup_ratio') or 0)}"
        q = lambda v: json.dumps(str(v), ensure_ascii=False)
        toml = f"""[paths]
image_dir = {q(p.get('dataset_dir') or '')}
output_dir = {q(os.path.join(task_dir, 'native_out'))}
model_format = "comfyui"
model_dir = {q(cfg_dir)}
model = {q(m.get('dit') or '')}
te = {q(m.get('te') or '')}
vae = {q(m.get('vae') or '')}
cache_dir = {q(os.path.join(task_dir, 'native_cache'))}

[dataset]
caption_extension = ".txt"
resolution = [{w}, {h}]
enable_bucket = {str(bool(p.get('enable_bucket', True))).lower()}
min_bucket_reso = {int(p.get('min_bucket_reso') or 256)}
max_bucket_reso = {int(p.get('max_bucket_reso') or 2048)}
bucket_reso_steps = 64
bucket_no_upscale = {str(bool(p.get('bucket_no_upscale', True))).lower()}
batch_size = {int(p.get('batch_size') or 1)}
num_repeats = {int(p.get('repeats') or 1)}

[cache]
max_sequence_length = {int(p.get('moss_max_token_length') or 4096)}

[network]
algo = {q(algo)}
dim = {int(p.get('network_dim') or 32)}
alpha = {float(p.get('network_alpha') or 32)}
factor = {int(p.get('lokr_factor') or 8)}
full_matrix = {str(bool(p.get('full_matrix'))).lower()}
decompose_both = false
train_text_projection = {str(bool(p.get('train_text_projection'))).lower()}
train_norm = {str(bool(p.get('train_norm'))).lower()}
target = "attention-mlp"
dropout = {float(p.get('network_dropout') or 0)}
rank_dropout = {float(p.get('rank_dropout') or 0)}
module_dropout = {float(p.get('module_dropout') or 0)}
block_size = 4
use_tucker = {str(bool(p.get('use_tucker'))).lower()}
dora_wd = false
constraint = 0.0

[training]
max_epochs = {max_epochs}
learning_rate = {float(p.get('learning_rate') or 1e-4)}
weight_decay = {float(p.get('weight_decay') or 0.01)}
gradient_accumulation_steps = {int(p.get('grad_accum') or 1)}
batch_mode = "packed"
max_grad_norm = {float(p.get('max_grad_norm') or 1.0)}
gradient_checkpointing = {str(bool(p.get('gradient_checkpointing'))).lower()}
checkpoint_policy = {q(p.get('checkpoint_policy') or 'full')}
fused_situ = {str(bool(p.get('fused_situ'))).lower()}
fused_rmsnorm = {str(bool(p.get('fused_rmsnorm'))).lower()}
compile_blocks = {str(bool(p.get('compile_blocks'))).lower()}
checkpoint_offload = {str(bool(p.get('checkpoint_offload'))).lower()}
checkpoint_blocks = -1
optimizer = {q(p.get('optimizer_type') or 'adamw')}
scheduler = {q(p.get('lr_scheduler') or 'cosine')}
cosine_restarts = 0
{warmup_line}
save_every_epochs = {int(p.get('save_every_n_epochs') or 1)}
seed = {int(p.get('seed') or 42)}
timestep_sampling = {q(p.get('timestep_sampling') or 'logit_normal')}
logit_mean = {float(p.get('logit_mean') or 0)}
logit_std = {float(p.get('logit_std') or 1)}
timestep_shift = {float(p.get('timestep_shift') or 1)}
caption_dropout = {float(p.get('caption_dropout') or 0)}
num_workers = {int(p.get('num_workers') or 0)}
tensorboard = true

[sample]
positive = {q(prompt)}
negative = {q(negative)}
cfg = {float(cfg)}
resolution = [{w}, {h}]
scheduler = "moss"
sampler = "euler"
every_epochs = {int(p.get('sample_every_epochs') or 0) if sample.get('enabled') else 0}
"""
        write_atomic(path, toml)

    @staticmethod
    def _resolution(p):
        if str(p.get("resolution") or "") == "custom":
            txt = str(p.get("resolution_wh") or "1024,1024").replace("×", ",").replace("x", ",")
            try:
                w, h = [int(x.strip()) for x in txt.split(",")[:2]]
            except (ValueError, IndexError):
                w, h = 1024, 1024
        else:
            w = h = int(p.get("resolution") or 1024)
        w = max(256, (w // 64) * 64)
        h = max(256, (h // 64) * 64)
        return w, h

# -*- coding: utf-8 -*-
"""底模类型的统一接口。

以后要支持新的底模（SDXL、Flux、Qwen-Image、Z-Image……）只需要在 app/models/ 下
新建一个文件，写一个继承 ModelSpec 的类并调用 register()，前端会自动出现：
  - 「模型设置」页的底模卡片和各个模型槽位（按 slots 生成，自动搜索按 kinds 过滤）
  - 「高级参数」页的表单（按 params() 生成）
  - 显存预设按钮（vram_presets()）
  - 「训练中测试」的默认出图参数（sample_defaults）
训练命令由 build_steps() 生成，引擎（sd-scripts / musubi-tuner / 以后别的）由 engine 字段区分。
"""
import math
import os
import re

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def F(key, label, type="text", default=None, section="adv", group="", **kw):
    """表单字段描述（前端 params.js 按它渲染）。

    type: int / float / text / bool / select / dir / file / textarea
    section: top（最上面）/ basic（基础）/ lr（学习率）/ adv（其他高级设置）
    group:  adv 里的小标题
    show_if: {"key": value 或 [values]} 满足才显示
    """
    d = {"key": key, "label": label, "type": type, "default": default, "section": section, "group": group}
    d.update(kw)
    return d


def write_atomic(path, text):
    """先写临时文件再替换：训练中途改测试提示词时，训练进程随时可能在读这个文件，不能让它读到写了一半的内容。"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    return path


class ParamError(ValueError):
    """开训前检查出的参数问题（build_steps 里抛），任务失败时按「参数需要调整」显示，不当成程序出错。"""


class Slot:
    """一个模型槽位，例如主模型 / 文本编码器 / VAE。"""

    def __init__(self, key, label, kinds, required=True, help="", downloads=None, allow_quant=False, cli=None):
        self.key = key
        self.label = label
        self.kinds = kinds              # 接受哪些分类（safetensors_meta.classify 的 kind）
        self.required = required
        self.help = help
        self.downloads = downloads or []  # [{label, repo, file, size_gb}] 找不到时可以一键下载
        self.allow_quant = allow_quant
        self.cli = cli

    def to_dict(self):
        return {"key": self.key, "label": self.label, "kinds": self.kinds, "required": self.required,
                "help": self.help, "downloads": self.downloads, "allow_quant": self.allow_quant}


class ModelSpec:
    id = ""
    label = ""
    subtitle = ""
    engine = ""            # sd-scripts / musubi-tuner
    caption_style = "tags"  # tags / natural / mixed —— 反推页的默认推荐
    slots = []
    sample_defaults = {}
    resume = False          # 支持冻结 / 续训（训练引擎能从保存的训练状态或某一轮的 LoRA 接着练）
    family = ""             # 同一个底模的不同训练引擎填同一个值：界面上合成一张卡片，卡片下面切换引擎
    engine_label = ""       # 切换引擎时显示的名字
    engine_note = ""
    linux_only = False      # 训练引擎只能在 Linux 上跑：Windows 下界面不列出
    # 主模型下拉框的分组：[variant 值, 组名, 选中这一组的模型时显示的提示]
    variant_groups = [
        ["official", "官方底模（推荐，练出来的 LoRA 在各个微调版上都通用）", ""],
        ["finetune", "微调 / 合并版", "用微调版训练：LoRA 在这个微调版上效果最好，换到官方底模或别的微调版上可能会变样；想通用就选官方底模。"],
        ["turbo", "Turbo / 加速蒸馏版（不建议拿来训练）", "⚠ 这是 Turbo / 加速蒸馏版，拿它训练 LoRA 效果通常不好，建议换成官方底模。"],
    ]

    # ---------- 自动选择 ----------
    def accept(self, slot, item):
        """这个文件能不能放进这个槽位（自选项只保留能用的文件）。默认：不接受量化模型。"""
        return item["kind"] in slot.kinds and (slot.allow_quant or not item.get("quant"))

    def score(self, slot_key, item):
        """候选文件打分，越高越优先。子类按「优先 base 模型」等规则覆盖。"""
        s = 0
        if item.get("quant"):
            s -= 50
        return s

    def variant(self, slot_key, item):
        """主模型下拉框分组用：official 官方底模 / finetune 微调·合并版 / turbo 加速·蒸馏版；其它槽位返回 ""。"""
        if slot_key != "dit":
            return ""
        n = item["name"].lower()
        if re.search(r"turbo|lightning|distill|dmd|lcm|hyper|[-_]\d+steps?\b", n):
            return "turbo"
        return "official" if self.is_official(n) else "finetune"

    def is_official(self, name):
        return False

    def sample_defaults_for(self, models):
        """测试出图的默认提示词（标准正向 + 标准负向）。同一底模不同系列写法不同时（SDXL 的 Illustrious / NoobAI / Pony）按选的主模型返回。"""
        return dict(self.sample_defaults)

    def base_label(self, task):
        """送到 WWY 启动器时的底模分类名（启动器按它分文件夹）。"""
        return self.label

    # ---------- 参数 ----------
    def params(self):
        return []

    def vram_presets(self):
        """[{id, label, min_gb, values:{...}, note}]，按显存从小到大。"""
        return []

    def recommend_preset(self, vram_gb, gpu=None):
        best = None
        for p in self.vram_presets():
            if vram_gb + 0.5 >= p["min_gb"]:
                best = p
        return best or (self.vram_presets() or [None])[0]

    def defaults(self):
        return {f["key"]: f["default"] for f in self.params() if f.get("default") is not None}

    # ---------- 命令 ----------
    def build_steps(self, task, task_dir):
        """返回 [{name, cmd, cwd}]。"""
        raise NotImplementedError

    def resume_args(self, r):
        """续训时追加到训练命令的参数。r = {"mode": "state", "state": 状态文件夹} 或 {"mode": "weights", "weights": LoRA 文件, "epoch": 已练完的轮数}"""
        return []

    def write_sample_prompts(self, task, task_dir):
        """把任务的测试提示词写进任务目录，返回文件路径（没开测试出图返回 None）。训练中途改提示词时也调它重写。"""
        return None

    def sample_line(self, p):
        """把一条测试提示词转成引擎认识的一行（kohya 系格式）。"""
        line = (p.get("prompt") or "").replace("\n", " ").strip()
        if p.get("negative"):
            line += f" --n {p['negative'].strip()}"
        for k, flag in (("width", "w"), ("height", "h"), ("steps", "s"), ("cfg", "l"), ("seed", "d")):
            if p.get(k) not in (None, ""):
                line += f" --{flag} {p[k]}"
        return line

    def sample_interval_args(self, task):
        """测试出图间隔。默认跟随「每几轮保存一次」：每存一个 LoRA 就出一组图，测试图和存下来的文件一一对应。"""
        smp = task.get("sample") or {}
        p = dict(self.defaults())
        p.update(task.get("params") or {})
        if smp.get("follow_save", True):
            n = int(p.get("save_every_n_epochs") or 0)
            if n > 0:
                return ["--sample_every_n_epochs", str(n)]
            # 只保存最终结果：只在训练结束时出一组
            if p.get("steps_mode") == "steps":
                return ["--sample_every_n_steps", str(int(p.get("max_train_steps") or 1))]
            return ["--sample_every_n_epochs", str(int(p.get("epochs") or 1))]
        a = []
        if int(smp.get("every_n_steps") or 0) > 0:
            a += ["--sample_every_n_steps", str(int(smp["every_n_steps"]))]
        if int(smp.get("every_n_epochs") or 0) > 0:
            a += ["--sample_every_n_epochs", str(int(smp["every_n_epochs"]))]
        return a

    def to_dict(self):
        params = self.params()
        for f in params:   # 「其他高级设置」按用途分页
            if f["section"] == "adv" and f["key"] in ADV_GROUPS:
                f["group"] = ADV_GROUPS[f["key"]]
        return {"id": self.id, "label": self.label, "subtitle": self.subtitle, "engine": self.engine,
                "caption_style": self.caption_style, "slots": [s.to_dict() for s in self.slots],
                "params": params, "presets": self.vram_presets(), "sample_defaults": self.sample_defaults,
                "adv_order": ADV_ORDER, "resume": self.resume, "variant_groups": self.variant_groups,
                "family": self.family or self.id, "engine_label": self.engine_label or self.engine, "engine_note": self.engine_note}


ADV_ORDER = ["训练细节", "标签处理", "数据增强", "分桶", "精度与保存", "时间步 / 噪声", "显存 / 速度", "LoRA 结构", "分层学习率",
             "文本编码", "正则化", "验证集", "多卡训练", "日志", "其他"]
ADV_GROUPS = {
    **{k: "标签处理" for k in ("caption_extension", "shuffle_caption", "keep_tokens", "keep_tokens_separator", "caption_dropout_rate",
                            "caption_dropout_every_n_epochs", "caption_tag_dropout_rate", "caption_prefix", "caption_suffix",
                            "secondary_separator", "enable_wildcard")},
    **{k: "数据增强" for k in ("flip_aug", "color_aug", "random_crop", "alpha_mask")},
    **{k: "分桶" for k in ("enable_bucket", "min_bucket_reso", "max_bucket_reso", "bucket_reso_steps", "bucket_no_upscale",
                          "resize_interpolation")},
    **{k: "精度与保存" for k in ("mixed_precision", "save_precision", "full_fp16", "save_model_as", "save_state", "save_last_n",
                            "save_last_n_epochs_state", "save_state_on_train_end", "resume", "export_comfy")},
}


# ---------------- 训练集 ----------------
def dataset_subsets(dataset_dir, repeats, use_folder_repeats=True):
    """把训练集拆成子集：根目录里的图片一组；每个含图片的子文件夹一组。

    子文件夹名是 kohya 习惯的「10_xxx」时，可以沿用前面的数字当重复次数；
    否则（或关闭沿用）一律用界面上设置的重复次数——不用再靠改文件夹名。"""
    subsets = []

    def count(d):
        try:
            return sum(1 for f in os.listdir(d) if f.lower().endswith(IMAGE_EXTS))
        except OSError:
            return 0

    if not dataset_dir or not os.path.isdir(dataset_dir):
        return subsets
    n = count(dataset_dir)
    if n:
        subsets.append({"dir": dataset_dir, "images": n, "repeats": int(repeats), "from_folder": False})
    for root, dirs, _files in os.walk(dataset_dir):
        dirs[:] = [d for d in dirs if not d.startswith(("_", "."))]
        if root == dataset_dir:
            continue
        n = count(root)
        if not n:
            continue
        r = int(repeats)
        m = re.match(r"^(\d+)_", os.path.basename(root))
        if m and use_folder_repeats:
            r = int(m.group(1))
        subsets.append({"dir": root, "images": n, "repeats": r, "from_folder": bool(m and use_folder_repeats)})
    return subsets


def estimate_steps(params, subsets):
    """估算总步数（分桶会让每轮步数略多一点，这里按理想情况算）。多卡时每张卡各跑一批，每轮步数按卡数平分。"""
    bs = max(1, int(params.get("batch_size") or 1)) * max(1, int(params.get("_num_gpus") or 1))
    ga = max(1, int(params.get("grad_accum") or 1))
    total_imgs = sum(s["images"] * s["repeats"] for s in subsets)
    per_epoch = math.ceil(total_imgs / bs / ga) if total_imgs else 0
    if params.get("steps_mode") == "steps":
        total = int(params.get("max_train_steps") or 0)
        epochs = math.ceil(total / per_epoch) if per_epoch else 0
    else:
        epochs = int(params.get("epochs") or 1)
        total = per_epoch * epochs
    return {"images": sum(s["images"] for s in subsets), "samples_per_epoch": total_imgs,
            "steps_per_epoch": per_epoch, "epochs": epochs, "total_steps": total}


def q(v):
    """TOML 字符串转义。"""
    return '"' + str(v).replace("\\", "/").replace('"', '\\"') + '"'


def split_args(s):
    """把「额外参数」文本拆成参数列表（支持引号）。"""
    import shlex
    out = []
    for a in shlex.split(s or "", posix=False):   # posix=False：保留 Windows 路径里的反斜杠
        if len(a) >= 2 and a[0] == a[-1] and a[0] in "\"'":
            a = a[1:-1]
        out.append(a)
    return out

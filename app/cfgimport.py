# -*- coding: utf-8 -*-
"""读取别的训练器导出的配置，换算成这个训练器的参数（启动训练页「导入外部配置」）。

认得的格式：
  - kohya-ss/sd-scripts、musubi-tuner 的 TOML 配置（--config_file）和数据集配置（--dataset_config）
  - kohya_ss GUI 保存的 JSON
  - 秋叶 lora-scripts（SD-Trainer）的 TOML、train.sh / train.ps1
  - ai-toolkit 的 YAML / JSON
  - OneTrainer 导出的 JSON（含单独的 concepts / samples 文件）
  - SimpleTuner 的 config.json / config.toml / multidatabackend.json
  - 训练命令行（accelerate launch … xxx_train_network.py --xxx …，.sh / .bat / .ps1 / 本训练器的 commands.txt）
  - sd-scripts 练出来的 LoRA 文件里记的训练参数（ss_ 元数据）
  - 本训练器自己的配置存档（data/presets.json）、草稿、任务

流程：读文件 → 按格式整理成 sd-scripts 风格的「参数名 → 值」→ 换成本训练器的参数名 → 按目标底模的参数表逐项检查。
只用标准库：Python 3.10 没有 tomllib、标准库没有 YAML，各带一个够用的小解析器。
"""
import json
import os
import re

import models
import scanner

# ======================= 读文件 =======================


def _load(text, name):
    """返回 (数据, 种类)。种类：data（JSON / TOML / YAML 读出来的字典或列表）/ cmd（命令行脚本，数据是原文）。"""
    text = (text or "").lstrip("\ufeff")
    if not text.strip():
        raise ValueError("文件是空的")
    ext = os.path.splitext((name or "").lower())[1]
    loaders = {".json": [_json], ".safetensors": [_json], ".toml": [_toml], ".yaml": [_yaml], ".yml": [_yaml],
               ".sh": [], ".bat": [], ".cmd": [], ".ps1": [], ".txt": []}.get(ext)
    if loaders is None:
        loaders = [_json, _toml] + ([_yaml] if re.search(r"^\s*[\w.\-\"']+\s*:(\s|$)", text, re.M) else [])
    err = None
    for fn in loaders:
        try:
            d = fn(text)
            if isinstance(d, (dict, list)) and d:
                return d, "data"
        except Exception as e:  # noqa: BLE001 —— 换下一种格式试
            err = err or e
    if re.search(r"(^|\s)--[A-Za-z]", text):
        return text, "cmd"
    if err and loaders and ext in (".json", ".toml", ".yaml", ".yml", ".safetensors"):
        raise ValueError(f"文件格式有误，读不出来：{err}")
    raise ValueError("认不出这个文件的格式。支持 TOML / JSON / YAML 配置、训练命令行和 LoRA 文件。")


def _json(text):
    return json.loads(text)


def _toml(text):
    try:
        import tomllib
        return tomllib.loads(text)
    except ImportError:
        return _MiniToml(text).parse()


def _yaml(text):
    try:
        import yaml  # 训练环境里一般有 PyYAML；界面用的便携 Python 里没有
        return yaml.safe_load(text)
    except ImportError:
        return _mini_yaml(text)


class _MiniToml:
    """Python 3.10 用的 TOML 解析（训练配置用得到的部分：表、表数组、字符串、数字、布尔、数组、内联表）。"""

    def __init__(self, s):
        self.s, self.i = s, 0

    def parse(self):
        root = {}
        cur = root
        while True:
            self._ws(nl=True)
            if self.i >= len(self.s):
                return root
            c = self.s[self.i]
            if c == "[":
                arr = self.s.startswith("[[", self.i)
                self.i += 2 if arr else 1
                keys = self._keys()
                self._ws()
                self._expect("]]" if arr else "]")
                tbl = root
                for k in keys[:-1]:
                    tbl = tbl.setdefault(k, {})
                    if isinstance(tbl, list):
                        tbl = tbl[-1]
                if arr:
                    tbl.setdefault(keys[-1], []).append({})
                    cur = tbl[keys[-1]][-1]
                else:
                    cur = tbl.setdefault(keys[-1], {})
            else:
                self._pair(cur)
            self._eol()

    def _pair(self, tbl):
        keys = self._keys()
        self._ws()
        self._expect("=")
        self._ws()
        v = self._value()
        for k in keys[:-1]:
            tbl = tbl.setdefault(k, {})
        tbl[keys[-1]] = v

    def _keys(self):
        keys = []
        while True:
            self._ws()
            c = self.s[self.i]
            if c in "\"'":
                keys.append(self._str())
            else:
                m = re.compile(r"[A-Za-z0-9_\-]+").match(self.s, self.i)
                if not m:
                    raise ValueError(f"第 {self._line()} 行：键名有误")
                keys.append(m.group())
                self.i = m.end()
            self._ws()
            if self.i < len(self.s) and self.s[self.i] == ".":
                self.i += 1
                continue
            return keys

    def _value(self):
        s, i = self.s, self.i
        c = s[i]
        if c in "\"'":
            return self._str()
        if c == "[":
            self.i += 1
            out = []
            while True:
                self._ws(nl=True)
                if s[self.i] == "]":
                    self.i += 1
                    return out
                out.append(self._value())
                self._ws(nl=True)
                if s[self.i] == ",":
                    self.i += 1
        if c == "{":
            self.i += 1
            out = {}
            while True:
                self._ws()
                if s[self.i] == "}":
                    self.i += 1
                    return out
                self._pair(out)
                self._ws()
                if s[self.i] == ",":
                    self.i += 1
        m = re.compile(r"[^\s,\]\}#]+").match(s, i)
        if not m:
            raise ValueError(f"第 {self._line()} 行：值有误")
        self.i = m.end()
        t = m.group()
        if t in ("true", "false"):
            return t == "true"
        t2 = t.replace("_", "")
        for conv in (int, float):
            try:
                return conv(t2)
            except ValueError:
                pass
        if re.match(r"0[xob]", t2):
            return int(t2, 0)
        return t   # 日期时间等原样当字符串

    def _str(self):
        s = self.s
        q = s[self.i]
        if s.startswith(q * 3, self.i):
            end = s.index(q * 3, self.i + 3)
            body = s[self.i + 3:end]
            self.i = end + 3
            if body.startswith("\n"):
                body = body[1:]
            return self._unescape(body) if q == '"' else body
        if q == "'":
            end = s.index("'", self.i + 1)
            body = s[self.i + 1:end]
            self.i = end + 1
            return body
        j = self.i + 1
        while s[j] != '"':
            j += 2 if s[j] == "\\" else 1
        body = s[self.i + 1:j]
        self.i = j + 1
        return self._unescape(body)

    @staticmethod
    def _unescape(b):
        return re.sub(r'\\(u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|.)',
                      lambda m: chr(int(m.group(1)[1:], 16)) if m.group(1)[0] in "uU" and len(m.group(1)) > 1
                      else {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f"}.get(m.group(1), m.group(1)), b, flags=re.S)

    def _ws(self, nl=False):
        s = self.s
        while self.i < len(s):
            c = s[self.i]
            if c in " \t" or (nl and c in "\r\n"):
                self.i += 1
            elif c == "#":
                while self.i < len(s) and s[self.i] != "\n":
                    self.i += 1
            else:
                break

    def _eol(self):
        self._ws()
        if self.i < len(self.s) and self.s[self.i] not in "\r\n":
            raise ValueError(f"第 {self._line()} 行：多余的内容")

    def _expect(self, t):
        if not self.s.startswith(t, self.i):
            raise ValueError(f"第 {self._line()} 行：缺少 {t}")
        self.i += len(t)

    def _line(self):
        return self.s.count("\n", 0, self.i) + 1


def _mini_yaml(text):
    """不依赖 PyYAML 的 YAML 解析（ai-toolkit 配置用得到的部分：缩进的字典 / 列表、行内 [] {}、引号字符串、| > 多行文本）。"""
    lines = []
    raw = text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(raw):
        ln = raw[i]
        i += 1
        if not ln.strip() or ln.strip().startswith("#") or ln.strip() in ("---", "..."):
            continue
        ind = len(ln) - len(ln.lstrip(" "))
        body = _yaml_uncomment(ln.strip())
        m = re.match(r"^(.*?)(:|-)\s*([|>])[-+]?$", body)
        if m and (m.group(2) == "-" and m.group(1) == "" or m.group(2) == ":"):
            # 多行文本：收下面缩进更深的行
            block = []
            while i < len(raw) and (not raw[i].strip() or len(raw[i]) - len(raw[i].lstrip(" ")) > ind):
                block.append(raw[i])
                i += 1
            pad = min((len(x) - len(x.lstrip(" ")) for x in block if x.strip()), default=0)
            joiner = "\n" if m.group(3) == "|" else " "
            val = joiner.join(x[pad:] for x in block).strip()
            body = (m.group(1) + m.group(2) + " " if m.group(2) == ":" else "- ") + json.dumps(val, ensure_ascii=False)
        lines.append((ind, body))
    if not lines:
        return {}
    val, _ = _yaml_block(lines, 0, lines[0][0])
    return val


def _yaml_uncomment(s):
    q = None
    for j, c in enumerate(s):
        if q:
            if c == q:
                q = None
        elif c in "\"'":
            q = c
        elif c == "#" and (j == 0 or s[j - 1] in " \t"):
            return s[:j].rstrip()
    return s


def _yaml_block(lines, i, ind):
    if lines[i][1].startswith("- ") or lines[i][1] == "-":
        out = []
        while i < len(lines) and lines[i][0] == ind and (lines[i][1].startswith("- ") or lines[i][1] == "-"):
            rest = lines[i][1][1:].strip()
            if not rest:
                i += 1
                if i < len(lines) and lines[i][0] > ind:
                    v, i = _yaml_block(lines, i, lines[i][0])
                else:
                    v = None
                out.append(v)
            elif re.match(r"^[^\"'\[{][^:]*:(\s|$)", rest) or re.match(r"^[\"'][^\"']*[\"']\s*:(\s|$)", rest):
                # 「- key: value」：后面缩进更深的行接着属于这个字典
                sub = [(ind + 2, rest)]
                i += 1
                while i < len(lines) and lines[i][0] > ind:
                    sub.append(lines[i])
                    i += 1
                v, _ = _yaml_block(sub, 0, ind + 2)
                out.append(v)
            else:
                out.append(_yaml_scalar(rest))
                i += 1
        return out, i
    out = {}
    while i < len(lines) and lines[i][0] == ind:
        body = lines[i][1]
        m = re.match(r"^(\"[^\"]*\"|'[^']*'|[^:]+?)\s*:(?:\s+(.*))?$", body)
        if not m:
            raise ValueError(f"YAML 格式有误：{body[:60]}")
        k = m.group(1).strip("\"'")
        rest = (m.group(2) or "").strip()
        i += 1
        if rest:
            out[k] = _yaml_scalar(rest)
        elif i < len(lines) and (lines[i][0] > ind or (lines[i][0] == ind and lines[i][1].startswith("-"))):
            out[k], i = _yaml_block(lines, i, lines[i][0])
        else:
            out[k] = None
    return out, i


def _yaml_scalar(s):
    s = s.strip()
    if not s:
        return None
    if s[0] in "[{":
        return _yaml_flow(s)
    if s[0] in "\"'" and s[-1] == s[0] and len(s) > 1:
        if s[0] == '"':
            try:
                return json.loads(s)
            except ValueError:
                pass
        return s[1:-1]
    low = s.lower()
    if low in ("null", "~"):
        return None
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    for conv in (int, float):
        try:
            return conv(s)
        except ValueError:
            pass
    return s


def _yaml_flow(s):
    """行内 [a, b] / {k: v}（可嵌套）。"""
    pos = [0]

    def val():
        while s[pos[0]] in " \t":
            pos[0] += 1
        c = s[pos[0]]
        if c in "[{":
            close = "]" if c == "[" else "}"
            pos[0] += 1
            out = [] if c == "[" else {}
            while True:
                while s[pos[0]] in " \t,":
                    pos[0] += 1
                if s[pos[0]] == close:
                    pos[0] += 1
                    return out
                if c == "[":
                    out.append(val())
                else:
                    k = atom(":")
                    pos[0] += 1
                    out[str(k)] = val()
        return atom(",]}")

    def atom(stops):
        st = pos[0]
        if s[st] in "\"'":
            end = s.index(s[st], st + 1)
            pos[0] = end + 1
            return s[st + 1:end]
        while pos[0] < len(s) and s[pos[0]] not in stops:
            pos[0] += 1
        return _yaml_scalar(s[st:pos[0]])

    return val()


# ======================= 各种格式 → 统一的「参数名 → 值」 =======================

class _R:
    """解析结果。kv：sd-scripts 风格参数；models：{槽位: 路径}；signals：判断底模的线索（按可信度排）。"""

    def __init__(self):
        self.fmt = ""
        self.kv = {}
        self.models = {}
        self.signals = []
        self.sample = None        # {"prompts": [...], "every_n_steps", "every_n_epochs", "at_first"}
        self.notes = []
        self.native = None        # 本训练器自己的格式：{"model_type", "params", "models", "sample"}
        self.presets = None       # data/presets.json：{名字: 草稿}
        self.output_dir = ""
        self.base_dir = ""        # 配置文件所在目录（解析相对路径用；浏览器上传的文件没有）

    def put(self, k, v, override=False):
        if v is None or v == "" or v == []:
            return
        if override or k not in self.kv:
            self.kv[k] = v


def _extract(R, data, kind):
    if kind == "cmd":
        return _from_cmd(R, data)
    if isinstance(data, list):
        dicts = [x for x in data if isinstance(x, dict)]
        if any("instance_data_dir" in x for x in dicts):
            R.fmt = "SimpleTuner 数据集配置（multidatabackend.json）"
            imgs = [x for x in dicts if x.get("instance_data_dir") and x.get("dataset_type", "image") == "image"]
            _datasets(R, [{"image_dir": x.get("instance_data_dir"), "num_repeats": x.get("repeats"),
                           "resolution": x.get("resolution") if x.get("resolution_type", "pixel") in ("pixel", "pixel_area") else None,
                           "caption_extension": ".txt" if x.get("caption_strategy") == "textfile" else None} for x in imgs])
            return
        if any("path" in x and ({"image", "text", "repeats", "concept_stats"} & x.keys()) for x in dicts):
            R.fmt = "OneTrainer 训练概念（concepts）"
            return _ot_concepts(R, dicts)
        if any("prompt" in x and ("diffusion_steps" in x or "cfg_scale" in x) for x in dicts):
            R.fmt = "OneTrainer 测试提示词（samples）"
            return _ot_samples(R, dicts)
        raise ValueError("认不出这个文件的格式")
    if not isinstance(data, dict):
        raise ValueError("认不出这个文件的格式")
    keys = set(data.keys())

    # ---- 本训练器 ----
    wt = data.get("wwy_lora_trainer")
    if isinstance(wt, dict) and isinstance(wt.get("config"), str):   # 「导出配置」的 TOML：用最后那份完整配置
        try:
            c = json.loads(wt["config"])
        except ValueError:
            raise ValueError("[wwy_lora_trainer] 里的配置坏了")
        R.fmt = "本训练器导出的配置"
        R.native = {"model_type": c.get("model_type") or "", "params": dict(c.get("params") or {}),
                    "models": dict(c.get("models") or {}), "sample": c.get("sample")}
        R.output_dir = c.get("output_dir") or ""
        R.signals.append(R.native["model_type"])
        return
    if data and all(isinstance(v, dict) and "by_model" in v for v in data.values()):
        R.fmt = "本训练器的配置存档文件"
        R.presets = data
        return
    if "by_model" in data or {"model_type", "params", "models"} <= keys:
        R.fmt = "本训练器的配置" if "by_model" in data else "本训练器的训练任务"
        mt = data.get("model_type") or ""
        c = (data.get("by_model") or {}).get(mt) if "by_model" in data else data
        if not c:
            raise ValueError("这份配置里没有选底模")
        R.native = {"model_type": mt, "params": dict(c.get("params") or {}), "models": dict(c.get("models") or {}),
                    "sample": c.get("sample")}
        R.output_dir = data.get("output_dir") or ""
        R.signals.append(mt)
        return

    # ---- LoRA 文件里的训练信息 ----
    if any(k.startswith("ss_") for k in keys) or any(k.startswith("modelspec.") for k in keys):
        return _from_ss(R, data)

    # ---- OneTrainer ----
    if "training_method" in keys or ("base_model_name" in keys and ("concepts" in keys or "concept_file_name" in keys)):
        return _from_onetrainer(R, data)

    # ---- ai-toolkit ----
    cfg = data.get("config") if isinstance(data.get("config"), dict) else data
    if isinstance(cfg.get("process"), list) and cfg["process"] and isinstance(cfg["process"][0], dict):
        return _from_aitk(R, cfg["process"][0], cfg)

    # ---- SimpleTuner ----
    if any(k.startswith("--") for k in keys) or "model_family" in keys:
        R.fmt = "SimpleTuner 配置"
        flat = {k.lstrip("-"): v for k, v in data.items()}
        R.signals.append(str(flat.get("model_family") or ""))
        if flat.get("validation_prompt"):
            R.sample = {"prompts": [_prompt_obj(flat.get("validation_prompt"), flat.get("validation_negative_prompt"),
                                                flat.get("validation_resolution"), flat.get("validation_resolution"),
                                                flat.get("validation_num_inference_steps"), flat.get("validation_guidance"),
                                                flat.get("validation_seed"))]}
            if flat.get("validation_steps"):
                R.sample["every_n_steps"] = _int(flat["validation_steps"])
        if flat.get("lora_type") == "lycoris":
            R.notes.append("原配置用 LyCORIS（lycoris_config 单独的文件），网络类型请自己确认。")
        if flat.get("data_backend_config"):
            R.notes.append(f"训练集写在 {flat['data_backend_config']} 里，可以再导入一次那个文件来带入训练集位置和重复次数。")
        for k in ("validation_prompt", "validation_negative_prompt", "validation_resolution", "validation_num_inference_steps",
                  "validation_guidance", "validation_seed", "validation_steps", "data_backend_config", "lora_type", "model_family",
                  "model_type", "validation_guidance_rescale", "validation_torch_compile", "validation_prompt_library"):
            flat.pop(k, None)
        return _from_kohya(R, flat, label=None)

    # ---- sd-scripts / musubi-tuner / kohya_ss GUI / lora-scripts ----
    return _from_kohya(R, data)


# ---------------- sd-scripts 系 ----------------

_SECTION_SKIP = ("datasets", "general")


def _from_kohya(R, data, label="auto"):
    flat = {}
    for k, v in data.items():
        # kohya_ss GUI 导出的 TOML / 秋叶的配置有的按小节分组（[model_arguments] [optimizer_arguments] …），摊平
        if isinstance(v, dict) and k not in _SECTION_SKIP and k not in ("optimizer_args", "network_args"):
            for k2, v2 in v.items():
                flat.setdefault(k2, v2)
        elif k not in _SECTION_SKIP:
            flat[k] = v
    if label == "auto":
        if "LoRA_type" in flat or "additional_parameters" in flat or "max_resolution" in flat:
            label = "kohya_ss GUI 配置"
        elif "model_train_type" in flat:
            label = "秋叶 lora-scripts 配置"
        elif "dit" in flat and "pretrained_model_name_or_path" not in flat:
            label = "musubi-tuner 配置"
        elif "datasets" in data and len(flat) < 12:
            label = "sd-scripts / musubi-tuner 数据集配置"
        else:
            label = "sd-scripts 配置"
    if label:
        R.fmt = R.fmt or label

    # 引用的其他文件（本机上找得到就一起读）
    for key in ("config_file", "dataset_config"):
        p = _local(R, flat.pop(key, None))
        if p:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    sub = _toml(f.read())
                if key == "dataset_config":
                    _dataset_cfg(R, sub)
                else:
                    for k2, v2 in _flatten_sections(sub).items():
                        flat.setdefault(k2, v2)   # 命令行上的值优先于 config_file 里的
                R.notes.append(f"已一起读取 {p}")
            except Exception as e:  # noqa: BLE001
                R.notes.append(f"读取 {p} 失败：{e}")
    if "datasets" in data or "general" in data:
        _dataset_cfg(R, data)

    # kohya_ss GUI：「额外参数」是一段命令行
    extra = flat.pop("additional_parameters", None)
    if isinstance(extra, str) and extra.strip():
        for k2, v2 in _parse_args(_split(extra)).items():
            flat.setdefault(k2, v2)

    # 底模线索：脚本名、GUI 勾选框、秋叶的训练类型、network_module、模型文件名
    for k in ("model_train_type", "network_module"):
        if flat.get(k):
            R.signals.append(str(flat[k]))
    for flag, fam in (("sdxl", "sdxl"), ("flux1_checkbox", "flux"), ("sd3_checkbox", "sd3"), ("v2", "sd2"), ("is_v2_model", "sd2")):
        if flat.get(flag) is True or str(flat.get(flag)).lower() in ("1", "true"):
            R.signals.insert(0, fam)
        flat.pop(flag, None)

    # 模型文件
    for k, slot in (("pretrained_model_name_or_path", "dit"), ("pretrained_model", "dit"), ("dit", "dit"), ("vae", "vae"),
                    ("qwen3", "te"), ("text_encoder", "te"), ("text_encoder1", "te")):
        v = flat.pop(k, None)
        if isinstance(v, str) and v.strip() and slot not in R.models:
            R.models[slot] = v.strip()
    if R.models.get("dit"):
        R.signals.append(R.models["dit"])
    R.output_dir = R.output_dir or (flat.pop("output_dir", "") or "")

    # 测试出图
    sp = flat.pop("sample_prompts", None)
    prompts = None
    if isinstance(sp, str) and sp.strip():
        p = _local(R, sp)
        if p:
            try:
                prompts = _read_prompt_file(p)
            except Exception as e:  # noqa: BLE001
                R.notes.append(f"测试提示词文件 {p} 读取失败：{e}")
        elif "\n" in sp or not re.search(r"\.(txt|toml|json)$", sp.strip(), re.I):
            prompts = _prompt_lines(sp)          # kohya_ss GUI 直接存的提示词内容
        else:
            R.notes.append(f"测试提示词文件 {sp} 在这台电脑上找不到，没有导入提示词。")
    elif isinstance(sp, list):
        prompts = [_prompt_obj(x) if isinstance(x, str) else _prompt_dict(x) for x in sp]
    pos = flat.pop("positive_prompts", None)           # 秋叶的预览图设置
    if isinstance(pos, str) and pos.strip():
        prompts = [_prompt_obj(pos, flat.pop("negative_prompts", ""), flat.pop("sample_width", None), flat.pop("sample_height", None),
                               flat.pop("sample_steps", None), flat.pop("sample_cfg", None), flat.pop("sample_seed", None))]
    smp = {}
    if prompts:
        smp["prompts"] = [p for p in prompts if p.get("prompt")]
    for k, sk in (("sample_every_n_steps", "every_n_steps"), ("sample_every_n_epochs", "every_n_epochs"), ("sample_at_first", "at_first")):
        v = _scalar(flat.pop(k, None))
        if v not in (None, "", 0, False):
            smp[sk] = v
    if smp:
        R.sample = dict(R.sample or {}, **smp)
    for k, v in flat.items():
        R.put(k, _scalar(v))


def _flatten_sections(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k not in _SECTION_SKIP:
            for k2, v2 in v.items():
                out.setdefault(k2, v2)
        elif k not in _SECTION_SKIP:
            out[k] = v
    return out


def _dataset_cfg(R, d):
    """sd-scripts / musubi-tuner 的数据集配置：[general] + [[datasets]] (+ [[datasets.subsets]])。"""
    gen = d.get("general") or {}
    subs = []
    for ds in d.get("datasets") or []:
        base = {**gen, **{k: v for k, v in ds.items() if k != "subsets"}}
        for sub in ds.get("subsets") or [{}]:
            subs.append({**base, **sub})
    if not subs and gen:
        subs = [dict(gen)]
    _datasets(R, subs)


def _datasets(R, subs):
    """多个子集：训练集取第一个（多个文件夹在同一个上级目录时取上级目录），正则化子集单独放。"""
    reg = [s for s in subs if s.get("is_reg")]
    subs = [s for s in subs if not s.get("is_reg")]
    if not subs:
        return
    first = subs[0]
    dirs = [s.get("image_dir") or s.get("image_directory") for s in subs]
    dirs = [x for x in dirs if x]
    if len(dirs) > 1:
        parents = {re.sub(r"[\\/][^\\/]+[\\/]?$", "", x) for x in dirs}
        if len(parents) == 1:
            R.put("dataset_dir", parents.pop(), True)
            R.notes.append(f"原配置有 {len(dirs)} 个子集，这里用它们共同的上级文件夹，子文件夹会各自作为一个子集。"
                           "各子集的重复次数不一样时，可以把文件夹改名成「10_名字」这种格式。")
        else:
            R.put("dataset_dir", dirs[0], True)
            R.notes.append(f"原配置有 {len(dirs)} 个不在同一处的子集，只导入了第一个：{dirs[0]}")
    elif dirs:
        R.put("dataset_dir", dirs[0], True)
    if reg and (reg[0].get("image_dir") or reg[0].get("image_directory")):
        R.put("reg_data_dir", reg[0].get("image_dir") or reg[0].get("image_directory"), True)
        if reg[0].get("num_repeats"):
            R.put("reg_repeats", reg[0]["num_repeats"], True)
    for k in ("num_repeats", "resolution", "batch_size", "caption_extension", "caption_extention", "shuffle_caption", "keep_tokens",
              "keep_tokens_separator", "flip_aug", "color_aug", "random_crop", "enable_bucket", "bucket_no_upscale",
              "min_bucket_reso", "max_bucket_reso", "bucket_reso_steps", "caption_dropout_rate", "caption_tag_dropout_rate",
              "caption_dropout_every_n_epochs", "caption_prefix", "caption_suffix", "secondary_separator", "enable_wildcard",
              "alpha_mask", "resize_interpolation"):
        if first.get(k) not in (None, ""):
            R.put(k, first[k], True)


# ---------------- 命令行 ----------------

def _split(s):
    """按空白拆参数，引号只用来包住带空格的部分（可以出现在中间：--a="D:\\my dir"）；反斜杠不当转义（Windows 路径）。"""
    toks, cur, q, has = [], "", None, False
    for c in s:
        if q:
            if c == q:
                q = None
            else:
                cur += c
        elif c in "\"'":
            q, has = c, True
        elif c.isspace():
            if cur or has:
                toks.append(cur)
            cur, has = "", False
        else:
            cur += c
    if cur or has:
        toks.append(cur)
    return toks


def _unquote(t):
    t = t.strip()
    return t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t


def _parse_args(toks):
    out = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        i += 1
        if not t.startswith("--") or len(t) < 3:
            continue
        k = t[2:]
        if "=" in k:
            k, v = k.split("=", 1)
            out[k.replace("-", "_")] = v
            continue
        k = k.replace("-", "_")
        vals = []
        while i < len(toks) and not (toks[i].startswith("--") and len(toks[i]) > 2) and not toks[i].lower().endswith(".py"):
            vals.append(toks[i])
            i += 1
        out[k] = True if not vals else vals[0] if len(vals) == 1 else vals
    return out


_VAR_KEYS = {   # 秋叶 train.sh / train.ps1 里的变量名
    "pretrained_model": "pretrained_model_name_or_path", "max_train_epoches": "max_train_epochs", "lr": "learning_rate",
    "lr_restart_cycles": "lr_scheduler_num_cycles", "train_unet_only": "network_train_unet_only",
    "train_text_encoder_only": "network_train_text_encoder_only", "algo": "lycoris_algo", "conv_dim": "conv_dim", "conv_alpha": "conv_alpha",
    "parameterization": "v_parameterization", "batch_size": "train_batch_size", "model_train_type": "model_train_type",
}


def _from_cmd(R, text):
    R.fmt = "训练命令行"
    t = text.replace("\r\n", "\n")
    t = re.sub(r"[ \t]*(\\|\^|`)[ \t]*\n", " ", t)    # 续行：sh 的 \、bat 的 ^、PowerShell 的 `
    vars_, args, scripts = {}, {}, []
    for line in t.split("\n"):
        s = line.strip()
        if s.startswith("$ ") and "--" in s:
            s = s[2:]                               # 本训练器的 commands.txt / 日志
        if not s or s.startswith(("#", "::", "//")) or re.match(r"^rem(\s|$)", s, re.I):
            continue
        m = (re.match(r'^set\s+"?([A-Za-z_]\w*)=(.*?)"?\s*$', s, re.I) or re.match(r"^\$([A-Za-z_]\w*)\s*=\s*(.*)$", s)
             or re.match(r"^(?:export\s+|declare\s+)?([A-Za-z_]\w*)=(.*)$", s))
        if m and not s.startswith("--"):
            v = re.sub(r"\s+#.*$", "", m.group(2)).strip()
            v = _unquote(v)
            if not v.startswith(("(", "@(", "$(")):
                vars_[m.group(1)] = v
            continue
        if "--" not in s:
            continue
        toks = [_expand(x, vars_) for x in _split(s)]
        scripts += [os.path.basename(x.replace("\\", "/")) for x in toks if x.lower().endswith(".py")]
        for k, v in _parse_args(toks).items():
            if isinstance(v, str) and "$" in v:
                continue                              # 没展开的变量
            args[k] = v                               # 后面的命令（训练）覆盖前面的（缓存）
    if not args and not vars_:
        raise ValueError("命令行里没有找到 --参数")
    # 脚本名最能说明底模：训练脚本优先
    scripts.sort(key=lambda x: 0 if "train" in x else 1)
    R.signals += scripts
    known = _known_keys()
    for k, v in vars_.items():
        k2 = _VAR_KEYS.get(k, k)
        if k2 in known or k2 in _ALIAS or k2 in _VAR_KEYS.values():
            args.setdefault(k2, v)
    if vars_ and not scripts:
        R.fmt = "训练脚本变量"
    lyc = args.pop("lycoris_algo", None)
    if lyc and "lycoris" in str(args.get("network_module", "")):
        na = args.get("network_args") or []
        args["network_args"] = ([na] if isinstance(na, str) else list(na)) + [f"algo={lyc}"]
    _from_kohya(R, args, label=None)


def _expand(tok, vars_):
    def rep(m):
        name = m.group(1) or m.group(2) or m.group(3)
        return vars_.get(name, m.group(0))
    return re.sub(r"\$\{(\w+)\}|\$(?:env:)?(\w+)|%(\w+)%", rep, tok)


# ---------------- LoRA 文件里的训练信息 ----------------

def _from_ss(R, meta):
    R.fmt = "LoRA 文件里的训练信息"
    ss = {k[3:]: v for k, v in meta.items() if k.startswith("ss_")}
    if not ss:
        # 没有 sd-scripts 的 ss_ 信息：看看有没有别的训练器把整份配置存成 JSON
        for k, v in meta.items():
            if isinstance(v, str) and v.strip().startswith("{"):
                try:
                    d = json.loads(v)
                except ValueError:
                    continue
                if isinstance(d, dict) and len(d) > 3:
                    sub = _R()
                    try:
                        _extract(sub, d, "data")
                    except ValueError:
                        continue
                    if sub.kv or sub.native:
                        sub.fmt = f"LoRA 文件里的训练配置（{sub.fmt}）"
                        sub.signals.append(meta.get("modelspec.architecture", ""))
                        R.__dict__.update(sub.__dict__)
                        return
        if not meta.get("modelspec.architecture"):
            raise ValueError("这个 LoRA 文件里没有记录训练参数（不是用 sd-scripts 系训练器练的，或者训练时关了元数据）")
        R.notes.append("这个 LoRA 只记录了底模类型，没有训练参数。")
    for k in ("base_model_version", "network_module", "sd_model_name"):
        if ss.get(k):
            R.signals.append(str(ss[k]))
    R.signals.append(str(meta.get("modelspec.architecture", "")))
    if ss.get("v2") in ("True", True):
        R.signals.insert(0, "sd2")
    kv = {}
    for k, v in ss.items():
        kv[k] = v
    # 键名换成训练参数名
    ren = {"num_epochs": "max_train_epochs", "batch_size_per_device": "train_batch_size", "max_train_steps": "max_train_steps",
           "debiased_estimation": "debiased_estimation_loss", "gradient_accumulation_steps": "gradient_accumulation_steps"}
    for a, b in ren.items():
        if a in kv:
            kv[b] = kv.pop(a)
    if kv.get("sd_model_name"):
        R.models["dit"] = kv.pop("sd_model_name")
    if kv.get("vae_name") and kv["vae_name"] not in ("None", ""):
        R.models["vae"] = kv.pop("vae_name")
    # 训练集：只记了文件夹名
    subs = []
    try:
        ds = json.loads(kv.pop("datasets", "") or "[]")
        for d in ds:
            for s in d.get("subsets") or []:
                subs.append({**{k: d.get(k) for k in ("resolution", "enable_bucket", "min_bucket_reso", "max_bucket_reso", "bucket_no_upscale")},
                             **s})
            if d.get("batch_size_per_device"):
                kv.setdefault("train_batch_size", d["batch_size_per_device"])
    except (ValueError, TypeError, AttributeError):
        pass
    dirs = {}
    try:
        dirs = json.loads(kv.pop("dataset_dirs", "") or "{}")
    except ValueError:
        pass
    names = [s.get("image_dir") for s in subs if s.get("image_dir")] or list(dirs)
    if subs:
        s0 = {k: v for k, v in subs[0].items() if k != "image_dir"}
        if isinstance(s0.get("resolution"), (list, tuple)):
            s0["resolution"] = ",".join(str(x) for x in s0["resolution"])
        _datasets(R, [s0])
    elif dirs:
        n = next(iter(dirs.values())).get("n_repeats")
        if n:
            R.put("num_repeats", n, True)
    if names:
        R.notes.append("LoRA 里只记了训练集的文件夹名（" + "、".join(str(x) for x in names[:5]) + "），没有完整路径，训练集位置请自己选。")
    # 优化器：「bitsandbytes.optim.adamw.AdamW8bit(weight_decay=0.01,betas=(0.9, 0.99))」
    opt = kv.pop("optimizer", "")
    m = re.match(r"^([\w.]+)(?:\((.*)\))?$", str(opt).strip(), re.S)
    if m:
        name = m.group(1)
        if name.startswith(("bitsandbytes.", "torch.optim.", "transformers.optimization.", "dadaptation.", "prodigyopt.",
                            "schedulefree.")):
            name = name.rsplit(".", 1)[-1]
        kv["optimizer_type"] = name
        if m.group(2):
            kv["optimizer_args"] = " ".join(_ss_opt_args(m.group(2)))
    for k in ("network_args",):
        if isinstance(kv.get(k), str) and kv[k].strip().startswith("{"):
            try:
                kv[k] = [f"{a}={b}" for a, b in json.loads(kv[k]).items()]
            except ValueError:
                pass
    if str(kv.get("resolution", "")).startswith("("):
        kv["resolution"] = kv["resolution"].strip("()")
    for k in ("tag_frequency", "bucket_info", "session_id", "training_started_at", "training_finished_at", "steps", "epoch", "num_train_images",
              "num_reg_images", "num_batches_per_epoch", "total_batch_size", "new_sd_model_hash", "sd_model_hash", "sd_scripts_commit_hash",
              "training_comment", "output_name", "lowram", "cache_latents", "network_dropout_rate"):
        kv.pop(k, None)
    _from_kohya(R, kv, label=None)


def _ss_opt_args(s):
    """「weight_decay=0.01,betas=(0.9, 0.99)」→ ["weight_decay=0.01", "betas=0.9,0.99"]（括号里的逗号不拆）。"""
    out, depth, cur = [], 0, ""
    for c in s:
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        if c == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += c
    out.append(cur)
    res = []
    for a in out:
        if "=" in a:
            k, v = a.split("=", 1)
            v = re.sub(r"[()\[\]\s]", "", v)
            res.append(f"{k.strip()}={v}")
    return res


# ---------------- OneTrainer ----------------

_OT_OPT = {"adamw_8bit": "AdamW8bit", "adam_8bit": "AdamW8bit", "adamw": "AdamW", "adam": "AdamW", "adafactor": "Adafactor",
           "prodigy": "Prodigy", "lion": "Lion", "lion_8bit": "Lion8bit", "came": "CAME", "dadapt_adam": "DAdaptAdam",
           "dadapt_lion": "DAdaptLion", "dadapt_sgd": "DAdaptSGD", "dadapt_adan": "DAdaptAdan", "dadapt_ada_grad": "DAdaptAdaGrad",
           "schedule_free_adamw": "AdamWScheduleFree", "schedule_free_sgd": "SGDScheduleFree",
           "prodigy_plus_schedule_free": "prodigyplus.ProdigyPlusScheduleFree", "sgd": "SGDNesterov", "sgd_8bit": "SGDNesterov8bit",
           "paged_adamw_8bit": "PagedAdamW8bit", "paged_adamw": "PagedAdamW", "paged_lion_8bit": "PagedLion8bit"}
_DTYPE = {"bfloat_16": "bf16", "float_16": "fp16", "float_32": "float", "bf16": "bf16", "fp16": "fp16", "float16": "fp16",
          "bfloat16": "bf16", "float32": "float", "fp32": "float"}


def _from_onetrainer(R, d):
    R.fmt = "OneTrainer 配置"
    R.signals.append(str(d.get("model_type") or ""))
    if str(d.get("training_method", "LORA")).upper() not in ("LORA", ""):
        R.notes.append(f"原配置的训练方式是 {d.get('training_method')}，这里按 LoRA 导入。")
    if d.get("base_model_name"):
        R.models["dit"] = d["base_model_name"]
    if (d.get("vae") or {}).get("model_name") if isinstance(d.get("vae"), dict) else False:
        R.models["vae"] = d["vae"]["model_name"]
    dest = d.get("output_model_destination") or ""
    if dest:
        R.output_dir = os.path.dirname(dest)
        R.put("output_name", os.path.splitext(os.path.basename(dest))[0])
    R.put("learning_rate", d.get("learning_rate"))
    R.put("max_train_epochs", d.get("epochs"))
    R.put("train_batch_size", d.get("batch_size"))
    R.put("gradient_accumulation_steps", d.get("gradient_accumulation_steps"))
    R.put("resolution", d.get("resolution"))
    R.put("network_dim", d.get("lora_rank"))
    R.put("network_alpha", d.get("lora_alpha"))
    R.put("seed", d.get("seed"))
    R.put("max_data_loader_n_workers", d.get("dataloader_threads"))
    R.put("enable_bucket", d.get("aspect_ratio_bucketing"))
    if d.get("dropout_probability"):
        R.put("network_dropout", d["dropout_probability"])
    gc = d.get("gradient_checkpointing")
    if gc is not None:
        R.put("gradient_checkpointing", gc is True or str(gc).upper() in ("ON", "CPU_OFFLOADED", "TRUE"))
    if d.get("train_dtype"):
        R.put("mixed_precision", _DTYPE.get(str(d["train_dtype"]).lower()))
    if d.get("output_dtype"):
        R.put("save_precision", _DTYPE.get(str(d["output_dtype"]).lower()))
    sch = str(d.get("learning_rate_scheduler") or "").lower()
    if sch:
        if sch == "constant" and int(d.get("learning_rate_warmup_steps") or 0) > 0:
            sch = "constant_with_warmup"
        R.put("lr_scheduler", {"cosine_with_hard_restarts": "cosine_with_restarts"}.get(sch, sch))
    R.put("lr_warmup_steps", d.get("learning_rate_warmup_steps"))
    R.put("lr_scheduler_num_cycles", d.get("learning_rate_cycles"))
    opt = d.get("optimizer") or {}
    if isinstance(opt, dict) and opt.get("optimizer"):
        name = str(opt["optimizer"]).lower()
        R.put("optimizer_type", _OT_OPT.get(name, name))
        oa = []
        if opt.get("weight_decay") not in (None, 0, 0.0):
            oa.append(f"weight_decay={opt['weight_decay']}")
        if opt.get("beta1") is not None and opt.get("beta2") is not None:
            oa.append(f"betas={opt['beta1']},{opt['beta2']}")
        if opt.get("eps") not in (None,):
            oa.append(f"eps={opt['eps']}")
        if oa:
            R.put("optimizer_args", " ".join(oa))
    te = d.get("text_encoder") or {}
    if isinstance(te, dict) and "train" in te:
        R.put("network_train_unet_only", not te.get("train"))
        if te.get("train") and te.get("learning_rate"):
            R.put("text_encoder_lr", te["learning_rate"])
    if d.get("save_every_unit") == "EPOCH":
        R.put("save_every_n_epochs", d.get("save_every"))
    elif d.get("save_every_unit") not in (None, "NEVER"):
        R.notes.append(f"原配置按 {d.get('save_every')} {d.get('save_every_unit')} 保存一次；这里只能按轮数保存，没有导入。")
    if d.get("noising_weight"):
        R.put("noise_offset", d["noising_weight"])
    if str(d.get("loss_weight_fn") or "").upper() == "MIN_SNR_GAMMA":
        R.put("min_snr_gamma", d.get("loss_weight_strength"))
    td = str(d.get("timestep_distribution") or "").lower()
    if td:
        R.put("timestep_sampling", td)
    R.put("masked_loss", d.get("masked_training") or None)
    if isinstance(d.get("concepts"), list) and d["concepts"]:
        _ot_concepts(R, d["concepts"])
    elif d.get("concept_file_name"):
        R.notes.append(f"训练集写在 {d['concept_file_name']} 里，可以再导入那个文件带入训练集位置和重复次数。")
    if isinstance(d.get("samples"), list) and d["samples"]:
        _ot_samples(R, d["samples"])
    elif d.get("sample_definition_file_name"):
        R.notes.append(f"测试提示词写在 {d['sample_definition_file_name']} 里，可以再导入那个文件。")
    unit = d.get("sample_after_unit")
    if R.sample is not None or d.get("sample_after"):
        if unit == "EPOCH":
            R.sample = dict(R.sample or {}, every_n_epochs=_int(d.get("sample_after")))
        elif unit == "STEP":
            R.sample = dict(R.sample or {}, every_n_steps=_int(d.get("sample_after")))
    R.notes.append("OneTrainer 和 kohya 系的学习率、alpha 习惯不完全一样，开训前看一眼学习率。")


def _ot_concepts(R, concepts):
    ok = [c for c in concepts if isinstance(c, dict) and c.get("path") and c.get("enabled", True)]
    if not ok:
        return
    subs = []
    for c in ok:
        img = c.get("image") or {}
        txt = c.get("text") or {}
        subs.append({"image_dir": c["path"], "num_repeats": max(1, round(float(c.get("repeats") or 1))),
                     "flip_aug": bool(img.get("enable_random_flip")) or None,
                     "caption_extension": ".txt" if txt.get("prompt_source", "sample") == "sample" else None})
    _datasets(R, subs)
    if any(c.get("type") == "PRIOR_PREDICTION" for c in ok):
        R.notes.append("原配置里有先验保持（prior prediction）概念，这里当普通训练集导入了。")


def _ot_samples(R, samples):
    ps = [_prompt_obj(s.get("prompt"), s.get("negative_prompt"), s.get("width"), s.get("height"), s.get("diffusion_steps"),
                      s.get("cfg_scale"), s.get("seed")) for s in samples if isinstance(s, dict) and s.get("enabled", True) and s.get("prompt")]
    if ps:
        R.sample = dict(R.sample or {}, prompts=ps)


# ---------------- ai-toolkit ----------------

def _from_aitk(R, proc, cfg):
    R.fmt = "ai-toolkit 配置"
    model = proc.get("model") or {}
    net = proc.get("network") or {}
    train = proc.get("train") or {}
    save = proc.get("save") or {}
    sample = proc.get("sample") or {}
    ds = [x for x in (proc.get("datasets") or []) if isinstance(x, dict)]
    arch = model.get("arch") or ("sdxl" if model.get("is_xl") else "flux" if model.get("is_flux") else "sd3" if model.get("is_v3")
                                 else "sd2" if model.get("is_v2") else "")
    R.signals += [str(arch), str(model.get("name_or_path") or "")]
    if model.get("name_or_path"):
        R.models["dit"] = model["name_or_path"]
    R.put("output_name", cfg.get("name"))
    if proc.get("training_folder"):
        R.output_dir = proc["training_folder"]
    nt = str(net.get("type") or "lora").lower()
    R.put("network_dim", net.get("linear"))
    R.put("network_alpha", net.get("linear_alpha"))
    if net.get("conv"):
        R.put("conv_dim", net["conv"])
        R.put("conv_alpha", net.get("conv_alpha"))
    if nt == "lokr":
        R.put("network_args", ["algo=lokr", f"factor={net.get('lokr_factor', -1)}"])
    R.put("train_batch_size", train.get("batch_size"))
    R.put("gradient_accumulation_steps", train.get("gradient_accumulation_steps") or train.get("gradient_accumulation"))
    R.put("gradient_checkpointing", train.get("gradient_checkpointing"))
    R.put("learning_rate", train.get("lr"))
    R.put("optimizer_type", train.get("optimizer"))
    op = train.get("optimizer_params")
    if isinstance(op, dict) and op:
        R.put("optimizer_args", " ".join(f"{k}={','.join(map(str, v)) if isinstance(v, list) else v}" for k, v in op.items()))
    R.put("lr_scheduler", train.get("lr_scheduler"))
    if train.get("steps"):
        R.put("max_train_steps", train["steps"])
    R.put("seed", train.get("seed") if train.get("seed") is not None else sample.get("seed"))
    R.put("max_grad_norm", train.get("max_grad_norm"))
    R.put("noise_offset", train.get("noise_offset") or None)
    R.put("min_snr_gamma", train.get("min_snr_gamma"))
    if train.get("dtype"):
        R.put("mixed_precision", _DTYPE.get(str(train["dtype"]).lower()))
    if save.get("dtype"):
        R.put("save_precision", _DTYPE.get(str(save["dtype"]).lower()))
    if train.get("train_text_encoder") is not None:
        R.put("network_train_unet_only", not train["train_text_encoder"])
    ts = str(train.get("timestep_type") or "").lower()
    if ts:
        R.put("timestep_sampling", {"linear": "uniform", "lognorm_blend": "sigmoid"}.get(ts, ts))
    if save.get("save_every"):
        R.notes.append(f"原配置每 {save['save_every']} 步保存一次；这里按轮数保存，没有导入。")
    if save.get("max_step_saves_to_keep"):
        R.put("save_last_n_epochs", save["max_step_saves_to_keep"])
    subs = []
    for x in ds:
        res = x.get("resolution")
        if isinstance(res, list) and res:
            res = max(int(r) for r in res if str(r).isdigit()) if all(str(r).isdigit() for r in res) else res[0]
        subs.append({"image_dir": x.get("folder_path"), "num_repeats": x.get("num_repeats"), "resolution": res,
                     "caption_extension": ("." + str(x["caption_ext"]).lstrip(".")) if x.get("caption_ext") else None,
                     "caption_dropout_rate": x.get("caption_dropout_rate"), "shuffle_caption": x.get("shuffle_tokens"),
                     "flip_aug": x.get("flip_x"), "is_reg": x.get("is_reg")})
    if subs:
        _datasets(R, subs)
        if isinstance(ds[0].get("resolution"), list) and len(ds[0]["resolution"]) > 1:
            R.notes.append(f"原配置同时用多种分辨率训练（{ds[0]['resolution']}），这里取最大的。")
    trig = proc.get("trigger_word") or ""
    if trig:
        R.notes.append(f"原配置的触发词是「{trig}」（ai-toolkit 训练时自动加进标签）：要用的话在「标签编辑」里加到每个标签文件开头。")
    prompts = sample.get("prompts") or []
    if prompts:
        ps = []
        for p in prompts:
            text = p.get("prompt") if isinstance(p, dict) else p
            if not text:
                continue
            text = str(text).replace("[trigger]", trig) if trig else str(text).replace("[trigger], ", "").replace("[trigger]", "")
            ps.append(_prompt_obj(text, sample.get("neg"), sample.get("width"), sample.get("height"), sample.get("sample_steps"),
                                  sample.get("guidance_scale"), sample.get("seed")))
        R.sample = {"prompts": ps}
        if sample.get("sample_every"):
            R.sample["every_n_steps"] = _int(sample["sample_every"])


# ---------------- 测试提示词 ----------------

def _prompt_obj(prompt, negative=None, w=None, h=None, steps=None, cfg=None, seed=None):
    p = {"prompt": str(prompt or "").strip()}
    if negative:
        p["negative"] = str(negative).strip()
    if isinstance(w, str) and re.match(r"^\d+x\d+$", w):   # SimpleTuner 的 validation_resolution 可以是 1024x768
        w, h = w.split("x")
    for k, v, conv in (("width", w, _int), ("height", h, _int), ("steps", steps, _int), ("cfg", cfg, _float), ("seed", seed, _int)):
        v = conv(v)
        if v is not None and v >= 0:
            p[k] = v
    return p


def _prompt_dict(d):
    return _prompt_obj(d.get("prompt") or d.get("text"), d.get("negative_prompt") or d.get("negative"), d.get("width"), d.get("height"),
                       d.get("sample_steps") or d.get("steps"), d.get("scale") or d.get("guidance_scale") or d.get("cfg"), d.get("seed"))


def _prompt_lines(text):
    """sd-scripts 的提示词文件：每行一条，「提示词 --n 反向 --w 1024 --h 1024 --s 28 --l 7 --d 42」。"""
    out = []
    for ln in text.replace("\r\n", "\n").split("\n"):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = re.split(r"\s+--(?=[a-z]+\s)", " " + ln)
        p = {"prompt": parts[0].strip()}
        opts = {}
        for part in parts[1:]:
            k, _, v = part.partition(" ")
            opts[k] = v.strip()
        out.append(_prompt_obj(p["prompt"], opts.get("n"), opts.get("w"), opts.get("h"), opts.get("s"), opts.get("l"), opts.get("d")))
    return [p for p in out if p["prompt"]]


def _read_prompt_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.lower().endswith(".json"):
        d = json.loads(text)
        return [_prompt_dict(x) for x in (d if isinstance(d, list) else d.get("prompt") or [])]
    if path.lower().endswith(".toml"):
        d = _toml(text)
        items = (d.get("prompt") or {}).get("subset") or []
        base = {k: v for k, v in (d.get("prompt") or {}).items() if k != "subset"}
        return [_prompt_dict({**base, **x}) for x in items]
    return _prompt_lines(text)


# ======================= 换成本训练器的参数名 =======================

_ALIAS = {   # 别的写法 → sd-scripts 参数名
    "train_data_dir": "dataset_dir", "image_dir": "dataset_dir", "image_directory": "dataset_dir", "instance_data_dir": "dataset_dir",
    "num_repeats": "repeats", "dataset_repeats": "repeats",
    "max_train_epochs": "epochs", "num_epochs": "epochs", "num_train_epochs": "epochs", "epoch": "epochs",
    "train_batch_size": "batch_size",
    "lora_rank": "network_dim", "rank": "network_dim", "lora_alpha": "network_alpha",
    "gradient_accumulation_steps": "grad_accum",
    "max_data_loader_n_workers": "num_workers", "dataloader_num_workers": "num_workers",
    "cache_text_encoder_outputs_to_disk": "cache_te_to_disk",
    "save_last_n_epochs": "save_last_n",
    "optimizer": "optimizer_type",
    "max_resolution": "resolution",
    "caption_extention": "caption_extension",
    "lr_warmup": "lr_warmup_percent",
    "train_unet_only": "network_train_unet_only", "train_text_encoder_only": "network_train_text_encoder_only",
    "factor": "lokr_factor", "dropout": "network_dropout", "lycoris_algo": "algo",
    "loraplus_unet_lr_ratio": "loraplus_lr_ratio",
}
# 这些由本训练器自己处理，导入时不提示「没导入」
_IGNORE = {"logging_dir", "cache_latents", "cache_text_encoder_outputs", "persistent_data_loader_workers", "output_config",
           "save_model_as_dir", "console_log_level", "console_log_file", "console_log_simple", "metadata_title", "metadata_author",
           "metadata_description", "metadata_license", "metadata_tags", "training_comment", "log_config", "no_metadata",
           "huggingface_repo_id", "huggingface_token", "save_n_epoch_ratio", "max_train_epoches", "LoRA_type", "LyCORIS_preset",
           "noise_offset_type", "model_list", "save_as_bool", "sample_prompts", "mem_eff_attn", "xformers", "sdpa", "flash_attn",
           "sage_attn", "network_module", "model_train_type", "algo", "use_8bit_adam", "use_lion_optimizer", "lr_warmup_percent",
           "v_parameterization", "network_train_unet_only", "network_train_text_encoder_only", "num_processes", "num_machines",
           "multi_gpu", "gpu_ids", "dynamo_backend", "ui_custom_params", "enable_preview", "task_id", "full_path", "save_toml",
           "shuffle_tokens", "caption_ext", "pretrained_model_name_or_path", "output_dir", "dataset_config", "config_file",
           "highvram", "lowram", "lowvram", "clip_skip", "masked_loss", "base_model_version", "vae_name", "v2"}
# 这几个是换算时顺带填的，目标底模没有也不用提示
_QUIET = {"network_type", "train_te", "steps_mode", "attn_mode", "num_cpu_threads_per_process", "timestep_sampling"}
_NUM = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$", re.I)


def _scalar(v):
    if not isinstance(v, str):
        return v
    s = v.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    if _NUM.match(s):
        f = float(s)
        return int(f) if re.match(r"^[+-]?\d+$", s) else f
    return s


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


_known_cache = None


def _known_keys():
    global _known_cache
    if _known_cache is None:
        _known_cache = {f["key"] for s in models.all_specs() for f in s.params()} | {"conv_dim", "conv_alpha", "lokr_factor", "v_pred"}
    return _known_cache


def _to_params(R):
    """sd-scripts 风格参数 → 本训练器参数（还没按底模检查）。返回 (参数, 没导入的 {键: 值})。"""
    kv = {}
    for k, v in R.kv.items():            # 正式参数名优先，别名只补空缺
        if k not in _ALIAS:
            kv[k] = _scalar(v)
    for k, v in R.kv.items():
        if k in _ALIAS and _ALIAS[k] not in kv:
            kv[_ALIAS[k]] = _scalar(v)
    p = {}
    take = kv.pop

    # 训练长度
    ep, st = _int(take("epochs", None)), _int(take("max_train_steps", None))
    if ep and ep > 0:
        p.update(steps_mode="epochs", epochs=ep)
    elif st and st > 0:
        p.update(steps_mode="steps", max_train_steps=st)
    # 分辨率：1024 / "1024,1024" / [1024, 768] / "1024x768"
    r = take("resolution", None)
    if r not in (None, ""):
        nums = [int(float(x)) for x in re.findall(r"\d+(?:\.\d+)?", ",".join(map(str, r)) if isinstance(r, (list, tuple)) else str(r))]
        if nums:
            w, h = nums[0], nums[1] if len(nums) > 1 else nums[0]
            if w == h:
                p["resolution"] = w
            else:
                p.update(resolution="custom", resolution_wh=f"{w},{h}")
    # 优化器
    opt = take("optimizer_type", None)
    if take("use_8bit_adam", None) is True:
        opt = opt or "AdamW8bit"
    if take("use_lion_optimizer", None) is True:
        opt = opt or "Lion"
    if opt:
        p["optimizer_type"] = str(opt)
    for k in ("optimizer_args", "lr_scheduler_args"):
        v = take(k, None)
        if v not in (None, ""):
            p[k] = " ".join(map(str, v)) if isinstance(v, list) else str(v)
    # 学习率
    lr = _float(kv.get("learning_rate"))
    ulr = take("unet_lr", None)
    if lr is None and _float(ulr):
        kv["learning_rate"] = _float(ulr)
    elif _float(ulr) and _float(ulr) != lr:
        p["unet_lr"] = str(ulr)
    te = take("text_encoder_lr", None)
    if isinstance(te, (list, tuple)):
        te = te[0] if te else None
    elif isinstance(te, str):
        te = _float(te.split()[0]) if te.split() else None
    if _float(te):
        p["text_encoder_lr"] = _float(te)
    uo, to = take("network_train_unet_only", None), take("network_train_text_encoder_only", None)
    if to is True:
        p.update(train_te=True, train_te_only=True)
    elif uo is not None:
        p["train_te"] = not uo
    elif _float(te) and "train_te" not in kv:
        p["train_te"] = True     # sd-scripts 默认文本编码器也一起练
    # 预热：比例 / 百分比 → 步数
    total = p.get("max_train_steps")
    wu = take("lr_warmup_steps", None)
    pct = take("lr_warmup_percent", None)
    if isinstance(wu, float) and 0 < wu < 1:
        pct = wu * 100
        wu = None
    if wu not in (None, ""):
        p["lr_warmup_steps"] = _int(wu)
    elif _float(pct):
        if total:
            p["lr_warmup_steps"] = int(total * _float(pct) / 100)
        else:
            R.notes.append(f"原配置预热占总步数的 {_float(pct):g}%，这里按步数填，请到「高级参数」里按总步数自己换算。")
    # 网络结构
    nm = str(take("network_module", "") or "")
    algo = take("algo", None)
    na = take("network_args", None)
    toks = na if isinstance(na, list) else _split(na) if isinstance(na, str) else []
    rest = []
    for t in toks:
        k, _, v = str(t).partition("=")
        k = k.strip()
        if k in ("conv_dim", "conv_rank"):
            kv.setdefault("conv_dim", _scalar(v))
        elif k == "conv_alpha":
            kv.setdefault("conv_alpha", _scalar(v))
        elif k == "algo":
            algo = algo or v.strip()
        elif k == "factor":
            kv.setdefault("lokr_factor", _scalar(v))
        elif k in ("rank_dropout", "module_dropout", "loraplus_lr_ratio", "loraplus_unet_lr_ratio"):
            kv.setdefault("loraplus_lr_ratio" if k.startswith("loraplus") else k, _scalar(v))
        elif k == "dropout":
            kv.setdefault("network_dropout", _scalar(v))
        elif k in ("preset",) and v.strip() in ("full", "attn-mlp"):
            continue
        else:
            rest.append(str(t))
    if rest:
        p["network_args"] = " ".join(rest)
    lt = str(take("LoRA_type", "") or "").lower()
    if lt == "standard":                 # kohya_ss GUI 选标准 LoRA 时 conv_dim / factor 还留着默认值，不生效
        for k in ("conv_dim", "conv_alpha", "lokr_factor"):
            kv.pop(k, None)
    algo = str(algo or "").lower() or ("lokr" if "lokr" in lt or nm.endswith("lokr") else "loha" if "loha" in lt or nm.endswith("loha")
                                      else "locon" if "locon" in lt else "")
    if algo:
        p["network_type"] = {"locon": "lora", "lora": "lora", "lokr": "lokr", "loha": "loha"}.get(algo, algo)
        if algo == "locon":
            p["_locon"] = True
    elif nm:
        p["network_type"] = "lora"
    # 注意力实现
    for flag in ("xformers", "sdpa", "mem_eff_attn", "flash_attn", "sage_attn"):
        v = take(flag, None)
        if v is True:
            p["attn_mode"] = flag
        elif isinstance(v, str) and v.lower() in ("xformers", "sdpa", "mem_eff_attn"):
            p["attn_mode"] = v.lower()
    if take("v_parameterization", None) is True:
        p["v_pred"] = "vpred"
    # 噪声偏移（kohya_ss GUI 两种都存着，只用选中的那种）
    nt = str(take("noise_offset_type", "") or "")
    if nt == "Multires":
        kv.pop("noise_offset", None)
    elif nt:
        kv.pop("multires_noise_iterations", None)
    ce = kv.get("caption_extension")
    if isinstance(ce, str) and ce and not ce.startswith("."):
        kv["caption_extension"] = "." + ce
    if str(kv.get("mixed_precision", "")).lower() == "no":
        kv.pop("mixed_precision")
        R.notes.append("原配置没开混合精度（fp32 训练），这里保持当前的混合精度设置。")
    for k in ("network_weights", "base_weights"):
        if isinstance(kv.get(k), list):
            kv[k] = kv[k][0] if kv[k] else ""
    if kv.get("save_every_n_steps"):
        R.notes.append(f"原配置每 {kv['save_every_n_steps']} 步保存一次；这里按轮数保存，没有导入。")
    known = _known_keys()
    unknown = {}
    for k, v in kv.items():
        if k in known:
            if v is not None and v != "":
                p.setdefault(k, v)
        elif k not in _IGNORE and v not in (None, "", False):
            unknown[k] = v
    return p, unknown


# ======================= 按目标底模检查 =======================

_VALUE_ALIAS = {
    "attn_mode": {"sdpa": "torch", "torch": "sdpa", "flash_attn": "flash", "flash": "flash_attn", "sage_attn": "torch"},
    "timestep_sampling": {"logit_normal": "sigmoid", "sigmoid": "logit_normal", "shift": "shift", "flux_shift": "shift"},
    "save_precision": {"fp32": "float", "float32": "float"},
}


def _norm_opt(s):
    return re.sub(r"[^a-z0-9]", "", str(s).rsplit(".", 1)[-1].lower())


def _fit(spec, params, R):
    fields = {f["key"]: f for f in spec.to_dict()["params"]}
    rows, skipped = [], []

    def add(k, v):
        f = fields[k]
        row = {"key": k, "label": f["label"], "value": v, "group": f.get("group") or {"top": "", "basic": "基础", "lr": "学习率"}.get(f["section"], "")}
        if f["type"] in ("dir", "file") and v:
            p = _local(R, v)
            if p:
                row["value"] = p
            else:
                row["missing"] = True
        rows.append(row)

    locon = params.pop("_locon", False)
    for k, v in params.items():
        f = fields.get(k)
        if not f:
            if k not in _QUIET:
                skipped.append({"key": k, "value": _short(v), "reason": f"{spec.label} 没有这个参数"})
            continue
        t = f["type"]
        try:
            if t == "int":
                v = int(float(v))
            elif t == "float":
                v = float(v)
            elif t == "bool":
                v = v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")
            elif t == "select":
                opts = [o[0] if isinstance(o, (list, tuple)) else o for o in f.get("options") or []]
                hit = _match_opt(opts, v)
                if hit is None:   # 不能用 or：选项值是空字符串（「默认」）时也算对上了
                    hit = _match_opt(opts, _VALUE_ALIAS.get(k, {}).get(str(v).lower()))
                if hit is None and k == "optimizer_type":
                    hit = next((o for o in opts if _norm_opt(o) == _norm_opt(v)), None)
                    if hit is None and "custom" in opts and "." in str(v):
                        add(k, "custom")
                        if "optimizer_custom" in fields:
                            add("optimizer_custom", str(v))
                        continue
                if hit is None and k == "resolution" and "custom" in opts and str(v).isdigit():
                    add("resolution", "custom")
                    if "resolution_wh" in fields:
                        add("resolution_wh", f"{v},{v}")
                    continue
                if hit is None:
                    skipped.append({"key": k, "value": _short(v), "reason": f"{f['label']}没有「{v}」这个选项"})
                    continue
                v = hit
            else:
                v = " ".join(map(str, v)) if isinstance(v, list) else str(v)
        except (TypeError, ValueError):
            skipped.append({"key": k, "value": _short(v), "reason": f"{f['label']}的值不对"})
            continue
        if t in ("int", "float"):
            lo, hi = f.get("min"), f.get("max")
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                nv = min(max(v, lo if lo is not None else v), hi if hi is not None else v)
                R.notes.append(f"{f['label']} {v} 超出范围，改成 {nv}")
                v = int(nv) if t == "int" else nv
        add(k, v)
    if locon and "conv_dim" not in params and "conv_dim" in fields:
        R.notes.append("原配置是 LoCon：卷积层 dim 请到「高级参数 → LoRA 结构」里设置。")
    if params.get("v_pred") == "vpred" and "v_pred" not in fields:
        R.notes.append("原配置是 v-pred 模型。")

    mrows = []
    for s in spec.slots:
        src = R.models.get(s.key)
        if not src:
            continue
        found = _find_model(R, src, spec, s)
        mrows.append({"key": s.key, "label": s.label, "src": src, "path": found or ""})
    for k in R.models:
        if k not in {s.key for s in spec.slots}:
            R.notes.append(f"{spec.label} 用不到原配置里的 {k} 模型：{R.models[k]}")
    return rows, skipped, mrows


def _match_opt(opts, v):
    if v is None:
        return None
    sv = str(v).strip().lower()
    for o in opts:
        if str(o).lower() == sv:
            return o
        try:
            if float(o) == float(v):
                return o
        except (TypeError, ValueError):
            pass
    return None


def _short(v):
    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    return s if len(s) <= 120 else s[:117] + "…"


def _local(R, p):
    """路径在本机存在就返回它（相对路径按配置文件所在目录找）。"""
    if not isinstance(p, str) or not p.strip():
        return None
    p = p.strip()
    cands = [p]
    if R.base_dir and not os.path.isabs(p):
        cands.append(os.path.join(R.base_dir, p))
    for c in cands:
        try:
            if os.path.exists(c):
                return os.path.abspath(c)
        except (OSError, ValueError):
            pass
    return None


def _dir_ok(d):
    """保存位置能不能用：本机的绝对路径，文件夹本身或上一级已经存在。"""
    if not d or not os.path.isabs(d):
        return False
    return os.path.isdir(d) or os.path.isdir(os.path.dirname(d.rstrip("\\/")))


def _find_model(R, src, spec, slot):
    """原路径在本机就直接用；不在（别的电脑导出的）就按文件名在本机模型库里找。"""
    p = _local(R, src)
    if p and os.path.isfile(p):
        return p
    base = re.split(r"[\\/]", src.strip())[-1].lower()
    if not re.search(r"\.(safetensors|sft|ckpt|pt|pth|bin|gguf)$", base):
        base += ".safetensors"
    try:
        items = scanner.list_models()
    except Exception:  # noqa: BLE001
        return None
    for it in items:
        if os.path.basename(it["path"]).lower() == base and spec.accept(slot, it):
            return it["path"]
    return None


# ======================= 判断底模 =======================

_FAMILIES = [   # (正则, 底模 id；!开头 = 这个训练器不支持的底模)
    (r"anima", "anima"),
    (r"krea[-_ ]?2", "krea2"),
    (r"moss", "moss"),
    (r"flux|chroma", "!FLUX"),
    (r"\bsd3|sd3_|stable[-_ ]?diffusion[-_ ]?3", "!SD3"),
    (r"qwen[-_]?image", "!Qwen-Image"),
    (r"hunyuan", "!HunyuanVideo"),
    (r"\bwan[-_]?2|\bwan\d|wan_video|\bwan\b", "!Wan"),
    (r"lumina", "!Lumina"),
    (r"hidream", "!HiDream"),
    (r"z[-_]?image", "!Z-Image"),
    (r"pixart", "!PixArt"),
    (r"sdxl|stable[-_ ]?diffusion[-_ ]?xl|\bxl\b|[-_]xl|xl[-_]|illustrious|noob|pony|animagine|stable_cascade", "sdxl"),
    (r"^sd2$|sd[-_]?v?2|stable[-_ ]?diffusion[-_ ]?v?2", "!SD 2.x"),
    (r"sd[-_]?v?1|sd15|stable[-_ ]?diffusion[-_ ]?(v?1|15)|^train_network\.py$|sd-lora|^sd$", "!SD 1.5"),
]


def _family(signals):
    for s in signals:
        s = str(s or "").strip().lower()
        if not s:
            continue
        if models.get(s):
            return s
        for pat, fam in _FAMILIES:
            if re.search(pat, s):
                return fam
    return ""


# ======================= 入口 =======================

def parse(text, name="", target="", current="", base_dir=""):
    """返回给界面的预览：识别结果、参数（逐项可勾选）、模型、测试提示词、注意事项。"""
    R = _R()
    R.base_dir = base_dir or ""
    data, kind = _load(text, name)
    _extract(R, data, kind)
    if R.presets is not None:
        return {"format": R.fmt, "presets": R.presets}

    fam = _family(R.signals)
    unsupported = fam[1:] if fam.startswith("!") else ""
    detected = models.visible_id(fam) if fam and not unsupported and models.get(fam) else ""
    cur = models.get(current)
    if target and models.get(target):
        spec = models.get(target)
    elif detected and cur and (cur.family or cur.id) == (models.get(detected).family or detected):
        spec = cur                                          # 同一个底模换了引擎：用现在选的引擎
    elif detected:
        spec = models.get(detected)
    elif cur:
        spec = cur
    else:
        spec = models.all_specs()[0]

    if R.native:
        params, unknown = dict(R.native["params"]), {}
        R.models = {k: v for k, v in (R.native.get("models") or {}).items() if v}
        R.sample = R.native.get("sample")
        if R.native["model_type"] and R.native["model_type"] != spec.id and models.get(R.native["model_type"]):
            R.notes.append(f"这份配置是给 {models.get(R.native['model_type']).label} 的，套到 {spec.label} 上只带过去两边都有的参数。")
    else:
        params, unknown = _to_params(R)
    rows, skipped, mrows = _fit(spec, params, R)

    sample = None
    if R.sample:
        sample = {k: v for k, v in R.sample.items() if k in ("prompts", "every_n_steps", "every_n_epochs", "at_first", "enabled", "follow_save")}
        if sample.get("prompts"):
            sample["prompts"] = [dict(spec.sample_defaults, **{k: v for k, v in p.items() if v is not None}) for p in sample["prompts"]][:20]
    if unsupported:
        R.notes.insert(0, f"原配置是给 {unsupported} 的，这个训练器不支持 {unsupported}：只能把通用的参数（学习率、轮数、dim 等）套到选的底模上，"
                          "学习率等数值不一定合适。")
    out_dir = R.output_dir if isinstance(R.output_dir, str) else ""
    return {
        "format": R.fmt, "family": fam.lstrip("!"), "detected": detected, "unsupported": unsupported, "target": spec.id,
        "rows": rows, "models": mrows, "sample": sample,
        "output_dir": out_dir, "output_dir_ok": _dir_ok(out_dir),
        "notes": list(dict.fromkeys(R.notes)), "skipped": skipped, "unknown": {k: _short(v) for k, v in unknown.items()},
    }

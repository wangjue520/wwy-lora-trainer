# -*- coding: utf-8 -*-
"""在电脑里找 WebUI / ComfyUI / WWY 启动器共享模型库，列出其中的底模文件并按内容分类。

结果缓存在 data/installs.json（安装位置）和 data/model_cache.json（每个文件的分类，按大小+修改时间失效）。"""
import json
import os
import re
import string
import threading
import time

import paths
import safetensors_meta as stm
import settings

INSTALLS_FILE = os.path.join(paths.DATA, "installs.json")
CACHE_FILE = os.path.join(paths.DATA, "model_cache.json")

SKIP_DIRS = {
    "windows", "program files", "program files (x86)", "programdata", "$recycle.bin", "system volume information",
    "appdata", "node_modules", ".git", "venv", ".venv", "env", "python", "site-packages", "__pycache__",
    "outputs", "output", "steamapps", "recovery", "msocache", "perflogs", "$windows.~ws", "$winreagent",
    "repositories", "extensions", "custom_nodes", "embeddings", "cache", ".cache", "tmp", "temp",
}
MAX_DEPTH = 4

# 各类安装里存放底模的子目录
WEBUI_DIRS = ["models/Stable-diffusion", "models/diffusion_models", "models/unet", "models/text_encoder",
              "models/text_encoders", "models/VAE", "models/vae", "models/clip", "models/MOSS-Image"]
COMFY_DIRS = ["models/checkpoints", "models/diffusion_models", "models/unet", "models/text_encoders",
              "models/clip", "models/vae"]
LIB_DIRS = ["checkpoints", "diffusion_models", "unet", "text_encoders", "clip", "vae", "Stable-diffusion",
            "text_encoder", "VAE", "MOSS-Image"]   # MOSS-Image：sd-forge-moss-image 插件放官方仓库 / 量化单文件的位置

_lock = threading.Lock()
_scan_state = {"running": False, "started": 0, "found": 0, "current": ""}


def _read_json(p, d):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return d


_wlock = threading.Lock()


def _write_json(p, d):
    with _wlock:
        tmp = f"{p}.{threading.get_ident()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        os.replace(tmp, p)


# ---------------- 找安装位置 ----------------
def _sig(path, names):
    try:
        entries = {e.lower() for e in os.listdir(path)}
    except OSError:
        return None
    if "webui.bat" in entries and "modules" in entries and "models" in entries:
        return "webui"
    if "main.py" in entries and "comfy" in entries and "folder_paths.py" in entries:
        return "comfyui"
    if "launcher_config.json" in entries and ("webview_api.py" in entries or "app" in entries):
        return "launcher"
    return None


def _walk(root, depth, out):
    if depth > MAX_DEPTH:
        return
    _scan_state["current"] = root
    kind = _sig(root, None)
    if kind:
        out.append({"type": kind, "path": root})
        _scan_state["found"] = len(out)
        if kind in ("webui", "comfyui"):
            # 安装目录里面不用再深找，但 WWY 启动器常被放在 WebUI 文件夹里，看一层子目录
            try:
                with os.scandir(root) as it:
                    for e in it:
                        if e.is_dir(follow_symlinks=False) and _sig(e.path, None) == "launcher":
                            out.append({"type": "launcher", "path": e.path})
            except OSError:
                pass
            return
    try:
        with os.scandir(root) as it:
            subs = [e.path for e in it if e.is_dir(follow_symlinks=False) and e.name.lower() not in SKIP_DIRS
                    and not e.name.startswith(".")]
    except OSError:
        return
    for s in subs:
        _walk(s, depth + 1, out)


def _drives():
    if os.name != "nt":
        return ["/"]
    return [f"{d}:\\" for d in string.ascii_uppercase if os.path.isdir(f"{d}:\\")]


def _parse_launcher(path):
    """WWY 启动器：取各实例根目录和共享模型库路径。"""
    res = []
    for p in (os.path.join(path, "launcher_config.json"), os.path.join(path, "launcher_data", "launcher_config.json")):
        cfg = _read_json(p, None)
        if not cfg:
            continue
        lib = cfg.get("model_library_path")
        if lib and cfg.get("model_library_enabled", True) and os.path.isdir(lib):
            res.append({"type": "library", "path": lib, "from": path})
        for inst in cfg.get("instances") or []:
            r = inst.get("webui_root")
            if r and os.path.isdir(r):
                t = "comfyui" if (inst.get("webui_branch") == "comfyui" or os.path.isfile(os.path.join(r, "main.py"))) else "webui"
                res.append({"type": t, "path": r, "from": path})
    return res


def scan_installs():
    """全盘找一遍（后台线程调用）。"""
    with _lock:
        if _scan_state["running"]:
            return
        _scan_state.update(running=True, started=time.time(), found=0)
    try:
        out = []
        for d in _drives():
            _walk(d, 0, out)
        extra = []
        for it in out:
            if it["type"] == "launcher":
                extra += _parse_launcher(it["path"])
        seen, res = set(), []
        for it in out + extra:
            key = os.path.normcase(os.path.abspath(it["path"]))
            if key in seen:
                continue
            seen.add(key)
            res.append(it)
        _write_json(INSTALLS_FILE, {"time": time.time(), "items": res})
    finally:
        _scan_state["running"] = False


def scan_state():
    d = dict(_scan_state)
    d["installs"] = _read_json(INSTALLS_FILE, {}).get("items", [])
    return d


def start_scan_async():
    threading.Thread(target=scan_installs, daemon=True).start()


# ---------------- 模型目录 ----------------
def _webui_arg_dirs(root):
    """webui-user.bat 里 --ckpt-dir(s) / --vae-dir / --text-encoder-dir 指定的目录。"""
    res = []
    try:
        txt = open(os.path.join(root, "webui-user.bat"), "r", encoding="utf-8", errors="ignore").read()
    except OSError:
        return res
    for m in re.finditer(r'--(?:ckpt-dirs?|vae-dirs?|text-encoder-dirs?|unet-dirs?)\s+("([^"]+)"|(\S+))', txt):
        p = m.group(2) or m.group(3)
        if p and os.path.isdir(p):
            res.append(p)
    return res


def _comfy_extra_dirs(root):
    """extra_model_paths.yaml（只做够用的简单解析，不依赖 PyYAML）。"""
    res = []
    p = os.path.join(root, "extra_model_paths.yaml")
    if not os.path.isfile(p):
        return res
    base = root
    for line in open(p, "r", encoding="utf-8", errors="ignore"):
        s = line.split("#", 1)[0].rstrip()
        m = re.match(r"\s+base_path:\s*(.+)$", s)
        if m:
            base = m.group(1).strip().strip("'\"")
            continue
        m = re.match(r"\s+(checkpoints|diffusion_models|unet|text_encoders|clip|vae):\s*(.*)$", s)
        if m:
            for v in re.split(r"[|\n]", m.group(2)):
                v = v.strip().strip("'\"")
                if v:
                    q = v if os.path.isabs(v) else os.path.join(base, v)
                    if os.path.isdir(q):
                        res.append(q)
    return res


def model_dirs():
    """所有要列文件的目录 [(dir, source_label)]。"""
    items = _read_json(INSTALLS_FILE, {}).get("items", [])
    res = []
    for it in items:
        r, t = it["path"], it["type"]
        if t == "webui":
            res += [(os.path.join(r, d), f"WebUI · {r}") for d in WEBUI_DIRS]
            res += [(d, f"WebUI 参数目录 · {r}") for d in _webui_arg_dirs(r)]
        elif t == "comfyui":
            res += [(os.path.join(r, d), f"ComfyUI · {r}") for d in COMFY_DIRS]
            res += [(d, f"ComfyUI 额外目录 · {r}") for d in _comfy_extra_dirs(r)]
        elif t == "library":
            res += [(os.path.join(r, d), f"共享模型库 · {r}") for d in LIB_DIRS]
    for d in settings.load().get("extra_search_dirs") or []:
        res.append((d, f"自定义目录 · {d}"))
    res.append((paths.DOWNLOADS, "训练器下载"))
    out, seen = [], set()
    for d, label in res:
        if not os.path.isdir(d):
            continue
        k = os.path.normcase(os.path.realpath(d))
        if k in seen:
            continue
        seen.add(k)
        out.append((d, label))
    return out


def list_models(force=False):
    """列出全部候选底模文件及分类。同一文件（大小相同、文件名相同）在多处出现只保留一个。"""
    cache = {} if force else _read_json(CACHE_FILE, {})
    res, seen_real = [], set()
    for d, label in model_dirs():
        for dirpath, dirnames, files in os.walk(d):
            dirnames[:] = [x for x in dirnames if x.lower() not in ("loras", "lora", "lycoris", "controlnet") and not x.startswith(".")]
            for fn in files:
                if not fn.lower().endswith((".safetensors", ".sft")):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                if st.st_size < 50 * 1024 * 1024:
                    continue
                real = os.path.normcase(os.path.realpath(p))
                # 官方 diffusers 分片（diffusion_pytorch_model-00001-of-00003 等）文件名千篇一律，Base / SFT 两版连大小都一样，
                # 名字带上上两级目录（例如 SFT/transformer/…）才分得清，也不会被当成同一个文件的副本去掉
                shard = bool(re.search(r"-\d{5}-of-\d{5}\.safetensors$", fn, re.I))
                if shard:
                    parts = os.path.normpath(dirpath).split(os.sep)
                    fn_show = "/".join(parts[-2:] + [fn])
                else:
                    fn_show = fn
                same = (fn_show.lower(), st.st_size)   # 同名同大小的副本（WebUI 和 ComfyUI 各放了一份）只列一次
                if real in seen_real or same in seen_real:
                    continue
                seen_real.add(real)
                seen_real.add(same)
                key = real
                c = cache.get(key)
                if not c or c.get("size") != st.st_size or c.get("mtime") != int(st.st_mtime) or c.get("v") != stm.VERSION:
                    info = stm.classify(p)
                    c = {"size": st.st_size, "mtime": int(st.st_mtime), "v": stm.VERSION, **info}
                    cache[key] = c
                res.append({"path": p, "name": fn_show, "size": st.st_size, "source": label,
                            "kind": c["kind"], "quant": c.get("quant", False), "qtype": c.get("qtype", ""), "vpred": c.get("vpred", False),
                            "error": c.get("error")})
    _write_json(CACHE_FILE, cache)
    return res


_side_cache = {}
PREVIEW_EXTS = (".preview.png", ".preview.jpg", ".preview.jpeg", ".preview.webp", ".png", ".jpg", ".jpeg", ".webp")


def sidecar(path):
    """模型旁边的附带信息（WebUI / ComfyUI / WWY 启动器 / Civitai 下载器的约定）：
    预览图 <名字>.preview.png 等；.civitai.info 里的模型名、版本、底模类型。模型多（SDXL 动辄上百个）时靠这些认。"""
    stem = os.path.splitext(path)[0]
    info_p = stem + ".civitai.info"
    try:
        key = (path, os.path.getmtime(info_p) if os.path.exists(info_p) else 0)
    except OSError:
        key = (path, 0)
    if key in _side_cache:
        return _side_cache[key]
    out = {"preview": next((stem + e for e in PREVIEW_EXTS if os.path.isfile(stem + e)), ""),
           "folder": os.path.basename(os.path.dirname(path))}
    if key[1]:
        d = _read_json(info_p, {}) or {}
        out.update(civitai_name=d.get("modelName") or "", civitai_version=d.get("versionName") or "", base_model=d.get("baseModel") or "")
    _side_cache[key] = out
    return out


def _launchers_by_recent():
    """电脑里的 WWY 启动器，最近用过的排前面（按 launcher_config.json 的修改时间）。"""
    res = []
    for it in _read_json(INSTALLS_FILE, {}).get("items", []):
        if it.get("type") != "launcher" or not os.path.isdir(it["path"]):
            continue
        t = 0
        for c in (os.path.join(it["path"], "launcher_config.json"), os.path.join(it["path"], "launcher_data", "launcher_config.json")):
            if os.path.isfile(c):
                t = max(t, os.path.getmtime(c))
        res.append((t, it["path"]))
    res.sort(reverse=True)
    return [p for _, p in res]


def launcher_wd14_models():
    """WWY 启动器「WD14 反推」已经下载好的模型：{目录名（= HF 仓库名，如 wd-eva02-large-tagger-v3）: 目录}。

    启动器把模型放在 <启动器根目录>/wd_tagger_models/<名字>/{model.onnx, selected_tags.csv}；
    老版本代码平铺在根目录，新版本代码在 app/ 里，两处都看。只复用模型文件，不用启动器的环境。
    多个启动器都有同一个模型时，用最近在用的那个启动器里的。"""
    found = {}
    for root in _launchers_by_recent():
        for base in (root, os.path.join(root, "app")):
            d = os.path.join(base, "wd_tagger_models")
            if not os.path.isdir(d):
                continue
            for name in sorted(os.listdir(d)):
                m = os.path.join(d, name)
                onnx = os.path.join(m, "model.onnx")
                if name not in found and os.path.isfile(onnx) and os.path.isfile(os.path.join(m, "selected_tags.csv"))                         and os.path.getsize(onnx) > 10 * 1024 * 1024:
                    found[name] = m
    return found


def launcher_wd14_root():
    """新下载的 WD 模型存到最近在用的启动器的 wd_tagger_models（两边共用）；没有启动器时返回 None。"""
    roots = _launchers_by_recent()
    return os.path.join(roots[0], "wd_tagger_models") if roots else None


def check_file(path):
    """单个文件的完整性检查（训练前体检用）。"""
    if not path or not os.path.isfile(path):
        return "文件不存在"
    if path.lower().endswith((".safetensors", ".sft")):
        _, err = stm.read_header(path)
        return err
    return None

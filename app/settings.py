# -*- coding: utf-8 -*-
"""训练器设置（data/settings.json）与网络环境（国内 / 国外）相关的地址。"""
import json
import os
import threading
import urllib.request

import paths

SETTINGS_FILE = os.path.join(paths.DATA, "settings.json")
DRAFT_FILE = os.path.join(paths.DATA, "draft.json")
PRESETS_FILE = os.path.join(paths.DATA, "presets.json")

DEFAULTS = {
    "port": 28100,
    "host": "127.0.0.1",
    "open_browser": True,          # 启动后自动打开界面
    "network": "auto",             # auto / cn / global
    "network_detected": "",        # auto 模式下检测出来的结果
    "proxy_url": "",               # 自定义代理（VPN 没开「系统代理」模式时填，例如 http://127.0.0.1:7890）
    "theme": "light",              # light / dark
    "torch_variant": "cu128",      # cu126 / cu128 / cu130
    "gpu_id": "0",
    "gpu_ids": [],                 # 多卡训练：选中的显卡编号；空 = 只用 gpu_id 一张
    "output_dir": paths.OUTPUT,    # LoRA 默认保存位置
    "extra_search_dirs": [],       # 额外的模型搜索目录
    "hf_token": "",
    "github_proxy": "https://ghfast.top/",
    "update_repo": "https://github.com/wangjue520/wwy-lora-trainer",   # 训练器自身的 git 仓库地址（一键更新用）
    "shutdown_after_queue": False,  # 队列全部完成后关机
    "auto_check_before_train": True,
    "sync_dataset_dir": True,  # 训练集文件夹同步：爬图保存位置 / 反推 / 标签编辑 / 高级参数改一处全跟着变  # 开训前体检（训练集 / 模型 / 显存）
    "api_presets": [],
    "crawl_presets": [],  # 爬图过滤预设（空 = 用内置的人物 / 画风 / 通用去噪；改过后整份存这里）
    "danbooru_login": "",          # 图片爬取用的 Danbooru 账号（可选）
    "danbooru_api_key": "",             # 自然语言反推 API：[{name, base_url, api_key, model}]
}

_lock = threading.Lock()
_cache = None


def _read(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


_wlock = threading.Lock()


def _write(path, data):
    # 界面会连续发多个保存请求，服务端多线程处理：必须串行写，否则两个线程同时写临时文件会把 JSON 写坏
    with _wlock:
        tmp = f"{path}.{threading.get_ident()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        for _ in range(5):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:   # Windows 上目标文件正被读取时会短暂失败
                import time
                time.sleep(0.05)


def load():
    global _cache
    with _lock:
        if _cache is None:
            d = dict(DEFAULTS)
            d.update(_read(SETTINGS_FILE, {}))
            _cache = d
        return dict(_cache)


def save(patch):
    global _cache
    with _lock:
        d = dict(DEFAULTS)
        d.update(_read(SETTINGS_FILE, {}))
        d.update(patch or {})
        _write(SETTINGS_FILE, d)
        _cache = d
        return dict(d)


def load_draft():
    return _read(DRAFT_FILE, {})


def save_draft(d):
    _write(DRAFT_FILE, d)


def set_draft_model(model_type, slot, path):
    """下载完成后把文件填进当前配置对应底模的槽位。"""
    if not model_type or not path:
        return
    d = load_draft()
    bm = d.setdefault("by_model", {}).setdefault(model_type, {})
    bm.setdefault("models", {})[slot] = path
    save_draft(d)


def load_presets():
    return _read(PRESETS_FILE, {})


def save_presets(d):
    _write(PRESETS_FILE, d)


# ---------------- 网络 ----------------
def _reachable(url, timeout=4):
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=timeout)
        return True
    except Exception as e:
        # 有 HTTP 状态码（403/405 等）说明连得上
        return hasattr(e, "code")


def detect_network():
    """huggingface.co 连不上就按国内网络处理。"""
    cn = not _reachable("https://huggingface.co/")
    v = "cn" if cn else "global"
    save({"network_detected": v})
    return v


def is_cn():
    s = load()
    if s["network"] == "cn":
        return True
    if s["network"] == "global":
        return False
    v = s.get("network_detected") or detect_network()
    return v == "cn"


def pip_index():
    return "https://pypi.tuna.tsinghua.edu.cn/simple" if is_cn() else "https://pypi.org/simple"


def torch_index():
    v = load().get("torch_variant") or "cu128"
    if is_cn():
        # 阿里云的 pytorch 镜像是 find-links 格式
        return ["-f", f"https://mirrors.aliyun.com/pytorch-wheels/{v}/"]
    return ["--index-url", f"https://download.pytorch.org/whl/{v}"]


def hf_endpoint():
    return "https://hf-mirror.com" if is_cn() else "https://huggingface.co"


def github_url(url):
    """GitHub 地址在国内走加速代理。"""
    if is_cn() and url.startswith("https://github.com/"):
        return (load().get("github_proxy") or "https://ghfast.top/") + url
    return url


def gpu_list():
    """训练用的显卡编号列表（至少一张）。"""
    s = load()
    ids = [str(x) for x in (s.get("gpu_ids") or []) if str(x).strip() != ""]
    return ids or [str(s.get("gpu_id") or "0")]


def child_env(extra=None, net=False):
    """给训练 / 反推子进程用的环境变量。net=True：子进程要从 HuggingFace 下载东西（反推模型），
    先测速选线路（镜像 / 官方 × 直连 / 代理），把排好序的线路通过 LORA_HF_ROUTES 交给子进程，失败时它会自己换。"""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_ENDPOINT"] = hf_endpoint()
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    s = load()
    if s.get("hf_token"):
        env["HF_TOKEN"] = s["hf_token"]
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list())
    # 训练环境的 Scripts 目录放到 PATH 最前（accelerate 等命令）
    if paths.env_python():
        env["PATH"] = os.path.dirname(paths.env_python()) + os.pathsep + env.get("PATH", "")
    env.pop("PYTHONHOME", None)
    env["LORA_TRAINER"] = "1"   # 训练器启动的子进程：app/vendor/sitecustomize.py 里的显存保护补丁只在这种进程里生效
    env["PYTHONPATH"] = paths.VENDOR   # 让引擎能按「lora_optim.Automagic」加载训练器自带的优化器
    if s.get("test_vram_limit_gb"):   # 测试用：在大显存卡上模拟小显存卡（界面上没有入口）
        env["LORA_VRAM_LIMIT_GB"] = str(s["test_vram_limit_gb"])
    if net:
        import json as _json
        import netpick
        r = netpick.best_hf()
        if r:
            env["HF_ENDPOINT"] = r[0][0]
            env = netpick.proxy_env(env, r[0][1])
            env["LORA_HF_ROUTES"] = _json.dumps(r)
    if extra:
        env.update(extra)
    return env

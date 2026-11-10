# -*- coding: utf-8 -*-
"""报错诊断用的 AI 接口配置（服务商、Key、模型、接口地址），和 WWY 启动器自动同步。

两边共用一个文件 %APPDATA%\\WWY\\ai_api.json（Linux：~/.config/WWY/ai_api.json）：谁改了就写进去，另一边下次读到就跟着改。
每边各自记着上次看到的共用文件内容（哈希），只有共用文件「变了」才拿过来，所以自己刚改的不会被旧值盖回去。
老版本启动器（3.0.172 起有诊断，但还不认识共用文件）只写自己的 launcher_config.json：
这里把它当成另一个来源，启动器配置里的诊断设置变了也拿过来（只读，不写启动器的配置文件——启动器开着时会被它自己的内存配置覆盖掉）。
键名和启动器一致（diag_llm_*），方便两边对照。"""
import hashlib
import json
import os
import threading
import time

import settings

FIELDS = ("diag_llm_provider", "diag_llm_api_key", "diag_llm_base_url", "diag_llm_model", "diag_llm_format")

# 和启动器 app/diagnose_llm.py 的 PROVIDERS 保持一致
PROVIDERS = {
    "deepseek": {"label": "DeepSeek（国内直连，默认）", "format": "openai", "base": "https://api.deepseek.com",
                 "model": "deepseek-v4-flash", "key_url": "https://platform.deepseek.com/api_keys"},
    "qwen": {"label": "通义千问（阿里云百炼，国内直连）", "format": "openai",
             "base": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus",
             "key_url": "https://bailian.console.aliyun.com/?apiKey=1"},
    "claude": {"label": "Claude（Anthropic，国内需要代理）", "format": "anthropic", "base": "https://api.anthropic.com",
               "model": "claude-sonnet-5-5", "key_url": "https://console.anthropic.com/settings/keys"},
    "custom": {"label": "自定义（OpenAI 兼容接口 / 中转）", "format": "openai", "base": "", "model": "", "key_url": ""},
}
DEFAULT_PROVIDER = "deepseek"
_lock = threading.Lock()


def shared_path():
    if os.environ.get("WWY_AI_API_FILE"):   # 测试用
        return os.environ["WWY_AI_API_FILE"]
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "WWY", "ai_api.json")


def _vals(d):
    d = d or {}
    out = {k: str(d.get(k) or "").strip() for k in FIELDS}
    out["diag_llm_provider"] = out["diag_llm_provider"] or DEFAULT_PROVIDER
    out["diag_llm_format"] = out["diag_llm_format"] or "openai"
    return out


def _hash(v):
    return hashlib.sha1(json.dumps(v, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _read_json(p):
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def read_shared():
    """{values, updated, by}；没有 / 坏了返回 None。"""
    d = _read_json(shared_path())
    if not d or not isinstance(d.get("values"), dict):
        return None
    return {"values": _vals(d["values"]), "updated": float(d.get("updated") or 0), "by": d.get("by") or ""}


def write_shared(values, by="lora_trainer"):
    p = shared_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"values": _vals(values), "updated": time.time(), "by": by}, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def launcher_config():
    """最近在用、且有诊断设置的 WWY 启动器配置：(values, 修改时间, 路径)；没有返回 None。"""
    try:
        import scanner
        roots = scanner._launchers_by_recent()
    except Exception:
        return None
    for root in roots:
        for p in (os.path.join(root, "launcher_config.json"), os.path.join(root, "launcher_data", "launcher_config.json")):
            d = _read_json(p)
            if d and "diag_llm_provider" in d:
                return _vals(d), os.path.getmtime(p), p
    return None


def sync():
    """三方对一下：共用文件变了 / 启动器配置里的变了 → 拿过来；训练器自己改过 → 写进共用文件。返回当前生效的值。"""
    with _lock:
        s = settings.load()
        local = _vals(s)
        seen = dict(s.get("diag_llm_seen") or {})
        cands = []   # (时间, 值, 来源)
        sh = read_shared()
        if sh and _hash(sh["values"]) != seen.get("shared"):
            seen["shared"] = _hash(sh["values"])
            cands.append((sh["updated"], sh["values"], "shared"))
        lc = launcher_config()
        if lc and _hash(lc[0]) != seen.get("launcher"):
            seen["launcher"] = _hash(lc[0])
            if lc[0]["diag_llm_api_key"]:   # 启动器里没填 Key（只有默认值）不算改动
                cands.append((lc[1], lc[0], "launcher"))
        new = local
        if cands:
            cands.sort(key=lambda c: c[0])
            pick = cands[-1][1]
            if pick["diag_llm_api_key"] or not local["diag_llm_api_key"]:
                new = pick
        if not sh or _hash(sh["values"]) != _hash(new):
            if new["diag_llm_api_key"] or sh:
                try:
                    write_shared(new)
                    seen["shared"] = _hash(new)
                except OSError:
                    pass
        patch = {}
        if new != local:
            patch.update(new)
        if seen != (s.get("diag_llm_seen") or {}):
            patch["diag_llm_seen"] = seen
        if patch:
            settings.save(patch)
        return new


def save(values):
    """训练器设置页改了 AI 配置：存到训练器设置，再写进共用文件（启动器下次读到就跟着改）。"""
    v = _vals(dict(current(), **{k: values[k] for k in FIELDS if k in values}))
    with _lock:
        try:
            write_shared(v)
            seen = dict(settings.load().get("diag_llm_seen") or {})
            seen["shared"] = _hash(v)
            settings.save(dict(v, diag_llm_seen=seen))
        except OSError:
            settings.save(v)
    return v


def current():
    return _vals(settings.load())


def resolve(cfg=None):
    """配置 → (format, base, model, key)；没填 Key 返回 None。和启动器 diagnose_llm.resolve 一样。"""
    cfg = _vals(cfg if cfg is not None else sync())
    pid = cfg["diag_llm_provider"]
    p = PROVIDERS.get(pid) or PROVIDERS[DEFAULT_PROVIDER]
    key = cfg["diag_llm_api_key"]
    base = cfg["diag_llm_base_url"] or p["base"]
    model = cfg["diag_llm_model"] or p["model"]
    fmt = (cfg["diag_llm_format"] or p["format"]) if pid == "custom" else p["format"]
    if not key or not base or not model:
        return None
    return fmt, base.rstrip("/"), model, key


def public():
    """给界面看的配置：Key 只给是否填了 + 尾号。"""
    v = sync()
    key = v["diag_llm_api_key"]
    sh = read_shared()
    return {"provider": v["diag_llm_provider"], "model": v["diag_llm_model"], "base_url": v["diag_llm_base_url"],
            "format": v["diag_llm_format"], "has_key": bool(key), "key_tail": key[-4:] if len(key) > 8 else "",
            "providers": [{"id": k, **p} for k, p in PROVIDERS.items()],
            "synced_from": (sh or {}).get("by") or "", "shared_file": shared_path()}


# ---------------- 给 AI 的长期说明（勾「记住」存下来的），按诊断对象记 ----------------
def note_key(target):
    t = target or {}
    if t.get("type") == "task":
        return "task"         # 训练任务的说明对所有任务都有用（比如「显存不够就降分辨率，别动 dim」）
    if t.get("type") == "job":
        return "job:" + str(t.get("kind") or "")
    return "trainer"


def get_note(target):
    return str((settings.load().get("diag_notes") or {}).get(note_key(target)) or "")


def set_note(target, text):
    notes = dict(settings.load().get("diag_notes") or {})
    text = str(text or "").strip()[:2000]
    if text:
        notes[note_key(target)] = text
    else:
        notes.pop(note_key(target), None)
    settings.save({"diag_notes": notes})
    return text

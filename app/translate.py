# -*- coding: utf-8 -*-
"""标签编辑里的翻译：D 站标签查中英字典（tagdict），自然语言标签用本地 Qwen3-VL（纯文字）或 API 翻成中文。

只用来看懂标签，不改标签文件。译文按原文的哈希存在 data/translate_cache.json，同一句只翻一次。"""
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import urllib.request

import jobs
import paths
import settings

CACHE = os.path.join(paths.DATA, "translate_cache.json")
_lock = threading.Lock()
_cache = None

PROMPT = ("Translate the following text-to-image training caption into natural Simplified Chinese. "
          "Keep character names, trigger words and any text quoted from the image exactly as written. "
          "Output only the translation, without explanations.\n\n")


def _key(text):
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


def _load():
    global _cache
    if _cache is None:
        try:
            _cache = json.load(open(CACHE, encoding="utf-8"))
        except Exception:
            _cache = {}
    return _cache


def _put(text, zh):
    with _lock:
        _load()[_key(text)] = zh
        tmp = CACHE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False)
        os.replace(tmp, CACHE)


def get(texts):
    """已经翻过的：{原文: 译文}。"""
    with _lock:
        c = _load()
        return {t: c[_key(t)] for t in texts if (t or "").strip() and _key(t) in c}


def start(texts, engine="local", model="qwen3vl4b", api=None, quant="auto"):
    texts = list(dict.fromkeys(t.strip() for t in texts if (t or "").strip()))
    done = get(texts)
    todo = [t for t in texts if t not in done]
    if not todo:
        raise ValueError("都已经翻译过了")
    if jobs.find_running("translate"):
        raise ValueError("已经有一个翻译任务在运行")
    if engine == "api":
        api = api or {}
        if not api.get("base_url") or not api.get("model"):
            raise ValueError("还没设置 API：到「标签反推 → API 自然语言」里填好接口地址、Key 和模型名")
        return jobs.start("translate", f"翻译标签（API）· {len(todo)} 条", _run_api, todo, api)
    return jobs.start("translate", f"翻译标签（本地模型）· {len(todo)} 条", _run_local, todo, model, quant)


def _run_local(job, todo, model, quant):
    py = paths.env_python()
    if not py:
        raise RuntimeError("本地翻译需要先安装训练环境（训练器设置 → 安装训练环境）；也可以改用 API 翻译。")
    cf = os.path.join(paths.DATA, f"translate_{job.id}.json")
    with open(cf, "w", encoding="utf-8") as f:
        json.dump({"texts": todo, "prompt": PROMPT, "models_dir": paths.TAGGER_MODELS,
                   "vlm": {"model": model, "quant": quant}}, f, ensure_ascii=False)
    n = [0]

    def parse(line):
        if not line.startswith("@@"):
            return True
        try:
            d = json.loads(line[2:])
        except Exception:
            return True
        if d.get("zh") is not None:
            _put(d["src"], d["zh"])
            n[0] += 1
            job.set_progress(n[0] / len(todo), f"{n[0]} / {len(todo)}")
        elif d.get("text"):
            job.set_progress(job.progress, d["text"])
            job.write(d["text"])
        return False

    try:
        rc = job.run_cmd([py, os.path.join(paths.APP, "workers", "translate_worker.py"), cf], cwd=paths.APP,
                         env=settings.child_env(net=True), parse=parse)
    finally:
        try:
            os.remove(cf)
        except OSError:
            pass
    if rc != 0 and not job.stop_flag:
        raise RuntimeError("本地翻译失败，详见日志（显存不够时可以改用 API 翻译）")
    return {"count": n[0]}


def _api_call(api, text):
    body = {"model": api["model"], "max_tokens": 1500, "temperature": 0.2,
            "messages": [{"role": "user", "content": PROMPT + text}]}
    req = urllib.request.Request(api["base_url"].rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {api.get('api_key', '')}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read().decode("utf-8"))
    c = d["choices"][0]["message"].get("content") or ""
    if isinstance(c, list):
        c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
    return re.sub(r"<think>.*?</think>", "", c, flags=re.S).strip()


def _run_api(job, todo, api):
    from tagging import _api_error_zh
    n, errs = 0, 0
    with concurrent.futures.ThreadPoolExecutor(max(1, int(api.get("concurrency") or 4))) as ex:
        futs = {ex.submit(_api_call, api, t): t for t in todo}
        for fu in concurrent.futures.as_completed(futs):
            if job.stop_flag:
                break
            try:
                _put(futs[fu], fu.result())
                n += 1
            except Exception as e:
                errs += 1
                job.write(f"[失败] {_api_error_zh(e)}")
            job.set_progress((n + errs) / len(todo), f"{n + errs} / {len(todo)}")
    if errs and not n:
        raise RuntimeError("API 翻译全部失败，详见日志")
    return {"count": n, "failed": errs}

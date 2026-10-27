# -*- coding: utf-8 -*-
"""反推打标作业：本地模型走训练环境里的 workers/tag_worker.py；API 自然语言直接在本进程里并发请求。"""
import base64
import json
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import captionutil as cu
import jobs
import paths
import scanner
import settings

API_PRESETS = [
    {"name": "OpenAI", "base_url": "https://api.openai.com/v1", "model": "gpt-4.1-mini"},
    {"name": "Google Gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-2.5-flash"},
    {"name": "硅基流动 SiliconFlow", "base_url": "https://api.siliconflow.cn/v1", "model": "Qwen/Qwen3-VL-32B-Instruct"},
    {"name": "阿里云百炼", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen3-vl-plus"},
    {"name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "model": "google/gemini-2.5-flash"},
    {"name": "本地 Ollama", "base_url": "http://127.0.0.1:11434/v1", "model": "qwen3-vl:8b"},
    {"name": "本地 LM Studio", "base_url": "http://127.0.0.1:1234/v1", "model": ""},
]

_last_dir = {"path": ""}


def last_dataset():
    return _last_dir["path"]


def start(cfg):
    d = cfg.get("dataset_dir")
    if not d or not os.path.isdir(d):
        raise ValueError("训练集文件夹不存在")
    if jobs.find_running("tag"):
        raise ValueError("已经有一个反推任务在运行")
    files = cu.list_images(d, cfg.get("recursive", True))
    if not files:
        raise ValueError("这个文件夹里没有图片")
    _last_dir["path"] = d
    mode = cfg.get("mode")
    title = {"wd14": "WD14 标签反推", "vlm": "本地自然语言反推", "hybrid": "WD14 + 自然语言反推",
             "api": "API 自然语言反推"}.get(mode, "反推")
    fn = _run_api if mode == "api" else _run_local
    job = jobs.start("tag", f"{title} · {len(files)} 张", fn, cfg, files)
    job.dataset_dir = d
    return job


def _prefetch(job, conf):
    """反推进程启动前，在这里用多连接下载器把要用的模型下好（hf-mirror / 魔搭 / HF 官方 × 直连 / 代理测速选线路，
    慢了 / 断了自动换线路）。反推进程看到下完的标记就不再联网。这里失败也不要紧，反推进程还会自己再试一遍。"""
    import captionutil as cu
    import downloader
    need = []
    mode = conf.get("mode")
    if mode in ("wd14", "hybrid"):
        w = conf.get("wd14") or {}
        repo = cu.WD14_MODELS.get(w.get("model"), {}).get("repo") or w.get("model")
        name = (repo or "").split("/")[-1]
        lm = conf.get("launcher_models") or {}
        if repo and not (lm.get(name) and os.path.isfile(os.path.join(lm[name], "model.onnx"))):
            local = os.path.join(conf["launcher_dir"], name) if conf.get("launcher_dir") and name in cu.LAUNCHER_WD14 \
                else os.path.join(conf["models_dir"], repo.replace("/", "__"))
            need.append((repo, local, ["model.onnx", "selected_tags.csv"]))
    if mode in ("vlm", "hybrid"):
        v = conf.get("vlm") or {}
        repo = v.get("repo") or cu.VLM_MODELS.get(v.get("model") or "toriigate", {}).get("repo")
        if repo:
            need.append((repo, os.path.join(conf["models_dir"], repo.replace("/", "__")), None))
    for repo, local, pats in need:
        if downloader.is_verified(local):   # 旧版只比大小的标记不算，补校验一次 sha256（坏了会自动重下）
            continue
        try:
            downloader.download_repo(job, repo, local, pats)
        except Exception as e:
            if job.stop_flag:
                raise
            job.write(f"⚠ 预先下载 {repo} 没成功（{e}），交给反推进程再试")


def _run_local(job, cfg, files):
    py = paths.env_python()
    if not py:
        raise RuntimeError("本地反推需要先安装训练环境（训练器设置 → 安装训练环境）。API 反推不需要。")
    conf = dict(cfg)
    conf["files"] = files
    conf["models_dir"] = paths.TAGGER_MODELS
    # 复用 WWY 启动器已经下载好的 WD 模型（只用模型文件，推理仍在训练环境里跑）
    import scanner
    conf["launcher_models"] = scanner.launcher_wd14_models()
    conf["launcher_dir"] = scanner.launcher_wd14_root()
    _prefetch(job, conf)
    cf = os.path.join(paths.DATA, f"tag_{job.id}.json")
    with open(cf, "w", encoding="utf-8") as f:
        json.dump(conf, f, ensure_ascii=False)
    results = []

    def parse(line):
        if not line.startswith("@@"):
            return True
        try:
            d = json.loads(line[2:])
        except Exception:
            return True
        if "n" in d and "i" in d:
            eta = d.get("eta")
            job.set_progress(d["i"] / max(1, d["n"]), f"{d['i']} / {d['n']}" + (f" · 剩余约 {int(eta // 60)} 分 {int(eta % 60)} 秒" if eta else ""))
            if d.get("caption") is not None:
                results.append({"file": d["file"], "caption": d["caption"]})
                job.write(f"{os.path.basename(d['file'])}：{(d['caption'] or '')[:160]}")
            if d.get("error"):
                job.write(f"[失败] {os.path.basename(d['file'])}：{d['error']}")
        elif d.get("dl") is not None:   # 模型下载进度（不写日志，只更新进度条）
            job.set_progress(d["dl"] / d["total"] if d.get("total") else None, d.get("text") or "")
        elif d.get("text"):
            job.set_progress(job.progress, d["text"])
            job.write(d["text"])
        return False

    try:
        rc = job.run_cmd([py, os.path.join(paths.APP, "workers", "tag_worker.py"), cf], cwd=paths.APP,
                         env=settings.child_env(net=True), parse=parse)
    finally:
        try:
            os.remove(cf)
        except OSError:
            pass
    if rc != 0 and not job.stop_flag:
        tail = "\n".join(list(job.log)[-40:])
        import diagnose
        d = diagnose.diagnose(tail, {})
        raise RuntimeError(f"{d['title']}：{d['reason']}")
    return {"count": len(results), "dataset_dir": cfg["dataset_dir"], "trigger": (cfg.get("out") or {}).get("trigger") or ""}


# ---------------- API ----------------
def _api_call(api, img_path, prompt, max_tokens):
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(img_path)[1].lower().lstrip(".")
    mime = {"jpg": "jpeg"}.get(ext, ext)
    body = {"model": api["model"], "max_tokens": int(max_tokens or 600), "temperature": 0.4,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}}]}]}
    url = api["base_url"].rstrip("/") + "/chat/completions"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {api.get('api_key', '')}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read().decode("utf-8"))
    msg = d["choices"][0]["message"]
    c = msg.get("content")
    if isinstance(c, list):
        c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
    c = c or ""
    c = re.sub(r"<think>.*?</think>", "", c, flags=re.S)
    return c.strip()


def _api_error_zh(e):
    s = str(e)
    if hasattr(e, "code"):
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        zh = {401: "API Key 不对或已失效", 403: "没有权限（Key 无效或地区不支持）", 404: "地址或模型名不对",
              429: "请求太频繁或额度用完", 400: "请求被拒绝（模型可能不支持图片输入）"}.get(e.code, f"HTTP {e.code}")
        return f"{zh}：{body}"
    if "timed out" in s:
        return "请求超时"
    if "getaddrinfo" in s or "Name or service" in s:
        return "连不上服务器（地址不对或网络不通）"
    return s


def _run_api(job, cfg, files):
    api = cfg.get("api") or {}
    if not api.get("base_url") or not api.get("model"):
        raise RuntimeError("请填写 API 地址和模型名")
    prompt = (cfg.get("vlm") or {}).get("prompt") or cu.PROMPT_PRESETS["detailed"]
    out = cfg.get("out") or {}
    ext = out.get("ext") or ".txt"
    conc = max(1, min(16, int(api.get("concurrency") or 4)))
    n = len(files)
    lock = threading.Lock()
    state = {"i": 0, "ok": 0, "fail": 0}
    t0 = time.time()

    def one(f):
        if job.stop_flag:
            return
        existing = cu.read_caption(f, ext)
        if out.get("existing") == "skip" and (existing or "").strip():
            text = None
        else:
            text = None
            for attempt in range(3):
                try:
                    text = _api_call(api, f, prompt, (cfg.get("vlm") or {}).get("max_tokens"))
                    break
                except Exception as e:
                    msg = _api_error_zh(e)
                    if attempt == 2 or (hasattr(e, "code") and e.code in (400, 401, 403, 404)):
                        with lock:
                            state["fail"] += 1
                            job.write(f"[失败] {os.path.basename(f)}：{msg}")
                        if hasattr(e, "code") and e.code in (401, 403, 404):
                            job.stop_flag = True
                            job.error = msg
                        break
                    time.sleep(2 * (attempt + 1))
            if text:
                final = cu.compose(existing, None, text, out)
                if final is not None:
                    cu.write_caption(f, final, ext)
                with lock:
                    state["ok"] += 1
                    job.write(f"{os.path.basename(f)}：{(final or '')[:160]}")
        with lock:
            state["i"] += 1
            el = time.time() - t0
            eta = el / state["i"] * (n - state["i"])
            job.set_progress(state["i"] / n, f"{state['i']} / {n} · 成功 {state['ok']} · 失败 {state['fail']} · 剩余约 {int(eta // 60)} 分 {int(eta % 60)} 秒")

    with ThreadPoolExecutor(conc) as ex:
        list(ex.map(one, files))
    if job.error:
        raise RuntimeError(job.error)
    return {"count": state["ok"], "failed": state["fail"], "dataset_dir": cfg["dataset_dir"],
            "trigger": (cfg.get("out") or {}).get("trigger") or ""}


def test_api(api):
    """发一个纯文本请求，确认地址 / Key / 模型能用。"""
    body = {"model": api["model"], "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    url = api["base_url"].rstrip("/") + "/chat/completions"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {api.get('api_key', '')}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            json.loads(r.read().decode())
        return {"ok": True, "message": "连接成功"}
    except Exception as e:
        return {"ok": False, "message": _api_error_zh(e)}

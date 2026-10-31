# -*- coding: utf-8 -*-
"""反推打标作业：本地模型走训练环境里的 workers/tag_worker.py；API 自然语言直接在本进程里并发请求。"""
import base64
import json
import os
import re
import threading
import time
import urllib.error
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
RESUME = os.path.join(paths.DATA, "tag_resume.json")   # 停止 / 出错时还没处理的图，「继续反推」从这里接着做


def last_dataset():
    return _last_dir["path"]


def start(cfg, files=None, base=0):
    """files / base 是「继续反推」用的：只做剩下的图，进度从已经做完的 base 张往后数。"""
    d = cfg.get("dataset_dir")
    if not d or not os.path.isdir(d):
        raise ValueError("训练集文件夹不存在")
    if jobs.find_running("tag"):
        raise ValueError("已经有一个反推任务在运行")
    if files is None:
        files = cu.list_images(d, cfg.get("recursive", True))
    if not files:
        raise ValueError("这个文件夹里没有图片")
    _last_dir["path"] = d
    mode = cfg.get("mode")
    title = {"wd14": "WD14 标签反推", "vlm": "本地自然语言反推", "hybrid": "WD14 + 自然语言反推",
             "api": "API 自然语言反推"}.get(mode, "反推")
    title += f" · {base + len(files)} 张" + ("（继续）" if base else "")
    fn = _run_api if mode == "api" else _run_local
    handled = set()   # 已经处理过的图（写好 / 跳过 / 失败都算）
    job = jobs.start("tag", title, fn, cfg, files, handled, base, on_done=lambda j: _save_resume(j, cfg, files, handled, base))
    job.dataset_dir = d
    return job


def _save_resume(job, cfg, files, handled, base):
    if job.status == "done":
        info = resume_info(full=True)
        if info and info["cfg"].get("dataset_dir") == cfg.get("dataset_dir"):
            os.remove(RESUME)
        return
    left = [f for f in files if f not in handled]
    if not left:
        return
    with open(RESUME, "w", encoding="utf-8") as f:
        json.dump({"cfg": cfg, "files": left, "base": base + len(files) - len(left), "time": time.time()}, f, ensure_ascii=False)


def resume_info(full=False):
    try:
        with open(RESUME, encoding="utf-8") as f:
            r = json.load(f)
    except (OSError, ValueError):
        return None
    if full:
        return r
    return {"dataset_dir": r["cfg"].get("dataset_dir"), "mode": r["cfg"].get("mode"), "left": len(r["files"]),
            "done": r["base"], "total": r["base"] + len(r["files"]), "time": r.get("time")}


def resume():
    r = resume_info(full=True)
    if not r:
        raise ValueError("没有可以继续的反推")
    files = [f for f in r["files"] if os.path.isfile(f)]
    return start(r["cfg"], files, r["base"] + len(r["files"]) - len(files))


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
        own = os.path.join(conf["models_dir"], repo.replace("/", "__")) if repo else ""
        if repo and not downloader.is_verified(own) and (conf.get("launcher_vlm") or {}).get(repo.replace("/", "__")):
            job.write(f"使用 WWY 启动器已下载的模型：{conf['launcher_vlm'][repo.replace('/', '__')]}")
        elif repo:
            need.append((repo, own, None))
    for repo, local, pats in need:
        if downloader.is_verified(local):   # 旧版只比大小的标记不算，补校验一次 sha256（坏了会自动重下）
            continue
        try:
            downloader.download_repo(job, repo, local, pats)
        except Exception as e:
            if job.stop_flag:
                raise
            job.write(f"⚠ 预先下载 {repo} 没成功（{e}），交给反推进程再试")


def _run_local(job, cfg, files, handled, base):
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
    conf["launcher_vlm"] = scanner.launcher_vlm_models()   # 启动器「图片反推」下好的自然语言模型也直接用
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
            handled.add(d["file"])
            eta = d.get("eta")
            i, n = base + d["i"], base + d["n"]
            job.set_progress(i / max(1, n), f"{i} / {n}" + (f" · 剩余约 {int(eta // 60)} 分 {int(eta % 60)} 秒" if eta else ""))
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
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # 安全审核直接拦下的（Gemini PROHIBITED_CONTENT、OpenAI content_policy_violation……）算拒答，不算请求出错
        text = ""
        try:
            text = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        if e.code in (400, 403, 422, 451) and _BLOCKED.search(text):
            raise Refused(f"被 API 的安全审核拦下（HTTP {e.code}）")
        e.body = text
        raise
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    c = msg.get("content")
    if isinstance(c, list):
        c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
    c = c or ""
    c = re.sub(r"<think>.*?</think>", "", c, flags=re.S).strip()
    why = refusal_reason(c, ch.get("finish_reason"), msg.get("refusal"))
    if why:
        raise Refused(why)
    return c


class Refused(Exception):
    """API 模型拒答（直接拒绝、被安全审核拦下，或者只给了一段刻意回避的描述）。"""


_BLOCKED = re.compile(r"safety|prohibited|content[_ ]?(policy|filter|management)|moderation|blocked|harm_category|"
                      r"sensitive|inappropriate|violat|敏感|违规|审核|不当内容", re.I)
# 拒答的说法。模型不一定干脆地说「不行」，常见的是先道歉 / 说明政策，或者给一段「不描述露骨细节」的笼统描述——都算拒答
_REFUSE = [re.compile(x, re.I) for x in (
    r"\bI(?:'m| am)\s+(?:sorry|unable|not able|not comfortable|afraid I can(?:'|no)t)\b",
    r"\bI\s+(?:can(?:'|no)t|cannot|won't|will not|must decline|am not going to|'m not going to|do not feel comfortable)\s+"
    r"(?:help|assist|provide|describe|create|generate|fulfil+|comply|do that|write|caption|process|analy[sz]e|engage|produce|"
    r"go into|detail|depict|discuss|continue|share)",
    r"\b(?:sorry|apolog(?:ize|ies))\b[,.!]?\s+(?:but\s+)?I\b",
    r"\bas an AI\b",
    r"\b(?:violates?|against|breach(?:es)?|not (?:allowed|permitted) (?:by|under))\s+(?:my|the|our|openai'?s?|google'?s?|content|usage|safety)?\s*"
    r"(?:content |usage |safety )?(?:guidelines|polic(?:y|ies)|rules|terms)",
    r"\b(?:not (?:able|allowed|permitted)|unable)\s+to\s+(?:describe|provide|generate|process|assist|help|analy[sz]e|caption)",
    r"\b(?:without|avoid(?:ing)?|won'?t go into|not go(?:ing)? into|omit(?:ting)?|refrain(?:ing)? from|leav(?:e|ing) out|exclud(?:e|ing)|skip(?:ping)?)\s+"
    r"(?:\w+\s+){0,4}(?:explicit|graphic|sexual|nsfw|intimate|lewd|suggestive)",
    r"\b(?:non-?explicit|non-?sexual|tasteful(?:ly)?|family[- ]friendly|safe[- ]for[- ]work|sfw|pg-?13|general audience)\s+"
    r"(?:way|manner|description|terms|version|overview|summary|fashion|level)",
    r"\bhere(?:'s| is)\s+(?:a |an )?(?:general|high[- ]level|non-?explicit|safe|tasteful|brief|limited|toned[- ]down|sanitized)\s+"
    r"(?:description|overview|summary|version)",
    r"[\[(]\s*(?:[\w-]+\s+){0,3}(?:redacted|removed|omitted)\s*[\])]",
    r"(?:抱歉|对不起|很遗憾|不好意思)[，,。！!\s]*(?:我)?(?:无法|不能|没法|不便|不可以)",
    r"(?:无法|不能|不便)(?:为你|为您|帮你|帮您)?(?:提供|描述|处理|协助|完成|生成|分析|识别)",
    r"(?:违反|不符合|违背)(?:了)?(?:相关|平台|内容|使用)?(?:的)?(?:政策|规定|准则|规范|条款)",
    r"(?:避免|不涉及|略过|省略|跳过|不描述|不展开)(?:任何)?(?:过于)?(?:露骨|色情|敏感|成人|性)(?:的)?(?:细节|内容|描写|部分)",
)]
_QUOTED = re.compile(r'"[^"\n]{0,400}"|“[^”\n]{0,400}”|「[^」\n]{0,400}」|『[^』\n]{0,400}』')


def refusal_reason(text, finish_reason=None, refusal=None):
    """API 的回答是不是拒答：是就返回原因，不是返回 ""。
    只看引号外面的文字（描述里照抄的图中文字可能就是「I'm sorry」），开头的道歉 / 政策说明和正文里的「不描述露骨细节」都算。"""
    if refusal:
        return "模型拒答：" + str(refusal)[:80]
    if str(finish_reason or "").lower() in ("content_filter", "safety", "prohibited_content", "blocklist", "spii", "recitation"):
        return f"被 API 的安全审核拦下（{finish_reason}）"
    t = (text or "").strip()
    if not t:
        return "API 返回了空内容（多半是被安全审核拦下）"
    bare = _QUOTED.sub(" ", t)
    for r in _REFUSE:
        m = r.search(bare)
        if m:
            return "模型拒答：" + bare[max(0, m.start() - 20):m.end() + 40].replace("\n", " ").strip()
    return ""


def _api_error_zh(e):
    s = str(e)
    if hasattr(e, "code"):
        body = getattr(e, "body", None)
        if body is None:
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
        body = body[:300]
        zh = {401: "API Key 不对或已失效", 403: "没有权限（Key 无效或地区不支持）", 404: "地址或模型名不对",
              429: "请求太频繁或额度用完", 400: "请求被拒绝（模型可能不支持图片输入）"}.get(e.code, f"HTTP {e.code}")
        return f"{zh}：{body}"
    if "timed out" in s:
        return "请求超时"
    if "getaddrinfo" in s or "Name or service" in s:
        return "连不上服务器（地址不对或网络不通）"
    return s


def _run_api(job, cfg, files, handled, base):
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
                except Refused as e:   # 不重试（换个说法拒绝而已，还多花钱），最后统一交给本地模型
                    with lock:
                        refused.append(f)
                        job.write(f"[拒答] {os.path.basename(f)}：{e}")
                    return progress()
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
            handled.add(f)
        progress()

    def progress():
        with lock:
            state["i"] += 1
            el = time.time() - t0
            eta = el / state["i"] * (n - state["i"])
            job.set_progress((base + state["i"]) / (base + n), f"{base + state['i']} / {base + n} · 成功 {state['ok']} · 失败 {state['fail']}"
                             + (f" · 拒答 {len(refused)}" if refused else "") + f" · 剩余约 {int(eta // 60)} 分 {int(eta % 60)} 秒")

    refused = []
    with ThreadPoolExecutor(conc) as ex:
        list(ex.map(one, files))
    if job.error:
        raise RuntimeError(job.error)
    local = 0
    if refused and not job.stop_flag:
        refused.sort(key=files.index)
        if not api.get("fallback_local", True):
            job.write(f"{len(refused)} 张被 API 拒答，没有开「拒答时用本地模型补上」，这些图没有写标签")
            state["fail"] += len(refused)
            handled.update(refused)
        else:
            key = api.get("fallback_model") or "qwen3vl8b"
            job.write(f"── {len(refused)} 张被 API 拒答，改用本地 {cu.VLM_MODELS.get(key, {}).get('label', key).split('（')[0]} 反推这些图 ──")
            vcfg = dict(cfg.get("vlm") or {}, model=key, quant=api.get("fallback_quant") or "auto", prompt=prompt)
            try:
                local = _run_local(job, dict(cfg, mode="vlm", vlm=vcfg), refused, handled, base + n - len(refused))["count"]
            except Exception as e:
                if job.stop_flag:
                    raise
                job.write(f"[失败] 本地模型没有补上拒答的图：{e}")
            state["fail"] += len(refused) - local
    return {"count": state["ok"] + local, "failed": state["fail"], "refused": len(refused), "local": local,
            "dataset_dir": cfg["dataset_dir"], "trigger": (cfg.get("out") or {}).get("trigger") or ""}


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

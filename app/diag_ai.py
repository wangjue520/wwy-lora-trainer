# -*- coding: utf-8 -*-
"""诊断面板：把训练器里出的问题（失败 / 被拦下的训练任务、失败的后台作业、训练器自己的内部报错）列出来，
先用本地规则（diagnose.py）认，认不出的也照样列出来（兜底），再可以交给 AI 深挖。

照 WWY 启动器 3.0.172 起的报错诊断（启动器 app/diagnose_llm.py）：
  - AI 只能用只读工具查（列目录、读文件、搜文本、看日志、看任务参数、看 pip 包、核对模型文件、测网址、看 HF 仓库）
  - 它推荐的修复只能是白名单里的动作（ACTIONS），校验过才变成按钮；执行前界面一定先让用户确认
  - 「AI 接管」：修 → 重新排队训练 → 看结果 → 再修，最多 5 轮（只对训练任务）
  - 发出去的日志 / 文件内容先脱敏（Key、Token、用户目录）
API 配置（服务商、Key、模型）见 aiconf.py，和启动器自动同步。只用标准库（urllib）。"""
import collections
import copy
import fnmatch
import json
import os
import re
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid

import aiconf
import diagnose
import fixes
import gpu
import jobs
import models
import paths
import settings
import tasks

MAX_STEPS = 8
TAKEOVER_ROUNDS = 5
NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# ======================= 训练器自己的内部报错（兜底：规则认不出也要报） =======================
ERRORS = collections.deque(maxlen=40)   # {id, time, where, text}
_err_lock = threading.Lock()


def record_error(where, text):
    """接口里没预料到的异常、界面脚本报错都记在这里，诊断面板上有角标。同样的报错 1 分钟内只记一次。"""
    text = str(text or "").strip()[-6000:]
    if not text:
        return
    with _err_lock:
        sig = (where, text.splitlines()[-1][:200] if text else "")
        for e in ERRORS:
            if (e["where"], e["text"].splitlines()[-1][:200]) == sig and time.time() - e["time"] < 60:
                return
        ERRORS.append({"id": uuid.uuid4().hex[:8], "time": time.time(), "where": where, "text": text})


# ======================= 脱敏 / 日志片段（和启动器一样） =======================
_SECRET_RES = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"), re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)((?:api[_-]?key|token|authorization|password|usertoken)[\"']?\s*[:=]\s*[\"']?)([^\s\"',}]{6,})"),
    re.compile(r"(?i)(Bearer\s+)([A-Za-z0-9._-]{12,})"),
]


def redact(text):
    if not text:
        return text
    text = str(text)
    home = os.path.expanduser("~")
    if home and len(home) > 3:
        text = re.sub(re.escape(home), "%USERPROFILE%", text, flags=re.IGNORECASE)
        text = re.sub(re.escape(home.replace("\\", "\\\\")), "%USERPROFILE%", text, flags=re.IGNORECASE)
    for r in _SECRET_RES:
        text = r.sub(lambda m: (m.group(1) + "***") if m.lastindex and m.lastindex >= 2 else "***", text)
    return text


def redact_obj(d):
    if isinstance(d, dict):
        return {k: ("***" if any(s in str(k).lower() for s in ("key", "token", "password", "secret")) and v else redact_obj(v))
                for k, v in d.items()}
    if isinstance(d, list):
        return [redact_obj(x) for x in d]
    return redact(d) if isinstance(d, str) else d


def snippet(text, max_chars=12000):
    """给 AI 的日志片段：报错附近的行优先，其余从尾部补，总长有上限。"""
    lines = [l for l in (text or "").splitlines() if l.strip()]
    if sum(len(l) + 1 for l in lines) <= max_chars:
        return "\n".join(lines)
    keep = set()
    err_re = re.compile(r"Traceback|Error|Exception|FAILED|失败|错误", re.IGNORECASE)
    idx = [i for i, l in enumerate(lines) if err_re.search(l)]
    for i in idx[-6:]:
        keep.update(range(max(0, i - 25), min(len(lines), i + 8)))
    keep.update(range(max(0, len(lines) - 60), len(lines)))
    out, total = [], 0
    for i in sorted(keep, reverse=True):
        total += len(lines[i]) + 1
        if total > max_chars:
            break
        out.append((i, lines[i]))
    out.sort()
    res, prev = [], -1
    for i, l in out:
        if prev >= 0 and i != prev + 1:
            res.append("……（省略）……")
        res.append(l[:600])
        prev = i
    return "\n".join(res)


_EXC_RE = re.compile(r"^\s*((?:[A-Za-z_]\w*\.)*[A-Z]\w*(?:Error|Exception|Exit|Fault))(?::\s*(.*))?$")


def last_error_line(text):
    """日志里最后一个异常行（ValueError: xxx），认不出的报错拿它当证据。"""
    for line in reversed((text or "").splitlines()):
        m = _EXC_RE.match(line)
        if m and not m.group(1).endswith(("KeyboardInterrupt", "SystemExit")):
            return line.strip()[:400]
    for line in reversed((text or "").splitlines()):
        if re.search(r"error|错误|失败|failed", line, re.I):
            return line.strip()[:400]
    return ""


# ======================= 诊断对象 =======================
def _job_text(j):
    return "\n".join(list(j.log)) + ("\n" + str(j.error) if j.error else "")


def _task_text(tid, tail=800):
    return "\n".join(tasks.read_log(tid, tail))


def problems():
    """现在有哪些问题（界面角标用）：[{key, target, name, time}]，新的在前。"""
    out = []
    for t in tasks.listing():
        if t.get("status") in ("failed", "blocked"):
            out.append({"key": "task:" + t["id"] + ":" + str(t.get("finished") or t.get("status")),
                        "target": {"type": "task", "id": t["id"]}, "name": "训练任务：" + str(t.get("name") or t["id"]),
                        "time": t.get("finished") or t.get("started") or 0})
    for j in list(jobs._jobs.values()):
        if j.status == "failed":
            out.append({"key": "job:" + j.id, "target": {"type": "job", "id": j.id, "kind": j.kind},
                        "name": "后台作业：" + j.title, "time": j.finished or j.started})
    for e in list(ERRORS):
        out.append({"key": "err:" + e["id"], "target": {"type": "trainer"}, "name": "训练器内部出错", "time": e["time"]})
    out.sort(key=lambda x: -(x["time"] or 0))
    return out


def targets():
    seen, out = set(), []
    for p in problems():
        k = json.dumps(p["target"], sort_keys=True)
        if k not in seen:
            seen.add(k)
            out.append(dict(p["target"], name=p["name"]))
    if not any(t["type"] == "trainer" for t in out):
        out.append({"type": "trainer", "name": "训练器本身"})
    return out


def target_name(t):
    t = t or {}
    if t.get("type") == "task":
        x = tasks.get(t.get("id"))
        return "训练任务：" + (x.get("name") if x else str(t.get("id")))
    if t.get("type") == "job":
        j = jobs.get(t.get("id"))
        return "后台作业：" + (j.title if j else str(t.get("id")))
    return "训练器本身"


def log_text(t):
    t = t or {}
    if t.get("type") == "task":
        return _task_text(t.get("id"))
    if t.get("type") == "job":
        j = jobs.get(t.get("id"))
        return _job_text(j) if j else ""
    return "\n\n".join(time.strftime("%H:%M:%S ", time.localtime(e["time"])) + f"[{e['where']}]\n" + e["text"] for e in ERRORS)


# ======================= 修复动作白名单 =======================
# risk：confirm = 执行前确认；danger = 单独再确认一次（结束进程）；ui = 只是打开页面
ACTIONS = {
    "set_params": {"desc": "改这个训练任务的参数后重新开始", "risk": "confirm", "task": True,
                   "params": "values：{参数名: 新值}，参数名必须是 task_info 里 param_keys 列出的"},
    "lower_vram": {"desc": "按显存档位自动调低显存占用后重新开始", "risk": "confirm", "task": True, "params": "无"},
    "workers_zero": {"desc": "数据加载线程改 0（减少内存占用）后重新开始", "risk": "confirm", "task": True, "params": "无"},
    "clear_cache": {"desc": "清掉这个任务的缓存和训练集里的 .npz 缓存后重新开始", "risk": "confirm", "task": True, "params": "无"},
    "retry": {"desc": "参数不变，重新排队训练", "risk": "confirm", "task": True, "params": "无"},
    "kill_gpu": {"desc": "结束占用显存的其他程序（WebUI / ComfyUI 等）后重新开始", "risk": "danger", "task": True, "params": "无"},
    "download_model": {"desc": "重新下载这个任务某个模型槽位的官方文件，下完自动替换并重试", "risk": "confirm", "task": True,
                       "params": "slot：槽位名（task_info 的 slots 里有下载项的）"},
    "pip_install": {"desc": "在训练环境里 pip install 一个包", "risk": "confirm",
                    "params": "pkg：包名（可带版本，如 bitsandbytes>=0.45）；torch / torchvision / xformers / triton 不能用这个装"},
    "repair_env": {"desc": "修复训练环境（重装全部依赖，不重下 PyTorch）", "risk": "confirm", "params": "无"},
    "reinstall_torch": {"desc": "按设置里的 CUDA 版本重装 PyTorch（约 3 GB）", "risk": "confirm", "params": "无"},
    "update_engine": {"desc": "把 sd-scripts / musubi-tuner 切到训练器测试过的版本并装好依赖", "risk": "confirm", "params": "无"},
    "switch_cn": {"desc": "网络改成国内（HF 走 hf-mirror，pip 走清华源，GitHub 走加速）", "risk": "confirm", "params": "无"},
    "open_page": {"desc": "打开训练器的某个页面", "risk": "ui",
                  "params": "page：train / model / params / crawl / tag / editor / sample / settings"},
}
_PAGES = {"train", "model", "params", "crawl", "tag", "editor", "sample", "settings"}
_PKG_RE = re.compile(r"^(-U\s+)?[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[\w,.-]+\])?(?:[<>=!~]=?[\w.*+-]+(?:,[<>=!~]=?[\w.*+-]+)*)?$")
_NEVER_PIP = {"torch", "torchvision", "torchaudio", "xformers", "triton"}


def actions_doc():
    return "\n".join(f"- {k}：{v['desc']}。参数：{v['params']}" + ("（只对训练任务）" if v.get("task") else "")
                     for k, v in ACTIONS.items())


def _coerce(field, v):
    t = field.get("type")
    if t == "int":
        return int(float(v))
    if t == "float":
        return float(v)
    if t == "bool":
        return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
    if t == "select":
        opts = [o[0] if isinstance(o, (list, tuple)) else o for o in field.get("options") or []]
        if opts and v not in opts and str(v) not in [str(o) for o in opts]:
            raise ValueError(f"{field['key']} 只能取 {opts}")
        return v
    return str(v)


def validate_action(target, aid, params):
    """返回 (params, None) 或 (None, 原因)。"""
    a = ACTIONS.get(aid)
    if not a:
        return None, "不在动作白名单里"
    params = dict(params or {})
    t = tasks.get((target or {}).get("id")) if (target or {}).get("type") == "task" else None
    if a.get("task") and not t:
        return None, "只能用在训练任务上"
    if aid == "set_params":
        spec = models.get(t["model_type"])
        fields = {f["key"]: f for f in (spec.params() if spec else [])}
        vals = {}
        for k, v in (params.get("values") or {}).items():
            if k not in fields:
                return None, f"没有参数 {k}"
            try:
                vals[k] = _coerce(fields[k], v)
            except (ValueError, TypeError) as e:
                return None, f"{k} 的值不对：{e}"
        if not vals:
            return None, "values 为空"
        return {"values": vals}, None
    if aid == "kill_gpu":
        procs = [p for p in gpu.processes(tasks.running_pids()) if p.get("suggest")]
        if not procs:
            return None, "没有找到占用显存的其他程序"
        return {"procs": procs}, None
    if aid == "download_model":
        spec = models.get(t["model_type"])
        slot = next((s for s in (spec.slots if spec else []) if s.key == params.get("slot") and s.downloads), None)
        if not slot:
            return None, "这个槽位没有可下载的官方文件"
        return dict(slot.downloads[0], slot=slot.key), None
    if aid == "pip_install":
        pkg = str(params.get("pkg") or "").strip()
        name = re.split(r"[<>=!~\[]", pkg.replace("-U ", ""))[0].strip().lower()
        if not _PKG_RE.match(pkg) or name in _NEVER_PIP:
            return None, f"不能用 pip 装「{pkg}」"
        return {"pkg": pkg}, None
    if aid == "open_page":
        if params.get("page") not in _PAGES:
            return None, "没有这个页面"
        return {"page": params["page"]}, None
    return {}, None


_UNDO = {}


def apply_action(target, aid, params, rule=False):
    """执行一个动作（界面上用户确认过了才会调到这里）。返回 {ok, text, undo?, job?}。
    rule=True：本地规则（diagnose.py）给的修复，数据是训练器自己算的，原样交给 fixes.apply；
    AI 推荐的动作只能是白名单里的，参数重新校验。"""
    if rule or aid not in ACTIONS:
        if not rule:
            return {"ok": False, "error": "不在动作白名单里"}
    elif aid != "kill_gpu":   # 结束进程：界面上让用户勾过要结束哪些
        params, err = validate_action(target, aid, params)
        if err:
            return {"ok": False, "error": err}
    a = ACTIONS.get(aid) or {"risk": "confirm"}
    if a["risk"] == "ui":
        return {"ok": True, "ui": True, "params": params, "text": "已打开"}
    tid = (target or {}).get("id") if (target or {}).get("type") == "task" else None
    undo = None
    if tid:
        t = tasks.get(tid)
        if t and t.get("status") in ("running", "checking"):
            return {"ok": False, "error": "这个任务正在训练，先停下来再修"}
        if aid in ("set_params", "lower_vram", "workers_zero"):
            old = copy.deepcopy(t.get("params") or {})
            undo = uuid.uuid4().hex[:10]
            _UNDO[undo] = {"kind": "params", "tid": tid, "old": old}
    if aid == "switch_cn":
        undo = uuid.uuid4().hex[:10]
        _UNDO[undo] = {"kind": "settings", "old": {"network": settings.load().get("network")}}
    if aid == "retry":
        tasks.requeue(tid)
        tasks.start_queue()
        return {"ok": True, "text": "已重新加入队列"}
    if aid == "kill_gpu":
        params = {"procs": [p for p in (params or {}).get("procs") or [] if isinstance(p, dict) and isinstance(p.get("pid"), int)]}
    r = fixes.apply(tid, aid, params)
    if not r.get("ok"):
        _UNDO.pop(undo, None)
        return {"ok": False, "error": r.get("message") or "执行失败"}
    if undo and _UNDO.get(undo, {}).get("kind") == "params":
        new = (tasks.get(tid) or {}).get("params") or {}
        old = _UNDO[undo]["old"]
        _UNDO[undo]["old"] = {k: old.get(k) for k in new if new.get(k) != old.get(k)}
        if not _UNDO[undo]["old"]:
            _UNDO.pop(undo)
            undo = None
    out = {"ok": True, "text": r.get("message") or "已完成", "job": r.get("job")}
    if undo:
        out["undo"] = undo
    if r.get("params"):
        out["params"], out["model_type"] = r["params"], r.get("model_type")
    return out


def undo(token):
    u = _UNDO.pop(token, None)
    if not u:
        return {"ok": False, "error": "已经撤销过了，或者训练器重启过"}
    if u["kind"] == "settings":
        settings.save(u["old"])
        return {"ok": True, "text": "已撤销"}
    t = tasks.get(u["tid"])
    if not t:
        return {"ok": False, "error": "任务已经删掉了"}
    if t.get("status") in ("running", "checking"):
        _UNDO[token] = u
        return {"ok": False, "error": "任务正在训练，停下来以后再撤销"}
    params = dict(t["params"], **u["old"])
    tasks.update_task(u["tid"], {"params": params})
    return {"ok": True, "text": "参数已改回：" + "，".join(f"{k}={v}" for k, v in u["old"].items()),
            "params": u["old"], "model_type": t["model_type"]}


def _fix_to_action(fx, code=""):
    """diagnose.py 规则给的修复 → 面板上的动作按钮（数据原样带上，执行走 fixes.apply）。"""
    return {"aid": fx["id"], "label": fx["label"], "desc": fx.get("desc") or "", "params": fx.get("data") or {},
            "risk": "danger" if fx.get("danger") else "confirm", "code": code, "rule": True}


# ======================= 规则诊断 =======================
def report(target):
    target = target or {"type": "trainer"}
    text = log_text(target)
    findings = []
    typ = target.get("type")
    if typ == "task":
        t = tasks.get(target.get("id"))
        if not t:
            return {"ok": False, "error": "找不到这个任务（可能已经删掉了）"}
        if t.get("status") == "blocked":
            for it in t.get("issues") or []:
                findings.append({"title": it["title"], "text": it.get("reason") or "", "evidence": "",
                                 "level": it.get("level") or "error",
                                 "actions": [_fix_to_action(f, it.get("code")) for f in it.get("fixes") or [] if f["id"] != "ignore"]})
        elif t.get("status") == "failed":
            err = t.get("error") or diagnose.diagnose(text, t)
            ev = last_error_line(err.get("detail") or text)
            f = {"title": err.get("title") or "训练失败", "text": err.get("reason") or "", "evidence": ev, "level": "error",
                 "actions": [_fix_to_action(x, err.get("code")) for x in err.get("fixes") or []]}
            if err.get("code") in ("unknown", "internal", None) and not f["actions"]:
                f["text"] = (f["text"] + " 本地规则没认出这个报错，可以让 AI 结合日志和本地文件再看看。").strip()
            f["actions"].append({"aid": "retry", "label": "不修改，直接重试", "params": {}, "risk": "confirm", "rule": True})
            findings.append(f)
        else:
            text = text or ""
    elif typ == "job":
        j = jobs.get(target.get("id"))
        if not j:
            return {"ok": False, "error": "找不到这个作业（训练器重启后作业记录会清空）"}
        if j.status == "failed":
            tail = "\n".join([x for x in text.splitlines() if x.strip()][-200:])
            hit = None
            for code, pat, title, reason, fx in diagnose.RULES:
                m = re.search(pat, tail, re.I)
                if m:
                    hit = (code, title, reason.format(*(m.groups() or ("",))), fx, m)
                    break
            acts = []
            if hit:
                try:
                    fx = diagnose._fixes_for(hit[3], hit[4], {}, None, {}, tail) if hit[3] else []
                except Exception:   # 有的修复要用到训练任务的参数，作业里没有
                    fx = []
                for x in fx:
                    if x["id"] in ACTIONS and not ACTIONS[x["id"]].get("task"):
                        acts.append(_fix_to_action(x, hit[0]))
                findings.append({"title": hit[1], "text": hit[2], "evidence": last_error_line(tail), "level": "error", "actions": acts})
            else:   # 兜底：认不出也要报出来
                findings.append({"title": f"「{j.title}」失败", "text": str(j.error or "作业出错了") +
                                 "。本地规则没认出这个报错，可以让 AI 看看。", "evidence": last_error_line(tail), "level": "error",
                                 "actions": []})
    else:
        for e in reversed(list(ERRORS)):
            findings.append({"title": "训练器内部出错（" + e["where"] + "）", "evidence": last_error_line(e["text"]) or e["text"][-300:],
                             "text": time.strftime("%m-%d %H:%M:%S", time.localtime(e["time"])) +
                             "：界面或后台遇到了没预料到的错误，规则没法自动处理。可以让 AI 看看，或者把诊断报告发给作者。",
                             "level": "error", "actions": [], "eid": e["id"]})
    conf = aiconf.resolve()
    return {"ok": True, "target": target, "name": target_name(target), "lines": len(text.splitlines()), "findings": findings,
            "ai_ready": bool(conf), "ai_model": conf[2] if conf else "", "note": aiconf.get_note(target),
            "can_takeover": typ == "task" and ((tasks.get(target.get("id")) or {}).get("status") in ("failed", "blocked")),
            "takeover": active_takeover(target) if typ == "task" else ""}


def clear_errors(ids=None):
    with _err_lock:
        keep = [e for e in ERRORS if ids is not None and e["id"] not in ids]
        ERRORS.clear()
        ERRORS.extend(keep)


# ======================= 大模型接口 =======================
class LlmError(Exception):
    pass


def _opener():
    proxy = (settings.load().get("proxy_url") or "").strip()
    if proxy:
        return urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener()   # 默认读系统代理（Windows 注册表 / 环境变量）


def _post(url, headers, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers=headers, method="POST")
    try:
        with _opener().open(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            status = r.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        status = e.code
    try:
        data = json.loads(raw)
    except ValueError:
        data = None
    return status, data, raw[:500]


def _net_error_text(e):
    s = str(e)
    if "proxy" in s.lower() or "Tunnel" in s:
        return f"连不上代理：{s[:200]}"
    if "timed out" in s.lower() or isinstance(e, TimeoutError):
        return "请求超时（模型接口没响应；Claude 在国内需要开梯子）"
    return f"连不上模型接口：{s[:200]}"


def _http_error_text(status, data, raw):
    msg = ""
    if isinstance(data, dict):
        err = data.get("error")
        msg = (err.get("message") if isinstance(err, dict) else err) or data.get("message") or ""
    msg = msg or raw
    if status in (401, 403):
        return f"API Key 无效或没有权限（{status}）：{msg}"
    if status == 402:
        return f"账户余额不足（402）：{msg}"
    if status == 404:
        return f"接口地址或模型名不对（404）：{msg}"
    if status == 429:
        return f"请求太频繁或额度用完（429）：{msg}"
    return f"模型接口返回错误 {status}：{msg}"


class Chat:
    """一次对话。send() 返回 (文本, [(调用id, 工具名, 参数dict)])，并把回复记进历史。和启动器 diagnose_llm.Chat 一样。"""

    def __init__(self, fmt, base, model, key, system, tools, timeout=120, post=None):
        self.fmt, self.base, self.model, self.key = fmt, base, model, key
        self.system, self.tools, self.timeout = system, tools, timeout
        self.messages = []
        self._post = post or _post

    def user(self, text):
        self.messages.append({"role": "user", "content": text})

    def tool_results(self, results):
        if self.fmt == "anthropic":
            self.messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": cid, "content": out} for cid, out in results]})
        else:
            for cid, out in results:
                self.messages.append({"role": "tool", "tool_call_id": cid, "content": out})

    def send(self, max_tokens=4096):
        if self.fmt == "anthropic":
            url = self.base + ("/messages" if self.base.endswith("/v1") else "/v1/messages")
            headers = {"x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            body = {"model": self.model, "max_tokens": max_tokens, "system": self.system, "messages": self.messages,
                    "temperature": 0.2}
            if self.tools:
                body["tools"] = [{"name": n, "description": d, "input_schema": s} for n, d, s in self.tools]
        else:
            url = self.base + "/chat/completions"
            headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
            body = {"model": self.model, "max_tokens": max_tokens, "temperature": 0.2,
                    "messages": [{"role": "system", "content": self.system}] + self.messages}
            if self.tools:
                body["tools"] = [{"type": "function", "function": {"name": n, "description": d, "parameters": s}}
                                 for n, d, s in self.tools]
        try:
            status, data, raw = self._post(url, headers, body, self.timeout)
        except Exception as e:
            raise LlmError(_net_error_text(e)) from e
        if status != 200 or not isinstance(data, dict):
            raise LlmError(_http_error_text(status, data, raw))
        if self.fmt == "anthropic":
            blocks = data.get("content") or []
            self.messages.append({"role": "assistant", "content": blocks})
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            return text, [(b["id"], b["name"], b.get("input") or {}) for b in blocks if b.get("type") == "tool_use"]
        msg = ((data.get("choices") or [{}])[0].get("message")) or {}
        keep = {k: v for k, v in msg.items() if v is not None and k in ("role", "content", "tool_calls", "reasoning_content")}
        keep["role"] = "assistant"
        keep.setdefault("content", "")
        self.messages.append(keep)
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {"_raw": fn.get("arguments")}
            calls.append((c.get("id"), fn.get("name"), args if isinstance(args, dict) else {}))
        return msg.get("content") or "", calls


def ping(values, post=None):
    """设置页「测试」：发一句话看通不通。返回 (ok, 说明)。"""
    conf = aiconf.resolve(values)
    if not conf:
        return False, "先填 API Key（自定义还要填接口地址和模型名）"
    fmt, base, model, key = conf
    chat = Chat(fmt, base, model, key, "你是测试助手。", [], timeout=30, post=post)
    chat.user("只回复两个字：正常")
    t0 = time.time()
    try:
        text, _ = chat.send(max_tokens=20)
    except LlmError as e:
        return False, str(e)
    return True, f"{model} 已连通（{time.time() - t0:.1f} 秒），回复：{text.strip()[:30]}"


# ======================= 只读工具 =======================
def _obj(props, required=()):
    return {"type": "object", "properties": props, "required": list(required)}


_S = {"type": "string"}
_I = {"type": "integer"}

TOOL_SPECS = [
    ("list_dir", "列出文件夹内容（名字、大小、是否文件夹）。只能看允许的目录。",
     _obj({"path": _S, "pattern": {"type": "string", "description": "可选，文件名通配符，如 *.txt"}}, ["path"])),
    ("read_file", "读文本文件的一段（按行）。图片和模型权重读不了。",
     _obj({"path": _S, "start_line": _I, "max_lines": {"type": "integer", "description": "默认 200，最多 400"}}, ["path"])),
    ("search_files", "在文件夹里按正则搜索文本文件内容，返回 文件:行号:内容。默认跳过 .git / env / models。",
     _obj({"path": _S, "pattern": _S, "glob": {"type": "string", "description": "文件名通配符，默认 *"}}, ["path", "pattern"])),
    ("get_log", "读日志。source 取值见系统说明里的「日志来源」。",
     _obj({"source": _S, "tail": {"type": "integer", "description": "最后多少行，默认 300"},
           "grep": {"type": "string", "description": "可选，只要包含这个正则的行（带前后 3 行）"}}, ["source"])),
    ("task_info", "训练任务的详细信息：底模、模型文件、全部参数、可以改的参数名（param_keys）、模型槽位、状态、报错、用的显卡。",
     _obj({"id": _S}, ["id"])),
    ("python_packages", "查训练环境里装的 Python 包版本。names 为空则列出全部。", _obj({"names": {"type": "array", "items": _S}})),
    ("check_model_file", "核对模型文件（safetensors）是否完整、是什么模型（Anima / Anima 2.9B / SDXL / MOSS …）。", _obj({"path": _S}, ["path"])),
    ("gpu_status", "显卡信息（型号、显存、驱动）和正在占用显卡的程序。", _obj({})),
    ("test_url", "从这台电脑（按训练器的代理设置）访问一个网址，看能不能连通、耗时多少。", _obj({"url": _S}, ["url"])),
    ("hf_files", "列出 HuggingFace 仓库里的文件。", _obj({"repo": {"type": "string", "description": "仓库名，如 circlestone-labs/Anima"}}, ["repo"])),
    ("finish", "诊断结束时调用，交最终结论。",
     _obj({"summary": {"type": "string", "description": "一句话结论"},
           "cause": {"type": "string", "description": "原因分析，引用日志 / 文件里的证据"},
           "steps": {"type": "array", "items": _S, "description": "用户要做的操作步骤，具体到训练器哪个页面哪个按钮"},
           "actions": {"type": "array", "description": "推荐的一键修复（只能用动作白名单里的）",
                       "items": _obj({"action": _S, "params": {"type": "object"}, "label": _S}, ["action"])},
           "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
           "verdict": {"type": "string", "enum": ["fixed", "continue", "need_user"],
                       "description": "只在「AI 接管」模式填：fixed=已经没问题了，continue=执行这轮 actions 后重新训练再看，"
                                      "need_user=要用户自己动手（写在 steps 里）"}},
          ["summary", "cause", "steps"])),
]

_SKIP_DIRS = {".git", "__pycache__", "node_modules", "env", "venv", ".venv", "models", "output", "python", "git"}
_STEP_LABELS = {"list_dir": "查看目录", "read_file": "读取文件", "search_files": "搜索", "get_log": "查看日志",
                "task_info": "查看任务参数", "python_packages": "查看 Python 包", "check_model_file": "核对模型文件",
                "gpu_status": "查看显卡", "test_url": "测试网址", "hf_files": "查看 HuggingFace 仓库"}


def _under(p, root):
    try:
        p, root = os.path.normcase(os.path.abspath(p)), os.path.normcase(os.path.abspath(root))
        return p == root or p.startswith(root.rstrip("\\/") + os.sep)
    except (TypeError, ValueError):
        return False


def allowed_roots():
    roots = [paths.ROOT]
    s = settings.load()
    roots += [s.get("output_dir") or ""] + list(s.get("extra_search_dirs") or [])
    for t in tasks.listing()[-30:]:
        p = t.get("params") or {}
        roots += [t.get("output_dir") or "", p.get("dataset_dir") or "", p.get("reg_data_dir") or ""]
        roots += [os.path.dirname(v) for v in (t.get("models") or {}).values() if isinstance(v, str) and v]
    try:
        import scanner
        roots += [it["path"] for it in scanner._read_json(scanner.INSTALLS_FILE, {}).get("items", []) if it.get("type") == "library"]
    except Exception:
        pass
    return [r for r in dict.fromkeys(os.path.normpath(r) for r in roots if r) if os.path.isdir(r)]


def log_sources():
    out = {"trainer": "训练器自己的内部报错（接口异常、界面脚本报错）"}
    for t in tasks.listing()[-15:]:
        out["task:" + t["id"]] = f"训练任务「{t.get('name')}」的训练日志（{t.get('status')}）"
    for j in list(jobs._jobs.values())[-15:]:
        out["job:" + j.id] = f"后台作业「{j.title}」（{j.status}）"
    return out


class Tools:
    def __init__(self, log_text=""):
        self.roots = allowed_roots()
        self.mentioned = {os.path.normcase(os.path.abspath(p)) for p in
                          re.findall(r"[A-Za-z]:[\\/][^\r\n\"'<>|*?:]+\.\w{1,12}", (log_text or "").replace("\\\\", "\\"))}

    def _ok_path(self, path):
        if not path:
            return None, "缺少 path"
        p = os.path.abspath(os.path.expandvars(str(path).replace("%USERPROFILE%", os.path.expanduser("~"))))
        if any(_under(p, r) for r in self.roots) or os.path.normcase(p) in self.mentioned:
            return p, None
        return None, "这个路径不在允许读取的范围里（允许的目录见系统说明）"

    def call(self, name, args):
        fn = getattr(self, "t_" + (name or ""), None)
        if not fn:
            return f"没有这个工具：{name}"
        try:
            out = fn(**{k: v for k, v in (args or {}).items() if not k.startswith("_")})
        except TypeError as e:
            return f"参数不对：{e}"
        except Exception as e:
            return f"出错了：{e}"
        return redact(out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, default=str))[:20000]

    def t_list_dir(self, path, pattern=""):
        p, err = self._ok_path(path)
        if err:
            return err
        if not os.path.isdir(p):
            return "不是文件夹或不存在"
        rows = []
        for name in sorted(os.listdir(p))[:400]:
            if pattern and not fnmatch.fnmatch(name.lower(), pattern.lower()):
                continue
            fp = os.path.join(p, name)
            if os.path.isdir(fp):
                rows.append(name + "/")
            else:
                try:
                    rows.append(f"{name}  {os.path.getsize(fp) / 1e6:.1f} MB")
                except OSError:
                    rows.append(name)
        return "\n".join(rows) or "（空）"

    def t_read_file(self, path, start_line=1, max_lines=200):
        p, err = self._ok_path(path)
        if err:
            return err
        if not os.path.isfile(p):
            return "文件不存在"
        if p.lower().endswith((".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".npz", ".png", ".jpg", ".jpeg", ".webp", ".onnx")):
            return "这是二进制文件，读不了；模型文件用 check_model_file"
        if os.path.basename(p).lower() == "settings.json":
            with open(p, encoding="utf-8") as f:
                return json.dumps(redact_obj(json.load(f)), ensure_ascii=False, indent=1)[:20000]
        start, n = max(1, int(start_line or 1)), min(400, max(1, int(max_lines or 200)))
        out = []
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                if i >= start + n:
                    break
                if i >= start:
                    out.append(f"{i}: {line.rstrip()[:500]}")
        return "\n".join(out) or "（这段没有内容）"

    def t_search_files(self, path, pattern, glob="*"):
        p, err = self._ok_path(path)
        if err:
            return err
        try:
            rx = re.compile(pattern, re.I)
        except re.error as e:
            return f"正则写错了：{e}"
        hits, scanned = [], 0
        walk = [(os.path.dirname(p), [], [os.path.basename(p)])] if os.path.isfile(p) else os.walk(p)
        for d, dirs, files in walk:
            dirs[:] = [x for x in dirs if x not in _SKIP_DIRS and not x.startswith(".")]
            for n in files:
                if not fnmatch.fnmatch(n.lower(), (glob or "*").lower()):
                    continue
                fp = os.path.join(d, n)
                try:
                    if os.path.getsize(fp) > 2_000_000:
                        continue
                    scanned += 1
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            if rx.search(line):
                                hits.append(f"{os.path.relpath(fp, p if os.path.isdir(p) else d)}:{i}: {line.strip()[:300]}")
                                if len(hits) >= 80:
                                    return "\n".join(hits) + "\n（结果太多，只列前 80 条）"
                except OSError:
                    pass
                if scanned > 3000:
                    return "\n".join(hits) + "\n（文件太多，只搜了前 3000 个）"
        return "\n".join(hits) or "没搜到"

    def t_get_log(self, source, tail=300, grep=""):
        source = str(source or "")
        if source.startswith("task:"):
            text = _task_text(source[5:], 3000)
        elif source.startswith("job:"):
            j = jobs.get(source[4:])
            text = _job_text(j) if j else None
        elif source == "trainer":
            text = log_text({"type": "trainer"})
        else:
            text = None
        if text is None:
            return "没有这个日志来源"
        lines = text.splitlines()
        if grep:
            try:
                rx = re.compile(grep, re.I)
            except re.error as e:
                return f"正则写错了：{e}"
            keep = set()
            for i, l in enumerate(lines):
                if rx.search(l):
                    keep.update(range(max(0, i - 3), min(len(lines), i + 4)))
            lines = [lines[i] for i in sorted(keep)]
        return "\n".join(lines[-min(1500, max(1, int(tail or 300))):]) or "（空）"

    def t_task_info(self, id):
        t = tasks.get(id)
        if not t:
            return "没有这个任务"
        spec = models.get(t["model_type"])
        info = {k: t.get(k) for k in ("id", "name", "status", "message", "model_type", "models", "params", "gpus", "output_dir",
                                      "started", "finished", "resume")}
        info["底模"] = spec.label if spec else t["model_type"]
        info["error"] = {k: (t.get("error") or {}).get(k) for k in ("code", "title", "reason")}
        info["progress"] = {k: v for k, v in (t.get("progress") or {}).items() if k != "loss_hist"}
        if spec:
            info["param_keys"] = {f["key"]: f.get("label") for f in spec.params()}
            info["slots"] = [{"key": s.key, "label": s.label, "有官方下载": bool(s.downloads)} for s in spec.slots]
            info["显存档位"] = [{"id": x["id"], "note": x["note"], "values": x["values"]} for x in spec.vram_presets()]
        return redact_obj(info)

    def t_python_packages(self, names=None):
        py = paths.env_python()
        if not py:
            return "训练环境还没安装"
        code = ("import json,sys,importlib.metadata as m\nnames=json.loads(sys.argv[1])\nout={}\n"
                "if names:\n  for n in names:\n    try: out[n]=m.version(n)\n    except Exception: out[n]=None\n"
                "else:\n  out={d.metadata['Name']:d.version for d in m.distributions()}\n"
                "try:\n  import torch; out['__torch_cuda__']=torch.version.cuda\nexcept Exception as e: out['__torch_cuda__']=str(e)\n"
                "print(json.dumps({'python':sys.version.split()[0],'packages':out}))")
        r = subprocess.run([py, "-s", "-c", code, json.dumps(list(names or []))], capture_output=True, text=True, timeout=120,
                           creationflags=NO_WINDOW, encoding="utf-8", errors="replace")
        return (r.stdout or r.stderr)[-15000:]

    def t_check_model_file(self, path):
        p, err = self._ok_path(path)
        if err:
            return err
        import safetensors_meta as stm
        import scanner
        bad = scanner.check_file(p)
        out = {"文件": p, "大小 GB": round(os.path.getsize(p) / 1e9, 2) if os.path.isfile(p) else None, "完整性": bad or "正常"}
        if not bad and p.lower().endswith(".safetensors"):
            out["分类"] = stm.classify(p)
        return out

    def t_gpu_status(self):
        return {"显卡": gpu.info(max_age=0), "占用显卡的程序": [{k: x.get(k) for k in ("pid", "name", "label", "used_mb")}
                                                     for x in gpu.processes()]}

    def t_test_url(self, url):
        if not re.match(r"^https?://", str(url or "")):
            return "只能测 http / https 网址"
        t0 = time.time()
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "wwy-lora-trainer"})
            with _opener().open(req, timeout=15) as r:
                return f"能连通：HTTP {r.status}，{time.time() - t0:.1f} 秒"
        except urllib.error.HTTPError as e:
            return f"连通了，但返回 HTTP {e.code}（{time.time() - t0:.1f} 秒）"
        except Exception as e:
            return f"连不上（{time.time() - t0:.1f} 秒）：{e}"

    def t_hf_files(self, repo):
        if not re.match(r"^[\w.-]+/[\w.-]+$", str(repo or "")):
            return "仓库名格式不对"
        base = settings.hf_endpoint()
        try:
            with _opener().open(urllib.request.Request(f"{base}/api/models/{repo}", headers={"User-Agent": "wwy-lora-trainer"}),
                                timeout=20) as r:
                d = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            return f"读不到仓库（{base}）：{e}"
        return "\n".join(s.get("rfilename", "") for s in d.get("siblings") or [])[:8000]

    def t_finish(self, **kw):
        return "ok"


# ======================= agent =======================
KNOWLEDGE = """\
- 训练器（WWY LoRA 训练器）：后端 app/server.py（只用标准库），训练在单独的训练环境 env/（PyTorch + sd-scripts + musubi-tuner）里跑。
  Anima / Anima 2.9B / SDXL / MOSS-Image 用 kohya sd-scripts，Krea 2 用 musubi-tuner。底模的配置在 app/models/*.py。
- 训练器给引擎打的补丁都在 app/vendor/sitecustomize.py（冻结续训、显存保护、Anima 2.9B 按 40 层加载、int8 内核提速等）。
- 「Torch not compiled with CUDA enabled」：训练环境里的 PyTorch 被换成了 CPU 版，用 reinstall_torch。
- 显存不足：优先 lower_vram（按显存档位调块交换 / 批量 / 分辨率）；有 WebUI / ComfyUI 在占显存时 kill_gpu。
  「CUDA error: out of memory」不一定是显存，内存（RAM）满了也会这样，先 workers_zero。
- 反推模型分片坏了（输出乱码 / 重复字）：下载器会校验 sha256 自动重下。
- 选显卡：设置页的编号和 nvidia-smi 一致（按 PCI 总线），子进程固定 CUDA_DEVICE_ORDER=PCI_BUS_ID。
- Anima 2.9B（40 层）和 Anima 2B（28 层）的 LoRA 不能混用。int8 ConvRot 的 Anima 2.9B 不能训练。
- 用户大多在中国大陆；网络问题优先 switch_cn（hf-mirror / 清华源 / GitHub 加速），或到「训练器设置 → 网络」填代理。
- 不要建议重装系统、删训练环境这种大动作，除非证据充分；PyTorch 只能用 reinstall_torch 装，不要让用户手动 pip install torch。
"""


def system_prompt():
    roots = "\n".join(f"- {r}" for r in allowed_roots())
    sources = "\n".join(f"- {k}：{v}" for k, v in log_sources().items())
    return (
        "你是 WWY LoRA 训练器内置的报错诊断助手。训练器是一个本地网页程序，用来给 Anima、SDXL、MOSS-Image、Krea 2 等底模训练 LoRA，"
        "还有图片爬取、标签反推、标签编辑。用户大多在中国大陆，不太懂技术。你的任务：根据日志和环境信息找出报错的真正原因，"
        "给出用户照着就能做的解决办法。\n\n"
        "工作方式：\n"
        "1. 先看给你的日志片段和环境信息；信息不够就用工具查（读文件、搜代码、看任务参数、看 pip 包、核对模型、测网址），最多查 6 步。\n"
        "2. 工具全是只读的，你不能直接改任何东西。要修就在 finish 的 actions 里推荐白名单动作，用户确认后才执行。"
        "白名单外的操作写进 steps 让用户手动做。\n"
        "3. 最后必须调用 finish 交结论。用中文，口语化，别堆术语；steps 要具体到训练器的哪个页面、哪个按钮。拿不准就说拿不准。\n\n"
        f"允许读取的目录：\n{roots}\n（日志里出现过的具体文件路径也可以读）\n\n"
        f"日志来源（get_log 的 source）：\n{sources}\n\n"
        f"修复动作白名单（finish.actions 里的 action / params）：\n{actions_doc()}\n\n"
        "===== 训练器知识 =====\n" + KNOWLEDGE)


def environment(target):
    import platform
    env = {"系统": f"{platform.system()} {platform.release()}"}
    try:
        import updater
        env["训练器版本"] = (updater.version() or {}).get("version")
    except Exception:
        pass
    g = gpu.info()
    env["显卡"] = [f"{x['index']}: {x['name']} {x['total_gb']} GB" for x in g.get("all") or []] or "没有检测到 NVIDIA 显卡"
    env["驱动"] = g.get("driver")
    env["设置里选的显卡"] = settings.gpu_list()
    s = settings.load()
    env["网络"] = {"设置": s.get("network"), "国内": settings.is_cn(), "代理": s.get("proxy_url") or "（没填，用系统代理）"}
    try:
        import envsetup
        st = envsetup.status()
        env["训练环境"] = {k: st.get(k) for k in ("env", "torch", "cuda", "sync_reason")}
    except Exception:
        pass
    try:
        import diagnose as _d
        env["内存 GB"] = _d.ram_gb()
    except Exception:
        pass
    if (target or {}).get("type") == "task":
        env["诊断对象（task_info 可看全部参数）"] = Tools().t_task_info(target.get("id"))
    return redact(json.dumps(env, ensure_ascii=False, indent=1, default=str))


def takeover_text(tk):
    hist = "\n".join(tk.get("history") or []) or "（这是第一轮，还没动过）"
    return (f"\n\n===== 现在是「AI 接管」模式（第 {tk['round']} 轮，最多 {tk['max']} 轮）=====\n"
            "用户把修复交给你了：你在 finish.actions 里给的动作会被直接执行，不再问用户（kill_gpu 除外，它要用户自己确认，别给）。"
            "执行完训练器会把这个任务重新排队训练，等它开始出训练步数（算修好）或者又失败，再把新日志交给你。\n"
            "- 已经没有需要处理的报错：verdict=fixed，actions 留空\n"
            "- 要用户自己动手（更新驱动、关掉别的程序、手动下载要登录的模型……）：verdict=need_user，写清 steps\n"
            "- 否则 verdict=continue，给出这一轮要执行的动作；一次只做有把握的几步，不用给 retry（会自动重新训练）\n"
            "- 前几轮做过但没解决的办法不要原样再来一次，换思路\n"
            "- 用户的说明（如果有）优先级最高\n"
            f"任务当前状态：{tk.get('state') or '未知'}\n之前几轮：\n{hist}")


def followup_text(prev, fix=False):
    out = ""
    if prev:
        steps = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(prev.get("steps") or []))
        acts = "；".join(f"{a.get('aid')} {json.dumps(a.get('params') or {}, ensure_ascii=False)}" for a in prev.get("actions") or [])
        out += (f"\n\n===== 你上一次的诊断 =====\n结论：{prev.get('summary') or ''}\n原因：{prev.get('cause') or ''}\n"
                f"步骤：\n{steps or '  （无）'}\n推荐的动作：{acts or '（无）'}")
    if fix:
        out += ("\n\n===== 用户点了「执行修复」=====\n照你的诊断直接动手：能用白名单动作做的都放进 finish.actions，"
                "用户确认一次就会全部执行。白名单做不到的才写进 steps。")
    return out


def run_agent(target, rep, on_step=lambda t: None, cancelled=lambda: False, post=None, note="", takeover=None,
              followup=None, fix=False):
    conf = aiconf.resolve()
    if not conf:
        raise LlmError("还没有设置 AI：到「训练器设置 → 报错诊断」填 API Key（启动器里填过的会自动同步过来）")
    fmt, base, model, key = conf
    text = log_text(target)
    tools = Tools(text)
    chat = Chat(fmt, base, model, key, system_prompt(), TOOL_SPECS, post=post)
    findings = (rep or {}).get("findings") or []
    rule_txt = "\n".join(f"- {f['title']}：{f.get('text', '')[:200]} 证据：{(f.get('evidence') or '')[:200]}" +
                         ("；规则建议：" + "；".join(f"{a['aid']} {json.dumps(a.get('params') or {}, ensure_ascii=False)[:200]}"
                                                for a in f.get("actions") or []) if f.get("actions") else "")
                         for f in findings) or "（本地规则没认出来）"
    memo = aiconf.get_note(target)
    note = str(note or "").strip()
    if note and note in memo:   # 刚勾「记住」存进长期说明的那句，别重复发
        note = ""
    note = "\n".join(x for x in (memo and "（长期说明）" + memo, str(note or "").strip()[:2000]) if x)
    ask = f"用户的描述 / 说明（优先照这个来）：\n{redact(note)}\n\n" if note else ""
    chat.user(f"{ask}要诊断的对象：{target_name(target)}\n\n环境信息：\n{environment(target)}\n\n本地规则的识别结果：\n{redact(rule_txt)}\n\n"
              f"日志片段（已脱敏）：\n```\n{redact(snippet(text))}\n```" +
              (redact(takeover_text(takeover)) if takeover else "") +
              (redact(followup_text(followup, fix)) if followup or fix else ""))
    on_step(f"正在请求 {model} ...")
    final_text = ""
    for rnd in range(1, MAX_STEPS + 2):
        if cancelled():
            raise LlmError("已取消")
        txt, calls = chat.send()
        final_text = txt or final_text
        if not calls:
            break
        results = []
        for cid, name, args in calls:
            if name == "finish":
                return _final(target, args, rnd)
            a = args or {}
            tgt = a.get("path") or a.get("url") or a.get("source") or a.get("id") or a.get("repo") or ""
            on_step(f"{_STEP_LABELS.get(name, name)} {redact(str(tgt))} {'「' + a['pattern'] + '」' if a.get('pattern') else ''}".strip())
            results.append((cid, tools.call(name, args) if rnd <= MAX_STEPS else "查询次数用完了，请根据已有信息调用 finish 给结论"))
        chat.tool_results(results)
    if final_text.strip():
        return {"summary": final_text.strip()[:300], "cause": final_text.strip(), "steps": [], "actions": [],
                "confidence": "low", "verdict": "need_user", "rounds": MAX_STEPS}
    raise LlmError("AI 没有给出结论")


def _final(target, args, rounds):
    acts, dropped = [], []
    for a in (args.get("actions") or [])[:6]:
        if not isinstance(a, dict):
            continue
        aid = a.get("action") or a.get("aid")
        params, err = validate_action(target, aid, a.get("params") or {})
        if err:
            dropped.append(f"{aid}：{err}")
            continue
        label = str(a.get("label") or ACTIONS[aid]["desc"])[:40]
        acts.append({"aid": aid, "label": label, "params": params, "risk": ACTIONS[aid]["risk"]})
    return {"summary": str(args.get("summary") or ""), "cause": str(args.get("cause") or ""),
            "steps": [str(s) for s in (args.get("steps") or []) if str(s).strip()], "actions": acts, "dropped": dropped,
            "confidence": args.get("confidence") or "medium",
            "verdict": args.get("verdict") if args.get("verdict") in ("fixed", "continue", "need_user") else
            ("continue" if acts else "need_user"), "rounds": rounds}


# ======================= 后台任务（AI 分析 / AI 接管），界面轮询 =======================
_RUNS = {}


class Run:
    def __init__(self, kind, target):
        self.id = uuid.uuid4().hex[:10]
        self.kind, self.target = kind, target
        self.steps, self.tk_steps = [], []
        self.result = self.error = self.tk_done = None
        self.done = False
        self.cancel = False
        self.fix = False
        self.started = time.time()

    def to_dict(self, since=0, tk_since=0):
        return {"id": self.id, "kind": self.kind, "target": self.target, "steps": self.steps[since:], "n": len(self.steps),
                "tk_steps": self.tk_steps[tk_since:], "tk_n": len(self.tk_steps), "done": self.done, "result": self.result,
                "error": self.error, "tk_done": self.tk_done, "fix": self.fix}


def _spawn(run, fn):
    _RUNS[run.id] = run
    for k in [k for k, r in _RUNS.items() if r.done and time.time() - r.started > 3600]:
        _RUNS.pop(k, None)

    def body():
        try:
            fn()
        except LlmError as e:
            run.error = str(e)
        except Exception as e:
            traceback.print_exc()
            run.error = f"出错了：{e}"
        finally:
            run.done = True
    threading.Thread(target=body, daemon=True).start()
    return run


def start_ai(target, note="", followup=None, fix=False, post=None):
    if not aiconf.resolve():
        raise ValueError("还没有设置 AI：到「训练器设置 → 报错诊断」填 API Key（启动器里填过的会自动同步过来）")
    run = Run("ai", target)
    run.fix = bool(fix)

    def go():
        rep = report(target)
        run.result = run_agent(target, rep, on_step=run.steps.append, cancelled=lambda: run.cancel, post=post, note=note,
                               followup=followup, fix=fix)
    return _spawn(run, go)


def start_takeover(target, note="", post=None, wait_poll=3, round_timeout=3 * 3600):
    """AI 接管训练任务：分析 → 直接执行动作（不再逐个确认；结束进程除外）→ 重新排队 → 等结果 → 再分析，最多 5 轮。"""
    if (target or {}).get("type") != "task" or not tasks.get(target.get("id")):
        raise ValueError("AI 接管只能用在训练任务上")
    if not aiconf.resolve():
        raise ValueError("还没有设置 AI：到「训练器设置 → 报错诊断」填 API Key")
    for r in _RUNS.values():
        if r.kind == "takeover" and not r.done and r.target.get("id") == target.get("id"):
            raise ValueError("这个任务已经在 AI 接管中了")
    run = Run("takeover", target)
    tid = target["id"]

    def step(text, kind="info", undo=None):
        run.tk_steps.append({"text": text, "kind": kind, "undo": undo})

    def wait_result():
        """等任务出结果：开始出训练步数 / 练完 = 修好；失败 / 被拦下 = 没修好。"""
        t0 = time.time()
        while time.time() - t0 < round_timeout:
            if run.cancel:
                raise LlmError("已停止接管")
            t = tasks.get(tid) or {}
            st = t.get("status")
            if st == "done" or (st == "running" and ((t.get("progress") or {}).get("step") or 0) >= 1):
                return True, st
            if st in ("failed", "blocked", "stopped"):
                return False, st
            time.sleep(wait_poll)
        return False, "timeout"

    def go():
        hist = []
        for rnd in range(1, TAKEOVER_ROUNDS + 1):
            if run.cancel:
                raise LlmError("已停止接管")
            t = tasks.get(tid) or {}
            step(f"第 {rnd} 轮：AI 分析中…", "round")
            rep = report(target)
            res = run_agent(target, rep, on_step=lambda s: step(s, "ai"), cancelled=lambda: run.cancel, post=post, note=note,
                            takeover={"round": rnd, "max": TAKEOVER_ROUNDS, "history": hist,
                                      "state": f"{t.get('status')}：{t.get('message') or ''}"})
            step("AI：" + (res.get("summary") or ""), "result")
            if res["verdict"] == "fixed":
                run.tk_done = {"ok": True, "text": res.get("summary") or "已经没有报错了", "steps": res.get("steps")}
                return
            if res["verdict"] == "need_user" or not res["actions"]:
                run.tk_done = {"ok": False, "text": res.get("summary") or "需要你手动处理", "steps": res.get("steps")}
                return
            did = []
            for a in res["actions"]:
                if a["risk"] in ("danger", "ui"):
                    step(f"跳过：{a['label']}（要你自己确认）", "warn")
                    continue
                r = apply_action(target, a["aid"], a["params"])
                if r.get("ok"):
                    step(f"{a['label']} → {r.get('text')}", "act", r.get("undo"))
                    did.append(a["label"])
                    if r.get("job"):   # 装东西 / 下载：等作业做完（完成后修复会自动重新排队）
                        j = jobs.get(r["job"])
                        while j and j.status == "running" and not run.cancel:
                            time.sleep(wait_poll)
                        if j and j.status != "done":
                            step(f"{a['label']} 没成功：{j.error or j.status}", "warn")
                else:
                    step(f"{a['label']} 没成功：{r.get('error')}", "warn")
            if not did:
                run.tk_done = {"ok": False, "text": "AI 给的修复都没能执行", "steps": res.get("steps")}
                return
            hist.append(f"第 {rnd} 轮：{res.get('summary')}；执行了：{'、'.join(did)}")
            t = tasks.get(tid) or {}
            if t.get("status") not in ("queued", "running", "checking"):
                tasks.requeue(tid)
            tasks.start_queue()
            step("已重新开始训练，等结果…", "info")
            ok, st = wait_result()
            if ok:
                run.tk_done = {"ok": True, "text": "修好了：训练已经开始正常出步数" if st == "running" else "修好了：训练完成",
                               "steps": []}
                return
            step(f"这一轮之后任务还是{ {'failed': '失败', 'blocked': '被训练前检查拦下', 'stopped': '停止', 'timeout': '没有结果（超时）'}.get(st, st)}",
                 "warn")
        run.tk_done = {"ok": False, "text": f"试了 {TAKEOVER_ROUNDS} 轮还没修好，需要你看看 AI 的分析手动处理", "steps": []}

    def body():
        try:
            go()
        except LlmError as e:
            run.tk_done = {"ok": False, "text": str(e), "steps": []}
            raise
    return _spawn(run, body)


def poll(rid, since=0, tk_since=0):
    r = _RUNS.get(rid)
    if not r:
        return {"done": True, "error": "这次分析已经过期（训练器重启过？）", "steps": [], "n": 0, "tk_steps": [], "tk_n": 0}
    return r.to_dict(int(since or 0), int(tk_since or 0))


def cancel(rid):
    r = _RUNS.get(rid)
    if r:
        r.cancel = True
    return True


def active_takeover(target):
    for r in _RUNS.values():
        if r.kind == "takeover" and (r.target or {}).get("id") == (target or {}).get("id") and \
                (not r.done or time.time() - r.started < 3600):
            return r.id
    return ""

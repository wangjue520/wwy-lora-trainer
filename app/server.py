# -*- coding: utf-8 -*-
"""本地 HTTP 服务：提供界面（web/）和 JSON 接口（POST /api/<名字>）。只用标准库。"""
import json
import mimetypes
import os
import re
import string
import subprocess
import threading
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import captionutil as cu
import dataset
import diagnose
import envsetup
import fixes
import gpu
import jobs
import models
import paths
import scanner
import settings
import tagging
import tasks
import thumbs
import updater
from models.base import dataset_subsets, estimate_steps

API = {}


def api(name):
    def deco(fn):
        API[name] = fn
        return fn
    return deco


# ======================= 接口 =======================
@api("boot")
def _boot(_):
    return {"version": updater.version(), "settings": settings.load(), "specs": [s.to_dict() for s in models.all_specs()],
            "draft": settings.load_draft(), "gpu": gpu.info(), "env": envsetup.status(), "scan": scanner.scan_state(),
            "queue": tasks.queue_state(), "presets": settings.load_presets(),
            "tagger": {"vlm": cu.VLM_MODELS, "wd14": cu.WD14_MODELS, "prompts": cu.PROMPT_PRESETS,
                       "api_presets": tagging.API_PRESETS, "last_dir": tagging.last_dataset(),
                       "launcher_wd14": scanner.launcher_wd14_models(), "launcher_wd14_dir": scanner.launcher_wd14_root()},
            "network": "cn" if settings.is_cn() else "global"}


@api("settings.save")
def _settings_save(a):
    return settings.save(a.get("patch") or {})


@api("draft.save")
def _draft_save(a):
    settings.save_draft(a.get("draft") or {})
    return True


@api("presets.save")
def _presets_save(a):
    p = settings.load_presets()
    p[a["name"]] = a["draft"]
    settings.save_presets(p)
    return p


@api("presets.delete")
def _presets_delete(a):
    p = settings.load_presets()
    p.pop(a["name"], None)
    settings.save_presets(p)
    return p


@api("gpu")
def _gpu(a):
    g = gpu.info(max_age=1)
    if a.get("procs"):
        g = dict(g, procs=gpu.processes(exclude_pids=tasks.running_pids()))
    return g


@api("gpu.kill")
def _gpu_kill(a):
    return gpu.kill([int(p) for p in a.get("pids") or [] if int(p) not in tasks.running_pids()])


@api("net.detect")
def _net_detect(_):
    return settings.detect_network()


@api("net.test")
def _net_test(_):
    """测速所有下载线路（设置页「测速」按钮）。"""
    import netpick
    netpick._cache.clear()
    row = lambda xs: [{"name": x["name"], "speed": x["speed"] / 1e6} for x in xs]
    pip = netpick.rank(
        [(f"{n}（{rn}）", u.rstrip("/") + "/pip/", p) for n, u in netpick.PIP for rn, p in netpick.routes()],
        prefer="清华" if settings.is_cn() else "官方")
    return {"routes": [n for n, _ in netpick.routes()],
            "hf": row(netpick.rank_hf("openai-community/gpt2", "config.json")),
            "github": row(netpick.rank_github("https://github.com/wangjue520/danbooru-zh-dict/raw/main/README.md")),
            "pip": row(pip)}


# ---- 模型搜索 ----
@api("scan.start")
def _scan_start(_):
    scanner.start_scan_async()
    return scanner.scan_state()


@api("scan.state")
def _scan_state(_):
    return scanner.scan_state()


@api("models.list")
def _models_list(a):
    spec = models.get(a.get("model_type"))
    if not spec:
        raise ValueError("未知底模类型")
    items = scanner.list_models(force=bool(a.get("force")))
    res = {}
    for s in spec.slots:
        cands = [dict(x, **scanner.sidecar(x["path"])) for x in items if not x.get("error") and spec.accept(s, x)]
        cands = [dict(x, score=spec.score(s.key, x), variant=spec.variant(s.key, x)) for x in cands]   # 分组要用 .civitai.info 的底模
        cands.sort(key=lambda x: (-x["score"], -x["size"]))
        res[s.key] = cands
    broken = [x for x in items if x["kind"] == "broken"]
    return {"slots": res, "broken": broken, "dirs": [{"dir": d, "label": l} for d, l in scanner.model_dirs()]}


@api("models.check")
def _models_check(a):
    import safetensors_meta as stm
    spec = models.get(a.get("model_type"))
    p = a.get("path") or ""
    if not os.path.isfile(p):
        return {"ok": False, "message": "文件不存在"}
    info = stm.classify(p)
    slot = next((s for s in spec.slots if s.key == a.get("slot")), None) if spec else None
    if info.get("error"):
        return {"ok": False, "message": info["error"], "info": info}
    if slot and info["kind"] not in slot.kinds:
        return {"ok": False, "message": f"这个文件不是{slot.label}（识别为：{info['kind']}）", "info": info}
    if slot and not spec.accept(slot, dict(info, name=os.path.basename(p))):
        return {"ok": False, "message": f"这个{slot.label}是量化版（{info.get('qtype') or '低精度'}），不能用来训练，请选 bf16 / fp16 原版", "info": info}
    return {"ok": True, "info": info}


# ---- 训练集 ----
@api("dataset.stats")
def _dataset_stats(a):
    p = dict(a.get("params") or {}, _num_gpus=len(settings.gpu_list()))
    subs = dataset_subsets(p.get("dataset_dir"), p.get("repeats") or 1, p.get("use_folder_repeats", True))
    return {"subsets": subs, "estimate": estimate_steps(p, subs), "num_gpus": p["_num_gpus"]}


@api("dataset.check")
def _dataset_check(a):
    issues, stats = diagnose.check_dataset(a.get("dir"), a.get("ext") or ".txt")
    return {"issues": issues, "stats": stats}


@api("dataset.clean")
def _dataset_clean(a):
    return fixes.clean_dataset(a.get("dir"), a.get("files"))


# ---- 文件选择 ----
@api("fs.list")
def _fs_list(a):
    p = a.get("path") or ""
    exts = tuple(a.get("exts") or ())
    if not p or not os.path.isdir(p):
        if os.name == "nt":
            return {"path": "", "parent": "", "dirs": [{"name": f"{d}:\\", "path": f"{d}:\\"} for d in string.ascii_uppercase
                                                       if os.path.isdir(f"{d}:\\")], "files": []}
        p = "/"
    dirs, files = [], []
    try:
        for e in sorted(os.scandir(p), key=lambda e: e.name.lower()):
            try:
                if e.is_dir():
                    if not e.name.startswith(("$", ".")):
                        dirs.append({"name": e.name, "path": e.path})
                elif exts and e.name.lower().endswith(exts):
                    files.append({"name": e.name, "path": e.path, "size": e.stat().st_size})
            except OSError:
                pass
    except OSError as e:
        raise ValueError(f"打不开这个文件夹：{e}")
    parent = os.path.dirname(p.rstrip("\\/")) if os.path.dirname(p.rstrip("\\/")) != p.rstrip("\\/") else ""
    if os.name == "nt" and len(p.rstrip("\\/")) <= 2:
        parent = ""
    imgs = sum(1 for f in os.listdir(p) if f.lower().endswith(cu.IMAGE_EXTS)) if os.path.isdir(p) else 0
    return {"path": p, "parent": parent, "dirs": dirs, "files": files, "images": imgs}


@api("fs.open")
def _fs_open(a):
    p = a.get("path")
    if p and os.path.exists(p):
        if os.path.isfile(p):
            subprocess.Popen(["explorer", "/select,", os.path.normpath(p)])
        else:
            os.startfile(p) if os.name == "nt" else subprocess.Popen(["xdg-open", p])
    return True


@api("fs.mkdir")
def _fs_mkdir(a):
    os.makedirs(a["path"], exist_ok=True)
    return True


# ---- 队列 ----
@api("queue.list")
def _queue_list(_):
    return {"tasks": tasks.listing(), "state": tasks.queue_state()}


@api("queue.add")
def _queue_add(a):
    t = tasks.add(a["draft"], a.get("name"), start=bool(a.get("start")))
    return t


@api("queue.remove")
def _queue_remove(a):
    return tasks.remove(a["id"])


@api("queue.move")
def _queue_move(a):
    tasks.move(a["id"], int(a.get("delta") or 0))
    return True


@api("queue.requeue")
def _queue_requeue(a):
    t = tasks.requeue(a["id"])
    if a.get("start"):
        tasks.start_queue()
    return t


@api("queue.update")
def _queue_update(a):
    return tasks.update_task(a["id"], a.get("patch") or {})


@api("queue.start")
def _queue_start(_):
    tasks.start_queue()
    return tasks.queue_state()


@api("queue.pause")
def _queue_pause(_):
    tasks.pause_queue()
    return tasks.queue_state()


@api("queue.freeze")
def _queue_freeze(_):
    return tasks.freeze_current()


@api("queue.resume")
def _queue_resume(a):
    return tasks.resume(a["id"])


@api("queue.stop")
def _queue_stop(_):
    tasks.stop_current()
    return True


@api("queue.clear")
def _queue_clear(_):
    tasks.clear_finished()
    return True


@api("queue.log")
def _queue_log(a):
    return tasks.read_log(a["id"], int(a.get("tail") or 400))


@api("queue.task")
def _queue_task(a):
    t = tasks.get(a["id"])
    if t:
        t["outputs"] = tasks.list_outputs(t)
    return t


@api("queue.metrics")
def _queue_metrics(a):
    """训练曲线：读任务目录 logs 里的 TensorBoard 日志；没有日志（旧任务）时退回进度条里解析的平均 loss。"""
    import tbreader
    t = tasks.get(a["id"])
    if not t:
        return None
    td = os.path.join(paths.TASKS, t["id"])
    d = tbreader.read_scalars(os.path.join(td, "logs"), t.get("tb_runs"))
    if not d and (t.get("progress") or {}).get("loss_hist"):
        d = {"loss/average": t["progress"]["loss_hist"]}
    # Automagic / EmoSens 的实际学习率（引擎日志里的 lr 只是调度器起始值）
    actual = []
    try:
        with open(os.path.join(td, "actual_lr.tsv"), encoding="utf-8") as f:
            for line in f:
                a_, b_ = line.split("	")
                actual.append([int(a_), float(b_)])
    except (OSError, ValueError):
        pass
    if actual:
        d["lr/actual"] = actual
    # 每轮学习率：每一轮结束那一步的学习率（优先用实际学习率）
    src = actual or next((v for k, v in sorted(d.items(), key=lambda kv: len(kv[0])) if k.startswith("lr/") and k != "lr/actual"), None)
    prog = t.get("progress") or {}
    try:
        n_ep = int(str(prog.get("epoch") or "").split("/")[1])
    except (IndexError, ValueError):
        n_ep = len(d.get("loss/epoch_average") or d.get("loss/epoch") or []) or int(t["params"].get("epochs") or 0)
    total = int(prog.get("total") or 0)
    if src and n_ep and total:
        spe = total / n_ep
        per_epoch, j = [], 0
        for e in range(1, n_ep + 1):
            end = round(e * spe)
            if src[-1][0] < end - 1:
                break
            while j + 1 < len(src) and src[j + 1][0] <= end:
                j += 1
            per_epoch.append([e, src[j][1]])
        if per_epoch:
            d["lr/epoch"] = per_epoch
    return {k: tbreader.downsample(v) for k, v in d.items()}


@api("fix.apply")
def _fix_apply(a):
    return fixes.apply(a.get("task"), a["fix"], a.get("data") or {})


# ---- 后台作业 ----
@api("jobs.list")
def _jobs_list(_):
    return jobs.listing()


@api("jobs.get")
def _jobs_get(a):
    j = jobs.get(a["id"])
    return j.to_dict(log_lines=int(a.get("log") or 300)) if j else None


@api("jobs.stop")
def _jobs_stop(a):
    return jobs.stop(a["id"])


@api("download.start")
def _download_start(a):
    import downloader

    def after(job):
        if job.status == "done" and a.get("slot"):
            settings.set_draft_model(a.get("model_type"), a["slot"], job.result)

    j = jobs.start("download", f"下载 {a.get('label') or a['file']}", downloader.download, a["repo"], a["file"], on_done=after)
    return j.id


# ---- 反推 ----
@api("tag.start")
def _tag_start(a):
    return tagging.start(a["cfg"]).id


@api("tag.test_api")
def _tag_test(a):
    return tagging.test_api(a["api"])


# ---- 图片爬取（Danbooru） ----
@api("crawl.start")
def _crawl_start(a):
    import crawler
    if jobs.find_running("crawl"):
        raise ValueError("已经有一个爬取任务在运行")
    cfg = a["cfg"]
    s = settings.load()
    cfg.setdefault("login", s.get("danbooru_login", ""))
    cfg.setdefault("api_key", s.get("danbooru_api_key", ""))

    def work(job):
        try:
            return crawler.run(job, cfg)
        except crawler.CrawlError as e:
            raise RuntimeError(str(e))
    return jobs.start("crawl", f"Danbooru 爬取 · {cfg.get('tags', '')[:30]}", work).id


@api("crawl.account")
def _crawl_account(a):
    import crawler

    class _J:
        stop_flag = False

        def write(self, *_):
            pass
    try:
        c = crawler.Client(_J(), a.get("login", ""), a.get("api_key", ""))
        level, limit = crawler.account_limit(c, a.get("login"))
        return {"ok": True, "level": level, "limit": limit}
    except crawler.CrawlError as e:
        return {"ok": False, "message": str(e)}


@api("crawl.autocomplete")
def _crawl_autocomplete(a):
    import crawler
    s = settings.load()
    try:
        return crawler.autocomplete(a.get("q", ""), s.get("danbooru_login", ""), s.get("danbooru_api_key", ""))
    except Exception:
        return []


# ---- Danbooru 中英字典 ----
@api("dict.status")
def _dict_status(_):
    import tagdict
    threading.Thread(target=tagdict.load, daemon=True).start()   # 后台读入，不卡界面
    return tagdict.status()


@api("dict.download")
def _dict_download(_):
    import tagdict
    if jobs.find_running("dict"):
        raise ValueError("字典正在下载")
    return jobs.start("dict", "下载 Danbooru 中英字典", tagdict.download).id


@api("dict.search")
def _dict_search(a):
    import tagdict
    return tagdict.search(a.get("q", ""), int(a.get("limit") or 12))


@api("dict.zh")
def _dict_zh(a):
    """标签编辑里的翻译：D 站标签 → 中文（空格 / 下划线写法、别名都认）。字典还没下载时返回空。"""
    import tagdict
    if not tagdict.load():
        return {}
    out = {}
    for t in a.get("tags") or []:
        for c in (t, t.replace(" ", "_"), t.replace("_", " ")):
            name = tagdict.resolve(c.strip().lower())
            row = tagdict.info([name]).get(name) if name else None
            if row and row.get("zh"):
                out[t] = row["zh"]
                break
    return out


@api("translate.get")
def _translate_get(a):
    import translate
    return translate.get(a.get("texts") or [])


@api("translate.start")
def _translate_start(a):
    import translate
    return translate.start(a.get("texts") or [], a.get("engine") or "local", a.get("model") or "qwen3vl4b",
                           a.get("api"), a.get("quant") or "auto").id


def _suggest(tagdict, t):
    """拼错的标签找最接近的正式写法（hatsunemiku_xyz → hatsune_miku）。"""
    return [x["name"] for x in tagdict.search(t.replace("_", " "), 4)] or tagdict.fuzzy(t, 4)


@api("dict.validate")
def _dict_validate(a):
    """检查一批输入是不是 D 站标签：字典里有的直接通过（别名换成正式名）；字典没有的去 Danbooru 在线确认。"""
    import tagdict
    out, unknown = [], []
    for raw in a.get("tags") or []:
        t = raw.strip().lower().replace(" ", "_").lstrip("-~")
        if not t:
            continue
        if "*" in t or "?" in t:
            out.append({"input": raw, "tag": t, "ok": True, "wild": True})
            continue
        r = tagdict.resolve(t)
        if r:
            info = tagdict.info([r]).get(r, {})
            out.append({"input": raw, "tag": r, "ok": True, "zh": info.get("zh", ""), "count": info.get("count", 0),
                        "category": info.get("category"), "renamed": r != t})
        else:
            out.append({"input": raw, "tag": t, "ok": False})
            unknown.append(t)
    if unknown:   # 字典只收录图数 ≥50 的标签，冷门标签在线确认
        import crawler

        class _J:
            stop_flag = False

            def write(self, *_):
                pass
        s = settings.load()
        try:
            c = crawler.Client(_J(), s.get("danbooru_login", ""), s.get("danbooru_api_key", ""))
            d = c.get("tags.json", {"search[name_comma]": ",".join(unknown), "only": "name,post_count,category", "limit": 100}, retries=2)
            found = {x["name"]: x for x in d} if isinstance(d, list) else {}
            offline = False
        except Exception:
            found, offline = {}, True
        for o in out:
            f = found.get(o["tag"])
            if not o["ok"] and offline:   # 连不上 Danbooru：先放行，开爬前还会再查一次
                o.update(ok=True, unverified=True)
            elif not o["ok"] and f and f.get("post_count"):
                o.update(ok=True, count=f["post_count"], category=f.get("category"), zh="", rare=True)
            elif not o["ok"]:
                o["suggest"] = _suggest(tagdict, o["tag"])
    return out


# ---- 触发词 ----
@api("trigger.analyze")
def _trigger_analyze(a):
    import triggers
    return triggers.analyze(a["dir"], a.get("ext") or ".txt", a.get("mode") or "freq", a.get("typical"), a.get("main"), a.get("hints"))


@api("trigger.apply")
def _trigger_apply(a):
    import triggers
    return triggers.apply(a["dir"], a.get("main"), a.get("features"), a.get("outfit"), a.get("ext") or ".txt",
                          a.get("prune"), a.get("mode") or "freq", a.get("typical"))


@api("trigger.sync_prompts")
def _trigger_sync_prompts(a):
    import triggers
    prompts, n = triggers.sync_prompts(a.get("dir"), a.get("prompts") or [], bool(a.get("only_with_main")))
    info = triggers.get(a.get("dir")) or {}
    return {"prompts": prompts, "changed": n, "prompt": info.get("prompt") or ""}


@api("sample.defaults")
def _sample_defaults(a):
    spec = models.get(a.get("model_type"))
    if not spec:
        raise ValueError("未知底模类型")
    return spec.sample_defaults_for(a.get("models") or {})


@api("trigger.get")
def _trigger_get(a):
    import triggers
    return triggers.get(a.get("dir"))


# ---- 标签编辑 ----
@api("editor.load")
def _editor_load(a):
    d = dataset.load(a["dir"], a.get("ext") or ".txt")
    thumbs.prepare([it["path"] for it in d["items"]])   # 后台先把缩略图都生成好
    return d


@api("editor.save")
def _editor_save(a):
    return dataset.save_one(a["dir"], a["path"], a.get("text") or "", a.get("ext") or ".txt")


@api("editor.batch")
def _editor_batch(a):
    kw = dict(a.get("args") or {})
    return dataset.batch(a["dir"], a["op"], a.get("files") or None, a.get("ext") or ".txt", **kw)


@api("editor.undo")
def _editor_undo(a):
    return dataset.undo(a["dir"], a.get("ext") or ".txt")


@api("editor.remove")
def _editor_remove(a):
    return dataset.remove_image(a["dir"], a["path"], a.get("ext") or ".txt")


# ---- 训练中测试 ----
@api("launcher.send_info")
def _launcher_send_info(a):
    import launcher_send
    t = tasks.get(a["id"])
    if not t:
        return None
    outs = tasks.list_outputs(t)
    path = a.get("path") or (outs[-1]["path"] if outs else "")
    if path and not os.path.isfile(path):   # 最后一轮保存的是不带轮数后缀的最终文件
        fin = os.path.join(t["output_dir"], (t["params"].get("output_name") or "lora") + ".safetensors")
        path = fin if os.path.isfile(fin) else ""
    final = [o for o in outs if not re.search(r"-\d{6}\.safetensors$", o["name"])]
    if not a.get("path") and final:
        path = final[0]["path"]
    d = launcher_send.info(a["id"], path) if path else {"targets": launcher_send.targets()}
    d.update(outputs=outs, path=path)
    return d


@api("launcher.send")
def _launcher_send(a):
    import launcher_send
    return launcher_send.send(a["id"], a["path"], a.get("targets") or [], a.get("type") or "", a.get("trigger") or "",
                              a.get("preview"), bool(a.get("overwrite")))


@api("samples.list")
def _samples_list(a):
    t = tasks.get(a["id"])
    if not t:
        return []
    return tasks.list_samples(t)


@api("samples.clear")
def _samples_clear(a):
    t = tasks.get(a["id"])
    if not t:
        raise ValueError("找不到这个任务")
    if t["status"] in ("running", "checking"):
        raise ValueError("训练中不能清空测试图")
    return tasks.clear_samples(t)


@api("samples.update_prompts")
def _samples_update_prompts(a):
    """训练中途改测试提示词（下一次出测试图起生效）。"""
    return tasks.update_sample(a["id"], a.get("prompts") or [])


# ---- 环境 / 更新 ----
@api("env.status")
def _env_status(_):
    return envsetup.status()


@api("env.install")
def _env_install(a):
    if jobs.find_running("env"):
        raise ValueError("已经有安装任务在进行")
    mode = a.get("mode") or "install"
    if mode == "torch":
        j = jobs.start("env", "重装 PyTorch", lambda job: envsetup.install_torch(job, force=True))
    elif mode == "update":
        j = jobs.start("env", "一键更新", updater.update_all)
    elif mode == "self":
        j = jobs.start("env", "只更新训练器", updater.update_trainer_only)
    else:
        j = jobs.start("env", "修复训练环境" if mode == "repair" else "安装训练环境", envsetup.install, repair=mode == "repair")
    return j.id


# ======================= HTTP =======================
class Handler(BaseHTTPRequestHandler):
    server_version = "LoraTrainer/1.0"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)) or body is None or isinstance(body, (bool, int, float, str)) and ctype.startswith("application/json"):
            body = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if "Cache-Control" not in (extra or {}):
            self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/file":
            return self._file(urllib.parse.parse_qs(u.query).get("path", [""])[0])
        if u.path == "/thumb":
            src = urllib.parse.parse_qs(u.query).get("path", [""])[0]
            if src.lower().endswith(cu.IMAGE_EXTS) and os.path.isfile(src):
                t = thumbs.get(src)
                if t:
                    with open(t, "rb") as f:
                        return self._send(200, f.read(), "image/jpeg", {"Cache-Control": "max-age=86400"})
            return self._file(src)
        rel = u.path.lstrip("/") or "index.html"
        p = os.path.normpath(os.path.join(paths.WEB, rel))
        if not p.startswith(os.path.normpath(paths.WEB)) or not os.path.isfile(p):
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(p)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        with open(p, "rb") as f:
            self._send(200, f.read(), ctype)

    def _file(self, p):
        # 只给图片：界面里显示训练集和测试图用
        if not p.lower().endswith(cu.IMAGE_EXTS) or not os.path.isfile(p):
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(p)[0] or "image/png"
        with open(p, "rb") as f:
            self._send(200, f.read(), ctype, {"Cache-Control": "max-age=60"})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if not u.path.startswith("/api/"):
            return self._send(404, {"ok": False, "error": "not found"})
        name = u.path[5:]
        fn = API.get(name)
        if not fn:
            return self._send(404, {"ok": False, "error": f"没有这个接口：{name}"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            args = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
            res = fn(args)
            self._send(200, {"ok": True, "data": res})
        except Exception as e:
            if not isinstance(e, ValueError):
                traceback.print_exc()
            self._send(200, {"ok": False, "error": str(e) or e.__class__.__name__})


def serve(host, port):
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd

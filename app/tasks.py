# -*- coding: utf-8 -*-
"""训练任务队列：多个任务排队，按顺序一个接一个训练。

任务状态：queued 排队中 / checking 体检中 / blocked 等用户确认 / running 训练中 /
         paused 已冻结（训练状态已保存、显存已释放，可继续）/ done 完成 / failed 失败 / stopped 已停止
队列状态（data/queue.json）重启训练器后保留；正在训练的任务在重启后标为「已中断」。"""
import copy
import json
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid

import diagnose
import gpu
import models
import paths
import settings
from jobs import kill_tree, decode_line, NO_WINDOW

QUEUE_FILE = os.path.join(paths.DATA, "queue.json")

_lock = threading.RLock()
_tasks = []
_state = {"running": False, "paused": False, "current": None, "proc": None, "stop_req": False, "freeze_req": False}
_cond = threading.Condition(_lock)

RE_STEPS = re.compile(r"(?<!Denoising )(?<!Sampling )steps:\s*(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:?]+)")
RE_SAMPLING = re.compile(r"(?:Denoising|Sampling) steps:\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)")   # 训练中出测试图的进度条
RE_LOSS = re.compile(r"avr_loss=([\d.eE+-]+|nan)")
RE_EPOCH = re.compile(r"\bepoch (\d+)/(\d+)", re.I)


def _hms(s):
    """进度条里的时间「MM:SS」或「H:MM:SS」→ 秒。"""
    try:
        parts = [int(x) for x in s.split(":")]
    except ValueError:
        return None
    sec = 0
    for x in parts:
        sec = sec * 60 + x
    return sec
ANSI = re.compile(r"\[[0-9;?]*[A-Za-z]")
RE_CACHE = re.compile(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)")
PAUSE_STATE = "paused_state"   # 冻结时训练状态存在任务目录的这个子文件夹


def _save():
    with _lock:
        data = [{k: v for k, v in t.items() if not k.startswith("_")} for t in _tasks]
    tmp = QUEUE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, QUEUE_FILE)


def load():
    global _tasks
    try:
        _tasks = json.load(open(QUEUE_FILE, encoding="utf-8"))
    except Exception:
        _tasks = []
    for t in _tasks:
        if t.get("status") in ("running", "checking"):
            t["status"] = "stopped"
            t["message"] = "训练器关闭时任务被中断"
    _save()


def _find(tid):
    for t in _tasks:
        if t["id"] == tid:
            return t
    return None


def task_dir(t):
    d = os.path.join(paths.TASKS, t["id"])
    os.makedirs(d, exist_ok=True)
    return d


def log_path(t):
    return os.path.join(task_dir(t), "train.log")


def listing():
    with _lock:
        out = [{k: v for k, v in t.items() if not k.startswith("_")} for t in _tasks]
    for t in out:   # 停止 / 失败 / 冻结的任务能不能一键续训
        spec = models.get(t["model_type"])
        if spec and spec.resume and t["status"] in ("paused", "stopped", "failed"):
            r = resume_info(t)
            t["can_resume"] = r["label"] if r else ""
    return out


def get(tid):
    with _lock:
        t = _find(tid)
        return copy.deepcopy({k: v for k, v in t.items() if not k.startswith("_")}) if t else None


def read_log(tid, tail=400):
    t = _find(tid)
    if not t:
        return []
    p = log_path(t)
    if not os.path.isfile(p):
        return []
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return [x.rstrip("\n") for x in lines[-tail:]]


# ---------------- 队列操作 ----------------
def add(draft, name=None, start=False):
    spec = models.get(draft.get("model_type"))
    if not spec:
        raise ValueError("请先在「模型设置」选择底模类型")
    p = dict(spec.defaults())
    p.update(draft.get("params") or {})
    out_dir = (draft.get("output_dir") or settings.load()["output_dir"]).strip()
    t = {
        "id": time.strftime("%m%d%H%M%S") + "_" + uuid.uuid4().hex[:4],
        "name": name or p.get("output_name") or "lora",
        "model_type": spec.id,
        "models": dict(draft.get("models") or {}),
        "params": p,
        "sample": copy.deepcopy(draft.get("sample") or {}),
        "output_dir": os.path.join(out_dir, p.get("output_name") or "lora") if draft.get("subdir_per_task", True) else out_dir,
        "status": "queued",
        "gpus": settings.gpu_list(),   # 加入队列时用哪几张卡（之后改设置不影响这个任务）
        "created": time.time(),
        "progress": None,
        "message": "",
    }
    with _lock:
        _tasks.append(t)
        _save()
        _cond.notify_all()
    if start:
        start_queue()
    return t


def remove(tid):
    with _lock:
        t = _find(tid)
        if not t:
            return False
        if t["status"] in ("running", "checking"):
            raise ValueError("正在训练的任务要先停止")
        _tasks.remove(t)
        _save()
    shutil.rmtree(os.path.join(paths.TASKS, tid), ignore_errors=True)
    return True


def move(tid, delta):
    with _lock:
        t = _find(tid)
        if not t:
            return
        i = _tasks.index(t)
        j = max(0, min(len(_tasks) - 1, i + delta))
        _tasks.insert(j, _tasks.pop(i))
        _save()


def requeue(tid, patch_params=None, patch_models=None):
    with _lock:
        t = _find(tid)
        if not t or t["status"] in ("running", "checking"):
            return None
        if patch_params:
            t["params"].update(patch_params)
        if patch_models:
            t["models"].update(patch_models)
        t.update(status="queued", message="", error=None, issues=None, progress=None, resume=None, tb_runs=[])
        _save()
        _cond.notify_all()
    return t


def update_task(tid, patch):
    with _lock:
        t = _find(tid)
        if not t or t["status"] in ("running", "checking"):
            return None
        for k in ("name", "params", "models", "sample", "output_dir"):
            if k in patch:
                t[k] = patch[k]
        _save()
        return t


def update_sample(tid, prompts):
    """训练中途改测试提示词：重写任务目录里的提示词文件，训练进程下一次出测试图时读到
    （新写的提示词由 app/vendor/sitecustomize.py 临时加载文本编码器编码）。"""
    with _lock:
        t = _find(tid)
        if not t:
            raise ValueError("找不到这个任务")
        if not [x for x in prompts if (x.get("prompt") or "").strip()]:
            raise ValueError("至少要有一条提示词")
        t["sample"] = dict(t.get("sample") or {}, prompts=copy.deepcopy(prompts))
        if t["status"] in ("running", "checking"):
            spec = models.get(t["model_type"])
            if spec:
                spec.write_sample_prompts(t, task_dir(t))
        _save()
        return t


def clear_samples(t):
    """删掉这个任务保存位置里的测试图（只删训练中测试图格式的文件名，别的文件不动）。返回删掉的张数。"""
    n = 0
    for x in list_samples(t):
        try:
            os.remove(x["file"])
            n += 1
        except OSError:
            pass
    return n


def clear_finished():
    with _lock:
        keep = [t for t in _tasks if t["status"] not in ("done",)]
        _tasks[:] = keep
        _save()


def start_queue():
    with _lock:
        _state["paused"] = False
        if not _state["running"]:
            _state["running"] = True
            threading.Thread(target=_worker, daemon=True).start()
        _cond.notify_all()


def pause_queue():
    """当前任务训完后不再开始下一个。"""
    _state["paused"] = True


def stop_current():
    with _lock:
        _state["stop_req"] = True
        proc = _state.get("proc")
    if proc:
        kill_tree(proc.pid)


# ---------------- 冻结 / 续训 ----------------
def freeze_current():
    """冻结正在训练的任务：训练进程在下一步开始前保存完整训练状态后退出，显存全部释放；队列也停下。
    还没进入训练（体检 / 缓存）时直接停掉，继续时从上次的进度（或从头）开始。"""
    with _lock:
        t = _find(_state.get("current") or "")
        if not t or t["status"] not in ("running", "checking"):
            raise ValueError("现在没有正在训练的任务")
        spec = models.get(t["model_type"])
        if not spec or not spec.resume:
            raise ValueError(f"{spec.label if spec else t['model_type']} 暂不支持冻结（训练引擎还不能从中间接着练）")
        _state["freeze_req"] = True
        _state["paused"] = True     # 冻结后不开始下一个任务，把显卡让出来
        training = (t["status"] == "running" and (t.get("progress") or {}).get("step") is not None
                    and _state.get("step_kind") == "train")
        t["message"] = "正在保存训练进度，这一步跑完就停…" if training else "正在停止…"
        proc = _state.get("proc")
    if training:
        open(os.path.join(task_dir(t), "pause.flag"), "w").close()
    elif proc:
        kill_tree(proc.pid)
    return True


def _spe(t):
    """每轮多少步（第一次训练时从进度条记下来）。"""
    p = t.get("progress") or {}
    try:   # 旧任务没记 steps_total / epochs_total 时用最后一次的进度条
        total = int(t.get("steps_total") or p.get("total"))
        ep = int(t.get("epochs_total") or str(p.get("epoch") or "").split("/")[1])
        return total / ep
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return 0


def resume_info(t):
    """这个任务能从哪里接着练：冻结保存的状态 > 每轮保存的训练状态 > 每轮保存的 LoRA。都没有返回 None。"""
    spe = _spe(t)
    try:
        total_ep = int(t.get("epochs_total") or str((t.get("progress") or {}).get("epoch") or "").split("/")[1])
    except (ValueError, IndexError):
        total_ep = 0
    if not spe:
        return None
    st = os.path.join(paths.TASKS, t["id"], PAUSE_STATE)
    frozen = None   # 冻结时保存的状态（冻结后续训失败 / 被停止也还能用）
    if os.path.isfile(os.path.join(st, "train_state.json")):
        try:
            with open(os.path.join(st, "train_state.json"), encoding="utf-8") as f:
                step = int(json.load(f)["current_step"])
            frozen = {"mode": "state", "state": st, "base": step, "epoch": int(step // spe), "label": f"从冻结时的第 {step} 步接着练"}
        except (OSError, ValueError, KeyError):
            pass
    if frozen and t.get("status") == "paused":
        return frozen
    name = t["params"].get("output_name") or "lora"
    best = None
    d = t.get("output_dir") or ""
    if os.path.isdir(d):
        for fn in os.listdir(d):
            m = re.fullmatch(re.escape(name) + r"-(\d{6})(-state|\.safetensors)", fn)
            if not m:
                continue
            ep = int(m.group(1))
            if ep >= total_ep:
                continue
            is_state = m.group(2) == "-state"
            if is_state and not os.path.isfile(os.path.join(d, fn, "train_state.json")):
                continue
            key = (ep, is_state)
            if best is None or key > best[0]:
                best = (key, os.path.join(d, fn))
    if not best:
        return frozen
    (ep, is_state), path = best
    base = int(round(ep * spe))
    if frozen and frozen["base"] >= base:   # 冻结的进度比每轮保存的更新
        return frozen
    if is_state:
        return {"mode": "state", "state": path, "base": base, "epoch": ep, "label": f"从第 {ep} 轮保存的训练状态接着练"}
    return {"mode": "weights", "weights": path, "base": base, "epoch": ep,
            "label": f"从第 {ep} 轮的 LoRA 接着练（没有保存训练状态，优化器会重新开始适应）"}


def resume(tid):
    """一键续训：放到队列最前面并开始。"""
    with _lock:
        t = _find(tid)
        if not t or t["status"] in ("running", "checking"):
            raise ValueError("这个任务正在训练")
        spec = models.get(t["model_type"])
        if not spec or not spec.resume:
            raise ValueError(f"{spec.label if spec else t['model_type']} 暂不支持续训")
        r = resume_info(t)
        if not r:
            raise ValueError("这个任务还没有保存过可以接着练的进度（训练状态或某一轮的 LoRA），只能重新训练")
        t.update(status="queued", message=r["label"], error=None, issues=None, resume=r)
        _tasks.remove(t)
        _tasks.insert(0, t)
        _save()
    start_queue()
    return t


def queue_state():
    return {"running": _state["running"], "paused": _state["paused"], "current": _state["current"]}


def running_pids():
    p = _state.get("proc")
    return [p.pid] if p else []


# ---------------- 执行 ----------------
def _next_task():
    for t in _tasks:
        if t["status"] == "queued":
            return t
    return None


def _worker():
    try:
        while True:
            import jobs
            with _lock:
                t = None if _state["paused"] else _next_task()
            if t and jobs.find_running("env"):   # 训练环境正在安装 / 更新 / 同步：等它做完再开始训练
                t["message"] = "等训练环境同步完成…"
                time.sleep(3)
                continue
            with _lock:
                if _state["paused"]:
                    break
                t = _next_task()
                if not t:
                    break
                _state["current"] = t["id"]
                _state["stop_req"] = False
            _run_task(t)
            with _lock:
                _state["current"] = None
                _state["proc"] = None
        if settings.load().get("shutdown_after_queue") and not _state["paused"] and any(x["status"] == "done" for x in _tasks):
            if not any(x["status"] == "queued" for x in _tasks):
                subprocess.run(["shutdown", "/s", "/t", "120"], creationflags=NO_WINDOW)
    finally:
        _state["running"] = False


def _set(t, **kw):
    with _lock:
        t.update(kw)
        _save()


def _run_task(t):
    spec = models.get(t["model_type"])
    td = task_dir(t)
    logf = open(log_path(t), "a", encoding="utf-8")

    def log(s):
        logf.write(s.rstrip("\n") + "\n")
        logf.flush()

    try:
        log(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} 开始任务：{t['name']}（{spec.label if spec else t['model_type']}）=====")
        # ---- 体检 ----
        if settings.load().get("auto_check_before_train", True) and not t.get("skip_check"):
            _set(t, status="checking", message="训练前检查…")
            issues = diagnose.preflight(t)
            ignored = set(t.get("ignored_issues") or [])
            issues = [i for i in issues if i["code"] not in ignored]
            blocking = [i for i in issues if i["level"] == "error" or i["fixes"]]
            if blocking:
                for i in issues:
                    log(f"[检查] {i['title']}：{i['reason']}")
                _set(t, status="blocked", issues=issues, message=blocking[0]["title"] + "，需要你确认处理方式")
                return
        os.makedirs(t["output_dir"], exist_ok=True)
        gpus = t.get("gpus") or settings.gpu_list()
        tb = dict(t, params=dict(t["params"], _num_gpus=len(gpus)))
        steps = spec.build_steps(tb, td)
        prep = prep_step(t["params"])
        if prep:
            steps.insert(0, prep)
        if len(gpus) > 1:
            log(f"[提示] 多卡训练：使用显卡 {', '.join(gpus)}，共 {len(gpus)} 个进程；每张卡各跑「单批数量」张图。")
        with open(os.path.join(td, "commands.txt"), "w", encoding="utf-8") as f:
            for s in steps:
                f.write(s["name"] + ":\n" + subprocess.list2cmdline([paths.env_python() or "python"] + s["cmd"]) + "\n\n")
        r = t.get("resume")
        lr_file = os.path.join(td, "actual_lr.tsv")
        if r:   # 续训：训练命令加上续训参数，曲线保留已经练过的部分
            for s in steps:
                if s["kind"] == "train":   # 开练前的对照图第一次已经出过了，续训不再重复出
                    s["cmd"] = [x for x in s["cmd"] if x != "--sample_at_first"] + spec.resume_args(r)
            base = int(r["base"])
            spe = _spe(t)
            ga = max(1, int(t["params"].get("grad_accum") or 1))
            if r["mode"] == "state" and spe:   # sd-scripts 续训时内部步数只从「这一轮里第几步」算起，记下差值好把曲线接上
                e = (base * ga) // spe
                r["tb_offset"] = int(round(base - (base * ga - e * spe * ga)))
            else:
                r["tb_offset"] = base
            t["tb_runs"] = (t.get("tb_runs") or []) + [[time.time(), r["tb_offset"]]]
            try:
                with open(lr_file, encoding="utf-8") as f:
                    keep = [ln for ln in f if int(ln.split("\t")[0]) <= base]
                with open(lr_file, "w", encoding="utf-8") as f:
                    f.writelines(keep)
            except (OSError, ValueError):
                pass
            hist = [x for x in ((t.get("progress") or {}).get("loss_hist") or []) if x[0] <= base]
            log(f"[续训] {r['label']}")
        else:
            t["tb_runs"] = []
            try:   # 每次重新开始训练都清空上次记录的实际学习率
                os.remove(lr_file)
            except OSError:
                pass
            hist = []
            if (t.get("sample") or {}).get("enabled") and t["sample"].get("clear_old"):
                n = clear_samples(t)
                if n:
                    log(f"[提示] 已清空上次的 {n} 张测试图")
        try:
            os.remove(os.path.join(td, "pause.flag"))
        except OSError:
            pass
        _set(t, status="running", started=time.time(), finished=None, error=None, issues=None,
             step_names=[s["name"] for s in steps], step_index=0, message="", progress={"loss_hist": hist})
        for i, s in enumerate(steps):
            if _state["freeze_req"]:   # 体检 / 上一步时就点了冻结
                _freeze_done(t, td, "", log)
                return
            for n in s.get("notes") or []:
                log("[提示] " + n)
            _set(t, step_index=i, message=s["name"] + "…")
            rc, tail = _run_step(t, s, log)
            if _state["freeze_req"]:
                _freeze_done(t, td, tail, log)
                return
            if _state["stop_req"]:
                _set(t, status="stopped", finished=time.time(), message="已手动停止")
                log("===== 已手动停止 =====")
                return
            if rc != 0 and s["kind"] == "prep":   # 预处理出问题不挡训练
                log("[提示] 图片预处理没有完成，跳过，继续训练")
                continue
            if rc != 0:
                d = diagnose.diagnose(tail, t)
                log(f"===== 失败：{d['title']} =====")
                _set(t, status="failed", finished=time.time(), error=d, message=d["title"])
                return
        _set(t, status="done", finished=time.time(), message="训练完成", outputs=list_outputs(t), resume=None)
        shutil.rmtree(os.path.join(td, PAUSE_STATE), ignore_errors=True)
        log(f"===== {time.strftime('%H:%M:%S')} 训练完成 =====")
    except Exception as e:
        log(traceback.format_exc())
        _set(t, status="failed", finished=time.time(), message=str(e),
             error={"code": "internal", "title": "训练器内部错误", "reason": str(e), "detail": traceback.format_exc(), "fixes": []})
    finally:
        logf.close()


def prep_step(p):
    """训练前自动把透明底图片改成白底（用透明通道当 loss 遮罩时不改）。"""
    if p.get("alpha_mask"):
        return None
    dirs = [d for d in (p.get("dataset_dir"), p.get("reg_data_dir")) if d and os.path.isdir(d)]
    if not dirs:
        return None
    return {"name": "检查透明底图片", "kind": "prep", "cwd": paths.APP,
            "cmd": prep_cmd(dirs)}


def prep_cmd(dirs):
    cmd = [os.path.join(paths.APP, "workers", "prep_images.py")]
    for d in dirs:
        cmd += ["--dir", d]
    return cmd + ["--backup", os.path.join(paths.DATA, "透明底原图备份"), "--cache", os.path.join(paths.DATA, "alpha_checked.json")]


def _reap_children(pid, proc, wait=20):
    """训练进程冻结退出后，结束它留下的 multiprocessing 子进程（命令行里带 parent_pid=训练进程）。"""
    time.sleep(wait)
    if proc.stdout.closed:
        return
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              f"Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                              f"Where-Object {{ $_.CommandLine -like '*parent_pid={pid},*' }} | ForEach-Object {{ $_.ProcessId }}"],
                             capture_output=True, text=True, timeout=30, creationflags=NO_WINDOW).stdout
        for x in out.split():
            if x.isdigit():
                kill_tree(int(x))
    except Exception:
        pass


def _freeze_done(t, td, tail, log):
    _state["freeze_req"] = False
    flag = os.path.join(td, "pause.flag")
    saved = os.path.exists(flag) and "LORA_PAUSED" in tail
    try:
        os.remove(flag)
    except OSError:
        pass
    if saved:
        info = resume_info(dict(t, status="paused"))
        msg = f"已冻结在第 {info['base']} 步，显存已释放；点「继续」接着练" if info else "已冻结；点「继续」接着练"
    elif t.get("resume"):
        msg = "已冻结（这次还没开始训练），继续时从上次的进度接着练"
    else:
        msg = "已冻结（还没开始训练），继续时从上次保存的进度接着练，没有进度就从头开始"
    _set(t, status="paused", finished=time.time(), message=msg)
    log(f"===== {msg} =====")


def _run_step(t, s, log):
    py = paths.env_python()
    if not py:
        raise RuntimeError("训练环境还没安装")
    gpus = t.get("gpus") or settings.gpu_list()
    p = t.get("params") or {}
    if s["kind"] == "train" and len(gpus) > 1:
        # 多卡：用 accelerate 启动多个进程，每张卡一个（Windows 用 gloo 通信，Linux 用 nccl，引擎内部自动选择）
        cmd = [py, os.path.join(paths.APP, "workers", "ddp_launch.py"), "--num_processes", str(len(gpus)), "--num_machines", "1",
               "--multi_gpu", "--mixed_precision", p.get("mixed_precision") or "bf16",
               "--main_process_port", str(int(p.get("main_process_port") or 29500)),
               "--num_cpu_threads_per_process", str(int(p.get("num_cpu_threads_per_process") or 1)),
               "--dynamo_backend", "no"] + s["cmd"]
        env = settings.child_env({"CUDA_VISIBLE_DEVICES": ",".join(gpus), "LORA_DDP_NO_LIBUV": "1", "USE_LIBUV": "0"})
    else:
        # 单卡训练、缓存、导出：只用第一张卡
        cmd = [py] + s["cmd"]
        env = settings.child_env({"CUDA_VISIBLE_DEVICES": gpus[0]})
    env.update(s.get("env") or {})   # 步骤自带的环境变量（例如 Krea 2 文本编码器分层卸载 LORA_TE_OFFLOAD）
    if s["kind"] == "train":   # 自适应优化器（Automagic / EmoSens）把每步实际学习率写到这里，训练页画图用
        env["LORA_ACTUAL_LR_FILE"] = os.path.join(task_dir(t), "actual_lr.tsv")
        spec = models.get(t["model_type"])
        if spec and spec.resume:   # 冻结：见 app/vendor/sitecustomize.py
            env["LORA_PAUSE_FILE"] = os.path.join(task_dir(t), "pause.flag")
            env["LORA_PAUSE_STATE"] = os.path.join(task_dir(t), PAUSE_STATE)
        r = t.get("resume")
        if r:
            env["LORA_STEP_OFFSET"] = str(r.get("tb_offset") or 0)
            env["LORA_ACTUAL_LR_BASE"] = str(r.get("base") or 0)
    with _lock:
        _state["step_kind"] = s["kind"]
    log("$ " + subprocess.list2cmdline(cmd))
    proc = subprocess.Popen(cmd, cwd=s["cwd"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
    with _lock:
        _state["proc"] = proc
    tail = []
    buf = b""
    last_prog_log = 0
    loss_hist = (t.get("progress") or {}).get("loss_hist") or []

    def handle(line):
        nonlocal last_prog_log
        m = RE_STEPS.search(line)
        if m and s["kind"] == "train":
            step, total = int(m.group(2)), int(m.group(3))
            r = t.get("resume")
            if r:   # 续训：进度条只数这次练的步数，加上之前练过的
                step, total = step + int(r["base"]), int(t.get("steps_total") or total)
            elif t.get("steps_total") != total:
                t["steps_total"] = total
            lm = RE_LOSS.search(line)
            loss = lm.group(1) if lm else None
            prog = dict(t.get("progress") or {})
            eta = m.group(5)
            run_steps = int(m.group(2))
            if r and run_steps > 0:   # 续训：引擎的剩余时间是按「这次的步数 / 全部步数」算的，偏大很多；按这次每步的耗时重算
                el = _hms(m.group(4))
                if el is not None:
                    left = int(el / run_steps * max(0, total - step))
                    eta = f"{left // 3600}:{left % 3600 // 60:02d}:{left % 60:02d}" if left >= 3600 else f"{left // 60:02d}:{left % 60:02d}"
            prog.update(step=step, total=total, elapsed=m.group(4), eta=eta)
            if loss and loss != "nan":
                prog["loss"] = float(loss)
                if not loss_hist or loss_hist[-1][0] != step:
                    loss_hist.append([step, float(loss)])
                    if len(loss_hist) > 2000:
                        del loss_hist[::2]
            prog["loss_hist"] = loss_hist
            em = RE_EPOCH.search(line)
            if em:
                prog["epoch"] = f"{em.group(1)}/{em.group(2)}"
            with _lock:
                t["progress"] = prog
            # 进度行很多，日志里每 2 秒记一次
            if time.time() - last_prog_log < 2:
                return
            last_prog_log = time.time()
        elif s["kind"] == "train" and RE_SAMPLING.search(line):
            ms = RE_SAMPLING.search(line)
            with _lock:
                t["message"] = f"出测试图中（{ms.group(1)}/{ms.group(2)} 步）"
            if time.time() - last_prog_log < 2:
                return
            last_prog_log = time.time()
        elif s["kind"] == "cache":
            m2 = RE_CACHE.search(line)
            if m2:
                with _lock:
                    t["progress"] = dict(t.get("progress") or {}, cache=f"{s['name']} {m2.group(2)}/{m2.group(3)}")
        if line.startswith("LORA_PAUSED"):   # 冻结：兜底清理没退出的数据加载子进程，免得一直等输出管道关闭
            pm = re.search(r"pid=(\d+)", line)
            if pm:
                threading.Thread(target=_reap_children, args=(int(pm.group(1)), proc), daemon=True).start()
        if "READY TO STOP" in line:   # EmoSens 判断 loss 已经收敛时打印的提示
            with _lock:
                pr = dict(t.get("progress") or {})
                pr["ready_stop"] = int(pr.get("ready_stop") or 0) + 1
                pr["ready_stop_step"] = pr.get("step")
                t["progress"] = pr
        em = RE_EPOCH.search(line)
        if em:
            with _lock:
                t["progress"] = dict(t.get("progress") or {}, epoch=f"{em.group(1)}/{em.group(2)}")
                if s["kind"] == "train":
                    t["epochs_total"] = int(em.group(2))
        log(line)
        tail.append(line)
        if len(tail) > 400:
            del tail[:100]

    while True:
        chunk = proc.stdout.read1(4096)
        if not chunk:
            break
        buf += chunk
        while True:
            j = min([x for x in (buf.find(b"\n"), buf.find(b"\r")) if x >= 0], default=-1)
            if j < 0:
                break
            line, buf = buf[:j], buf[j + 1:]
            line = ANSI.sub("", decode_line(line)).rstrip()
            if line.strip():
                handle(line)
    if buf.strip():
        handle(decode_line(buf))
    rc = proc.wait()
    with _lock:
        _state["proc"] = None
        _save()
    return rc, "\n".join(tail)


def list_outputs(t):
    d = t.get("output_dir")
    res = []
    if d and os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".safetensors"):
                p = os.path.join(d, fn)
                res.append({"name": fn, "path": p, "size": os.path.getsize(p), "mtime": os.path.getmtime(p)})
    return res


def list_samples(t):
    """训练中测试图：output_dir/sample/{name}_{000500|e000002}_{序号}_{时间}_{种子}.png"""
    d = os.path.join(t.get("output_dir") or "", "sample")
    res = []
    if os.path.isdir(d):
        for fn in os.listdir(d):
            if not fn.lower().endswith((".png", ".jpg", ".webp")):
                continue
            m = re.search(r"_(e?)(\d{6})_(\d{2})_(\d{14})(?:_(\d+))?", fn)
            if not m:
                continue
            res.append({"file": os.path.join(d, fn), "name": fn, "epoch": bool(m.group(1)), "num": int(m.group(2)),
                        "prompt": int(m.group(3)), "time": m.group(4), "seed": m.group(5)})
    res.sort(key=lambda x: (x["time"], x["prompt"]))
    return res

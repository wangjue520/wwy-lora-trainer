# -*- coding: utf-8 -*-
"""执行用户确认过的修复动作，然后把任务重新放回队列。"""
import glob
import os
import shutil

import diagnose
import downloader
import envsetup
import gpu
import jobs
import models
import paths
import settings
import tasks


def _inside(d, f):
    try:
        return bool(d) and os.path.commonpath([os.path.abspath(d), os.path.abspath(f)]) == os.path.abspath(d)
    except ValueError:
        return False


def clean_dataset(dataset_dir, files):
    """删除空标签 / 空图片；损坏的图片移到「_已移除」。只处理训练集目录里面的文件。"""
    removed, moved = 0, 0
    trash = os.path.join(dataset_dir, "_已移除")
    for it in files or []:
        f = it["path"] if isinstance(it, dict) else it
        if not _inside(dataset_dir, f) or not os.path.isfile(f):
            continue
        what = it.get("what", "") if isinstance(it, dict) else ""
        try:
            if "损坏" in what and os.path.getsize(f) > 0:
                os.makedirs(trash, exist_ok=True)
                shutil.move(f, os.path.join(trash, os.path.basename(f)))
                cap = os.path.splitext(f)[0] + ".txt"
                if os.path.isfile(cap):
                    shutil.move(cap, os.path.join(trash, os.path.basename(cap)))
                moved += 1
            else:
                os.remove(f)
                removed += 1
        except OSError:
            pass
    return f"已删除 {removed} 个空文件" + (f"，{moved} 张损坏图片移到了「_已移除」" if moved else "")


def apply(tid, fid, data=None):
    """返回 {ok, message, job?}。tid 可以为空（例如在高级参数页手动检查训练集）。"""
    data = data or {}
    t = tasks.get(tid) if tid else None
    spec = models.get(t["model_type"]) if t else None

    def requeue(patch=None, mpatch=None):
        if t:
            tasks.requeue(tid, patch, mpatch)
            tasks.start_queue()

    def changed(vals):
        # 改了哪些参数也返回给前端：同一个底模的「当前训练配置」跟着改，不然那里还显示旧值（例如分辨率还是 1024），下次加入队列又用回旧值
        return {"params": vals, "model_type": t["model_type"]} if t and vals else {}

    if fid == "ignore":
        if t:
            ign = set(t.get("ignored_issues") or [])
            ign.update(data.get("codes") or [])
            with tasks._lock:
                tt = tasks._find(tid)
                tt["ignored_issues"] = sorted(ign)
            requeue()
        return {"ok": True, "message": "已忽略，继续训练"}

    if fid == "clean_dataset":
        d = data.get("dataset_dir") or (t["params"].get("dataset_dir") if t else None)
        msg = clean_dataset(d, data.get("files"))
        requeue()
        return {"ok": True, "message": msg}

    if fid == "kill_gpu":
        own = set(tasks.running_pids())
        pids = [p["pid"] for p in data.get("procs") or [] if p["pid"] not in own and p["pid"] != os.getpid()]
        done = gpu.kill(pids)
        requeue()
        return {"ok": True, "message": f"已结束 {len(done)} 个进程"}

    if fid == "lower_vram":
        vals = diagnose.lower_vram_values(t, spec) if t else {}
        if not vals:
            return {"ok": False, "message": "已经是最省显存的设置了，只能换显存更大的显卡或降低分辨率 / dim。"}
        requeue(vals)
        return {"ok": True, "message": "已调整：" + ", ".join(f"{k}={v}" for k, v in vals.items()), **changed(vals)}

    if fid == "sync_sample_prompts":
        if t:
            import triggers
            with tasks._lock:
                tt = tasks._find(tid)
                tt["sample"]["prompts"], _ = triggers.sync_prompts(tt["params"].get("dataset_dir"), tt["sample"].get("prompts"), True)
            requeue()
        return {"ok": True, "message": "测试提示词已改好，继续训练"}

    if fid == "set_params":
        vals = data.get("values") or {}
        requeue(vals)
        return {"ok": True, "message": "参数已修改，重新开始", **changed(vals)}

    if fid == "workers_zero":
        patch = {"num_workers": 0}
        if spec and spec.id == "krea2":
            p = t["params"]
            patch["blocks_to_swap"] = max(int(p.get("blocks_to_swap") or 0), 8)
        requeue(patch)
        return {"ok": True, "message": "已减少内存占用，重新开始", **changed(patch)}

    if fid == "switch_cn":
        settings.save({"network": "cn"})
        requeue()
        return {"ok": True, "message": "已切换到国内网络"}

    if fid == "clear_cache":
        if t:
            shutil.rmtree(os.path.join(paths.TASKS, tid, "cache"), ignore_errors=True)
            d = t["params"].get("dataset_dir")
            if d and os.path.isdir(d):
                for f in glob.glob(os.path.join(d, "**", "*.npz"), recursive=True):
                    try:
                        os.remove(f)
                    except OSError:
                        pass
        requeue()
        return {"ok": True, "message": "缓存已清除"}

    # ---- 需要后台作业的修复：作业完成后自动重新排队 ----
    def after(job):
        if job.status == "done":
            if fid == "download_model" and job.result:
                if t:
                    tasks.requeue(tid, None, {data["slot"]: job.result})
                    tasks.start_queue()
                if t:
                    settings.set_draft_model(t["model_type"], data["slot"], job.result)
            else:
                requeue()

    if fid == "download_model":
        job = jobs.start("download", f"下载 {data.get('label') or data.get('file')}", downloader.download,
                         data["repo"], data["file"], on_done=after)
        return {"ok": True, "message": "开始下载，完成后自动继续", "job": job.id}
    if fid in ("install_env", "repair_env"):
        if jobs.find_running("env"):
            return {"ok": False, "message": "环境安装已经在进行中"}
        job = jobs.start("env", "安装训练环境" if fid == "install_env" else "修复训练环境", envsetup.install,
                         repair=fid == "repair_env", on_done=after)
        return {"ok": True, "message": "开始安装，完成后自动继续", "job": job.id}
    if fid == "update_engine":
        job = jobs.start("env", "同步训练引擎和环境", envsetup.sync, on_done=after)
        return {"ok": True, "message": "开始同步训练引擎和环境，完成后自动继续", "job": job.id}
    if fid == "reinstall_torch":
        job = jobs.start("env", "重装 PyTorch", lambda j: envsetup.install_torch(j, force=True), on_done=after)
        return {"ok": True, "message": "开始重装 PyTorch", "job": job.id}
    if fid == "pip_install":
        job = jobs.start("env", f"安装 {data.get('pkg')}", envsetup.pip_install, data.get("pkg"), on_done=after)
        return {"ok": True, "message": f"开始安装 {data.get('pkg')}", "job": job.id}
    return {"ok": False, "message": f"未知的修复动作：{fid}"}

# -*- coding: utf-8 -*-
"""一键更新：训练器本体 + 两个训练引擎 + 环境依赖，三样一起更新、版本对应。

训练器的 version.json 里记着测试过的引擎版本（engines），更新时引擎切到这个版本，再按新的依赖清单装依赖。"""
import os
import shutil
import subprocess
import urllib.request
import zipfile

import envsetup
import paths
import settings

NO_WINDOW = 0x08000000 if os.name == "nt" else 0
VERSION_FILE = os.path.join(paths.ROOT, "version.json")


def version():
    import json
    try:
        return json.load(open(VERSION_FILE, encoding="utf-8"))
    except Exception:
        return {"version": "0.1.0"}


def update_self(job):
    g = envsetup._git()
    repo = (settings.load().get("update_repo") or "").strip()
    if os.path.isdir(os.path.join(paths.ROOT, ".git")) and g:
        job.write("更新训练器本体（git pull）…")
        rc = job.run_cmd([g, "-C", paths.ROOT, "pull", "--ff-only"], env=settings.child_env())
        if rc != 0:
            raise RuntimeError("训练器本体更新失败：本地文件有改动或网络不通，详见日志")
        return True
    if not repo:
        job.write("没有配置训练器的更新地址（训练器设置 → 更新地址），跳过本体更新")
        return False
    import netpick
    url = repo.rstrip("/").removesuffix(".git") + "/archive/refs/heads/main.zip"
    tmp = os.path.join(paths.DATA, "self_update.zip")
    netpick.download_file(job, netpick.rank_github(url), tmp, "训练器更新包")
    with zipfile.ZipFile(tmp) as z:
        names = z.namelist()
        top = names[0].split("/")[0] + "/"
        for n in names:
            rel = n[len(top):]
            if not rel or rel.endswith("/"):
                continue
            if not rel.startswith(("app/", "web/", "scripts/")) and rel not in (
                    "启动LoRA训练器.bat", "version.json", "README.md",
                    "run.sh", "install.sh", "start.sh", "stop.sh", "README-Linux.md"):
                continue
            dst = os.path.join(paths.ROOT, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with z.open(n) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out)
    os.remove(tmp)
    job.write("训练器本体已更新，重启训练器后生效")
    return True


def update_all(job):
    job.set_progress(0.05, "更新训练器本体")
    self_updated = update_self(job)
    # 用刚更新下来的新代码同步引擎和环境：新版本要求的引擎版本、新加的依赖当场就装好，不用等下次
    import importlib
    import netpick
    for m in (settings, netpick, envsetup):
        importlib.reload(m)
    job.set_progress(0.3, "同步训练引擎和环境")
    envsetup.sync(job)
    job.set_progress(1, "更新完成" + ("，请重启训练器" if self_updated else ""))
    return {"restart": bool(self_updated)}


def update_trainer_only(job):
    """只更新训练器本体，不动引擎和训练环境：训练器还新、修 bug 勤，引擎一般是成熟的，不用每次都跟着动。
    新版训练器要是要求了别的引擎版本 / 新依赖，重启后启动检查（auto_sync）会照常同步，保证版本对应。"""
    job.set_progress(0.1, "更新训练器本体")
    self_updated = update_self(job)
    import importlib
    importlib.reload(envsetup)   # 用刚下下来的 version.json / 依赖清单检查
    why = envsetup.needs_sync()
    if why:
        job.write(f"新版训练器要求的训练环境有变化（{why}），重启训练器后会自动同步引擎和依赖")
    job.set_progress(1, "更新完成" + ("，请重启训练器" if self_updated else ""))
    return {"restart": bool(self_updated), "sync": why}


def auto_sync():
    """启动时检查：引擎版本 / 依赖和这个版本的训练器对不上（不管训练器是怎么更新的），就自动在后台同步。"""
    import jobs
    why = envsetup.needs_sync()
    if not why or jobs.find_running("env"):
        return None
    j = jobs.start("env", "同步训练引擎和环境", _auto_sync_job, why)
    return j.id


def _auto_sync_job(job, why):
    job.write(f"训练器和训练环境对不上（{why}），自动同步…训练任务会等同步完成再开始")
    return envsetup.sync(job)

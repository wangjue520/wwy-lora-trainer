# -*- coding: utf-8 -*-
"""一键更新：训练器本体 + 两个训练引擎 + 环境依赖，三样一起更新、版本对应。

训练器的 version.json 里记着测试过的引擎版本（engines），更新时引擎切到这个版本，再按新的依赖清单装依赖。"""
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request
import zipfile

import envsetup
import paths
import settings

NO_WINDOW = 0x08000000 if os.name == "nt" else 0
VERSION_FILE = os.path.join(paths.ROOT, "version.json")
# 这份训练器是哪个提交：git archive / GitHub 打包下载时自动填成提交号（.gitattributes 里的 export-subst）
COMMIT_FILE = os.path.join(paths.ROOT, "commit.txt")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def version():
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
        _check_cache.clear()
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
                    "启动LoRA训练器.bat", "version.json", "commit.txt", "README.md",
                    "run.sh", "install.sh", "start.sh", "stop.sh", "README-Linux.md"):
                continue
            dst = os.path.join(paths.ROOT, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with z.open(n) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out)
        if not _read_commit():   # GitHub 没替换 commit.txt 时，用压缩包注释里的提交号（GitHub 打包会写进去）
            c = (z.comment or b"").decode("ascii", "ignore").strip()
            if HEX40.match(c):
                open(COMMIT_FILE, "w", encoding="utf-8").write(c + "\n")
    os.remove(tmp)
    _check_cache.clear()
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


# ---------------- 新版本提示 ----------------
def _read_commit():
    try:
        c = open(COMMIT_FILE, encoding="utf-8").read().strip()
    except OSError:
        return None
    return c if HEX40.match(c) else None


def _git_out(*args):
    g = envsetup._git()
    if not g:
        return None
    try:
        r = subprocess.run([g, "-C", paths.ROOT, *args], capture_output=True, text=True, timeout=10, creationflags=NO_WINDOW)
    except Exception:
        return None
    return r


def local_commit():
    """这份训练器是哪个提交：目录是 git 仓库就问 git，整合包 / 一键更新下来的看 commit.txt；拿不准返回 None。"""
    if os.path.isdir(os.path.join(paths.ROOT, ".git")):
        r = _git_out("rev-parse", "HEAD")
        if r and HEX40.match(r.stdout.strip()):
            return r.stdout.strip()
    return _read_commit()


def _github_repo():
    repo = (settings.load().get("update_repo") or "").strip()
    if not repo and os.path.isdir(os.path.join(paths.ROOT, ".git")):   # 更新地址留空的 git 仓库：用它的 origin
        r = _git_out("remote", "get-url", "origin")
        repo = r.stdout.strip() if r and r.returncode == 0 else ""
    repo = repo.rstrip("/").removesuffix(".git")
    m = re.match(r"^https://github\.com/([\w.-]+)/([\w.-]+)$", repo)
    return (repo, m.group(1), m.group(2)) if m else None


def _remote(repo, owner, name, deadline):
    """GitHub 上 main 分支最近的提交：先问 GitHub API（带更新说明），不通再读 git 的 info/refs（能走加速代理，只有提交号）。"""
    import netpick
    routes = netpick.routes()
    last = None
    for _, proxy in routes:
        if time.time() > deadline:
            break
        try:
            url = f"https://api.github.com/repos/{owner}/{name}/commits?sha=main&per_page=30"
            with netpick.open_url(url, proxy, {"Accept": "application/vnd.github+json"}, timeout=6) as r:
                d = json.loads(r.read(4 << 20).decode("utf-8"))
            return [{"sha": c["sha"], "msg": (c["commit"]["message"] or "").split("\n")[0].strip(),
                     "date": c["commit"]["committer"]["date"]} for c in d]
        except Exception as e:
            last = e
    refs = repo + ".git/info/refs?service=git-upload-pack"
    custom = (settings.load().get("github_proxy") or "").strip()
    prefixes = [""] + [gp.rstrip("/") + "/" for gp in dict.fromkeys(([custom] if custom else []) + netpick.GH_PROXIES)]
    if settings.is_cn():
        prefixes = prefixes[1:] + prefixes[:1]
    for _, proxy in routes:
        for pre in prefixes:
            if time.time() > deadline:
                raise RuntimeError(f"连不上 GitHub：{last}")
            try:
                with netpick.open_url(pre + refs, proxy, timeout=6) as r:
                    m = re.search(rb"([0-9a-f]{40}) refs/heads/main\b", r.read(1 << 20))
                if m:
                    return [{"sha": m.group(1).decode(), "msg": "", "date": ""}]
            except Exception as e:
                last = e
    raise RuntimeError(f"连不上 GitHub：{last}")


_check_cache = {}
_check_lock = threading.Lock()


def check(force=False):
    """有没有新版本：{new, count, commits, local, remote, checked_at, error}。成功的结果记 1 小时，连不上的记 10 分钟。"""
    with _check_lock:
        c = _check_cache.get("r")
        if c and not force and time.time() - c["checked_at"] < (3600 if not c.get("error") else 600):
            return c
        local = local_commit()
        res = {"new": False, "count": None, "commits": [], "local": local, "remote": None, "checked_at": time.time(), "error": None}
        gh = _github_repo()
        if not gh:
            res["error"] = "训练器更新地址不是 GitHub 仓库，没法检查新版本"
        else:
            try:
                commits = _remote(*gh, deadline=time.time() + 40)
                remote = res["remote"] = commits[0]["sha"]
                shas = [x["sha"] for x in commits]
                if not local or remote == local:
                    pass
                elif local in shas:
                    res["new"], res["count"] = True, shas.index(local)
                    res["commits"] = commits[:res["count"]][:10]
                else:
                    # 本地是 git 仓库、GitHub 的最新提交本地已经有了（开发机上本地比 GitHub 新）就不算有更新
                    r = _git_out("merge-base", "--is-ancestor", remote, "HEAD") if os.path.isdir(os.path.join(paths.ROOT, ".git")) else None
                    if not (r and r.returncode == 0):
                        res["new"] = True
                        res["commits"] = [x for x in commits if x["msg"]][:10]
            except Exception as e:
                res["error"] = str(e)
        _check_cache["r"] = res
        return res


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

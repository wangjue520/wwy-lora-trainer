# -*- coding: utf-8 -*-
"""显卡信息与占用显存的进程（nvidia-smi）。"""
import os
import subprocess
import time

import plat

_cache = {"t": 0, "v": None}
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _smi(args, timeout=8):
    try:
        r = subprocess.run(["nvidia-smi"] + args, capture_output=True, text=True, timeout=timeout,
                           creationflags=NO_WINDOW, encoding="utf-8", errors="replace")
        return r.stdout if r.returncode == 0 else None
    except Exception:
        return None


def info(max_age=5):
    """{name, total_gb, used_gb, free_gb, compute_cap, driver, util}；没有 N 卡时返回 {}。"""
    if _cache["v"] is not None and time.time() - _cache["t"] < max_age:
        return _cache["v"]
    out = _smi(["--query-gpu=index,name,memory.total,memory.used,memory.free,compute_cap,driver_version,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits"])
    res = {}
    if out:
        try:
            import settings
            want = settings.gpu_list()[0]
        except Exception:
            want = "0"
        rows = [[x.strip() for x in line.split(",")] for line in out.strip().splitlines() if line.strip()]
        row = next((r for r in rows if r[0] == want), rows[0] if rows else None)
        if row:
            def fl(x):
                try:
                    return float(x)
                except ValueError:
                    return 0.0
            res = {"index": row[0], "name": row[1], "total_gb": round(fl(row[2]) / 1024, 1),
                   "used_gb": round(fl(row[3]) / 1024, 1), "free_gb": round(fl(row[4]) / 1024, 1),
                   "compute_cap": fl(row[5]), "driver": row[6], "util": fl(row[7]), "temp": fl(row[8]),
                   "all": [{"index": r[0], "name": r[1], "total_gb": round(fl(r[2]) / 1024, 1)} for r in rows]}
    _cache.update(t=time.time(), v=res)
    return res


# 这些进程绝不能结束：系统组件、桌面、浏览器（训练器界面就开在浏览器里）
PROTECTED_WIN = {"explorer.exe", "dwm.exe", "csrss.exe", "winlogon.exe", "shellhost.exe", "searchhost.exe", "startmenuexperiencehost.exe",
                 "textinputhost.exe", "shellexperiencehost.exe", "applicationframehost.exe", "widgets.exe", "lockapp.exe",
                 "crossdeviceresume.exe", "phoneexperiencehost.exe", "systemsettings.exe", "msedge.exe", "chrome.exe", "firefox.exe",
                 "msedgewebview2.exe", "brave.exe", "opera.exe", "claude.exe", "nvcontainer.exe", "nvidia share.exe",
                 "nvidia overlay.exe", "nvdisplay.container.exe", "rtss.exe", "obs64.exe"}
# Linux：桌面 / 远程桌面 / Jupyter（仙宫云、AutoDL 这类镜像里就靠它开终端）/ 容器运行时
PROTECTED_NIX = {"xorg", "xwayland", "gnome-shell", "kwin_x11", "kwin_wayland", "plasmashell", "systemd", "sshd",
                 "jupyter", "jupyter-lab", "jupyterhub", "jupyter-notebook", "code-server", "dockerd", "containerd"}
PROTECTED = PROTECTED_WIN if os.name == "nt" else PROTECTED_NIX


def _cmdlines(pids):
    """取进程命令行，用来认出 WebUI / ComfyUI（Linux 读 /proc，Windows 用 PowerShell）。"""
    if not pids:
        return {}
    return {pid: plat.cmdline(pid) for pid in pids}


def _label(name, cmd):
    c = (cmd or "").lower()
    if "comfy" in c or "main.py" in c and "comfy" in c:
        return "ComfyUI"
    if "webui" in c or "launch.py" in c or "forge" in c:
        return "WebUI"
    if "train" in c or "sd-scripts" in c or "musubi" in c or "kohya" in c:
        return "另一个训练程序"
    if name.lower().startswith("python"):
        return "Python 程序"
    return ""


def processes(exclude_pids=()):
    """占用显卡的进程 [{pid, name, used_mb, label, suggest}]。

    Windows 上 nvidia-smi 会把所有用到显卡的进程（包括桌面、浏览器）都列出来，显存用量通常拿不到。
    这里排除系统 / 浏览器进程，只有 WebUI、ComfyUI、Python 程序默认建议结束（suggest=True），其余要用户自己勾。"""
    out = _smi(["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    rows = []
    if out:
        for line in out.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            pid = int(parts[0])
            if pid in exclude_pids or pid == os.getpid():
                continue
            name = os.path.basename(parts[1].replace("\\", "/"))
            path = parts[1].lower()
            if name.lower() in PROTECTED:
                continue
            if os.name == "nt" and ("\\windows\\" in path.replace("/", "\\") or "\\windowsapps\\" in path.replace("/", "\\")):
                continue
            try:
                mb = int(parts[2])
            except (ValueError, IndexError):
                mb = None
            rows.append({"pid": pid, "name": name, "path": parts[1], "used_mb": mb})
    cmds = _cmdlines([r["pid"] for r in rows])
    for r in rows:
        r["cmd"] = cmds.get(r["pid"], "")[:300]
        r["label"] = _label(r["name"], r["cmd"])
        r["suggest"] = bool(r["label"]) or (r["used_mb"] or 0) > 1024
    rows.sort(key=lambda r: (not r["suggest"], -(r["used_mb"] or 0)))
    return rows


def kill(pids):
    """只结束确实在用显卡、且不在保护名单里的进程。"""
    allowed = {p["pid"] for p in processes()}
    done = []
    for pid in pids:
        if pid not in allowed:
            continue
        try:
            # 只结束这个进程：Linux 上它可能和 Jupyter / 桌面在同一个进程组，整组杀会误伤
            if os.name == "nt":
                plat.kill_tree(pid)
            else:
                os.kill(pid, 9)
            done.append(pid)
        except Exception:
            pass
    _cache["t"] = 0
    return done

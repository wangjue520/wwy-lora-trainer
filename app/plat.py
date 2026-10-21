# -*- coding: utf-8 -*-
"""Windows / Linux 的差异都收在这里，别的模块不直接判断系统。

训练器原来是给 Windows 整合包写的：子进程用 CREATE_NO_WINDOW 隐藏黑框、结束进程用 taskkill /T、
用 PowerShell 取命令行、文件选择框从盘符开始。Linux 上没有这些，但有一一对应的做法：
start_new_session（独立进程组，好整组结束）、/proc/<pid>/cmdline、从 / 和 home 开始浏览。"""
import os
import shlex
import shutil
import signal
import subprocess
import sys

IS_WIN = os.name == "nt"
NO_WINDOW = 0x08000000 if IS_WIN else 0


def popen_kwargs():
    """启动子进程的参数：Windows 隐藏控制台窗口；Linux 让子进程自成进程组，
    这样停止训练时能连它 fork 出来的 dataloader / accelerate 子进程一起结束。"""
    if IS_WIN:
        return {"creationflags": NO_WINDOW}
    return {"start_new_session": True}


def detach_kwargs():
    """「重启训练器」用：新进程脱离当前终端 / 会话，旧进程退出后它还在。"""
    if IS_WIN:
        return {"creationflags": 0x00000010}   # DETACHED_PROCESS
    return {"start_new_session": True}


def venv_python(env_dir):
    """训练环境里的 python 解释器（Windows：Scripts\\python.exe；Linux：bin/python）。"""
    p = os.path.join(env_dir, "Scripts", "python.exe") if IS_WIN else os.path.join(env_dir, "bin", "python")
    return p if os.path.isfile(p) else None


def _pgid(pid):
    try:
        return os.getpgid(pid)
    except OSError:
        return None


def kill_tree(pid):
    """结束一个进程和它下面的所有子进程。"""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return
    try:
        if IS_WIN:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=NO_WINDOW)
            return
        pgid = _pgid(pid)
        # 千万别把自己所在的进程组一起杀了：只有子进程是独立进程组时才 killpg
        if pgid is not None and pgid != _pgid(0):
            os.killpg(pgid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def cmdline(pid):
    """进程命令行。Windows 用 PowerShell，Linux 读 /proc。拿不到返回 ""。"""
    if IS_WIN:
        try:
            ps = (f"Get-CimInstance Win32_Process -Filter \"ProcessId={int(pid)}\" | "
                  f"ForEach-Object {{ $_.CommandLine }}")
            r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                               timeout=10, creationflags=NO_WINDOW, encoding="utf-8", errors="replace")
            return (r.stdout or "").strip()
        except Exception:
            return ""
    try:
        with open(f"/proc/{int(pid)}/cmdline", "rb") as f:
            return f.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    except (OSError, ValueError):
        return ""


def pids_with(pattern):
    """命令行里带 pattern 的进程（Linux 用 /proc，Windows 用 PowerShell）。"""
    pattern = str(pattern)
    if IS_WIN:
        try:
            ps = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "
                  f"'*{pattern}*' }} | ForEach-Object {{ $_.ProcessId }}")
            r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                               timeout=30, creationflags=NO_WINDOW, encoding="utf-8", errors="replace")
            return [int(x) for x in (r.stdout or "").split() if x.isdigit()]
        except Exception:
            return []
    res = []
    try:
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            try:
                with open(f"/proc/{name}/cmdline", "rb") as f:
                    if pattern.encode() in f.read():
                        res.append(int(name))
            except OSError:
                continue
    except OSError:
        pass
    return res


def browse_roots():
    """文件选择框的起点（Windows 是盘符，Linux 是常用目录）。"""
    if IS_WIN:
        import string
        return [{"name": f"{d}:\\", "path": f"{d}:\\"} for d in string.ascii_uppercase if os.path.isdir(f"{d}:\\")]
    home = os.path.expanduser("~")
    cands = [home, "/root", "/workspace", "/data", "/mnt", "/opt", "/home", "/"]
    res, seen = [], set()
    for p in cands:
        if not p or not os.path.isdir(p):
            continue
        k = os.path.realpath(p)
        if k in seen:
            continue
        seen.add(k)
        res.append({"name": p if p != home else f"{p}（主目录）", "path": p})
    return res


def open_path(path):
    """在文件管理器里打开（Windows 资源管理器 / Linux xdg-open）。"""
    try:
        if IS_WIN:
            subprocess.Popen(["explorer", "/select,", os.path.normpath(path)], creationflags=NO_WINDOW)
        else:
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
        return True
    except Exception:
        return False


def shutdown():
    """队列跑完后关机。"""
    if IS_WIN:
        subprocess.run(["shutdown", "/s", "/t", "120"], creationflags=NO_WINDOW)
    else:
        subprocess.run(["shutdown", "-h", "+2"])


def system_python():
    """能用来创建训练环境的 Python。"""
    for name in ("python3", "python"):
        p = shutil.which(name)
        if p:
            return p
    return sys.executable


def quote_cmd(cmd):
    """把命令拼成能直接复制到终端里的一行（只用于显示 / 日志）。"""
    if IS_WIN:
        return subprocess.list2cmdline([str(c) for c in cmd])
    return " ".join(shlex.quote(str(c)) for c in cmd)

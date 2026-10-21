# -*- coding: utf-8 -*-
"""后台作业：下载模型、安装环境、反推打标等。训练任务另有队列（tasks.py）。

每个作业有自己的日志（内存里保留最后 3000 行）和进度，前端轮询 /api/jobs 显示。"""
import collections
import re
import os
import subprocess
import threading
import time
import traceback
import uuid

NO_WINDOW = 0x08000000 if os.name == "nt" else 0

_jobs = collections.OrderedDict()
_lock = threading.Lock()


class Job:
    def __init__(self, kind, title):
        self.id = uuid.uuid4().hex[:10]
        self.kind = kind
        self.title = title
        self.status = "running"   # running / done / failed / stopped
        self.progress = None      # 0..1
        self.progress_text = ""
        self.log = collections.deque(maxlen=3000)
        self.result = None
        self.error = None
        self.started = time.time()
        self.finished = None
        self.stop_flag = False
        self.proc = None
        self.on_done = None

    def write(self, line):
        line = str(line).rstrip("\r\n")
        if line:
            self.log.append(time.strftime("%H:%M:%S ") + line)

    def set_progress(self, v, text=""):
        self.progress = v
        if text:
            self.progress_text = text

    def run_cmd(self, cmd, cwd=None, env=None, parse=None):
        """跑一个子进程，输出逐行写进日志；返回退出码。parse(line) 可以从输出里解析进度。"""
        self.write("$ " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
        self.proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
        buf = b""
        while True:
            chunk = self.proc.stdout.read1(4096) if hasattr(self.proc.stdout, "read1") else self.proc.stdout.read(1)
            if not chunk:
                break
            buf += chunk
            while True:
                # tqdm 用 \r 刷新同一行，所以 \r 和 \n 都当换行
                j = min([x for x in (buf.find(b"\n"), buf.find(b"\r")) if x >= 0], default=-1)
                if j < 0:
                    break
                line, buf = buf[:j], buf[j + 1:]
                s = re.sub(r"\[[0-9;?]*[A-Za-z]", "", decode_line(line)).strip()
                if not s:
                    continue
                if parse:
                    try:
                        if parse(s) is False:
                            continue
                    except Exception:
                        pass
                self.write(s)
            if self.stop_flag:
                kill_tree(self.proc.pid)
        if buf.strip():
            self.write(buf.decode("utf-8", errors="replace"))
        rc = self.proc.wait()
        self.proc = None
        return rc

    def to_dict(self, log_lines=0):
        d = {"id": self.id, "kind": self.kind, "title": self.title, "status": self.status, "progress": self.progress,
             "progress_text": self.progress_text, "error": self.error, "result": self.result,
             "started": self.started, "finished": self.finished}
        if log_lines:
            d["log"] = list(self.log)[-log_lines:]
        return d


def decode_line(b):
    """子进程输出：Python 程序是 UTF-8；Windows 自带命令 / 系统报错可能是 GBK。"""
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return b.decode("gbk")
        except UnicodeDecodeError:
            return b.decode("utf-8", errors="replace")


def kill_tree(pid):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=NO_WINDOW)
        else:
            os.killpg(os.getpgid(pid), 9)
    except Exception:
        pass


def start(kind, title, fn, *args, on_done=None, **kw):
    """fn(job, *args, **kw) 在后台线程执行；返回值存进 job.result，抛异常算失败。"""
    job = Job(kind, title)
    job.on_done = on_done
    with _lock:
        _jobs[job.id] = job
        while len(_jobs) > 60:
            _jobs.popitem(last=False)

    def run():
        try:
            job.result = fn(job, *args, **kw)
            job.status = "stopped" if job.stop_flag else "done"
        except Exception as e:
            job.status = "stopped" if job.stop_flag else "failed"
            job.error = str(e) or e.__class__.__name__
            job.write("出错：" + job.error)
            job.write(traceback.format_exc().splitlines()[-1])
        job.finished = time.time()
        if job.on_done:
            try:
                job.on_done(job)
            except Exception:
                traceback.print_exc()

    threading.Thread(target=run, daemon=True).start()
    return job


def get(jid):
    return _jobs.get(jid)


def find_running(kind):
    return [j for j in _jobs.values() if j.kind == kind and j.status == "running"]


def listing():
    return [j.to_dict() for j in reversed(_jobs.values())]


def stop(jid):
    j = _jobs.get(jid)
    if j and j.status == "running":
        j.stop_flag = True
        if j.proc:
            kill_tree(j.proc.pid)
    return bool(j)

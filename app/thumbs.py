# -*- coding: utf-8 -*-
"""标签编辑页的缩略图缓存（data/thumbs）。

打开训练集时在后台用训练环境的 Pillow 批量生成；/thumb 请求时如果还没生成好，最多等几秒，
仍然没有就退回原图。训练环境没装时一律用原图。"""
import hashlib
import json
import os
import subprocess
import threading
import time

import paths
import settings

THUMB_DIR = os.path.join(paths.DATA, "thumbs")
SIZE = 320
NO_WINDOW = 0x08000000 if os.name == "nt" else 0
_lock = threading.Lock()
_pending = set()
_failed = set()   # 生成失败的缩略图（坏图 / 格式不支持）：不再反复生成，直接用原图


def thumb_path(src):
    try:
        st = os.stat(src)
    except OSError:
        return None
    key = f"{os.path.abspath(src)}|{st.st_size}|{int(st.st_mtime)}|{SIZE}"
    h = hashlib.sha1(key.encode("utf-8", "surrogatepass")).hexdigest()
    return os.path.join(THUMB_DIR, h[:2], h + ".jpg")


def prepare(files):
    """后台生成一批缩略图（已存在的跳过）。"""
    py = paths.env_python()
    if not py:
        return
    todo = []
    for f in files:
        t = thumb_path(f)
        if t and not os.path.isfile(t) and t not in _pending and t not in _failed:
            todo.append([f, t, SIZE])
    if not todo:
        return
    with _lock:
        _pending.update(t for _, t, _ in todo)

    def run():
        job = os.path.join(paths.DATA, f"thumbs_{os.getpid()}_{threading.get_ident()}.json")
        try:
            with open(job, "w", encoding="utf-8") as fh:
                json.dump(todo, fh, ensure_ascii=False)
            subprocess.run([py, os.path.join(paths.APP, "workers", "thumbs.py"), job], env=settings.child_env(),
                           capture_output=True, creationflags=NO_WINDOW, timeout=600)
        except Exception:
            pass
        finally:
            with _lock:
                _pending.difference_update(t for _, t, _ in todo)
                _failed.update(t for _, t, _ in todo if not os.path.isfile(t))
            try:
                os.remove(job)
            except OSError:
                pass

    threading.Thread(target=run, daemon=True).start()


def get(src, wait=6.0):
    """返回可以直接发给浏览器的缩略图路径；没有就返回 None（调用方退回原图）。"""
    t = thumb_path(src)
    if not t:
        return None
    if os.path.isfile(t):
        return t
    if t not in _pending:
        prepare([src])
    end = time.time() + wait
    while time.time() < end:
        if os.path.isfile(t):
            return t
        if t not in _pending and not os.path.isfile(t):
            break
        time.sleep(0.1)
    return t if os.path.isfile(t) else None


def lookup(src, wait=0.3):
    """不阻塞的版本：返回 (缩略图路径 or None, 还在生成中)。
    (None, False) = 生成不了（没装训练环境 / 坏图），调用方直接用原图。"""
    t = thumb_path(src)
    if not t:
        return None, False
    if os.path.isfile(t):
        return t, False
    if t in _failed:
        return None, False
    if t not in _pending:
        prepare([src])
    end = time.time() + wait
    while time.time() < end and t in _pending:
        if os.path.isfile(t):
            return t, False
        time.sleep(0.05)
    if os.path.isfile(t):
        return t, False
    return None, t in _pending

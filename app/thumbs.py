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
        if t and not os.path.isfile(t) and t not in _pending:
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

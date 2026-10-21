# -*- coding: utf-8 -*-
"""Danbooru 中英标签字典（作者的 wangjue520/danbooru-zh-dict，MIT）。

用途：爬图页的标签输入只能选字典里的 D 站正式标签——中文、英文、别名、日文都能搜，
别名（miku_hatsune）自动换成正式标签（hatsune_miku）。字典只收录图数 ≥50 的标签，
不在字典里的冷门标签再去 Danbooru 在线确认一次。

字典文件第一次用到时自动下载到 data/dicts/（约 17 MB，国内网络走 GitHub 加速代理）。"""
import csv
import math
import os
import threading
import urllib.request

import paths
import settings

URL = "https://github.com/wangjue520/danbooru-zh-dict/raw/main/dist/danbooru_zh_dict.csv"
DIR = os.path.join(paths.DATA, "dicts")
FILE = os.path.join(DIR, "danbooru_zh_dict.csv")
CAT = {"通用": 0, "画师": 1, "作品": 3, "角色": 4, "元标签": 5}

_lock = threading.Lock()
_db = None          # 列表 [(tag, cat, count, zh, syn, aliases_lower, note)]
_by_tag = {}        # tag → 行
_alias = {}         # 别名（小写）→ 正式标签
_state = {"status": "missing", "error": ""}


def status():
    d = dict(_state)
    d["file"] = FILE if os.path.isfile(FILE) else ""
    d["count"] = len(_db) if _db else 0
    return d


def download(job=None):
    os.makedirs(DIR, exist_ok=True)
    tmp = FILE + ".part"
    import netpick
    if job:
        job.write("测试下载线路…")
    ranked = netpick.rank_github(URL)   # GitHub 直连 / 各加速代理 × 直连 / VPN 代理，按速度排
    if job:
        job.write("线路测速：" + netpick.describe(ranked))
    netpick.download_file(job, ranked, tmp, "Danbooru 中英字典", min_size=1_000_000)
    os.replace(tmp, FILE)
    global _db
    with _lock:
        _db = None
    load()
    return status()


def load():
    """读入内存（约 1 秒）。没有字典文件时返回 False。"""
    global _db, _by_tag, _alias
    with _lock:
        if _db is not None:
            return True
        if not os.path.isfile(FILE):
            _state["status"] = "missing"
            return False
        _state["status"] = "loading"
        db, by, alias = [], {}, {}
        with open(FILE, encoding="utf-8-sig", newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if len(row) < 9:
                    continue
                tag, _prompt, cat, cnt, zh, syn, en_al, other_al, note = row[:9]
                try:
                    cnt = int(cnt)
                except ValueError:
                    cnt = 0
                al = [a for a in (en_al + "|" + other_al).split("|") if a]
                item = (tag, CAT.get(cat, 0), cnt, zh, syn.replace("|", "、"), "|".join(al).lower(), note)
                db.append(item)
                by[tag] = item
                for a in en_al.split("|"):
                    a = a.strip().lower()
                    if a and a not in alias:
                        alias[a] = tag
        db.sort(key=lambda x: -x[2])
        _db, _by_tag, _alias = db, by, alias
        _state["status"] = "ready"
        return True


def _row(item, why=""):
    tag, cat, cnt, zh, syn, _al, note = item
    return {"name": tag, "category": cat, "count": cnt, "zh": zh, "syn": syn, "note": note, "why": why}


def search(q, limit=12):
    """中文 / 英文 / 别名 / 日文都能搜，按匹配程度和图数排序。"""
    if not load():
        return []
    q = (q or "").strip().lower()
    if not q:
        return []
    qt = q.replace(" ", "_")
    scored = []
    for it in _db:
        tag, _c, cnt, zh, syn, al, _n = it
        s = 0
        if tag == qt:
            s = 1000
        elif zh == q:
            s = 950
        elif tag.startswith(qt):
            s = 600
        elif zh.startswith(q):
            s = 560
        elif q in syn.split("、"):
            s = 520
        elif qt in al.split("|"):
            s = 500
        elif qt in tag:
            s = 300
        elif q in zh or q in syn:
            s = 260
        elif len(q) >= 2 and q in al:
            s = 120
        if s:
            scored.append((s + math.log10(cnt + 1) * 20, it))
            if len(scored) > 400 and s < 300:
                continue
    scored.sort(key=lambda x: -x[0])
    return [_row(it) for _, it in scored[:limit]]


def resolve(tag):
    """把输入换成 D 站正式标签：正式标签原样返回；别名换成正式名；查不到返回 None。"""
    if not load():
        return None
    t = (tag or "").strip().lower().replace(" ", "_")
    if t in _by_tag:
        return t
    return _alias.get(t)


def info(tags):
    if not load():
        return {}
    return {t: _row(_by_tag[t]) for t in tags if t in _by_tag}


def fuzzy(t, limit=4):
    """拼写近似：去掉下划线后比前缀，越长越像；再用 difflib 兜底。按图数排序。"""
    if not load():
        return []
    flat = (t or "").lower().replace("_", "").replace(" ", "")
    if len(flat) < 3:
        return []
    for n in range(len(flat), 3, -1):
        p = flat[:n]
        hit = [it[0] for it in _db if it[0].replace("_", "").startswith(p)][:limit]   # _db 已按图数排序
        if hit:
            return hit
    import difflib
    return difflib.get_close_matches(t, [it[0] for it in _db[:20000]], limit, 0.75)

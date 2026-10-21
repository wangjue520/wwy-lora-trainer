# -*- coding: utf-8 -*-
"""标签编辑：读取训练集、逐张保存、批量操作（每次批量操作前自动备份，可撤销）。"""
import collections
import json
import os
import re
import shutil
import time

import captionutil as cu
import paths

SENTENCE_WORDS = 7   # 超过这么多个词的片段当成自然语言句子，不计入标签统计


def _is_sentence(t):
    return len(t.split()) >= SENTENCE_WORDS or t.endswith(".")


def load(d, ext=".txt"):
    files = cu.list_images(d)
    items, freq = [], collections.Counter()
    for f in files:
        cap = cu.read_caption(f, ext)
        tags = cu.split_tags(cap or "")
        for t in tags:
            if not _is_sentence(t):
                freq[t] += 1
        try:
            st = os.stat(f)
            size = st.st_size
        except OSError:
            size = 0
        items.append({"path": f, "rel": os.path.relpath(f, d), "caption": cap, "has_caption": cap is not None,
                      "size": size})
    return {"dir": d, "items": items, "tags": freq.most_common(), "count": len(items),
            "no_caption": sum(1 for x in items if not (x["caption"] or "").strip()),
            "undo": bool(_latest_backup(d))}


def _inside(d, f):
    try:
        return os.path.commonpath([os.path.abspath(d), os.path.abspath(f)]) == os.path.abspath(d)
    except ValueError:
        return False


def save_one(d, img, text, ext=".txt"):
    if not _inside(d, img):
        raise ValueError("文件不在训练集里")
    cu.write_caption(img, text, ext)
    return True


# ---------------- 备份 / 撤销 ----------------
def _bk_dir(d):
    key = re.sub(r"[^\w]+", "_", os.path.abspath(d))[-80:]
    p = os.path.join(paths.BACKUPS, key)
    os.makedirs(p, exist_ok=True)
    return p


def _latest_backup(d):
    p = _bk_dir(d)
    fs = sorted(x for x in os.listdir(p) if x.endswith(".json"))
    return os.path.join(p, fs[-1]) if fs else None


def _backup(d, files, ext, label):
    snap = {"label": label, "time": time.time(), "files": {f: cu.read_caption(f, ext) for f in files}}
    p = os.path.join(_bk_dir(d), time.strftime("%Y%m%d_%H%M%S_") + f"{int(time.time() * 1000) % 1000:03d}.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, ensure_ascii=False)
    # 每个训练集最多留 30 份
    allb = sorted(x for x in os.listdir(_bk_dir(d)) if x.endswith(".json"))
    for x in allb[:-30]:
        os.remove(os.path.join(_bk_dir(d), x))


def undo(d, ext=".txt"):
    p = _latest_backup(d)
    if not p:
        return {"ok": False, "message": "没有可以撤销的批量操作"}
    snap = json.load(open(p, encoding="utf-8"))
    for f, txt in snap["files"].items():
        if txt is None:
            try:
                os.remove(cu.caption_path(f, ext))
            except OSError:
                pass
        else:
            with open(cu.caption_path(f, ext), "w", encoding="utf-8") as fh:
                fh.write(txt)
    os.remove(p)
    return {"ok": True, "message": f"已撤销：{snap.get('label')}（{len(snap['files'])} 个文件）"}


# ---------------- 批量操作 ----------------
def batch(d, op, files=None, ext=".txt", **kw):
    """op: add / remove / replace / find_replace / dedupe / trigger / sort / clear"""
    targets = [f for f in (files or cu.list_images(d)) if _inside(d, f)]
    if not targets:
        return {"changed": 0}
    labels = {"add": "添加标签", "remove": "删除标签", "replace": "替换标签", "find_replace": "查找替换",
              "dedupe": "去除重复标签", "trigger": "设置触发词", "sort": "整理标签顺序", "clear": "清空标签"}
    _backup(d, targets, ext, labels.get(op, op))
    changed = 0
    freq = collections.Counter()
    if op == "sort":
        for f in cu.list_images(d):
            for t in cu.split_tags(cu.read_caption(f, ext) or ""):
                freq[t] += 1
    for f in targets:
        old = cu.read_caption(f, ext)
        cap = old or ""
        tags = cu.split_tags(cap)
        if op == "add":
            new = [t for t in cu.split_tags(kw.get("tags", "")) if t not in tags]
            tags = new + tags if kw.get("position") == "front" else tags + new
            cap = ", ".join(tags)
        elif op == "remove":
            rm = set(cu.split_tags(kw.get("tags", "")))
            tags = [t for t in tags if t not in rm]
            cap = ", ".join(tags)
        elif op == "replace":
            a, b = kw.get("old", "").strip(), kw.get("new", "").strip()
            out = []
            for t in tags:
                t2 = b if t == a else t
                if t2 and t2 not in out:
                    out.append(t2)
            cap = ", ".join(out)
        elif op == "find_replace":
            find, rep = kw.get("find", ""), kw.get("replace", "")
            if not find:
                continue
            if kw.get("regex"):
                cap = re.sub(find, rep, cap)
            else:
                cap = cap.replace(find, rep)
            cap = re.sub(r"\s*,\s*(,\s*)+", ", ", cap).strip(" ,")
        elif op == "dedupe":
            out = []
            for t in tags:
                if t.lower() not in [x.lower() for x in out]:
                    out.append(t)
            cap = ", ".join(out)
        elif op == "trigger":
            trig = cu.split_tags(kw.get("trigger", ""))
            rest = [t for t in tags if t not in trig and t not in cu.split_tags(kw.get("old_trigger", ""))]
            cap = ", ".join(trig + rest)
        elif op == "sort":
            keep = int(kw.get("keep", 1))
            head, rest = tags[:keep], tags[keep:]
            words = [t for t in rest if not _is_sentence(t)]
            sents = [t for t in rest if _is_sentence(t)]
            words.sort(key=lambda t: (-freq[t], t))
            cap = ", ".join(head + words + sents)
        elif op == "clear":
            cap = ""
        if cap != (old or ""):
            cu.write_caption(f, cap, ext)
            changed += 1
    return {"changed": changed}


def remove_image(d, img, ext=".txt"):
    """把图片和标签移到训练集里的「_已移除」文件夹（不会真的删掉）。"""
    if not _inside(d, img):
        raise ValueError("文件不在训练集里")
    dst = os.path.join(d, "_已移除")
    os.makedirs(dst, exist_ok=True)
    for f in (img, cu.caption_path(img, ext)):
        if os.path.exists(f):
            shutil.move(f, os.path.join(dst, os.path.basename(f)))
    return True

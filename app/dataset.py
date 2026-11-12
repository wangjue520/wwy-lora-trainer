# -*- coding: utf-8 -*-
"""标签编辑：读取训练集、逐张保存、批量操作（每次批量操作前自动备份，可撤销）。

标签比较一律用 captionutil.norm_tag：大小写、下划线 / 空格、转义括号不同的写法算同一个标签。
标签文件按「标签段 + 自然语言段」拆开处理（captionutil.parse_caption），批量增删只动标签段，句子原样保留。"""
import collections
import json
import os
import re
import shutil
import time

import captionutil as cu
import paths

SENTENCE_WORDS = cu.SENTENCE_WORDS
MAX_BACKUPS = 100   # 每个训练集最多留这么多份批量备份


def _is_sentence(t):
    return cu.is_prose(t)


def _trigger_main(d):
    try:
        import triggers
        info = triggers.get(d) or {}
    except Exception:
        return None
    return info


def _guess_trigger(caps):
    """没有触发词记录时（老版本「设置触发词」不写记录）：≥90% 的标签文件第一个标签都一样（至少 3 张），就当它是触发词。"""
    firsts, n = collections.Counter(), 0
    for cap in caps:
        tags = cu.parse_caption(cap or "")[0]
        if tags:
            n += 1
            firsts[cu.norm_tag(tags[0])] += 1
    if n >= 3 and firsts:
        k, c = firsts.most_common(1)[0]
        if c >= n * 0.9:
            return k
    return ""


def trigger_info(d, caps=None, ext=".txt"):
    """{"main", "features", "outfit", "guessed"}；没有记录也推测不出来时返回 None。"""
    info = _trigger_main(d) or {}
    if info.get("main"):
        return {"main": info["main"], "features": info.get("features") or [], "outfit": info.get("outfit") or [], "guessed": False}
    if caps is None:
        caps = [cu.read_caption(f, ext) for f in cu.list_images(d)]
    g = _guess_trigger(caps)
    return {"main": g, "features": [], "outfit": [], "guessed": True} if g else None


def load(d, ext=".txt"):
    if not d or not os.path.isdir(d):
        raise ValueError("训练集文件夹不存在：" + (d or ""))
    files = cu.list_images(d)
    items, freq = [], collections.Counter()
    for f in files:
        cap, st = cu.read_caption_ex(f, ext)
        tags, _text, _sep = cu.parse_caption(cap or "")
        for t in tags:
            if "\ufffd" not in t:   # 编码不对的文件里读坏的标签不计入统计
                freq[t] += 1
        items.append({"path": f, "rel": os.path.relpath(f, d), "caption": cap, "has_caption": cap is not None,
                      "enc": st})
    return {"dir": d, "items": items, "tags": freq.most_common(), "count": len(items),
            "no_caption": sum(1 for x in items if not (x["caption"] or "").strip()),
            "undo": bool(_latest_backup(d)),
            "trigger": trigger_info(d, [x["caption"] for x in items if x["enc"] != "bad"])}


def _inside(d, f):
    try:
        return os.path.commonpath([os.path.abspath(d), os.path.abspath(f)]) == os.path.abspath(d)
    except ValueError:
        return False


def _same(a, b):
    return (a or "").strip() == (b or "").strip()


def save_one(d, img, text, ext=".txt", base=None):
    """base：界面上次从文件读到 / 写进去的内容。文件被别的程序改过（比如反推正在写）时不覆盖，把文件里的新内容返回给界面。"""
    if not _inside(d, img):
        raise ValueError("文件不在训练集里")
    cur, st = cu.read_caption_ex(img, ext)
    if st == "bad":
        raise ValueError("这个标签文件不是 UTF-8 编码，直接保存会变成乱码。先在「标签体检」里把它转成 UTF-8")
    if base is not None and cur is not None and not _same(cur, base) and not _same(cur, text):
        return {"ok": False, "conflict": True, "current": cur}
    cu.write_caption(img, text, ext)
    return {"ok": True}


# ---------------- 备份 / 撤销 ----------------
def _bk_dir(d):
    key = re.sub(r"[^\w]+", "_", os.path.abspath(d))[-80:]
    p = os.path.join(paths.BACKUPS, key)
    os.makedirs(p, exist_ok=True)
    return p


def _backups(d):
    return sorted(x for x in os.listdir(_bk_dir(d)) if x.endswith(".json"))


def _latest_backup(d):
    fs = _backups(d)
    return os.path.join(_bk_dir(d), fs[-1]) if fs else None


def _backup(d, files, ext, label, after=None):
    """files：要备份的图片；after：{图片: 改完的内容}，撤销前用来判断之后有没有手动改过。返回备份 id。"""
    snap = {"label": label, "time": time.time(), "files": {f: cu.read_caption(f, ext) for f in files}}
    if after is not None:
        snap["after"] = after
    bid = time.strftime("%Y%m%d_%H%M%S_") + f"{int(time.time() * 1000) % 1000:03d}.json"
    p = os.path.join(_bk_dir(d), bid)
    while os.path.exists(p):   # 同一毫秒连续两次
        bid = bid[:-5] + "_.json"
        p = os.path.join(_bk_dir(d), bid)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, ensure_ascii=False)
    for x in _backups(d)[:-MAX_BACKUPS]:
        try:
            os.remove(os.path.join(_bk_dir(d), x))
        except OSError:
            pass
    return bid


def history(d):
    out = []
    for x in reversed(_backups(d)):
        try:
            snap = json.load(open(os.path.join(_bk_dir(d), x), encoding="utf-8"))
        except Exception:
            continue
        out.append({"id": x, "label": snap.get("label") or "", "time": snap.get("time") or 0, "count": len(snap.get("files") or {})})
    return out


def undo(d, ext=".txt", bid=None, dry=False):
    """撤销到某次批量操作之前（bid 为空 = 最近一次）。比它新的批量操作会一起撤销（从新到旧依次还原）。
    dry：只检查，返回这些批量操作之后又被手动改过的文件数。"""
    allb = _backups(d)
    if not allb:
        return {"ok": False, "message": "没有可以撤销的批量操作"}
    bid = bid or allb[-1]
    if bid not in allb:
        return {"ok": False, "message": "这条撤销记录已经不在了（可能已经撤销过）"}
    chain = list(reversed(allb[allb.index(bid):]))   # 从新到旧
    snaps = []
    for x in chain:
        try:
            snaps.append((x, json.load(open(os.path.join(_bk_dir(d), x), encoding="utf-8"))))
        except Exception:
            snaps.append((x, {"files": {}}))
    if dry:
        seen, modified = set(), []
        for _x, s in snaps:
            for f, after in (s.get("after") or {}).items():
                if f in seen:
                    continue
                seen.add(f)
                if not _same(cu.read_caption(f, ext), after):
                    modified.append(f)
        return {"ok": True, "modified": len(modified), "files": modified[:50], "count": len(chain)}
    items = {}
    for x, s in snaps:
        for f, txt in (s.get("files") or {}).items():
            if txt is None:
                try:
                    os.remove(cu.caption_path(f, ext))
                except OSError:
                    pass
                items[f] = None
            else:
                cu.write_caption(f, txt, ext)
                items[f] = txt
        try:
            os.remove(os.path.join(_bk_dir(d), x))
        except OSError:
            pass
    first = snaps[-1][1]
    more = f"，以及之后的 {len(snaps) - 1} 次批量操作" if len(snaps) > 1 else ""
    return {"ok": True, "message": f"已撤销：{first.get('label')}{more}（{len(items)} 个文件）", "items": items,
            "undo": bool(_latest_backup(d))}


# ---------------- 批量操作 ----------------
LABELS = {"add": "添加标签", "remove": "删除标签", "replace": "替换标签", "rename_map": "统一标签写法", "find_replace": "查找替换",
          "dedupe": "去除重复标签", "trigger": "设置触发词", "sort": "整理标签顺序", "clear": "清空标签"}


def _dedupe(tags):
    out, seen = [], set()
    for t in tags:
        k = cu.norm_tag(t)
        if k and k not in seen:
            seen.add(k)
            out.append(t)
    return out


def _lead(tags, d, kw):
    """「加到最前」插在哪：开头的触发词不动（训练时一般固定前 1 个标签不打乱）。"""
    if "keep" in kw:
        return max(0, min(int(kw.get("keep") or 0), len(tags)))
    main = kw.get("_trigger") or ""
    return 1 if main and tags and cu.norm_tag(tags[0]) == main else 0


def _apply(op, d, tags, text, sep, cap, kw, freq):
    """返回 (新标签, 新句子, 新分隔符, 新整条 or None)；整条不为 None 时直接用整条（查找替换 / 清空）。"""
    if op == "add":
        new = cu.split_input(kw.get("tags", ""))
        front = kw.get("position") == "front"
        if front:
            want = {cu.norm_tag(t) for t in new}
            lead = _lead(tags, d, kw)
            head, rest = tags[:lead], tags[lead:]
            have = {cu.norm_tag(t): t for t in rest}
            moved = [have.get(cu.norm_tag(t), t) for t in new]   # 已经有的标签挪到最前（保留原来的写法）
            head = [t for t in head if cu.norm_tag(t) not in want or t in moved]
            tags = _dedupe(head + moved + [t for t in rest if cu.norm_tag(t) not in want])
        else:
            have = {cu.norm_tag(t) for t in tags}
            tags = tags + [t for t in _dedupe(new) if cu.norm_tag(t) not in have]
    elif op == "remove":
        rm = {cu.norm_tag(t) for t in cu.split_input(kw.get("tags", ""))}
        tags = [t for t in tags if cu.norm_tag(t) not in rm]
    elif op in ("replace", "rename_map"):
        mp = {cu.norm_tag(kw.get("old", "")): kw.get("new", "").strip()} if op == "replace" else \
            {cu.norm_tag(k): (v or "").strip() for k, v in (kw.get("map") or {}).items()}
        tags = _dedupe([mp.get(cu.norm_tag(t), t) for t in tags])
    elif op == "dedupe":
        tags = _dedupe(tags)
    elif op == "trigger":
        trig = cu.split_input(kw.get("trigger", ""))
        drop = {cu.norm_tag(t) for t in trig + cu.split_input(kw.get("old_trigger", ""))}
        tags = trig + [t for t in tags if cu.norm_tag(t) not in drop]
    elif op == "sort":
        keep = int(kw.get("keep", 1))
        head, rest = tags[:keep], tags[keep:]
        rest.sort(key=lambda t: (-freq[cu.norm_tag(t)], cu.norm_tag(t)))
        tags = head + rest
    elif op == "find_replace":
        find, rep = kw.get("find", ""), kw.get("replace", "")
        if not find:
            return tags, text, sep, cap
        if kw.get("regex"):
            try:
                new = re.sub(find, rep, cap)
            except re.error as e:
                raise ValueError(f"正则表达式写错了：{e}")
        else:
            new = cap.replace(find, rep)
        new = re.sub(r"\s*,\s*(,\s*)+", ", ", new).strip(" ,")
        return tags, text, sep, new
    elif op == "clear":
        return [], "", "", ""
    else:
        raise ValueError("不认识的批量操作：" + op)
    return tags, text, sep, None


def batch(d, op, files=None, ext=".txt", **kw):
    """op: add / remove / replace / rename_map / find_replace / dedupe / trigger / sort / clear

    先把每个文件改完的内容都算出来，只备份、只写真正有变化的文件。
    返回 changed（改了几个）、total（范围内几张）、skipped（编码不对没敢改的）、items（{图片: 新内容}）、message。"""
    targets = [f for f in (cu.list_images(d) if files is None else files) if _inside(d, f)]
    if not targets:
        return {"changed": 0, "total": 0, "items": {}, "skipped": [], "message": "范围里没有图片"}
    freq = collections.Counter()
    if op == "sort":
        for f in cu.list_images(d):
            for t in set(cu.norm_tag(x) for x in cu.parse_caption(cu.read_caption(f, ext) or "")[0]):
                freq[t] += 1
    if op == "add" and kw.get("position") == "front" and "keep" not in kw:
        ti = trigger_info(d, ext=ext)
        kw = dict(kw, _trigger=cu.norm_tag(ti["main"]) if ti else "")
    plan, skipped, had = {}, [], 0
    want = {cu.norm_tag(t) for t in cu.split_input(kw.get("tags", ""))}
    for f in targets:
        old, st = cu.read_caption_ex(f, ext)
        if st == "bad":
            skipped.append(f)
            continue
        cap = old or ""
        tags, text, sep = cu.parse_caption(cap)
        if want and any(cu.norm_tag(t) in want for t in tags):
            had += 1
        nt, ntext, nsep, whole = _apply(op, d, list(tags), text, sep, cap, kw, freq)
        new = whole if whole is not None else (cap if (nt == tags and ntext == text) else cu.build_caption(nt, ntext, nsep))
        if not _same(new, cap) or (op == "clear" and old):
            plan[f] = new
    total = len(targets)
    label = LABELS.get(op, op)
    if not plan:
        msg = _nothing_msg(op, total, had, kw)
        return {"changed": 0, "total": total, "items": {}, "skipped": skipped, "message": msg + _skip_msg(skipped)}
    bid = _backup(d, list(plan), ext, label + _label_detail(op, kw), after=plan)
    for f, new in plan.items():
        cu.write_caption(f, new, ext)
    msg = f"{label}：改了 {len(plan)} / {total} 张"
    if op == "remove":
        msg = f"删除：{len(plan)} 张里有这些标签，已删掉" + (f"；{total - len(plan)} 张本来就没有" if total > len(plan) else "")
    elif op == "add":
        msg = f"添加：改了 {len(plan)} 张" + (f"；{total - len(plan)} 张已经有了（位置也对）" if total > len(plan) else "")
    return {"changed": len(plan), "total": total, "items": plan, "skipped": skipped, "undo_id": bid,
            "message": msg + _skip_msg(skipped)}


def _label_detail(op, kw):
    if op in ("add", "remove"):
        s = ", ".join(cu.split_input(kw.get("tags", "")))
        return f"「{s[:40]}」" if s else ""
    if op == "replace":
        return f"「{kw.get('old', '')}」→「{kw.get('new', '')}」"
    if op == "find_replace":
        return f"「{kw.get('find', '')[:30]}」"
    return ""


def _nothing_msg(op, total, had, kw):
    if op == "remove":
        return f"范围内 {total} 张都没有这些标签（大小写、下划线 / 空格写法不同也算同一个标签）"
    if op == "add":
        return f"范围内 {total} 张都已经有这些标签了" + ("，位置也已经在最前" if kw.get("position") == "front" else "")
    if op == "find_replace":
        return f"范围内 {total} 张都没找到「{kw.get('find', '')}」"
    return f"范围内 {total} 张都不需要修改"


def _skip_msg(skipped):
    return f"。另有 {len(skipped)} 个标签文件不是 UTF-8 编码，没敢改（用「标签体检」转换）" if skipped else ""


def convert_encoding(d, files, encoding, ext=".txt"):
    """把不是 UTF-8 的标签文件按指定编码读出来、存成 UTF-8；原文件先原样复制到备份文件夹。"""
    keep = os.path.join(_bk_dir(d), "encoding_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(keep, exist_ok=True)
    items, failed = {}, []
    for f in files:
        if not _inside(d, f):
            continue
        p = cu.caption_path(f, ext)
        try:
            raw = open(p, "rb").read()
            if raw.startswith(b"\xef\xbb\xbf"):
                raw = raw[3:]
            text = raw.decode(encoding)
        except (OSError, UnicodeDecodeError, LookupError):
            failed.append(f)
            continue
        shutil.copy2(p, os.path.join(keep, f"{len(items):05d}_" + os.path.basename(p)))
        cu.write_caption(f, text, ext)
        items[f] = text
    msg = f"已把 {len(items)} 个标签文件转成 UTF-8（原文件备份在 {keep}）"
    if failed:
        msg += f"；{len(failed)} 个用这个编码也读不出来，没动"
    return {"changed": len(items), "items": items, "message": msg}


def strip_bom(d, files, ext=".txt"):
    items = {}
    for f in files:
        if not _inside(d, f):
            continue
        txt, st = cu.read_caption_ex(f, ext)
        if st == "bom":
            cu.write_caption(f, txt, ext)
            items[f] = txt
    return {"changed": len(items), "items": items, "message": f"已去掉 {len(items)} 个标签文件开头的 BOM"}


def remove_image(d, img, ext=".txt"):
    """把图片和标签移到训练集里的「_已移除」文件夹（不会真的删掉）。"""
    if not _inside(d, img):
        raise ValueError("文件不在训练集里")
    dst = os.path.join(d, "_已移除")
    os.makedirs(dst, exist_ok=True)
    pair = [img, cu.caption_path(img, ext)]
    stem = os.path.join(dst, os.path.splitext(os.path.basename(img))[0])
    suffix, k = "", 2
    # 不同子文件夹里有同名文件：图片和标签一起加同一个序号，不覆盖之前移出的
    while any(os.path.exists(stem + suffix + os.path.splitext(f)[1]) for f in pair):
        suffix, k = f"_{k}", k + 1
    for f in pair:
        if os.path.exists(f):
            shutil.move(f, stem + suffix + os.path.splitext(f)[1])
    return True

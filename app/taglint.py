# -*- coding: utf-8 -*-
"""标签体检：训练前把训练集的标签问题找出来，能自动修的给出一键修复（走 dataset 的批量操作，可撤销）。

问题：编码（不是 UTF-8 / 带 BOM）、同一个标签多种写法、别名没换成正式名、字典里没有（疑似拼错）、
同一张图里重复、人数标签互相矛盾、触发词缺失或不在最前、标签太长（估算 token）、没有标签的图，
以及几乎每张都有 / 只出现一次的标签（仅提示）。"""
import collections
import re

import captionutil as cu
import dataset

# 人数标签：同一张图里出现两个不同的数就是矛盾
_GIRLS = ["1girl", "2girls", "3girls", "4girls", "5girls", "6+girls"]
_BOYS = ["1boy", "2boys", "3boys", "4boys", "5boys", "6+boys"]
_MULTI = {"multiple girls", "multiple boys", "2girls", "3girls", "4girls", "5girls", "6+girls", "2boys", "3boys", "4boys", "5boys", "6+boys"}


def _tokens(cap):
    """粗略估算 token 数（CLIP / T5 这类分词器大致是：一个英文单词 1~2 个 token，标点 1 个）。"""
    words = re.findall(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]", cap or "")
    return int(sum(1 if len(w) <= 6 else 2 for w in words) * 1.05) + 2


def _style(stats):
    """训练集里多词标签的写法习惯：空格（long hair）还是下划线（long_hair）。"""
    sp = sum(n for raw, n in stats if " " in raw)
    us = sum(n for raw, n in stats if "_" in raw and len(raw) > 3)
    return "_" if us > sp else " "


def _in_style(tag, style):
    return tag.replace("_", " ") if style == " " and len(tag) > 3 else tag


def run(d, ext=".txt", limit=None, trigger=None):
    data = dataset.load(d, ext)
    items = data["items"]
    n = len(items)
    issues = []
    if not n:
        return {"issues": [], "count": 0}
    info = data.get("trigger") or {}
    main = cu.norm_tag(trigger or info.get("main") or "")

    parsed = {}
    raw_forms = collections.defaultdict(collections.Counter)   # norm → {原写法: 张数}
    df = collections.Counter()                                 # norm → 张数
    for it in items:
        tags, text, _sep = cu.parse_caption(it["caption"] or "")
        parsed[it["path"]] = (tags, text)
        seen = set()
        for t in tags:
            k = cu.norm_tag(t)
            raw_forms[k][t] += 1
            if k not in seen:
                df[k] += 1
                seen.add(k)
    stats = [(raw, c) for k in raw_forms for raw, c in raw_forms[k].items()]
    style = _style(stats)

    def disp(k):   # 这个标签在训练集里最常见的写法
        return raw_forms[k].most_common(1)[0][0] if k in raw_forms else k

    # 1. 编码
    bad = [it["path"] for it in items if it.get("enc") == "bad"]
    if bad:
        prev = {}
        for enc, name in (("gb18030", "GBK（简体中文 Windows）"), ("cp932", "Shift-JIS（日文 Windows）"), ("big5", "Big5（繁体）")):
            ok, sample = 0, ""
            for f in bad:
                try:
                    raw = open(cu.caption_path(f, ext), "rb").read()
                    s = raw.decode(enc)
                    ok += 1
                    if not sample and re.search(r"[^\x00-\x7f]", s):
                        sample = s.strip()[:120]
                except (OSError, UnicodeDecodeError):
                    pass
            if ok:
                prev[enc] = {"name": name, "ok": ok, "sample": sample}
        issues.append({"code": "encoding", "level": "bad", "title": f"{len(bad)} 个标签文件不是 UTF-8 编码",
                       "desc": "训练程序按 UTF-8 读标签，这些文件里的中文 / 日文会变成乱码；标签编辑也不会改它们（改了会写坏）。"
                               "看下面的预览哪种编码读出来是对的，就按哪种转换（原文件会先备份）。",
                       "files": bad, "encodings": prev,
                       "fixes": [{"id": "convert", "label": f"按 {v['name']} 转成 UTF-8", "data": {"encoding": e, "files": bad}}
                                 for e, v in prev.items()]})
    bom = [it["path"] for it in items if it.get("enc") == "bom"]
    if bom:
        issues.append({"code": "bom", "level": "warn", "title": f"{len(bom)} 个标签文件开头带 BOM",
                       "desc": "记事本另存的 UTF-8 文件开头会多一个看不见的字符，训练时第一个标签（往往是触发词）会被读成另一个词。",
                       "files": bom, "fixes": [{"id": "strip_bom", "label": "去掉 BOM", "data": {"files": bom}}]})

    # 2. 没有标签
    empty = [it["path"] for it in items if not (it["caption"] or "").strip()]
    if empty:
        issues.append({"code": "empty", "level": "warn", "title": f"{len(empty)} 张图没有标签",
                       "desc": "训练时这些图只能学到触发词（或什么都学不到）。可以去「标签反推」补上，或者移出训练集。", "files": empty, "fixes": []})

    # 3. 同一个标签多种写法
    variants = [(k, raw_forms[k]) for k in raw_forms if len(raw_forms[k]) > 1]
    if variants:
        mp, rows, files = {}, [], set()
        for k, forms in sorted(variants, key=lambda x: -sum(x[1].values())):
            best = max(forms, key=lambda f: (f == f.lower() and "\\" not in f, forms[f]))   # 优先全小写、没转义的写法
            to = _in_style(best, style)
            rows.append({"forms": forms.most_common(), "to": to})
            for f in forms:
                if f != to:
                    mp[f] = to
        for p, (tags, _t) in parsed.items():
            if any(t in mp for t in tags):
                files.add(p)
        issues.append({"code": "variants", "level": "warn", "title": f"{len(variants)} 个标签有好几种写法",
                       "desc": "大小写、下划线 / 空格、转义括号不同，训练时会被当成不同的词，学习效果被分散。",
                       "rows": rows, "files": sorted(files),
                       "fixes": [{"id": "rename", "label": "统一成最常见的写法", "data": {"map": mp}}]})

    # 4. 同一张图里重复
    dup = [p for p, (tags, _t) in parsed.items() if len({cu.norm_tag(t) for t in tags}) < len(tags)]
    if dup:
        issues.append({"code": "dupes", "level": "warn", "title": f"{len(dup)} 张图里有重复的标签",
                       "desc": "重复的标签会挤占 token，也会让这个概念的权重变高。", "files": dup,
                       "fixes": [{"id": "dedupe", "label": "去掉重复", "data": {"files": dup}}]})

    # 5. 人数标签矛盾
    conflict = []
    for p, (tags, _t) in parsed.items():
        ks = {cu.norm_tag(t) for t in tags}
        g = [x for x in _GIRLS if x in ks]
        b = [x for x in _BOYS if x in ks]
        if len(g) > 1 or len(b) > 1 or ("solo" in ks and ks & _MULTI):
            conflict.append(p)
    if conflict:
        issues.append({"code": "conflict", "level": "warn", "title": f"{len(conflict)} 张图的人数标签互相矛盾",
                       "desc": "比如同时有 1girl 和 2girls，或者 solo 和 multiple girls。多半是反推看错了，点「筛选这些图」逐张看。",
                       "files": conflict, "fixes": []})

    # 6. 触发词
    if main:
        capmap = {it["path"]: it["caption"] for it in items}
        miss = [p for p, (tags, _t) in parsed.items() if (capmap[p] or "").strip() and (not tags or cu.norm_tag(tags[0]) != main)]
        if miss:
            trig = disp(main) if main in raw_forms else (trigger or info.get("main"))
            guess = "（没有触发词记录，按 90% 以上的图第一个标签都是它推测的）" if info.get("guessed") else ""
            issues.append({"code": "trigger", "level": "warn", "title": f"{len(miss)} 张图的第一个标签不是触发词「{trig}」",
                           "desc": "训练时一般固定前 1 个标签不打乱，触发词不在最前面就学不到它。" + guess, "files": miss,
                           "fixes": [{"id": "trigger", "label": f"把「{trig}」放到最前面", "data": {"trigger": trig, "files": miss}}]})

    # 7. 太长
    if limit:
        long_ = [it["path"] for it in items if _tokens(it["caption"]) > int(limit)]
        if long_:
            issues.append({"code": "long", "level": "warn", "title": f"{len(long_)} 张图的标签估计超过 {limit} 个 token",
                           "desc": "超出的部分训练时会被截掉（估算值，大致准确）。可以删掉不重要的标签，或者在高级参数里调大标签最大长度。",
                           "files": long_, "fixes": []})

    # 8. 字典：别名 / 疑似拼错
    try:
        import tagdict
        have_dict = tagdict.load()
    except Exception:
        have_dict = False
    skip = {main} | {cu.norm_tag(t) for t in (info.get("features") or []) + (info.get("outfit") or [])}
    if not have_dict:
        issues.append({"code": "nodict", "level": "info", "title": "没有中英字典，没检查拼写和别名",
                       "desc": "下载 Danbooru 中英字典（约 17 MB）后，能找出拼错的标签和没换成正式名的别名，标签也能显示中文。",
                       "files": [], "fixes": [{"id": "dict_download", "label": "下载字典", "data": {}}]})
    else:
        alias, unknown = [], []
        for k in raw_forms:
            if k in skip or not k or _has_cjk(k) or cu.is_prose(k):
                continue
            q = k.replace(" ", "_")
            if q in tagdict._by_tag:
                continue
            r = tagdict.resolve(q)
            if r and r != q:
                alias.append({"tag": disp(k), "to": _in_style(r, style), "count": df[k]})
            elif not r:
                unknown.append(k)
        if alias:
            alias.sort(key=lambda x: -x["count"])
            files = [p for p, (tags, _t) in parsed.items() if any(cu.norm_tag(t) in {cu.norm_tag(a["tag"]) for a in alias} for t in tags)]
            issues.append({"code": "alias", "level": "warn", "title": f"{len(alias)} 个标签是别名，不是 D 站正式写法",
                           "desc": "底模是用正式标签训练的，用正式写法 LoRA 更容易和底模已有的知识对上。",
                           "rows": alias, "files": files,
                           "fixes": [{"id": "rename", "label": "全部换成正式写法", "data": {"map": {a["tag"]: a["to"] for a in alias}}}]})
        if unknown:
            unknown.sort(key=lambda k: -df[k])
            rows = []
            for n, k in enumerate(unknown[:200]):
                sug = []
                if n < 30:   # 找近似写法要扫一遍 12 万条字典，只给出现最多的 30 个找（每个约 0.1 秒）
                    try:
                        sug = [x["name"] for x in tagdict.search(k, 4)] or tagdict.fuzzy(k.replace(" ", "_"), 3)
                    except Exception:
                        sug = []
                sug = [_in_style(s, style) for s in sug if cu.norm_tag(s) != k][:3]
                rows.append({"tag": disp(k), "count": df[k], "suggest": sug})
            files = [p for p, (tags, _t) in parsed.items() if any(cu.norm_tag(t) in set(unknown) for t in tags)]
            issues.append({"code": "unknown", "level": "info", "title": f"{len(unknown)} 个标签字典里没有",
                           "desc": "可能是拼错了，也可能是自己加的触发词 / 冷门标签（字典只收录图数 ≥50 的标签）。点建议的写法就整体改名。",
                           "rows": rows, "files": files, "fixes": []})

    # 9. 仅提示：几乎每张都有 / 只出现一次
    if n >= 10:
        hi = [{"tag": disp(k), "count": c} for k, c in df.most_common() if c >= n * 0.95 and k != main]
        if hi:
            issues.append({"code": "frequent", "level": "info", "title": f"{len(hi)} 个标签几乎每张都有",
                           "desc": "每张都有的标签会和触发词一起被学成「这个 LoRA 的一部分」。想让它变成 LoRA 自带的特征就删掉；想以后能用提示词控制就留着。",
                           "rows": hi, "files": [], "fixes": []})
        once = [{"tag": disp(k), "count": 1} for k, c in df.items() if c == 1 and not cu.is_prose(k)]
        if len(once) >= 5:
            issues.append({"code": "once", "level": "info", "title": f"{len(once)} 个标签只出现在 1 张图里",
                           "desc": "只出现一次的标签基本学不到，多数可以不管；数量特别多时，可能是反推阈值太低。",
                           "rows": once[:300], "files": [], "fixes": []})
    order = {"bad": 0, "warn": 1, "info": 2}
    issues.sort(key=lambda x: order[x["level"]])
    return {"issues": issues, "count": n, "style": style}


def _has_cjk(s):
    return bool(re.search(r"[぀-ヿ㐀-鿿가-힯]", s))


def fix(d, fid, data, ext=".txt"):
    data = data or {}
    if fid == "convert":
        return dataset.convert_encoding(d, data.get("files") or [], data.get("encoding") or "gb18030", ext)
    if fid == "strip_bom":
        return dataset.strip_bom(d, data.get("files") or [], ext)
    if fid == "rename":
        return dataset.batch(d, "rename_map", None, ext, map=data.get("map") or {})
    if fid == "dedupe":
        return dataset.batch(d, "dedupe", data.get("files") or None, ext)
    if fid == "trigger":
        return dataset.batch(d, "trigger", data.get("files") or None, ext, trigger=data.get("trigger") or "")
    if fid == "remove":
        return dataset.batch(d, "remove", data.get("files") or None, ext, tags=data.get("tags") or "")
    raise ValueError("不认识的修复：" + fid)

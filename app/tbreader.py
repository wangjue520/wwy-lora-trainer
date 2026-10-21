# -*- coding: utf-8 -*-
"""只用标准库读取 TensorBoard 事件文件（events.out.tfevents.*），取出标量曲线给训练页画图。

文件格式：TFRecord（每条 = 8 字节长度 + 4 字节 CRC + 数据 + 4 字节 CRC），数据是 protobuf 编码的 Event：
  Event   { double wall_time = 1; int64 step = 2; Summary summary = 5; }
  Summary { repeated Value value = 1; }
  Value   { string tag = 1; float simple_value = 2; TensorProto tensor = 8; }
  TensorProto { float float_val = 5 (packed); double double_val = 6 (packed); bytes tensor_content = 4; int dtype = 1; }
只解析用得到的字段，其余跳过。"""
import glob
import os
import struct

_cache = {}   # 文件路径 → (大小, 已解析的字节数, {tag: [[step, value], ...]})


def _varint(b, i):
    r = shift = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << shift
        if c < 0x80:
            return r, i
        shift += 7


def _fields(b):
    """遍历一段 protobuf，产出 (字段号, 线型, 值)。"""
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1:
            v = b[i:i + 8]
            i += 8
        elif wt == 2:
            ln, i = _varint(b, i)
            v = b[i:i + ln]
            i += ln
        elif wt == 5:
            v = b[i:i + 4]
            i += 4
        else:   # 不认识的线型（老式 group），后面没法继续解析
            return
        yield fn, wt, v


def _tensor_value(b):
    dtype, content = 1, None
    for fn, wt, v in _fields(b):
        if fn == 1 and wt == 0:
            dtype = v
        elif fn == 5:   # float_val
            return struct.unpack("<f", v[:4])[0] if wt == 2 or wt == 5 else None
        elif fn == 6:   # double_val
            return struct.unpack("<d", v[:8])[0]
        elif fn == 4 and wt == 2:
            content = v
    if content:
        if dtype == 2 and len(content) >= 8:   # DT_DOUBLE
            return struct.unpack("<d", content[:8])[0]
        if len(content) >= 4:
            return struct.unpack("<f", content[:4])[0]
    return None


def _parse_event(b, out):
    step = 0
    summary = None
    for fn, wt, v in _fields(b):
        if fn == 2 and wt == 0:
            step = v
        elif fn == 5 and wt == 2:
            summary = v
    if summary is None:
        return
    for fn, wt, v in _fields(summary):
        if fn != 1 or wt != 2:
            continue
        tag, val = None, None
        for f2, w2, v2 in _fields(v):
            if f2 == 1 and w2 == 2:
                tag = v2.decode("utf-8", "replace")
            elif f2 == 2 and w2 == 5:
                val = struct.unpack("<f", v2)[0]
            elif f2 == 8 and w2 == 2 and val is None:
                val = _tensor_value(v2)
        if tag and val is not None and val == val:   # 跳过 NaN
            out.setdefault(tag, []).append([step, val])


def _read_file(path):
    size = os.path.getsize(path)
    old = _cache.get(path)
    if old and old[0] == size:
        return old[2]
    data = old[2] if old else {}
    pos = old[1] if old else 0
    with open(path, "rb") as f:
        f.seek(pos)
        buf = f.read()
    i = 0
    while i + 12 <= len(buf):
        ln = struct.unpack("<Q", buf[i:i + 8])[0]
        end = i + 12 + ln + 4
        if end > len(buf):   # 最后一条还没写完，下次再读
            break
        try:
            _parse_event(buf[i + 12:i + 12 + ln], data)
        except (IndexError, struct.error):
            pass
        i = end
    _cache[path] = (size, pos + i, data)
    return data


def _file_time(p):
    try:   # events.out.tfevents.<创建时间>.<主机>...
        return float(os.path.basename(p).split(".")[3])
    except (IndexError, ValueError):
        return os.path.getmtime(p)


def read_scalars(log_dir, runs=None):
    """{tag: [[step, value], ...]}，同一 tag 多个文件时合并并按 step 排序。
    runs = [[开始时间, 步数偏移], ...]：续训时每次运行的步数偏移（sd-scripts 续训后内部步数不是总步数），
    后一次运行的点覆盖前一次运行里同样或更靠后的步数。"""
    merged = {}
    files = sorted(glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True), key=_file_time)
    for p in files:
        ft = _file_time(p)
        off = 0
        for start, o in runs or []:
            if ft >= start - 5:
                off = o
        try:
            for tag, pts in _read_file(p).items():
                # 按轮记录的曲线（loss/epoch_average 等）横轴本来就是真实轮数，续训后也接着记，不加步数偏移
                d = 0 if "epoch" in tag.lower() else off
                pts = [[s + d, v] for s, v in pts]
                cur = merged.setdefault(tag, [])
                if runs and pts and cur:
                    first = min(x[0] for x in pts)
                    cur[:] = [x for x in cur if x[0] < first]
                cur.extend(pts)
        except OSError:
            continue
    for tag in merged:
        merged[tag].sort(key=lambda x: x[0])
    return merged


def downsample(pts, n=1500):
    """点太多时按区间抽样（保留每段的最大最小值，尖峰不会被抹掉）。"""
    if len(pts) <= n:
        return pts
    out, k = [], len(pts) / (n / 2)
    i = 0.0
    while int(i) < len(pts):
        seg = pts[int(i):int(i + k)] or pts[int(i):int(i) + 1]
        lo = min(seg, key=lambda x: x[1])
        hi = max(seg, key=lambda x: x[1])
        out.extend(sorted([lo, hi], key=lambda x: x[0]) if lo is not hi else [lo])
        i += k
    return out

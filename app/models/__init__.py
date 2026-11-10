# -*- coding: utf-8 -*-
"""底模注册表。新增底模：在本目录新建模块，定义 ModelSpec 子类，然后在下面 _BUILTIN 里加一行。"""
import plat
from .base import ModelSpec, Slot, F, dataset_subsets, estimate_steps  # noqa: F401
from .anima import Anima
from .anima29 import Anima29
from .krea2 import Krea2
from .sdxl import SDXL
from .moss import Moss
from .moss_native import MossNative

_REGISTRY = {}


def register(spec):
    _REGISTRY[spec.id] = spec
    return spec


for _cls in (Anima, Anima29, Krea2, SDXL, Moss, MossNative):   # _BUILTIN
    register(_cls())


def get(model_id):
    return _REGISTRY.get(model_id)


def all_specs():
    """界面上列出的底模。只支持 Linux 的引擎在 Windows 下不列出（get 仍然取得到，旧任务能报出明确原因）。"""
    return [s for s in _REGISTRY.values() if not (s.linux_only and plat.IS_WIN)]


def visible_id(model_id):
    """草稿里选的引擎在本机不可用时，换成同一个底模下可用的引擎；底模本身没了返回原值，交给前端清空。"""
    spec = get(model_id)
    shown = all_specs()
    if not spec or spec in shown:
        return model_id
    fam = spec.family or spec.id
    return next((s.id for s in shown if (s.family or s.id) == fam), model_id)


def runnable(model_id):
    """这个引擎本机能不能跑（未知的底模也算「能」，交给体检报未知底模）。"""
    spec = get(model_id)
    return not spec or spec in all_specs()


def hidden_aliases():
    """本机不列出的引擎 → 同一个底模下能用的引擎（给前端载入旧任务 / 旧存档时换引擎用）。"""
    out = {}
    for mid in _REGISTRY:
        vid = visible_id(mid)
        if vid != mid:
            out[mid] = vid
    return out


def _option_ok(field, value):
    if field.get("type") != "select":
        return True
    return any(str(o[0] if isinstance(o, (list, tuple)) else o) == str(value) for o in field.get("options") or [])


def convert(src_id, params, model_files, vram_gb=0):
    """把一份按 src_id 引擎写的参数 / 模型换成本机能用的同底模引擎。
    返回 (新引擎 spec, 新参数, 新模型, 说明)；本来就能用、或者没有可替换的引擎时返回 None。

    新参数 = 新引擎默认值 → 按显存推荐的档位（显存小时自动开 int8 量化等）→ 原任务里自己改过的、两个引擎都有的参数。
    模型文件按槽位照搬：sd-scripts 引擎 bf16 和 int8 ConvRot 量化单文件都能直接练。"""
    src = get(src_id)
    if not src or src in all_specs():
        return None
    dst = get(visible_id(src_id))
    if not dst or dst is src:
        return None
    src_defaults = src.defaults()
    fields = {f["key"]: f for f in dst.params()}
    p = dict(dst.defaults())
    preset = dst.recommend_preset(vram_gb) if vram_gb else None
    if preset:
        p.update(preset["values"])
    kept = []
    for k, v in (params or {}).items():
        if k.startswith("_") or k not in fields or not _option_ok(fields[k], v):
            continue
        if k in ("dataset_dir", "output_name") or v != src_defaults.get(k):
            p[k] = v
            kept.append(k)
    slot_keys = {s.key for s in dst.slots}
    m = {k: v for k, v in (model_files or {}).items() if k in slot_keys and v}
    note = (f"{src.engine_label or src.engine}只支持 Linux，已自动改用 {dst.engine_label or dst.engine} 引擎"
            + (f"（按 {vram_gb:g} GB 显存用「{preset['label']}」档）" if preset else "")
            + "；两个引擎都有的参数和选好的模型已带过去。")
    return dst, p, m, note


def migrate_task(t, vram_gb=0):
    """队列里本机跑不了的任务（比如 Windows 上排进去的 MOSS 专用引擎任务）就地换成能用的引擎。改了返回说明，没改返回 ""。"""
    r = convert(t.get("model_type"), t.get("params"), t.get("models"), vram_gb)
    if not r:
        return ""
    dst, p, m, note = r
    t["migrated_from"] = t.get("model_type")
    t["model_type"] = dst.id
    t["params"] = p
    t["models"] = m
    return note

# -*- coding: utf-8 -*-
"""底模注册表。新增底模：在本目录新建模块，定义 ModelSpec 子类，然后在下面 _BUILTIN 里加一行。"""
import plat
from .base import ModelSpec, Slot, F, dataset_subsets, estimate_steps  # noqa: F401
from .anima import Anima
from .krea2 import Krea2
from .sdxl import SDXL
from .moss import Moss
from .moss_native import MossNative

_REGISTRY = {}


def register(spec):
    _REGISTRY[spec.id] = spec
    return spec


for _cls in (Anima, Krea2, SDXL, Moss, MossNative):   # _BUILTIN
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

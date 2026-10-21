# -*- coding: utf-8 -*-
"""底模注册表。新增底模：在本目录新建模块，定义 ModelSpec 子类，然后在下面 _BUILTIN 里加一行。"""
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
    return list(_REGISTRY.values())

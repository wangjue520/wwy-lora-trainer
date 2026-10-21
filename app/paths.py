# -*- coding: utf-8 -*-
"""训练器用到的所有目录。代码在 app/，其余目录都相对项目根目录。"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app")
WEB = os.path.join(ROOT, "web")
DATA = os.path.join(ROOT, "data")            # 设置、队列、缓存
TASKS = os.path.join(DATA, "tasks")          # 每个任务一个目录：生成的配置、日志
BACKUPS = os.path.join(DATA, "caption_backups")
ENGINES = os.path.join(ROOT, "engines")      # sd-scripts / musubi-tuner
ENV = os.path.join(ROOT, "env")              # 训练环境 venv（torch + 两个引擎 + 反推）
TAGGER_MODELS = os.path.join(ROOT, "models", "taggers")
DOWNLOADS = os.path.join(ROOT, "models", "base")   # 训练器自己下载的底模
OUTPUT = os.path.join(ROOT, "output")

VENDOR = os.path.join(APP, "vendor")         # 训练器自带的额外优化器等（lora_optim）
SD_SCRIPTS = os.path.join(ENGINES, "sd-scripts")
MUSUBI = os.path.join(ENGINES, "musubi-tuner")

for _d in (DATA, TASKS, BACKUPS, ENGINES, TAGGER_MODELS, DOWNLOADS, OUTPUT):
    os.makedirs(_d, exist_ok=True)


def env_python():
    """训练环境的 python（还没装好时返回 None）。"""
    p = os.path.join(ENV, "Scripts", "python.exe") if os.name == "nt" else os.path.join(ENV, "bin", "python")
    return p if os.path.isfile(p) else None



def base_python():
    """创建训练环境用的 Python：优先项目自带的便携版（Windows 整合包 / Linux 上 conda 建在 python/），否则用系统 Python。"""
    for p in (os.path.join(ROOT, "python", "python.exe"), os.path.join(ROOT, "python", "bin", "python")):
        if os.path.isfile(p):
            return p
    exe = sys.executable
    # 别拿训练环境自己的解释器去重建训练环境
    if os.path.abspath(exe).startswith(os.path.abspath(ENV) + os.sep):
        exe = shutil.which("python3") or shutil.which("python") or exe
    return exe

# -*- coding: utf-8 -*-
"""多卡训练启动入口：等同于 `accelerate launch ...`。

Windows 上的 libuv 问题由 app/vendor/sitecustomize.py 处理（训练器启动多卡时设置 LORA_DDP_NO_LIBUV=1，
启动器进程和每个工作进程启动时都会自动打补丁）。用法：python ddp_launch.py <accelerate launch 的参数...>"""
import sys

from accelerate.commands.launch import main

sys.argv = ["accelerate-launch"] + sys.argv[1:]
main()

# -*- coding: utf-8 -*-
"""批量生成缩略图（在训练环境里运行，需要 Pillow）。

用法：python thumbs.py <任务.json>   任务 = [[原图, 缩略图, 边长], ...]
每完成一张输出一行「OK <缩略图>」，界面进程据此知道哪些已就绪。"""
import json
import os
import sys

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

for src, dst, size in json.load(open(sys.argv[1], encoding="utf-8")):
    try:
        if os.path.isfile(dst):
            print("OK " + dst, flush=True)
            continue
        im = Image.open(src)
        im.draft("RGB", (size * 2, size * 2))   # JPEG 直接按小尺寸解码，快很多
        im = ImageOps.exif_transpose(im)
        im.thumbnail((size, size), Image.LANCZOS)
        if im.mode not in ("RGB", "L"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            im = im.convert("RGBA")
            bg.paste(im, mask=im.split()[-1])
            im = bg
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".tmp"
        im.convert("RGB").save(tmp, "JPEG", quality=85)
        os.replace(tmp, dst)
        print("OK " + dst, flush=True)
    except Exception as e:  # 坏图不影响其他图
        print(f"FAIL {src} {e}", flush=True)

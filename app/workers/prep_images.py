# -*- coding: utf-8 -*-
"""训练集图片预处理（在训练环境里运行，需要 Pillow）。用户不用管，训练前 / 爬图后自动跑。

透明底 → 白底：训练引擎读图时直接丢掉透明通道，透明部分会变成黑色或乱色，模型会学到黑边。
这里把带透明像素的图合成到白底上、原地保存（格式不变，WebP 用无损），原图备份到 data/backups/透明底原图/。
已经检查过的图记在缓存里（路径 + 大小 + 修改时间），下次直接跳过，几千张图也只要一两秒。

用法：python prep_images.py --dir 训练集 [--dir 另一个] [--backup 备份根目录] [--cache 缓存文件]"""
import argparse
import hashlib
import json
import os
import shutil
import sys

from PIL import Image

EXTS = (".png", ".webp", ".bmp")   # 训练引擎能读、又可能带透明通道的格式（jpg 没有透明）
SKIP_DIRS = {"sample", ".thumbs", "__pycache__", "_已移除"}


def has_alpha(im):
    if im.mode in ("RGBA", "LA", "PA"):
        return im.getchannel("A").getextrema()[0] < 255
    if im.mode == "P" and "transparency" in im.info:
        return im.convert("RGBA").getchannel("A").getextrema()[0] < 255
    if im.mode in ("RGB", "L") and "transparency" in im.info:
        return True
    return False


def to_white(path, backup_root, root):
    with Image.open(path) as im:
        im.load()
        fmt = im.format
        if getattr(im, "n_frames", 1) > 1 or not has_alpha(im):
            return False
        rgba = im.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    out = Image.alpha_composite(bg, rgba).convert("RGB")
    if backup_root:
        dst = os.path.join(backup_root, os.path.relpath(path, root))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            shutil.copy2(path, dst)
    tmp = path + ".tmp"
    if fmt == "WEBP":
        out.save(tmp, "WEBP", lossless=True, quality=100)
    elif fmt == "BMP":
        out.save(tmp, "BMP")
    else:
        out.save(tmp, "PNG")
    os.replace(tmp, path)
    return True


def fix_dirs(dirs, backup_base=None, cache_file=None, log=print):
    cache = {}
    if cache_file:
        try:
            with open(cache_file, encoding="utf-8") as f:
                cache = json.load(f)
        except (OSError, ValueError):
            cache = {}
    fixed = checked = 0
    for root in dirs:
        if not root or not os.path.isdir(root):
            continue
        root = os.path.abspath(root)
        tag = os.path.basename(root.rstrip("\\/")) + "_" + hashlib.md5(root.lower().encode("utf-8")).hexdigest()[:6]
        backup_root = os.path.join(backup_base, tag) if backup_base else None
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith(".") and not d.endswith("-state")]
            for fn in fns:
                if not fn.lower().endswith(EXTS):
                    continue
                p = os.path.join(dp, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                key = os.path.normcase(p)
                sig = f"{st.st_size}:{int(st.st_mtime)}"
                if cache.get(key) == sig:
                    continue
                checked += 1
                try:
                    if to_white(p, backup_root, root):
                        fixed += 1
                        st = os.stat(p)
                        sig = f"{st.st_size}:{int(st.st_mtime)}"
                except Exception as e:   # 坏图交给训练前体检处理，这里不中断
                    log(f"[透明底] 跳过 {p}：{e}")
                    continue
                cache[key] = sig
    if cache_file:
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            tmp = cache_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cache, f)
            os.replace(tmp, cache_file)
        except OSError:
            pass
    if fixed:
        log(f"[透明底] {fixed} 张透明底图片已自动改成白底（原图备份在 {backup_base}）")
    elif checked:
        log(f"[透明底] 检查了 {checked} 张新图片，没有透明底")
    return fixed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", default=[])
    ap.add_argument("--backup")
    ap.add_argument("--cache")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    fix_dirs(a.dir, a.backup, a.cache)

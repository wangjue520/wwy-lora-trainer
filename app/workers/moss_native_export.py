# -*- coding: utf-8 -*-
"""把专用引擎（moss_trainer）的 checkpoint 导出成训练器认识的 LoRA 文件。

用法：python moss_native_export.py --src <native_out> --dest <output_dir> --name <名字>
专用引擎每个保存点存一个 epoch-000001/adapter.safetensors（ComfyUI 可直接加载的 LyCORIS 格式）。
和 sd-scripts 的命名一样导出：中间的轮次 <名字>-000001.safetensors，最后一轮 <名字>.safetensors。
预览图训练时已经由 tasks.py 复制到 <output_dir>/sample/。
"""
import argparse
import os
import re
import shutil


def _checkpoints(src):
    """[(编号, 目录)]，按编号排序；只要 epoch-* / step-* 里有 adapter.safetensors 的。"""
    res = []
    for name in os.listdir(src):
        m = re.fullmatch(r"(?:epoch-(\d+)|step-(\d+))", name)
        path = os.path.join(src, name)
        if m and os.path.isfile(os.path.join(path, "adapter.safetensors")):
            res.append((int(m.group(1) or m.group(2)), path))
    return sorted(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    src = a.src
    if not os.path.isdir(src):
        raise SystemExit(f"找不到专用引擎的输出目录：{src}")
    found = _checkpoints(src)
    if not found:
        raise SystemExit(f"{src} 里没有 adapter.safetensors（训练可能没跑完）")
    os.makedirs(a.dest, exist_ok=True)
    for i, (num, ckpt) in enumerate(found):
        last = i == len(found) - 1
        dst = os.path.join(a.dest, a.name + ("" if last else f"-{num:06d}") + ".safetensors")
        shutil.copy2(os.path.join(ckpt, "adapter.safetensors"), dst)
        print(f"导出 {dst}（{os.path.getsize(dst) / 1e6:.1f} MB，来自 {os.path.basename(ckpt)}）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

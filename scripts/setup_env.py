# -*- coding: utf-8 -*-
"""命令行安装 / 修复训练环境，不用开界面（Linux 云主机、容器里用）。

用法（在项目根目录）：
    python3 scripts/setup_env.py                # 安装（已经装好的部分会很快跳过）
    python3 scripts/setup_env.py --torch cu128  # 指定 PyTorch 的 CUDA 版本
    python3 scripts/setup_env.py --repair       # 修复：重装依赖
    python3 scripts/setup_env.py --reinstall-torch
    python3 scripts/setup_env.py --check        # 只看状态，不装
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))

# 输出一律 UTF-8：被 WWY 启动器（python -I）调用时 PYTHONIOENCODING 不生效，中文 Windows 上会按 GBK 输出，
# 打印「⚠」这类字符直接 UnicodeEncodeError（换源 / 换线路的提示里就有），安装中途崩掉
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")


class ConsoleJob:
    """把 envsetup 的进度和日志直接打到终端。"""

    def __init__(self):
        self.status = "running"
        self.proc = None
        self.progress = 0
        self._last = ""

    def write(self, line):
        for ln in str(line).rstrip("\r\n").splitlines():
            print(ln, flush=True)

    def set_progress(self, v, text=""):
        try:
            pct = int(float(v or 0) * 100)
        except (TypeError, ValueError):
            pct = 0
        self.progress = v
        if text.startswith("下载 PyTorch"):   # 下载进度每 0.1 GB 打一行
            m = re.search(r"([\d.]+) GB", text)
            key = f"{float(m.group(1)):.1f}" if m else text
            if key != self._last:
                print(f"  {text}", flush=True)
                self._last = key
            return
        print(f"\n[{pct:3d}%] {text}", flush=True)

    def run_cmd(self, cmd, cwd=None, env=None, parse=None):
        from jobs import decode_line
        import plat
        print("$ " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd), flush=True)
        p = self.proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, **plat.popen_kwargs())
        buf = b""
        while True:
            chunk = p.stdout.read1(4096) if hasattr(p.stdout, "read1") else p.stdout.read(1)
            if not chunk:
                break
            buf += chunk
            while True:
                j = min([x for x in (buf.find(b"\n"), buf.find(b"\r")) if x >= 0], default=-1)
                if j < 0:
                    break
                line, buf = buf[:j], buf[j + 1:]
                s = decode_line(line).strip()
                if s and parse:
                    try:
                        if parse(s) is False:
                            continue
                    except Exception:
                        pass
                if s:
                    print(s, flush=True)
        if buf.strip():
            print(decode_line(buf), flush=True)
        return p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch", choices=["cu126", "cu128", "cu130"], help="PyTorch 的 CUDA 版本")
    ap.add_argument("--repair", action="store_true", help="重装依赖（不重装 PyTorch）")
    ap.add_argument("--reinstall-torch", action="store_true", help="重装 PyTorch")
    ap.add_argument("--check", action="store_true", help="只查看训练环境状态")
    args = ap.parse_args()

    import envsetup
    import settings

    if args.torch:
        settings.save({"torch_variant": args.torch})
        print(f"PyTorch 版本设为 {args.torch}")
    if args.check:
        print(envsetup.status())
        return 0

    job = ConsoleJob()
    try:
        info = envsetup.install(job, repair=args.repair, reinstall_torch=args.reinstall_torch)
    except Exception as e:
        print(f"\n安装失败：{e}", file=sys.stderr)
        return 1
    print("\n训练环境安装完成：")
    print(f"  PyTorch {info.get('torch')} · CUDA {info.get('cuda')}")
    for n, e in envsetup.ENGINES.items():
        print(f"  {n}: {envsetup.head(n) or '未安装'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

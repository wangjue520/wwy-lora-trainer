# -*- coding: utf-8 -*-
"""版本号走一位：version.json 的 version 末位 +1（0.1.2 → 0.1.3，0.1.9 → 0.1.10），updated_at 改成今天（日本时间）。

每次把改动推到 main（用户「一键更新」拿到的一次更新）之前运行一次：
    python scripts/bump_version.py

一次更新只升一位：如果现在的版本号已经比 GitHub 上 main 的大（这次更新里已经升过了），就不再升，
只把 updated_at 改成今天。确实要再升一位时加 --force。engines（测试过的引擎版本）不会动。"""
import datetime
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE = os.path.join(ROOT, "version.json")
JST = datetime.timezone(datetime.timedelta(hours=9))


def bump(v):
    """末位 +1，不进位。"""
    parts = v.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"版本号格式不对（应该是 X.Y.Z）：{v}")
    parts[-1] = str(int(parts[-1]) + 1)
    return ".".join(parts)


def key(v):
    return tuple(int(x) for x in v.split("."))


def remote_version():
    """GitHub 上 main 的版本号（先 fetch 一下）；拿不到返回 None。"""
    try:
        subprocess.run(["git", "-C", ROOT, "fetch", "-q", "origin", "main"], capture_output=True, timeout=60)
        r = subprocess.run(["git", "-C", ROOT, "show", "origin/main:version.json"], capture_output=True, timeout=30)
        if r.returncode == 0:
            return json.loads(r.stdout.decode("utf-8"))["version"]
    except Exception:
        pass
    return None


def main():
    force = "--force" in sys.argv
    with open(FILE, encoding="utf-8") as f:
        data = json.load(f)
    old = data["version"]
    remote = remote_version()
    if remote and key(old) > key(remote) and not force:
        print(f"这次更新已经升过版本号了（GitHub 上是 {remote}，现在是 {old}），不再升；只更新日期。确实要再升一位加 --force")
        new = old
    else:
        new = bump(old)
        print(f"版本号 {old} → {new}" + (f"（GitHub 上是 {remote}）" if remote else "（没连上 GitHub，按本地版本号升）"))
    data["version"] = new
    data["updated_at"] = datetime.datetime.now(JST).strftime("%Y-%m-%d")
    with open(FILE, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

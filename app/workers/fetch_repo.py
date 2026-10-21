# -*- coding: utf-8 -*-
"""训练前下载模型仓库里的一部分文件（例如 MOSS-Image 的分词器）。
用法：python fetch_repo.py <仓库> <保存目录> [文件通配符 ...]
线路选择、多连接、慢了 / 断了换线路都用训练器自己的下载器（hf-mirror / 魔搭 / HF 官方 × 直连 / 代理）。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import downloader  # noqa: E402


class Job:
    stop_flag = False
    progress = 0

    def write(self, s):
        print(s, flush=True)

    def set_progress(self, p, text=""):
        self.progress = p


if __name__ == "__main__":
    repo, dest, pats = sys.argv[1], sys.argv[2], sys.argv[3:] or None
    downloader.download_repo(Job(), repo, dest, pats)

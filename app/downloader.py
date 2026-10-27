# -*- coding: utf-8 -*-
"""下载模型文件（单个文件，或整个仓库里需要的文件），下完校验 safetensors 文件头。

线路自动选择（netpick.py）：hf-mirror / 魔搭 / HF 官方 × 直连 / 系统代理 / 本机代理端口 / 自定义代理，下载前先测速排序；
下载中某条线路连续出错（连不上、被重置、CDN 地址过期……）或者持续很慢（测速时快、真下起来慢、半路被限速），
就换到下一条线路接着下，已经下好的部分不用重下。

多连接分块下载：国内网络常见「每条连接限速、总带宽够」，单连接几百 KB/s，开 8 条连接能快好几倍。
文件按 64 MB 分块，8 个线程各下一块，下完的块记在 <文件>.part.json 里，断了再开从没下完的块接着下。
服务器不支持分段（没有 206）时退回单连接下载（同样支持续传）。"""
import json
import os
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor

import netpick
import paths
import safetensors_meta as stm
import settings

CHUNK = 64 * 1024 * 1024
CONNECTIONS = 8


def hf_url(repo, file, endpoint=None):
    return f"{endpoint or settings.hf_endpoint()}/{repo}/resolve/main/{file}"


SLOW_WINDOW = 45      # 按最近 45 秒的平均速度判断「持续很慢」
SLOW_MIN_BPS = 300 * 1024


class Sources:
    """测速排好序的线路，出错多了 / 持续很慢就换下一条。"""

    def __init__(self, job, ranked):
        self.job, self.list, self.i, self.fails = job, ranked, 0, 0
        self.lock = threading.Lock()
        self.since = time.time()   # 换到当前线路的时间

    @property
    def cur(self):
        return self.list[self.i]

    def failed(self, err, limit=3):
        """记一次出错；同一条线路累计出错 limit 次就换线路。返回 False = 已经没有线路可换。"""
        with self.lock:
            self.fails += 1
            if self.fails < limit:
                return True
            return self._next(f"「{self.cur['name']}」连续出错（{err}）")

    def switch(self, why):
        with self.lock:
            return self._next(why)

    def _next(self, why):
        if self.i + 1 >= len(self.list):
            return False
        # 优先换连接方式不同的线路（代理断了时，同一个代理上的其它源多半也不通）
        netpick.record(self.cur["url"], self.cur["proxy"], failed=True)
        rest = range(self.i + 1, len(self.list))
        self.i = next((j for j in rest if self.list[j]["proxy"] != self.cur["proxy"]), self.i + 1)
        self.fails = 0
        self.since = time.time()
        self.job.write(f"⚠ {why}，换到「{self.cur['name']}」继续下载")
        return True


def download(job, repo, file, dest_dir=None, dest_name=None, ranked=None, quiet=False):
    dest_dir = dest_dir or paths.DOWNLOADS
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, dest_name or os.path.basename(file))
    part = dest + ".part"
    if ranked is None:
        job.write("测试下载线路…")
        ranked = netpick.rank_hf(repo, file)
        if ranked:
            job.write("线路测速：" + netpick.describe(ranked))
    else:   # 整个仓库一起下：线路是用仓库里最大的文件测的，这里换成这个文件的地址
        ranked = [dict(x, url=_same_route_url(x["url"], repo, file)) for x in ranked]
    if not ranked:
        raise RuntimeError("所有下载线路都连不上（hf-mirror、魔搭、HF 官方都试过了，直连和代理也都试过了）。"
                           "请检查网络；挂梯子的话确认梯子已连接，或者到「训练器设置 → 网络」填写代理地址。")
    src = Sources(job, ranked)
    last_err = None
    while True:
        try:
            if not _fetch_multi(job, src, part):
                _fetch(job, src, part)
            break
        except Exception as e:
            if job.stop_flag:
                raise RuntimeError("已取消下载")
            last_err = e
            job.write(f"下载出错（{src.cur['name']}）：{e}")
            if not src.switch("这条线路下不动"):
                raise RuntimeError(f"所有线路都试过了，下载失败：{last_err}。已下载的部分保留着，重新点下载会接着下。")
            time.sleep(2)
    os.replace(part, dest)
    try:
        os.remove(part + ".json")
    except OSError:
        pass
    if dest.endswith(".safetensors"):
        _, err = stm.read_header(dest)
        if err:
            os.remove(dest)
            raise RuntimeError("下载的文件校验失败：" + err)
    if not quiet:
        job.write(f"下载完成：{dest}")
    return dest


def _same_route_url(url, repo, file):
    """把某条线路上 A 文件的地址换成同一个源上 B 文件的地址。"""
    import urllib.parse
    for name, tpl, _ in netpick.HF_SOURCES:
        head = tpl.split("{repo}")[0]
        if url.startswith(head):
            return tpl.format(repo=repo, file=urllib.parse.quote(file))
    return url


MARK = ".lora_complete"
MARK_VERSION = 2   # 1 = 空文件（只比过大小）；2 = 记了每个文件校验过的 sha256


def sha256_file(path, job=None):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            if job is not None and job.stop_flag:
                raise RuntimeError("已取消")
            b = f.read(16 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def is_verified(local_dir):
    """仓库是否已经下完并校验过 sha256。以前的标记是空文件（只比大小，分块拼错了也认不出来），算没校验。"""
    try:
        with open(os.path.join(local_dir, MARK), encoding="utf-8") as f:
            return int(json.load(f).get("v") or 0) >= MARK_VERSION
    except (OSError, ValueError, AttributeError):
        return False


def download_repo(job, repo, local_dir, patterns=None):
    """把模型仓库里需要的文件下到 local_dir（反推模型等多文件仓库）。已经下好、大小和 sha256 都对得上的文件跳过。
    用仓库里最大的文件测一次线路，所有文件共用；大文件多连接分块下载，出错 / 变慢自动换线路。
    每个文件下完都和官方 sha256 比一遍（多连接分块 + 中途换线路时，拼错一块文件大小照样对，只有哈希能发现）；
    不对就删掉换线路重下。都对了写 .lora_complete 标记，反推进程看到它就不再联网检查。"""
    import fnmatch
    os.makedirs(local_dir, exist_ok=True)
    mark = os.path.join(local_dir, MARK)
    files = netpick.list_repo_files_sha(repo)
    files = [f for f in files if not patterns or any(fnmatch.fnmatch(f[0], pt) for pt in patterns)]
    files = [(p, s, h) for p, s, h in files if not p.startswith(".") and not p.endswith((".md", ".gitattributes"))
             and (patterns or not p.lower().endswith((".bin", ".pt", ".pth", ".msgpack", ".h5", ".ckpt"))
                  or not any(q.endswith(".safetensors") for q, _, _ in files))]   # 有 safetensors 时不下重复的旧格式权重
    if not files:
        raise RuntimeError(f"模型仓库 {repo} 里没有找到需要的文件")
    try:
        with open(mark, encoding="utf-8") as f:
            checked = json.load(f).get("sha256") or {}
    except (OSError, ValueError, AttributeError):
        checked = {}
    todo = []
    for p, s, h in files:
        fp = os.path.join(local_dir, p)
        if not os.path.isfile(fp) or (s and os.path.getsize(fp) != s):
            todo.append((p, s, h))
        elif h and checked.get(p) != h:
            job.write(f"校验已下载的 {p}（{s / 1e9:.2f} GB）…")
            if sha256_file(fp, job) == h:
                checked[p] = h
            else:
                job.write(f"⚠ {p} 和官方文件不一致（之前下载时损坏），删掉重下")
                os.remove(fp)
                todo.append((p, s, h))
    total = sum(s for _, s, _ in files)
    if todo:
        big = max(todo, key=lambda x: x[1])
        job.write(f"下载 {repo}：{len(todo)} 个文件，共 {sum(s for _, s, _ in todo) / 1e9:.2f} GB；测试下载线路…")
        ranked = netpick.rank_hf(repo, big[0])
        if not ranked:
            raise RuntimeError(f"{repo} 所有下载线路都连不上（hf-mirror、魔搭、HF 官方 × 直连 / 代理都试过了）。"
                               "请检查网络；挂梯子的话确认梯子已连接，或者到「训练器设置 → 网络」填写代理地址。")
        job.write("线路测速：" + netpick.describe(ranked))
        for k, (p, s, h) in enumerate(sorted(todo, key=lambda x: x[1])):
            d = os.path.join(local_dir, os.path.dirname(p))
            job.write(f"[{k + 1}/{len(todo)}] {p}（{s / 1e6:.1f} MB）")
            for attempt in range(3):
                dest = download(job, repo, p, d, os.path.basename(p), ranked=ranked, quiet=True)
                ranked = _promote(ranked, job)
                if not h:
                    break
                if sha256_file(dest, job) == h:
                    checked[p] = h
                    break
                os.remove(dest)
                if attempt == 2:
                    raise RuntimeError(f"{p} 连续 3 次下载都和官方 sha256 对不上，线路可能在篡改或截断文件，换个网络再试")
                job.write(f"⚠ {p} 下载后校验不通过（和官方 sha256 不一致），换一条线路重下")
                ranked = ranked[1:] + ranked[:1]
    with open(mark, "w", encoding="utf-8") as f:
        json.dump({"v": MARK_VERSION, "repo": repo, "sha256": checked}, f)
    if todo:
        job.write(f"{repo} 下载完成（{total / 1e9:.2f} GB）")
    return local_dir


def _promote(ranked, job):
    """上一个文件下载中换过线路的话，后面的文件直接从换到的那条开始。"""
    i = getattr(job, "_dl_route", 0)
    job._dl_route = 0   # 用一次就清掉：下一个文件没再换线路的话不要再转
    return ranked[i:] + ranked[:i] if i else ranked


def _headers(url):
    h = {"User-Agent": "lora-trainer/1.0"}
    tok = settings.load().get("hf_token")
    if tok and "huggingface.co" in url:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _probe(url, proxy):
    """问一下文件大小、能不能分段，顺便拿到跳转后的真实地址（HF 会跳到 CDN）。"""
    with netpick.open_url(url, proxy, dict(_headers(url), Range="bytes=0-0"), timeout=30) as r:
        cr = r.headers.get("Content-Range") or ""
        if r.status != 206 or "/" not in cr:
            return None, None
        return int(cr.rsplit("/", 1)[1]), r.geturl()


def _fetch_multi(job, src, part):
    total, real = _probe(src.cur["url"], src.cur["proxy"])
    if not total or total < CHUNK * 2:
        return False   # 小文件或服务器不支持分段：单连接
    n = (total + CHUNK - 1) // CHUNK
    state_file = part + ".json"
    done = set()
    try:
        with open(state_file, encoding="utf-8") as f:
            st = json.load(f)
        if st.get("total") == total and os.path.isfile(part):
            done = set(st.get("done") or [])
    except (OSError, ValueError):
        if os.path.isfile(part):   # 旧版单连接下了一半的文件：前面整块的部分算下完了
            have = os.path.getsize(part)
            if have <= total:
                done = set(range(have // CHUNK))
    if not os.path.isfile(part):
        open(part, "wb").close()
    with open(part, "r+b") as f:
        f.truncate(total)
    lock = threading.Lock()
    prog = {"bytes": sum(min(CHUNK, total - i * CHUNK) for i in done), "t0": time.time()}
    prog["b0"] = prog["bytes"]
    real_urls = {src.i: real or src.cur["url"]}   # 每条线路跳转后的真实地址
    job.write(f"开始下载（{src.cur['name']}，{CONNECTIONS} 条连接分块下载"
              + (f"，从 {prog['bytes'] / 1e9:.2f} GB 处续传" if done else "") + "）")

    def save_state():
        tmp = state_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"total": total, "done": sorted(done)}, fh)
        os.replace(tmp, state_file)

    def report():
        dt = max(0.001, time.time() - prog["t0"])
        speed = (prog["bytes"] - prog["b0"]) / dt
        eta = (total - prog["bytes"]) / speed if speed > 0 else 0
        job.set_progress(prog["bytes"] / total, f"{prog['bytes'] / 1e9:.2f} / {total / 1e9:.2f} GB · {speed / 1e6:.1f} MB/s · "
                                                f"剩余 {int(eta // 60)} 分 {int(eta % 60)} 秒 · {src.cur['name']}")

    def url_for(k):
        if k not in real_urls:   # 换了线路：重新拿这条线路的真实地址
            c = src.list[k]
            try:
                real_urls[k] = _probe(c["url"], c["proxy"])[1] or c["url"]
            except Exception:
                real_urls[k] = c["url"]
        return real_urls[k]

    class Rerouted(Exception):
        pass

    def one(i):
        start, end = i * CHUNK, min(total, (i + 1) * CHUNK) - 1
        pos = start   # 这一块已经写到哪里了：断了 / 换线路都从这里接着下，不整块重下
        for attempt in range(12):
            if job.stop_flag:
                raise RuntimeError("已取消下载")
            k = src.i
            try:
                u = url_for(k)
                hdr = dict(_headers(u), Range=f"bytes={pos}-{end}")
                with netpick.open_url(u, src.list[k]["proxy"], hdr, timeout=60) as r, open(part, "r+b") as f:
                    if r.status != 206:
                        raise RuntimeError(f"服务器没有按分段返回（{r.status}）")
                    f.seek(pos)
                    while True:
                        if job.stop_flag:
                            raise RuntimeError("已取消下载")
                        if src.i != k:
                            raise Rerouted()   # 看门狗换了线路：剩下的部分从新线路下
                        b = r.read(1024 * 1024)
                        if not b:
                            break
                        f.write(b)
                        pos += len(b)
                        with lock:
                            prog["bytes"] += len(b)
                            hist.append((time.time(), prog["bytes"]))
                            report()
                if pos != end + 1:
                    raise RuntimeError("连接中断")
                with lock:
                    done.add(i)
                    save_state()
                return
            except Rerouted:
                continue
            except Exception as e:
                if job.stop_flag:
                    raise
                if isinstance(e, urllib.error.HTTPError) and e.code in (403, 410):
                    real_urls.pop(k, None)   # CDN 签名地址过期了，下次重新拿
                if not src.failed(e) and attempt >= 3:
                    raise RuntimeError(f"第 {i + 1} 块下载失败：{e}")
                time.sleep(2 + min(attempt, 4) * 2)
        raise RuntimeError(f"第 {i + 1} 块下载失败")

    from collections import deque
    hist = deque(maxlen=4000)
    stop_watch = threading.Event()

    def watchdog():
        """持续很慢（比后面线路测速时的速度慢一半以上，或者低于 300 KB/s）就换线路。刚换过的线路先给 45 秒。"""
        while not stop_watch.wait(5):
            now = time.time()
            if now - src.since < SLOW_WINDOW + 10 or src.i + 1 >= len(src.list):
                continue
            with lock:
                recent = [(t, b) for t, b in hist if now - t <= SLOW_WINDOW]
            bps = (recent[-1][1] - recent[0][1]) / max(1.0, recent[-1][0] - recent[0][0]) if len(recent) > 1 else 0
            nxt = max(x["speed"] for x in src.list[src.i + 1:])
            # 测速是单连接的速度，这里是 8 条连接的总和：按单条连接比（国内常见「每条连接限速」，换线路能快好几倍）
            per_conn = bps / CONNECTIONS
            # 带宽本来就跑满时每条连接自然慢，总速度也比不过对方单连接两倍才换，免得来回换
            if bps < SLOW_MIN_BPS or (per_conn < nxt * 0.5 and bps < nxt * 2):
                src.switch(f"「{src.cur['name']}」最近 {SLOW_WINDOW} 秒只有 {bps / 1e6:.2f} MB/s"
                           f"（每条连接 {per_conn / 1e3:.0f} KB/s，后面的线路测速 {nxt / 1e6:.1f} MB/s）")
                job._dl_route = src.i

    todo = [i for i in range(n) if i not in done]
    threading.Thread(target=watchdog, daemon=True).start()
    try:
        with ThreadPoolExecutor(CONNECTIONS) as ex:
            for fut in [ex.submit(one, i) for i in todo]:
                fut.result()
    finally:
        stop_watch.set()
    if len(done) != n:
        raise RuntimeError("还有分块没下完")
    dt = time.time() - prog["t0"]
    if prog["bytes"] - prog["b0"] > 32 << 20 and dt > 5:   # 记下这条线路实际的速度，下次排序参考
        netpick.record(src.cur["url"], src.cur["proxy"], (prog["bytes"] - prog["b0"]) / dt)
    return True


def _fetch(job, src, part):
    url, proxy = src.cur["url"], src.cur["proxy"]
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = _headers(url)
    if have:
        headers["Range"] = f"bytes={have}-"
    job.write(f"开始下载（{src.cur['name']}）" + (f"，从 {have / 1e9:.2f} GB 处续传" if have else ""))
    with netpick.open_url(url, proxy, headers, timeout=60) as r:
        if have and r.status != 206:
            have = 0   # 服务器不支持续传，重新下
        total = have + int(r.headers.get("Content-Length") or 0)
        mode = "ab" if have else "wb"
        t0, done0 = time.time(), have
        with open(part, mode) as f:
            done = have
            while True:
                if job.stop_flag:
                    raise RuntimeError("已取消下载")
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                dt = max(0.001, time.time() - t0)
                speed = (done - done0) / dt
                if total:
                    eta = (total - done) / speed if speed > 0 else 0
                    job.set_progress(done / total, f"{done / 1e9:.2f} / {total / 1e9:.2f} GB · {speed / 1e6:.1f} MB/s · "
                                                   f"剩余 {int(eta // 60)} 分 {int(eta % 60)} 秒 · {src.cur['name']}")
        if total and done < total:
            raise RuntimeError("连接中断")

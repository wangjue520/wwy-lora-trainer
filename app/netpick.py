# -*- coding: utf-8 -*-
"""下载线路自动选择：国内裸连、国内挂梯子、国外网络都能用。

线路 = 下载源 × 连接方式：
  下载源：模型 —— HF 镜像 hf-mirror / 魔搭 ModelScope（同名仓库）/ HF 官方；GitHub —— 直连 / 几个加速代理；
          pip —— 清华 / 阿里云 / 中科大 / 官方；PyTorch —— 南大 / 阿里云 / 官方。
  连接方式：直连 / 系统代理（梯子开了「系统代理」）/ 本机常见代理端口（梯子是 PAC、规则模式时系统代理读不到，
           直接探测 Clash 7890、Clash Verge 7897、v2rayN 10809 等端口）/ 设置里手动填的代理。
每次下载前并行测速：从文件中间读几 MB，去掉首包延迟后算「持续速度」（只测开头 1 MB 会被首包爆发骗，
测得快、真下起来几百 KB/s）。实际下载的速度和出错记在 data/net_stats.json，下次排序时一起参考：
真下得快的线路排前面，最近卡过 / 断过的往后放。连不上的线路直接跳过——
比如梯子关了但系统代理还开着、镜像站暂时挂了、官方站被墙。

「训练器设置 → 网络」里的「国内 / 国外」只决定速度差不多时谁优先，不是非此即彼。"""
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import paths
import settings

# 模型下载源：(名字, 文件地址模板, 是否 HuggingFace 接口兼容)
HF_SOURCES = [("HF 镜像 hf-mirror", "https://hf-mirror.com/{repo}/resolve/main/{file}", True),
              ("魔搭 ModelScope", "https://www.modelscope.cn/models/{repo}/resolve/master/{file}", False),
              ("HF 官方", "https://huggingface.co/{repo}/resolve/main/{file}", True)]
HF = [("HF 镜像 hf-mirror", "https://hf-mirror.com"), ("HF 官方", "https://huggingface.co")]   # huggingface_hub 能用的端点
GH_PROXIES = ["https://ghfast.top/", "https://gh-proxy.com/", "https://ghproxy.net/"]
PIP = [("清华", "https://pypi.tuna.tsinghua.edu.cn/simple"), ("阿里云", "https://mirrors.aliyun.com/pypi/simple"),
       ("中科大", "https://mirrors.ustc.edu.cn/pypi/simple"), ("PyPI 官方", "https://pypi.org/simple")]
# 梯子软件常见的本机 HTTP / 混合代理端口
LOCAL_PROXY_PORTS = [(7890, "Clash"), (7897, "Clash Verge"), (10809, "v2rayN"), (20171, "v2rayA"), (2080, "NekoRay / sing-box")]
UA = "lora-trainer/1.0"
STATS_FILE = os.path.join(paths.DATA, "net_stats.json")

_cache = {}
_lock = threading.Lock()
_routes_cache = [0, None]


# ---------------- 连接方式 ----------------
def _alive(proxy, timeout=1.5):
    """代理端口连得上吗（梯子关了但系统代理没撤的情况）。"""
    try:
        u = urllib.parse.urlparse(proxy if "://" in proxy else "http://" + proxy)
        with socket.create_connection((u.hostname, u.port or 80), timeout=timeout):
            return True
    except Exception:
        return False


def routes(ttl=60):
    """[(名字, 代理地址或 None)]：直连 + 系统代理 + 本机常见代理端口 + 设置里手动填的代理（连不上的不列）。"""
    with _lock:
        if _routes_cache[1] is not None and time.time() - _routes_cache[0] < ttl:
            return list(_routes_cache[1])
    res = [("直连", None)]
    seen = set()
    sysp = urllib.request.getproxies()   # Windows 上会读注册表里的系统代理（梯子的「系统代理」模式）
    cands = [("系统代理", sysp.get("https") or sysp.get("http")), ("自定义代理", settings.load().get("proxy_url"))]
    cands += [(f"本机代理 {name}", f"http://127.0.0.1:{port}") for port, name in LOCAL_PROXY_PORTS]
    for name, p in cands:
        p = (p or "").strip()
        if not p:
            continue
        p = p if "://" in p else "http://" + p
        key = p.replace("localhost", "127.0.0.1").rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        if _alive(p, 0.4 if "127.0.0.1" in p else 1.5):
            res.append((f"{name} {p.split('://', 1)[1]}", p))
    with _lock:
        _routes_cache[:] = [time.time(), res]
    return list(res)


def opener(proxy):
    if proxy:
        return urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))   # 明确不走代理


def open_url(url, proxy, headers=None, timeout=60):
    return opener(proxy).open(urllib.request.Request(url, headers=dict({"User-Agent": UA}, **(headers or {}))), timeout=timeout)


# ---------------- 实际下载速度记录 ----------------
def _skey(url, proxy):
    return f"{urllib.parse.urlparse(url).hostname}|{proxy or 'direct'}"


def _load_stats():
    try:
        return json.load(open(STATS_FILE, encoding="utf-8"))
    except Exception:
        return {}


def record(url, proxy, bps=None, failed=False):
    """记下一条线路实际下载的持续速度（bps），或者一次出错 / 卡死（failed）。排序时参考。"""
    with _lock:
        st = _load_stats()
        k = _skey(url, proxy)
        e = st.get(k) or {}
        now = time.time()
        if bps:
            e["bps"] = bps if not e.get("bps") or now - e.get("t", 0) > 7 * 86400 else e["bps"] * 0.5 + bps * 0.5
            e["t"] = now
        if failed:
            e["fail_t"] = now
        st[k] = e
        try:
            tmp = STATS_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(st, f)
            os.replace(tmp, STATS_FILE)
        except OSError:
            pass


# ---------------- 测速 ----------------
def probe(url, proxy, nbytes=4 << 20, timeout=6, headers=None, start=32 << 20):
    """从 start 处读 nbytes，返回 (持续速度 字节/秒, 跳转后的真实地址)；连不上返回 (0, None)。
    速度从第一块数据到达之后开始算，不把握手 / 首包等待算进去；文件比 start 小时从头读。"""
    t0 = time.time()
    try:
        try:
            r = open_url(url, proxy, dict(headers or {}, Range=f"bytes={start}-{start + nbytes - 1}"), timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code != 416 or not start:
                raise
            r = open_url(url, proxy, dict(headers or {}, Range=f"bytes=0-{nbytes - 1}"), timeout=timeout)
        with r:
            got, t1, g1 = 0, None, 0
            while got < nbytes and time.time() - t0 < timeout:
                b = r.read(64 * 1024)
                if not b:
                    break
                got += len(b)
                if t1 is None:
                    t1, g1 = time.time(), got
            real = r.geturl()
        t2 = time.time()
        if not got:
            return 0, None
        if t1 is not None and t2 - t1 > 0.3 and got > g1:
            return (got - g1) / (t2 - t1), real
        return got / max(0.05, t2 - t0), real   # 小文件一下就读完了：按总时间算
    except Exception:
        return 0, None


def rank(candidates, ttl=600, prefer=None, small=False):
    """candidates：[(线路名, 网址, 代理)]。并行测速，按速度从快到慢返回 [{name, url, proxy, speed, real}]，测不通的去掉。
    prefer：速度差不多（相差 30% 以内）时优先的线路名关键字。small：测的是小文件 / 网页（只看连不连得上、快不快）。
    排序还参考以前实际下载的速度：真下得快的往前，最近一小时卡过 / 断过的往后。"""
    key = tuple((n, u, p) for n, u, p in candidates)
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    with ThreadPoolExecutor(min(12, len(candidates) or 1)) as ex:
        kw = {"nbytes": 256 * 1024, "start": 0} if small else {}
        speeds = list(ex.map(lambda c: probe(c[1], c[2], **kw), candidates))
    res = [{"name": n, "url": u, "proxy": p, "speed": s, "real": real} for (n, u, p), (s, real) in zip(candidates, speeds) if s > 0]
    st = _load_stats()
    now = time.time()

    def score(x):
        s = x["speed"]
        e = st.get(_skey(x["url"], x["proxy"])) or {}
        if e.get("bps") and now - e.get("t", 0) < 3 * 86400:
            s = (s + e["bps"]) / 2    # 测速只有几秒，实际下载的速度更可信
        if e.get("fail_t") and now - e["fail_t"] < 3600:
            s *= 0.3
        if prefer and prefer in x["name"]:
            s *= 1.3
        return -s
    res.sort(key=score)
    with _lock:
        _cache[key] = (time.time(), res)
    return res


def describe(ranked, n=4):
    return "，".join(f"{x['name']} {x['speed'] / 1e6:.1f} MB/s" for x in ranked[:n]) or "全部连不上"


# ---------------- 各类下载的候选线路 ----------------
def hf_candidates(repo, file, hf_only=False):
    return [(f"{name}（{rname}）", tpl.format(repo=repo, file=urllib.parse.quote(file)), proxy)
            for name, tpl, compat in HF_SOURCES if compat or not hf_only for rname, proxy in routes()]


def rank_hf(repo, file):
    """模型文件：hf-mirror / 魔搭 / HF 官方 × 各连接方式。魔搭上没有这个仓库时测速会失败，自动去掉。"""
    return rank(hf_candidates(repo, file), prefer="镜像" if settings.is_cn() else "官方")


def best_hf():
    """给用 huggingface_hub 的子进程：[(endpoint, 代理)] 从快到慢。用一个几百 MB 的文件测持续速度。"""
    r = rank(hf_candidates("openai-community/gpt2", "model.safetensors", hf_only=True), prefer="镜像" if settings.is_cn() else "官方")
    out = []
    for x in r:
        ep = x["url"].split("/openai-community/")[0]
        if (ep, x["proxy"]) not in out:
            out.append((ep, x["proxy"]))
    return out


def list_repo_files(repo, routes_ranked=None):
    """列出模型仓库里的文件 [(路径, 大小)]：依次试 hf-mirror / 魔搭 / HF 官方的接口（各连接方式），哪个通用哪个。"""
    tries = []
    for rname, proxy in routes():
        tries += [(f"hf-mirror（{rname}）", f"https://hf-mirror.com/api/models/{repo}/tree/main?recursive=true", proxy, "hf"),
                  (f"魔搭（{rname}）", f"https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Recursive=true", proxy, "ms"),
                  (f"HF 官方（{rname}）", f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true", proxy, "hf")]
    last = None
    for name, url, proxy, kind in tries:
        try:
            with open_url(url, proxy, timeout=12) as r:
                d = json.loads(r.read().decode("utf-8"))
            if kind == "hf":
                files = [(x["path"], int(x.get("size") or 0)) for x in d if x.get("type") == "file"]
            else:
                files = [(x["Path"], int(x.get("Size") or 0)) for x in ((d.get("Data") or {}).get("Files") or []) if x.get("Type") == "blob"]
            if files:
                return files
        except Exception as e:
            last = e
    raise RuntimeError(f"读不到模型仓库 {repo} 的文件列表（hf-mirror / 魔搭 / HF 官方都试过了）：{last}")


def github_candidates(url):
    """GitHub 下载：直连 + 几个加速代理，各自再分直连 / 代理。"""
    srcs = [("GitHub 直连", url)]
    custom = (settings.load().get("github_proxy") or "").strip()
    for gp in ([custom] if custom else []) + GH_PROXIES:
        if gp and ("GitHub 加速 " + gp, gp + url) not in srcs:
            srcs.append(("GitHub 加速 " + gp, gp.rstrip("/") + "/" + url))
    return [(f"{n}（{rn}）", u, p) for n, u in srcs for rn, p in routes()]


def rank_github(url):
    return rank(github_candidates(url), prefer="加速" if settings.is_cn() else "直连", small=True)


def rank_pip():
    """pip 镜像：测各镜像上一个小页面的速度。"""
    cands = [(f"{n}（{rn}）", u.rstrip("/") + "/pip/", p) for n, u in PIP for rn, p in routes()]
    r = rank(cands, prefer="清华" if settings.is_cn() else "官方", small=True)
    return [(x["url"][:-len("/pip/")], x["proxy"], x["name"]) for x in r]


def rank_torch(variant):
    """PyTorch 下载源：南大 / 阿里云 / 官方 × 各连接方式，测的是各源上 torch 的索引页。
    返回 [(pip 参数, 代理, 名字)]，从快到慢。torch 包 2–3 GB，选对源能差十几倍。"""
    srcs = [("南大镜像", f"https://mirror.nju.edu.cn/pytorch/whl/{variant}/torch/",
             lambda idx: ["--index-url", f"https://mirror.nju.edu.cn/pytorch/whl/{variant}", "--extra-index-url", idx]),
            ("阿里云镜像", f"https://mirrors.aliyun.com/pytorch-wheels/{variant}/",
             lambda idx: ["-f", f"https://mirrors.aliyun.com/pytorch-wheels/{variant}/", "-i", idx]),
            ("PyTorch 官方", f"https://download.pytorch.org/whl/{variant}/torch/",
             lambda idx: ["--index-url", f"https://download.pytorch.org/whl/{variant}", "--extra-index-url", idx])]
    cands = [(f"{n}（{rn}）", u, p) for n, u, _ in srcs for rn, p in routes()]
    r = rank(cands, prefer="镜像" if settings.is_cn() else "官方", small=True)
    pips = rank_pip()
    out = []
    for x in r:
        f = next(fn for n, u, fn in srcs if x["url"] == u)
        idx = next((u for u, p, _ in pips if p == x["proxy"]), settings.pip_index())   # 其余依赖用同一种连接方式下最快的 pip 源
        out.append((f(idx), x["proxy"], x["name"]))
    return out


# ---------------- 通用：按排好的线路下载整个文件（小文件：字典、zip 包） ----------------
def download_file(job, ranked, dest, label="", min_size=0):
    """ranked 来自 rank_xxx()；一条线路失败就换下一条。job 可以是 None。"""
    if not ranked:
        raise RuntimeError(f"{label or '文件'}所有下载线路都连不上，请检查网络；挂梯子的话确认梯子已连接，"
                           "或者到「训练器设置 → 网络」填写代理地址。")
    last = None
    for x in ranked:
        tmp = dest + ".part"
        try:
            if job:
                job.write(f"下载{label}（{x['name']}）")
            t0 = time.time()
            with open_url(x["url"], x["proxy"], timeout=120) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    if job is not None and getattr(job, "stop_flag", False):
                        raise RuntimeError("已取消")
                    b = r.read(1 << 18)
                    if not b:
                        break
                    f.write(b)
                    done += len(b)
                    if job and total:
                        job.set_progress(done / total, f"{done / 1e6:.1f} / {total / 1e6:.1f} MB · {x['name']}")
            if total and done < total or done < min_size:
                raise RuntimeError("下载不完整")
            os.replace(tmp, dest)
            if done > 8 << 20:
                record(x["url"], x["proxy"], done / max(0.1, time.time() - t0))
            return dest
        except Exception as e:
            last = e
            if job is not None and getattr(job, "stop_flag", False):
                raise
            record(x["url"], x["proxy"], failed=True)
            if job:
                job.write(f"⚠ {x['name']} 下载失败（{e}），换下一条线路")
            try:
                os.remove(tmp)
            except OSError:
                pass
    raise RuntimeError(f"{label or '文件'}所有线路都下载失败：{last}")


def proxy_env(env, proxy):
    """子进程（git / pip / huggingface_hub）用哪种连接：proxy 为 None 时明确直连（不让它自己去读系统代理）。"""
    env = dict(env)
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        env.pop(k, None)
    if proxy:
        env["HTTP_PROXY"] = env["HTTPS_PROXY"] = proxy
        env.pop("NO_PROXY", None)
    else:
        env["NO_PROXY"] = "*"
    return env


def github_prefixes(url):
    """GitHub 仓库地址的可用前缀（直连 / 各加速代理）及连接方式，从快到慢：[(前缀, 代理, 名字)]。"""
    out = []
    for x in rank_github(url):
        prefix = x["url"][:-len(url)] if x["url"].endswith(url) else ""
        out.append((prefix, x["proxy"], x["name"]))
    return out

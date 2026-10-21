# -*- coding: utf-8 -*-
"""训练环境：一个 venv 同时装 PyTorch + sd-scripts + musubi-tuner + 反推依赖。

sd-scripts 和 musubi-tuner 都是 kohya-ss 的项目，依赖版本（accelerate / transformers / diffusers / safetensors）
完全一致，可以共用一个环境，省一份 PyTorch 的下载和硬盘空间。"""
import json
import os
import time
import re
import shutil
import subprocess
import zipfile
import urllib.request

import netpick
import paths
import settings

ENGINES = {
    "sd-scripts": {"repo": "https://github.com/kohya-ss/sd-scripts", "branch": "main", "dir": paths.SD_SCRIPTS},
    "musubi-tuner": {"repo": "https://github.com/kohya-ss/musubi-tuner", "branch": "main", "dir": paths.MUSUBI},
}
# flash-linear-attention：MOSS-Image 的文本编码器 Qwen3.5 的 Gated DeltaNet 层用它的 Triton 内核（没有就退回很慢的 PyTorch 实现）
EXTRA_PKGS = ["pillow", "huggingface-hub", "numpy", "dadaptation", "flash-linear-attention==0.5.2",
              "lycoris-lora==4.0.0"]   # MOSS 专用引擎（moss_trainer）的 LoRA / LoKr / LoHa
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _git():
    if os.name == "nt":
        g = os.path.join(paths.ROOT, "git", "cmd", "git.exe")   # 整合包自带的 git
        if os.path.isfile(g):
            return g
    return shutil.which("git")


def status():
    py = paths.env_python()
    d = {"env": bool(py), "python": py, "engines": {}, "torch": None, "cuda": None}
    pv = pins()
    for k, e in ENGINES.items():
        d["engines"][k] = {"installed": os.path.isdir(e["dir"]), "commit": _commit(e["dir"]), "pinned": pv.get(k),
                           "match": not pv.get(k) or head(k) == pv.get(k)}
    info = os.path.join(paths.ENV, "lora_trainer_env.json")
    if os.path.isfile(info):
        try:
            d.update(json.load(open(info, encoding="utf-8")))
        except Exception:
            pass
    d["sync_reason"] = needs_sync()
    return d


def _commit(d):
    g = _git()
    if not g or not os.path.isdir(os.path.join(d, ".git")):
        v = os.path.join(d, ".zip_version")
        return open(v, encoding="utf-8").read().strip() if os.path.isfile(v) else None
    try:
        r = subprocess.run([g, "log", "-1", "--format=%h %cd", "--date=short"], cwd=d, capture_output=True, text=True,
                           creationflags=NO_WINDOW, timeout=10)
        return r.stdout.strip() or None
    except Exception:
        return None


def _env():
    e = settings.child_env()
    e["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    e["PIP_NO_INPUT"] = "1"
    return e


_pip_routes = {}


def _pip(job, args, cwd=None, proxy=None, label=""):
    """pip 安装：镜像源按测速排序（清华 / 阿里云 / 中科大 / 官方 × 直连 / 代理），一个失败换下一个。
    调用方自己指定了源（PyTorch）时，proxy / label 是那个源测出来的连接方式。"""
    py = paths.env_python()
    cmd = [py, "-m", "pip", "install", "--progress-bar", "off"] + args
    explicit = any(a in ("-i", "--index-url", "-f", "--find-links") for a in args)
    if explicit:
        routes = [(None, proxy, label)]
    else:
        if not _pip_routes.get("list") or time.time() - _pip_routes.get("t", 0) > 600:
            _pip_routes.update(list=netpick.rank_pip() or [(settings.pip_index(), None, "默认")], t=time.time())
        routes = _pip_routes["list"]
    rc = None
    for i, (idx, proxy, name) in enumerate(routes):
        c = cmd + (["-i", idx] if idx else [])
        env = netpick.proxy_env(_env(), proxy) if idx or explicit else _env()
        if name:
            job.write(f"pip 源：{name}")
        rc = job.run_cmd(c, cwd=cwd, env=env)
        if rc == 0:
            if i:   # 换过源的话，下次直接用这个
                _pip_routes["list"] = [routes[i]] + [r for r in routes if r is not routes[i]]
            return
        if i + 1 < len(routes):
            job.write(f"⚠ 用这个源安装失败（退出码 {rc}），换下一个源重试")
    raise RuntimeError(f"pip 安装失败（退出码 {rc}，所有源都试过了）：{' '.join(args[:3])}")


def relocate_env():
    """整合包解压到别的文件夹 / 电脑后，训练环境里记着的旧路径改成现在的位置：venv 的 Python 位置（pyvenv.cfg），
    两个引擎的可编辑安装（.pth / __editable__ finder / direct_url.json）。没搬过就什么都不做。返回是否改过。"""
    import glob
    if os.name != "nt":
        return False   # Linux 的 venv 脚本里写的是绝对路径，搬家后请直接重建训练环境（比改路径可靠）
    cfg = os.path.join(paths.ENV, "pyvenv.cfg")
    try:
        text = open(cfg, encoding="utf-8").read()
    except OSError:
        return False
    m = re.search(r"^home\s*=\s*(.+?)\s*$", text, re.M)
    new_home = os.path.join(paths.ROOT, "python")
    if not m or not os.path.isfile(os.path.join(new_home, "python.exe")):
        return False
    old_home = m.group(1)
    if os.path.normcase(os.path.normpath(old_home)) == os.path.normcase(new_home):
        return False
    old_root = os.path.dirname(os.path.normpath(old_home))   # 训练环境是用训练器自带的 python 建的：home = <旧位置>\python
    pairs = [(old_root, paths.ROOT), (old_root.replace("\\", "\\\\"), paths.ROOT.replace("\\", "\\\\")),
             (old_root.replace("\\", "/"), paths.ROOT.replace("\\", "/"))]
    site = os.path.join(paths.ENV, "Lib", "site-packages")
    files = [cfg] + glob.glob(os.path.join(site, "*.pth")) + glob.glob(os.path.join(site, "__editable__*finder.py")) \
        + glob.glob(os.path.join(site, "*.dist-info", "direct_url.json"))
    for f in files:
        try:
            s = open(f, encoding="utf-8").read()
        except (OSError, UnicodeDecodeError):
            continue
        n = s
        for a, b in pairs:
            n = n.replace(a, b)
        if n != s:
            with open(f, "w", encoding="utf-8", newline="") as fh:
                fh.write(n)
    return True


# ---------------- 引擎代码 ----------------
def pins():
    """这个版本的训练器测试过的引擎版本（version.json 的 engines：{名字: 提交号}）。
    更新时引擎切到这个版本，不直接追上游最新——上游改了参数名训练器就会报错，训练器和引擎要一起更新。"""
    try:
        v = json.load(open(os.path.join(paths.ROOT, "version.json"), encoding="utf-8"))
        return {k: x for k, x in (v.get("engines") or {}).items() if k in ENGINES and x}
    except Exception:
        return {}


def head(name):
    """引擎当前的完整提交号（git 仓库）或 zip 包记下的版本。"""
    d = ENGINES[name]["dir"]
    g = _git()
    if g and os.path.isdir(os.path.join(d, ".git")):
        try:
            r = subprocess.run([g, "rev-parse", "HEAD"], cwd=d, capture_output=True, text=True, creationflags=NO_WINDOW, timeout=10)
            return r.stdout.strip() or None
        except Exception:
            return None
    v = os.path.join(d, ".zip_version")
    return open(v, encoding="utf-8").read().strip().split()[-1] if os.path.isfile(v) else None


def fetch_engine(job, name, update=False, ref=None):
    """准备引擎代码。update：更新到 ref（提交号；None = 远端最新）。"""
    e = ENGINES[name]
    d = e["dir"]
    g = _git()
    if os.path.isdir(os.path.join(d, ".git")) and g:
        if update:
            if ref and head(name) == ref:
                job.write(f"{name} 已经是训练器要求的版本 {ref[:7]}")
                return
            job.write(f"更新 {name} 到 {ref[:7] if ref else '最新版本'} …")
            url = settings.github_url(e["repo"] + ".git")
            job.run_cmd([g, "-C", d, "remote", "set-url", "origin", url], env=_env())
            rc = job.run_cmd([g, "-C", d, "fetch", "--depth", "1", "origin", ref or e["branch"]], env=_env())
            if rc == 0:
                rc = job.run_cmd([g, "-C", d, "reset", "--hard", "FETCH_HEAD"], env=_env())
            if rc != 0:
                job.write(f"⚠ git 更新失败，改用 zip 下载 {name}")
                _zip_download(job, name, ref)
        return
    if os.path.isdir(d) and not update:
        return
    if update and ref and head(name) == ref:
        return
    if g:
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
        for prefix, proxy, label in netpick.github_prefixes(e["repo"]) or [("", None, "GitHub")]:
            shutil.rmtree(d, ignore_errors=True)
            job.write(f"git clone（{label}）")
            rc = job.run_cmd([g, "clone", "--depth", "1", "-b", e["branch"], prefix + e["repo"] + ".git", d],
                             env=netpick.proxy_env(_env(), proxy))
            if rc == 0:
                if prefix:   # 加速代理克隆下来的，remote 改回 GitHub 原地址
                    job.run_cmd([g, "-C", d, "remote", "set-url", "origin", e["repo"] + ".git"], env=_env())
                if ref and head(name) != ref:   # 切到训练器要求的版本
                    job.run_cmd([g, "-C", d, "fetch", "--depth", "1", "origin", ref], env=netpick.proxy_env(_env(), proxy))
                    if job.run_cmd([g, "-C", d, "reset", "--hard", "FETCH_HEAD"], env=_env()) != 0:
                        job.write(f"⚠ 切换到 {ref[:7]} 失败，先用最新版本")
                return
            job.write("⚠ 这条线路 git clone 失败，换下一条")
        job.write("git clone 全部失败，改用 zip 下载")
    _zip_download(job, name, ref)


def _zip_download(job, name, ref=None):
    e = ENGINES[name]
    url = f"{e['repo']}/archive/{ref}.zip" if ref else f"{e['repo']}/archive/refs/heads/{e['branch']}.zip"
    tmp = os.path.join(paths.ENGINES, name + ".zip")
    netpick.download_file(job, netpick.rank_github(url), tmp, f" {name} 源码包")
    tmpdir = os.path.join(paths.ENGINES, name + "_tmp")
    shutil.rmtree(tmpdir, ignore_errors=True)
    with zipfile.ZipFile(tmp) as z:
        z.extractall(tmpdir)
    inner = os.path.join(tmpdir, os.listdir(tmpdir)[0])
    shutil.rmtree(e["dir"], ignore_errors=True)
    shutil.move(inner, e["dir"])
    shutil.rmtree(tmpdir, ignore_errors=True)
    os.remove(tmp)
    with open(os.path.join(e["dir"], ".zip_version"), "w", encoding="utf-8") as f:
        f.write("zip " + time.strftime("%Y-%m-%d") + (" " + ref if ref else ""))


# ---------------- 安装 ----------------
def torch_sources():
    v = settings.load().get("torch_variant") or "cu128"
    official = ["--index-url", f"https://download.pytorch.org/whl/{v}"]
    if settings.is_cn():
        return [["--index-url", f"https://mirror.nju.edu.cn/pytorch/whl/{v}", "--extra-index-url", settings.pip_index()],
                ["-f", f"https://mirrors.aliyun.com/pytorch-wheels/{v}/", "-i", settings.pip_index()],
                official]
    return [official]


def _torch_ok(job):
    py = paths.env_python()
    r = subprocess.run([py, "-c", "import torch,json;print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
                                  "'ok':torch.cuda.is_available()}))"],
                       capture_output=True, text=True, env=_env(), creationflags=NO_WINDOW)
    try:
        d = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        job.write((r.stderr or "").strip()[-500:])
        return None
    job.write(f"PyTorch {d['torch']} · CUDA {d['cuda']} · 显卡可用：{'是' if d['ok'] else '否'}")
    return d if d.get("cuda") else None


def install_torch(job, force=False):
    if not force:
        d = _torch_ok(job)
        if d:
            return d
    last = None
    v = settings.load().get("torch_variant") or "cu128"
    job.write("测试 PyTorch 下载源…")
    srcs = netpick.rank_torch(v) or [(s, None, "默认") for s in torch_sources()]
    job.write("PyTorch 下载源：" + "，".join(n for _, _, n in srcs[:4]))
    for src, proxy, name in srcs:
        try:
            job.write(f"安装 PyTorch（{name}）")
            extra = ["--force-reinstall", "--no-cache-dir"] if force else []
            _pip(job, ["torch", "torchvision"] + extra + src, proxy=proxy, label=name)
            d = _torch_ok(job)
            if d:
                return d
            job.write("装上的是 CPU 版 PyTorch，换下一个源重装")
            force = True
        except Exception as e:
            last = e
            job.write(f"这个源失败：{e}")
    raise RuntimeError(f"PyTorch 安装失败：{last or '只能装上 CPU 版本'}")


def onnxruntime_spec():
    """WD14 用的 onnxruntime-gpu 要和 PyTorch 的 CUDA 大版本一致（它会复用 PyTorch 自带的 CUDA 库）。
    PyPI 上 1.27 起默认是 CUDA 13 版，CUDA 12 的 PyTorch 要用 1.26 及以下，否则只能跑 CPU。"""
    v = settings.load().get("torch_variant") or "cu128"
    return "onnxruntime-gpu>=1.27" if v.startswith("cu13") else "onnxruntime-gpu>=1.20,<1.27"


def _venv_bases(base):
    """能用来建 venv 的解释器，按优先级：指定的那个 → 项目自带的 → conda / 系统的其它版本。"""
    cands = [base]
    extra = [os.path.join(paths.ROOT, "python", "bin", "python"),
             "/root/miniconda3/bin/python", "/opt/conda/bin/python",
             shutil.which("python3.13"), shutil.which("python3.12"), shutil.which("python3.11"),
             shutil.which("python3.10"), shutil.which("python3")]
    for p in extra:
        if p and os.path.isfile(p) and os.path.abspath(p) not in {os.path.abspath(x) for x in cands}:
            cands.append(p)
    return cands


def _create_venv(job, base):
    """建训练环境。Ubuntu 的系统 Python 常常没装 python3-venv（ensurepip 缺失），
    这时自动换一个带 venv 的解释器（项目自带的 python/、miniconda 等）再试。"""
    for cand in _venv_bases(base):
        shutil.rmtree(paths.ENV, ignore_errors=True)
        job.write(f"用 {cand} 创建训练环境…")
        rc = job.run_cmd([cand, "-m", "venv", paths.ENV], env=_env())
        if rc == 0 and paths.env_python():
            if job.run_cmd([paths.env_python(), "-m", "pip", "--version"], env=_env()) == 0:
                return
            job.write("这个 Python 建出来的环境里没有 pip，换一个再试")
        elif rc == 0:
            job.write("环境目录建出来了但没有 python，换一个再试")
        else:
            job.write("这个 Python 不能建 venv（缺 ensurepip / python3-venv），换一个再试")
    shutil.rmtree(paths.ENV, ignore_errors=True)
    raise RuntimeError("创建虚拟环境失败：系统 Python 缺 venv / ensurepip。"
                       "Ubuntu / Debian 装一下 python3-venv（apt install -y python3-venv），"
                       "或用 PYTHON=/root/miniconda3/bin/python ./install.sh 指定一个带 venv 的 Python。")


def install(job, repair=False, reinstall_torch=False):
    """完整安装 / 修复。可重复执行，已经装好的部分会很快跳过。"""
    job.set_progress(0.02, "准备训练引擎代码")
    pv = pins()
    for n in ENGINES:
        fetch_engine(job, n, update=bool(pv.get(n)) and head(n) != pv.get(n), ref=pv.get(n))
    job.set_progress(0.08, "创建虚拟环境")
    if not paths.env_python():
        base = paths.base_python()
        job.write(f"用 {base} 创建训练环境 {paths.ENV}")
        _create_venv(job, base)
        _pip(job, ["-U", "pip", "setuptools", "wheel"])
    job.set_progress(0.15, "下载安装 PyTorch（约 2.7 GB，这一步没有进度条，10 MB/s 网速约 5 分钟）")
    tinfo = install_torch(job, force=reinstall_torch)
    _deps(job, tinfo.get("torch"), upgrade=repair, p0=0.6)
    info = {"torch": tinfo.get("torch"), "cuda": tinfo.get("cuda"), "variant": settings.load().get("torch_variant")}
    _write_stamp(info)
    job.set_progress(1, "训练环境安装完成")
    return info


def _deps(job, torch_ver, upgrade=False, p0=0.6):
    """引擎依赖 + 训练器自己的依赖（反推、加速组件）+ 环境检查。安装和更新共用。"""
    job.set_progress(p0, "安装 sd-scripts 依赖")
    _pip(job, ["-r", "requirements.txt"] + (["--upgrade"] if upgrade else []), cwd=paths.SD_SCRIPTS)
    job.set_progress(p0 + (1 - p0) * 0.4, "安装 musubi-tuner")
    _pip(job, ["-e", paths.MUSUBI])
    job.set_progress(p0 + (1 - p0) * 0.65, "安装反推和加速组件")
    try:
        _pip(job, EXTRA_PKGS + [onnxruntime_spec()])
    except Exception as e:
        # flash-linear-attention 是 MOSS-Image 文本编码器的加速项，装不上只影响速度，不该挡住整个环境
        job.write(f"⚠ 加速组件没全装上（{e}），先装其余依赖")
        _pip(job, [p for p in EXTRA_PKGS if not p.startswith("flash-linear-attention")] + [onnxruntime_spec()])
    m = re.match(r"(\d+)\.(\d+)", torch_ver or "")
    if os.name == "nt" and m and int(m.group(1)) == 2 and int(m.group(2)) >= 6:
        tm = int(m.group(2)) - 4
        try:   # int8 ConvRot 量化的加速内核；装不上不影响其它功能
            _pip(job, [f"triton-windows>=3.{tm},<3.{tm + 1}"])
        except Exception as e:
            job.write(f"triton 没装上（不影响训练，只是 int8 量化会慢一些）：{e}")
    job.set_progress(p0 + (1 - p0) * 0.9, "检查环境")
    rc = job.run_cmd([paths.env_python(), "-c", "import accelerate, transformers, diffusers, safetensors, bitsandbytes; "
                                                "import library.anima_train_utils; import musubi_tuner; print('环境检查通过')"],
                     cwd=paths.SD_SCRIPTS, env=_env())
    if rc != 0:
        raise RuntimeError("环境检查没通过，详见日志")


# ---------------- 训练器和引擎 / 环境同步 ----------------
STAMP = os.path.join(paths.ENV, "lora_trainer_env.json")


def deps_fingerprint():
    """依赖清单的指纹：两个引擎的依赖文件 + 训练器自己的依赖。任何一个变了（训练器或引擎更新带来新依赖）就要重新装。"""
    import hashlib
    h = hashlib.sha1()
    for f in (os.path.join(paths.SD_SCRIPTS, "requirements.txt"), os.path.join(paths.MUSUBI, "pyproject.toml")):
        try:
            h.update(open(f, "rb").read())
        except OSError:
            h.update(b"-")
    h.update(json.dumps(EXTRA_PKGS + [onnxruntime_spec()]).encode())
    return h.hexdigest()[:16]


def _write_stamp(extra=None):
    """记下这次同步后的状态：引擎版本 + 依赖指纹。启动时和它比，对不上就自动同步。"""
    info = {}
    try:
        info = json.load(open(STAMP, encoding="utf-8"))
    except Exception:
        pass
    info.update(extra or {})
    info.update(deps=deps_fingerprint(), engine_heads={n: head(n) for n in ENGINES}, synced_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(STAMP, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False)


def needs_sync():
    """训练环境装好了、但和当前训练器对不上时返回原因（中文）；对得上返回 ""。
    训练器不管怎么更新的（一键更新 / git pull / 手动覆盖文件），启动时都会发现并自动同步。"""
    if not paths.env_python() or not all(os.path.isdir(e["dir"]) for e in ENGINES.values()):
        return ""
    why = []
    for n, ref in pins().items():
        h = head(n)
        if h and h != ref:
            why.append(f"{n} 是 {h[:7]}，这个版本的训练器要求 {ref[:7]}")
    try:
        st = json.load(open(STAMP, encoding="utf-8"))
    except Exception:
        st = {}
    if st.get("deps") != deps_fingerprint():
        why.append("依赖清单有变化" if st.get("deps") else "还没有记录过依赖版本")
    return "；".join(why)


def sync(job):
    """把引擎切到训练器要求的版本，并装好对应的依赖。一键更新、启动时自动同步、报错修复都走这里。"""
    pv = pins()
    job.set_progress(0.05, "同步训练引擎版本")
    for n in ENGINES:
        fetch_engine(job, n, update=True, ref=pv.get(n))
    if not paths.env_python():
        job.write("训练环境还没安装，跳过依赖同步")
        return status()
    tv = None
    try:
        tv = json.load(open(STAMP, encoding="utf-8")).get("torch")
    except Exception:
        pass
    _deps(job, tv, p0=0.3)
    _write_stamp()
    job.set_progress(1, "训练引擎和环境已同步")
    return status()


def update_engines(job):
    return sync(job)


def pin_current():
    """开发者用：把两个引擎现在的版本写进 version.json（测试通过后再发布）。"""
    p = os.path.join(paths.ROOT, "version.json")
    v = json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {"version": "0.1.0"}
    v["engines"] = {n: head(n) for n in ENGINES}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(v, f, ensure_ascii=False, indent=1)
    return v["engines"]


def pip_install(job, pkg):
    _pip(job, pkg.split())

# -*- coding: utf-8 -*-
"""反推打标子进程（在训练环境里运行，需要 torch / onnxruntime / transformers）。

用法：python tag_worker.py <config.json>
进度以「@@{json}」行输出，界面进程解析后显示；其余输出进日志。"""
import csv
import json
import os
import threading
import sys
import time
import traceback
import warnings

# bitsandbytes 8bit 推理时每层每次都会警告一遍，几千行刷掉日志里有用的信息
warnings.filterwarnings("ignore", message=r".*MatMul8bitLt.*")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import captionutil as cu  # noqa: E402


MAX_PIXELS = 1024 * 1024   # 自然语言反推时图片缩到的最大像素数


def vram_free_gb():
    """现在还能用多少显存（GB）。设置了 LORA_VRAM_LIMIT_GB（测试用：在大显存卡上模拟小显存卡）时按上限算。"""
    import torch
    free, total = torch.cuda.mem_get_info()
    lim = float(os.environ.get("LORA_VRAM_LIMIT_GB") or 0)
    if lim:
        free = min(free, lim * 2 ** 30 - torch.cuda.memory_reserved())
    return free / 2 ** 30


def apply_vram_limit():
    """测试用：LORA_VRAM_LIMIT_GB=8 时把这个进程能用的显存锁在 8 GB（超出直接报显存不足，不会偷偷借内存）。"""
    lim = float(os.environ.get("LORA_VRAM_LIMIT_GB") or 0)
    if lim:
        import torch
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, lim / total))
        log(f"[测试] 显存上限锁定为 {lim:g} GB（模拟 {lim:g} GB 显卡）")


def auto_quant(size_gb):
    """精度选「自动」：按剩余显存选 bf16 / 4bit。size_gb 是 bf16 模型文件大小；另外留 1.5 GB 给图片和生成。
    不自动选 8bit：bitsandbytes 的 8bit 推理比 4bit 慢 2–3 倍（实测 Qwen3-VL-4B 5 张图 304 秒 vs 4bit 约 120 秒），质量差别很小。"""
    free = vram_free_gb()
    q = "bf16" if free >= size_gb * 1.05 + 1.5 else "4bit"
    log(f"精度自动选择：可用显存 {free:.1f} GB，模型 bf16 约 {size_gb:g} GB → 用 {q}"
        + ("" if q == "bf16" else "（显存不够放原精度，量化后质量略降）"))
    if q == "4bit" and free < size_gb * 0.35 + 1.2:
        log(f"⚠ 显存可能连 4bit 都放不下（约需 {size_gb * 0.35 + 1.2:.1f} GB），可以换小一点的模型（Qwen3-VL-4B）或关掉占显存的程序")
    return q


def emit(**kw):
    print("@@" + json.dumps(kw, ensure_ascii=False), flush=True)


def log(*a):
    print(*a, flush=True)


def fetch(repo, models_dir, patterns=None, local=None):
    """下载 / 复用 HuggingFace 模型到 models/taggers/<repo 名>（或指定的 local 目录）。"""
    local = local or os.path.join(models_dir, repo.replace("/", "__"))
    if os.path.isfile(os.path.join(local, ".lora_complete")):   # 训练器已经用多连接下载器下好了，不用再联网检查
        log(f"使用已下载的模型：{local}")
        return local
    emit(stage="download", text=f"下载 / 检查模型 {repo}")
    # 训练器测好速排好序的线路（镜像 / 官方 × 直连 / 代理），一条失败换下一条；没有就按 HF_ENDPOINT 来
    try:
        routes = json.loads(os.environ.get("LORA_HF_ROUTES") or "[]")
    except ValueError:
        routes = []
    routes = routes or [[os.environ.get("HF_ENDPOINT", "https://huggingface.co"), None]]
    stop = threading.Event()
    _DL["total"] = 0
    threading.Thread(target=_dl_progress, args=(repo, local, patterns, routes[0][0], stop), daemon=True).start()
    try:
        return _fetch_routes(repo, local, patterns, routes)
    finally:
        stop.set()


_DL = {"total": 0}   # 要下载的总字节数（进度线程查到后填进来，看门狗用它判断还剩多少）


def _dir_size(d):
    n = 0
    for dp, _, fs in os.walk(d):
        for f in fs:
            try:
                n += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return n


def _dl_progress(repo, local, patterns, endpoint, stop):
    """huggingface_hub 下大文件时只显示「第几个文件」，这里按文件夹大小报「已下载 / 总大小 · 速度」。"""
    import fnmatch
    total = 0
    try:
        from huggingface_hub import HfApi
        info = HfApi(endpoint=endpoint).model_info(repo, files_metadata=True)
        for s in info.siblings or []:
            if not patterns or any(fnmatch.fnmatch(s.rfilename, p) for p in patterns):
                total += s.size or 0
    except Exception:
        pass
    _DL["total"] = total
    base = _dir_size(local) if os.path.isdir(local) else 0
    t0, last = time.time(), base
    while not stop.wait(2):
        cur = _dir_size(local) if os.path.isdir(local) else 0
        speed = (cur - base) / max(0.1, time.time() - t0)
        if cur == last and cur >= total > 0:
            continue
        last = cur
        txt = f"下载模型 {repo.split('/')[-1]}：{cur / 1e9:.2f}" + (f" / {total / 1e9:.2f}" if total else "") + f" GB · {speed / 1e6:.1f} MB/s"
        if total and speed > 0:
            eta = max(0, total - cur) / speed
            txt += f" · 剩余 {int(eta // 60)} 分 {int(eta % 60)} 秒"
        emit(dl=cur, total=total, text=txt)


SLOW_BPS = 1 << 20     # 持续低于 1 MB/s 算太慢（还有别的线路时换掉）
SLOW_WINDOW = 45       # 按最近 45 秒的平均速度判断
STALL_SEC = 90         # 90 秒一个字节都没下到 = 卡死

_CHILD = """import json, sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2], allow_patterns=json.loads(sys.argv[3]) or None, endpoint=sys.argv[4])
"""


def _fetch_routes(repo, local, patterns, routes):
    """按线路顺序下载。每条线路在子进程里跑：出错、卡死、或者明显太慢（测速时快、真下起来慢的线路很常见）
    就结束子进程换下一条；huggingface_hub 会接着上次没下完的部分继续，不会从头下。"""
    import subprocess
    from collections import deque
    last = None
    for i, (ep, proxy) in enumerate(routes):
        more = i + 1 < len(routes)
        env = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONIOENCODING="utf-8")
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy"):
            env.pop(k, None)
        if proxy:
            env["HTTP_PROXY"] = env["HTTPS_PROXY"] = proxy
        else:
            env["NO_PROXY"] = "*"
        log(f"准备模型 {repo}（{ep}，{'代理 ' + proxy if proxy else '直连'}）…")
        p = subprocess.Popen([sys.executable, "-c", _CHILD, repo, local, json.dumps(patterns or []), ep], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        tail = deque(maxlen=20)
        threading.Thread(target=lambda: [tail.append(x.rstrip()) for x in p.stdout], daemon=True).start()
        t0 = time.time()
        hist = deque()           # (时间, 已下载字节)
        moved = t0
        why = ""
        while p.poll() is None:
            time.sleep(3)
            now, cur = time.time(), _dir_size(local) if os.path.isdir(local) else 0
            if hist and cur > hist[-1][1]:
                moved = now
            hist.append((now, cur))
            while hist and now - hist[0][0] > SLOW_WINDOW:
                hist.popleft()
            if now - moved > STALL_SEC:
                why = f"{STALL_SEC} 秒没有下载到数据"
            elif more and now - t0 > SLOW_WINDOW + 15 and len(hist) > 2:
                bps = (hist[-1][1] - hist[0][1]) / max(1.0, hist[-1][0] - hist[0][0])
                left = _DL["total"] - cur if _DL["total"] else 1 << 30
                if bps < SLOW_BPS and left > 100 << 20:
                    why = f"速度只有 {bps / 1e6:.2f} MB/s"
            if why:
                p.kill()
                p.wait()
                break
        if not why and p.returncode == 0:
            return local
        err = why or (next((x for x in reversed(tail) if "Error" in x), "") or (tail[-1] if tail else f"退出码 {p.returncode}"))
        last = err
        if more:
            log(f"⚠ 这条线路{'太慢' if why.startswith('速度') else ('卡住了' if why else '下载失败')}（{err[:160]}），换下一条线路，已下载的部分接着下")
    raise RuntimeError(f"模型 {repo} 所有线路都下载失败：{last}")


# ---------------- WD14 ----------------
class WD14:
    def __init__(self, cfg, models_dir):
        import numpy as np  # noqa: F401
        try:
            import torch  # noqa: F401  先导入 torch，让 onnxruntime 用上它自带的 CUDA / cuDNN
        except Exception:
            pass
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
        repo = cu.WD14_MODELS.get(cfg.get("model"), {}).get("repo") or cfg.get("model")
        name = repo.split("/")[-1]
        local = (cfg.get("launcher_models") or {}).get(name)
        if local and os.path.isfile(os.path.join(local, "model.onnx")):
            log(f"使用 WWY 启动器已下载的模型：{local}")
            emit(stage="load", text=f"使用 WWY 启动器里的 {name}")
            d = local
        elif cfg.get("launcher_dir") and name in cu.LAUNCHER_WD14:
            # 启动器也认识的模型：下载到启动器的 wd_tagger_models，两边共用一份
            d = os.path.join(cfg["launcher_dir"], name)
            log(f"下载 {repo} 到 WWY 启动器的模型目录：{d}")
            emit(stage="download", text=f"下载 {name}（存到 WWY 启动器，两边共用）")
            fetch(repo, models_dir, ["model.onnx", "selected_tags.csv"], local=d)
            import shutil
            shutil.rmtree(os.path.join(d, ".cache"), ignore_errors=True)   # 不在启动器目录里留下载缓存
        else:
            d = fetch(repo, models_dir, ["model.onnx", "selected_tags.csv"])
        provs = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(os.path.join(d, "model.onnx"), providers=provs)
        log("WD14 使用：" + ", ".join(self.sess.get_providers()))
        self.size = self.sess.get_inputs()[0].shape[1]
        self.tags, self.cats = [], []
        with open(os.path.join(d, "selected_tags.csv"), encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.tags.append(row["name"])
                self.cats.append(int(row["category"]))
        self.cfg = cfg
        self.exclude = set(t.strip().replace(" ", "_") for t in cu.split_tags(cfg.get("exclude") or ""))

    def __call__(self, img):
        import numpy as np
        from PIL import Image
        im = img.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
        w, h = im.size
        s = max(w, h)
        sq = Image.new("RGB", (s, s), (255, 255, 255))
        sq.paste(im, ((s - w) // 2, (s - h) // 2))
        sq = sq.resize((self.size, self.size), Image.BICUBIC)
        x = np.asarray(sq, dtype=np.float32)[:, :, ::-1]   # RGB → BGR
        x = np.expand_dims(x, 0)
        probs = self.sess.run(None, {self.sess.get_inputs()[0].name: x})[0][0]
        gthr = float(self.cfg.get("general_threshold") or 0.35)
        cthr = float(self.cfg.get("character_threshold") or 0.85)
        chars, general, rating = [], [], None
        best_r = -1
        for i, p in enumerate(probs):
            name, cat = self.tags[i], self.cats[i]
            if name in self.exclude:
                continue
            if cat == 9:
                if p > best_r:
                    best_r, rating = p, name
            elif cat == 4 and p >= cthr:
                chars.append((p, name))
            elif cat == 0 and p >= gthr:
                general.append((p, name))
        chars.sort(reverse=True)
        general.sort(reverse=True)
        out = [n for _, n in chars] + [n for _, n in general]
        if self.cfg.get("include_rating") and rating:
            out.insert(0, rating)
        if self.cfg.get("replace_underscore", True):
            out = [t.replace("_", " ") if len(t) > 3 else t for t in out]   # ^_^ 这类表情标签保留下划线
        if self.cfg.get("escape_brackets", False):
            out = [t.replace("(", r"\(").replace(")", r"\)") for t in out]
        return out


# ---------------- 自然语言 VLM ----------------
class VLM:
    def __init__(self, cfg, models_dir):
        import torch
        from transformers import AutoProcessor
        key = cfg.get("model") or "toriigate"
        info = cu.VLM_MODELS.get(key, {})
        repo = cfg.get("repo") or info.get("repo")
        self.key = key
        d = fetch(repo, models_dir)
        kw = {"dtype": torch.bfloat16, "device_map": "cuda"}
        q = cfg.get("quant") or "auto"
        if q == "auto":
            q = auto_quant(info.get("size_gb") or 17)
        if q in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                                            bnb_4bit_quant_type="nf4") if q == "4bit"
                                         else BitsAndBytesConfig(load_in_8bit=True))
        emit(stage="load", text="加载模型到显卡…")
        model = None
        errs = []
        import transformers
        for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "LlavaForConditionalGeneration"):
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(d, **kw)
                break
            except Exception as e:
                errs.append(f"{cls_name}: {e}")
        if model is None:
            raise RuntimeError("模型加载失败：\n" + "\n".join(errs))
        self.model = model.eval()
        pkw = {}
        if key == "toriigate":
            pkw = {"min_pixels": 256 * 32 * 32, "max_pixels": 1024 * 1024}
        self.proc = AutoProcessor.from_pretrained(d, **pkw)
        self.cfg = cfg
        self.torch = torch
        # 停止符：有的模型（ToriiGate 0.5）没带 generation_config.json，只认 <|endoftext|>，
        # 可对话模板每句话以 <|im_end|> 结尾——写完描述也停不下来，每张图都生成满 max_tokens（被当特殊符号删掉，看不出来）
        tok = getattr(self.proc, "tokenizer", self.proc)
        ids = [model.generation_config.eos_token_id, getattr(tok, "eos_token_id", None)]
        ids = [i for x in ids for i in (x if isinstance(x, list) else [x]) if i is not None]
        for t in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>"):
            i = tok.convert_tokens_to_ids(t) if t in tok.get_vocab() else None
            if i is not None:
                ids.append(i)
        self.eos = sorted(set(ids)) or None
        # 成批生成：短的提示词在左边补齐，生成的新字才都接在最右边
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.pad = tok.pad_token_id
        self.bs = pick_batch(cfg)

    def prompt(self, tags=None):
        cfg = self.cfg
        if self.key == "toriigate":
            system = cu.TORII_SYSTEM
            user = cu.torii_query(cfg.get("torii_format") or "long", tags if cfg.get("feed_tags", True) else None,
                                  cfg.get("use_names", True))
        else:
            system = "You are a helpful image captioner."
            user = cfg.get("prompt") or cu.PROMPT_PRESETS["detailed"]
            if tags and cfg.get("feed_tags", True):
                user += "\nReference tags (may contain mistakes): " + ", ".join(tags)
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]}]
        if self.key == "joycaption":
            # JoyCaption 的对话模板会自动插入图片占位符（官方用法）
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @staticmethod
    def prep(img):
        img = img.convert("RGB")
        # 大图先缩到约 100 万像素：Qwen-VL 系列按原图大小切视觉 token，D 站 4000×6000 的图会变成几万个 token，
        # 显存爆了以后 Windows 驱动会悄悄借用内存（共享显存），结果是吃掉几十 GB 内存、每张图要几分钟。
        if img.width * img.height > MAX_PIXELS:
            from PIL import Image
            s = (MAX_PIXELS / (img.width * img.height)) ** 0.5
            img = img.resize((max(28, int(img.width * s)), max(28, int(img.height * s))), Image.LANCZOS)
        return img

    def batch(self, imgs, tags_list):
        """几张图一起生成。一次只生成一张时，每个字都要 Python 给显卡发几百个小指令，CPU 一个核跑满、显卡在等；
        一起生成时指令数不变、每个指令干几张图的活，显卡才吃得满。"""
        torch = self.torch
        cfg = self.cfg
        texts = [self.prompt(t) for t in tags_list]
        inputs = self.proc(text=texts, images=list(imgs), return_tensors="pt", padding=True).to(self.model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        try:
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=int(cfg.get("max_tokens") or 512),
                                          do_sample=float(cfg.get("temperature") or 0) > 0,
                                          temperature=float(cfg.get("temperature") or 0.6) or None, top_p=0.9,
                                          repetition_penalty=1.05, eos_token_id=self.eos, pad_token_id=self.pad)
            res = self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        finally:
            del inputs
            out = None
            torch.cuda.empty_cache()   # 每批之后还回显存，避免越用越多
        return [(r.split("</think>", 1)[1] if "</think>" in r else r).strip() for r in res]

    def __call__(self, img, tags=None):
        return self.batch([self.prep(img)], [tags])[0]


def pick_batch(cfg):
    """一次生成几张：设置了就用设置的（LORA_TAG_BATCH 环境变量优先，测试用），否则按模型加载后剩的显存估，最多 8 张。
    每多一张大约多占 1 GB（看图的视觉部分最占）；估多了也不要紧，显存不足时会自动减半重来。"""
    v = os.environ.get("LORA_TAG_BATCH") or cfg.get("batch")
    if v:
        bs = max(1, int(v))
        log(f"成批生成：一次 {bs} 张（手动设置）")
        return bs
    free = vram_free_gb()
    bs = max(1, min(8, int(free - 1.5)))
    log(f"成批生成：模型加载后还剩 {free:.1f} GB 显存 → 一次 {bs} 张")
    return bs


def main():
    cfg = json.load(open(sys.argv[1], encoding="utf-8"))
    apply_vram_limit()
    from PIL import Image
    files = cfg["files"]
    mode = cfg["mode"]          # wd14 / vlm / hybrid
    out = cfg.get("out") or {}
    ext = out.get("ext") or ".txt"
    models_dir = cfg["models_dir"]
    wcfg = dict(cfg.get("wd14") or {}, launcher_models=cfg.get("launcher_models"), launcher_dir=cfg.get("launcher_dir"))
    wd = WD14(wcfg, models_dir) if mode in ("wd14", "hybrid") else None
    vlm = VLM(cfg.get("vlm") or {}, models_dir) if mode in ("vlm", "hybrid") else None
    n = len(files)
    keep = mode == "wd14" or (mode == "hybrid" and cfg.get("hybrid_keep_tags", True))
    st = {"count": 0, "worked": 0, "done": 0, "t0": time.time()}

    def report(f, **kw):
        st["count"] += 1
        if not kw.get("skipped"):
            st["worked"] += 1
        el = time.time() - st["t0"]
        emit(i=st["count"], n=n, file=f, eta=el / max(1, st["worked"]) * (n - st["count"]), **kw)

    def finish(it, text):
        f, existing, tags = it[0], it[1], it[3]
        try:
            final = cu.compose(existing, tags if keep else None, text, out)
            if final is not None:
                cu.write_caption(f, final, ext)
            st["done"] += 1
            report(f, caption=final)
        except Exception as e:
            fail(f, e)

    def fail(f, e):
        log(f"[失败] {f}：{e}")
        report(f, error=str(e))

    def run(items):
        """生成一批。显存不够就把以后每批的张数减半、这批拆开重来；一批里有张图出错就一张张单跑，找出是哪张。"""
        if not items:
            return
        import torch
        try:
            texts = vlm.batch([x[2] for x in items], [x[3] for x in items])
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            if len(items) == 1:
                return fail(items[0][0], e)
            vlm.bs = max(1, len(items) // 2)
            log(f"⚠ 显存不够一次生成 {len(items)} 张，改成一次 {vlm.bs} 张")
            for k in range(0, len(items), vlm.bs):
                run(items[k:k + vlm.bs])
            return
        except Exception as e:
            if len(items) == 1:
                return fail(items[0][0], e)
            log(f"⚠ 这一批出错（{e}），改成一张张生成")
            for x in items:
                run([x])
            return
        for it, text in zip(items, texts):
            finish(it, text)

    # 读图、缩图、WD14 放在另一个线程里提前做，显卡生成这一批时下一批已经准备好了
    import queue
    q = queue.Queue(maxsize=(vlm.bs if vlm else 1) * 2 + 2)

    def produce():
        for f in files:
            try:
                existing = cu.read_caption(f, ext)
                if out.get("existing") == "skip" and (existing or "").strip():
                    q.put(("skip", f))
                    continue
                img = Image.open(f)
                img.load()
                tags = wd(img) if wd else None
                q.put(("ok", (f, existing, vlm.prep(img) if vlm else None, tags)))
            except Exception as e:
                q.put(("err", (f, e)))
        q.put(None)

    threading.Thread(target=produce, daemon=True).start()
    pend = []
    while True:
        x = q.get()
        if x is None:
            break
        kind, it = x
        if kind == "skip":
            report(it, skipped=True)
        elif kind == "err":
            fail(*it)
        elif not vlm:
            finish(it, None)
        else:
            pend.append(it)
            if len(pend) >= vlm.bs:
                run(pend)
                pend = []
    if vlm:
        run(pend)
    emit(stage="done", done=st["done"], n=n)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

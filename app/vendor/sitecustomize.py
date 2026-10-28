# -*- coding: utf-8 -*-
"""训练子进程启动时由 Python 自动加载（app/vendor 在训练子进程的 PYTHONPATH 里）。

平时什么都不做；只有多卡训练时（训练器设置 LORA_DDP_NO_LIBUV=1）才在 Windows 上打一个补丁：
Windows 版 PyTorch 没有 libuv，但 torch elastic 的工作进程在复用启动器存储时新建 TCPStore 不传 use_libuv，
默认去用 libuv 直接报错（PyTorch 2.11 的问题）。这里把 TCPStore 默认改成 use_libuv=False。"""
import os

if os.name == "nt" and os.environ.get("LORA_DDP_NO_LIBUV") == "1":
    try:
        import sys as _sys
        import torch.distributed as _dist
        import torch.distributed.rendezvous  # noqa: F401
        _rdzv = _sys.modules["torch.distributed.rendezvous"]   # 注意：torch.distributed.rendezvous 这个属性是同名函数，不是模块

        _TCPStore = _dist.TCPStore

        def _tcpstore_no_libuv(*args, **kwargs):
            kwargs.setdefault("use_libuv", False)
            return _TCPStore(*args, **kwargs)

        _dist.TCPStore = _tcpstore_no_libuv
        _rdzv.TCPStore = _tcpstore_no_libuv
        from torch.distributed.elastic.rendezvous import static_tcp_rendezvous as _s, c10d_rendezvous_backend as _c
        _s.TCPStore = _tcpstore_no_libuv
        _c.TCPStore = _tcpstore_no_libuv
    except Exception:   # 补丁失败也不影响单卡训练
        pass


# ---- 冻结训练（训练器「冻结」按钮）----
# 训练器在任务目录放一个 pause.flag，训练进程在下一步开始前把完整训练状态（LoRA 权重、优化器、调度器、随机数、
# 当前步数）存到 LORA_PAUSE_STATE，然后退出、释放显存；点「继续」时用 --resume 从这一步接着练。
# 挂在 accelerate.Accelerator.accumulate 上：sd-scripts 每一步都会进这个 with，此时上一步已经完整结束。
# 只在训练器启动的训练进程里生效（有 LORA_PAUSE_FILE 环境变量），等 accelerate 被导入时才打补丁，不提前加载 torch。
if os.environ.get("LORA_PAUSE_FILE") or os.environ.get("LORA_VRAM_LIMIT_GB"):
    import sys as _sys2

    def _install_pause(mod):
        import contextlib
        import json
        A = mod.Accelerator
        if getattr(A, "_lora_pause", False):
            return
        _orig = A.accumulate
        flag, state_dir = os.environ["LORA_PAUSE_FILE"], os.environ.get("LORA_PAUSE_STATE", "")
        offset = int(os.environ.get("LORA_STEP_OFFSET") or 0)

        def _want(self):
            want = os.path.exists(flag)
            if self.num_processes > 1:   # 多卡：所有进程必须在同一步一起停
                import torch
                t = torch.tensor([1.0 if want else 0.0], device=self.device)
                want = float(self.reduce(t, "sum").item()) > 0
            return want

        def _swap_in(tmp):
            """存完整的新进度换掉上次冻结的进度；换的时候出错把旧的放回去。"""
            import shutil
            old = state_dir + ".old"
            shutil.rmtree(old, ignore_errors=True)
            if os.path.isdir(state_dir):
                os.replace(state_dir, old)
            try:
                os.replace(tmp, state_dir)
            except OSError:
                if os.path.isdir(old) and not os.path.isdir(state_dir):
                    os.replace(old, state_dir)
                raise
            shutil.rmtree(old, ignore_errors=True)

        def _pause(self):
            # 先存到旁边的文件夹，存完整了再换进来：存到一半报错退出时，上次冻结的进度还能用
            tmp = state_dir + ".saving"
            if self.is_main_process:
                import shutil
                shutil.rmtree(tmp, ignore_errors=True)
            self.wait_for_everyone()
            self.save_state(tmp)
            self.wait_for_everyone()
            if self.is_main_process:
                p, step = os.path.join(tmp, "train_state.json"), "?"
                try:   # sd-scripts 按「保存发生在步数 +1 之前」写的 current_step+1，这里是在步与步之间，要减回去；再换算成真实总步数
                    d = json.load(open(p, encoding="utf-8"))
                    d["current_step"] = int(d.get("current_step") or 0) - 1 + offset
                    json.dump(d, open(p, "w", encoding="utf-8"))
                    step = d["current_step"]
                except Exception:
                    pass
                _swap_in(tmp)
                print(f"LORA_PAUSED step={step} pid={os.getpid()}", flush=True)
            self.wait_for_everyone()
            try:
                self.end_training()
            except Exception:
                pass
            _sys2.stdout.flush()
            try:   # 数据加载子进程（num_workers > 0）不会跟着 os._exit 退出，还占着输出管道，训练器会一直等
                import multiprocessing
                for c in multiprocessing.active_children():
                    c.terminate()
            except Exception:
                pass
            os._exit(0)

        @contextlib.contextmanager
        def accumulate(self, *models):
            if state_dir and self.sync_gradients and _want(self):
                _pause(self)
            with _orig(self, *models) as x:
                yield x

        A.accumulate = accumulate
        A._lora_pause = True

    class _PauseFinder:
        """accelerate.accelerator 模块第一次导入完成后给 Accelerator 打补丁。"""
        def find_spec(self, name, path, target=None):
            if name != "accelerate.accelerator":
                return None
            import importlib.util
            _sys2.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return spec
            _exec = spec.loader.exec_module

            def exec_module(module):
                _exec(module)
                try:
                    _lim = float(os.environ.get("LORA_VRAM_LIMIT_GB") or 0)
                    if _lim:   # 测试用：把训练进程能用的显存锁在 N GB，模拟小显存卡
                        import torch
                        _tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
                        torch.cuda.set_per_process_memory_fraction(min(1.0, _lim / _tot))
                        print(f"[测试] 显存上限锁定为 {_lim:g} GB", flush=True)
                except Exception as e:
                    print(f"[lora] 显存上限设置失败：{e}", flush=True)
                try:
                    if os.environ.get("LORA_PAUSE_FILE"):
                        _install_pause(module)
                except Exception as e:   # 打补丁失败只是不能冻结，不影响训练
                    print(f"[lora] 冻结功能不可用：{e}", flush=True)
            spec.loader.exec_module = exec_module
            return spec

    _sys2.meta_path.insert(0, _PauseFinder())


# ---------------- 分层卸载用的锁页内存：限量 ----------------
# 锁页内存不能换出到虚拟内存，Windows 上最多只能锁大约一半物理内存，而且 PyTorch 按 2 的幂向上取整分配。
# 15 GB 内存的电脑把 MOSS 文本编码器 9 GB 的权重全锁的话会锁到上限、把可用内存挤到 0.3 GB，
# 随后把 VAE 搬进显存时驱动申请不到内存，报 "CUDA error: out of memory"（看起来像显存不足，其实是内存）。
# 所以只锁一部分：不超过物理内存的 1/4，并且至少留 6 GB 可用内存；其余的层用普通内存（搬进显存稍慢）。
class _PinBudget:
    def __init__(self):
        try:
            import psutil
            vm = psutil.virtual_memory()
            self.left = max(0, min(vm.total // 4, vm.available - (6 << 30)))
        except Exception:
            self.left = 0
        self.used = 0

    def pin(self, t):
        if t.device.type != "cpu":
            t = t.cpu()
        size = t.numel() * t.element_size()
        alloc = 1 << max(0, size - 1).bit_length()   # 实际占用按 2 的幂算
        if alloc > self.left:
            return t
        try:
            t = t.pin_memory()
        except Exception:
            self.left = 0   # 锁不动了，后面的都不再尝试
            return t
        self.left -= alloc
        self.used += alloc
        return t


# ---------------- Krea 2 文本编码器分层卸载（8 GB 显卡用） ----------------
# Qwen3-VL-4B 文本编码器 bf16 要约 9 GB 显存，8 GB 卡放不下；放 CPU 上跑一条标签要一分多钟。
# 有 LORA_TE_OFFLOAD 时：权重留在内存（锁页），36 个解码层用到哪层才搬进显存、算完切回内存那份；
# 词表嵌入 / 视觉部分 / 最后的 norm 常驻显存。显存约 2–4 GB，速度每条一两秒。缓存文本编码和训练中出测试图都走这里。
# 训练中途改了测试提示词、要重新编码时（_RT["te_offload"]，见下面「训练中途改测试提示词」）也临时走这里：DiT 还在显存里，整个搬进去会爆。
_RT = {"te_offload": False}
if os.environ.get("LORA_TE_OFFLOAD") or os.environ.get("LORA_TRAINER"):
    import sys as _sys3

    def _offload_qwen(model):
        import torch
        lim = float(os.environ.get("LORA_VRAM_LIMIT_GB") or 0)
        if lim:   # 测试用：缓存脚本不导入 accelerate.accelerator，这里也锁一下显存
            tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            torch.cuda.set_per_process_memory_fraction(min(1.0, lim / tot))
        dev = torch.device("cuda")
        lm = model.model.language_model
        model.model.visual.to(dev)
        for m in (lm.embed_tokens, lm.norm, lm.rotary_emb):
            m.to(dev)
        model.lm_head.to(dev)   # 和词表嵌入共用权重

        pins = _PinBudget()
        for layer in lm.layers:
            ps = list(layer.parameters()) + list(layer.buffers())
            for p in ps:
                p._lora_cpu = pins.pin(p.data)
                p.data = p._lora_cpu

            def pre(mod, args, ps=ps):
                for p in ps:
                    p.data = p._lora_cpu.to(dev, non_blocking=True)

            def post(mod, args, out, ps=ps):
                for p in ps:   # 权重没变，直接切回内存那份，不用从显存拷回
                    p.data = p._lora_cpu
                return out

            layer.register_forward_pre_hook(pre)
            layer.register_forward_hook(post)
        print(f"[lora] 文本编码器分层卸载：权重在内存，逐层搬进显存计算（8 GB 显卡模式，锁页 {pins.used / 2**30:.1f} GB）", flush=True)
        return model

    class _TEOffloadFinder:
        def find_spec(self, name, path, target=None):
            if name != "musubi_tuner.krea2.krea2_encoder":
                return None
            import importlib.util
            _sys3.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return spec
            _exec = spec.loader.exec_module

            def exec_module(module):
                _exec(module)
                orig = module._load_qwen3_vl_model

                def load(model_path, *, dtype, device, disable_mmap=True):
                    if str(device).startswith("cpu") or not (os.environ.get("LORA_TE_OFFLOAD") or _RT["te_offload"]):
                        return orig(model_path, dtype=dtype, device=device, disable_mmap=disable_mmap)
                    return _offload_qwen(orig(model_path, dtype=dtype, device="cpu", disable_mmap=disable_mmap))

                module._load_qwen3_vl_model = load
            spec.loader.exec_module = exec_module
            return spec

    _sys3.meta_path.insert(0, _TEOffloadFinder())


# ---------------- SDXL fp8 大模型出测试图（6 / 8 GB 显卡用） ----------------
# sd-scripts 出测试图前会把整个 fp8 UNet 转成 fp16（多占 2.6 GB，6 GB 卡直接显存不足），出完再转回去。
# 出图本来就在 autocast 里，fp8 权重会像训练时一样逐层临时转精度，所以跳过这次整体转换。
if os.environ.get("LORA_PAUSE_FILE"):
    import sys as _sys4

    def _patch_lpw(module):
        P = module.SdxlStableDiffusionLongPromptWeightingPipeline
        orig = P.__call__

        def __call__(self, *a, **kw):
            u = self.unet
            dt = getattr(u, "dtype", None)
            if dt is None or getattr(dt, "itemsize", 2) != 1:
                return orig(self, *a, **kw)
            real_to = u.to

            def to(*args, **kwargs):   # 只拦「整体转精度」，搬设备照常
                if len(args) == 1 and not kwargs and isinstance(args[0], type(dt)):
                    return u
                return real_to(*args, **kwargs)
            u.to = to
            try:
                return orig(self, *a, **kw)
            finally:
                del u.to
        P.__call__ = __call__

    class _LpwFinder:
        def find_spec(self, name, path, target=None):
            if name != "library.sdxl_lpw_stable_diffusion":
                return None
            import importlib.util
            _sys4.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return spec
            _exec = spec.loader.exec_module

            def exec_module(module):
                _exec(module)
                try:
                    _patch_lpw(module)
                except Exception as e:
                    print(f"[lora] SDXL fp8 出图补丁没打上：{e}", flush=True)
            spec.loader.exec_module = exec_module
            return spec

    _sys4.meta_path.insert(0, _LpwFinder())


# ---------------- Qwen-Image VAE 解码显存不足时降级（Krea 2 小显存出测试图） ----------------
# musubi 出测试图时 12B 的 DiT 还在显存里，最后 VAE 解码 1024×1024 又要好几 GB，8 / 12 GB 显卡在这一步爆显存，
# 整个训练失败。这里：先正常解码；显存不够就清缓存、开分块解码再试；还不够就把 VAE 挪到 CPU 上解码（慢十几秒）。
if os.environ.get("LORA_TRAINER"):
    import sys as _sys5

    def _patch_qwen_vae(module):
        C = module.AutoencoderKLQwenImage
        orig = C.decode_to_pixels

        def decode_to_pixels(self, latents):
            import torch
            try:
                return orig(self, latents)
            except torch.OutOfMemoryError:
                pass
            torch.cuda.empty_cache()
            print("[lora] VAE 解码显存不足，改用分块解码", flush=True)
            try:
                self.enable_tiling()
                return orig(self, latents)
            except torch.OutOfMemoryError:
                pass
            finally:
                self.use_tiling = False
            torch.cuda.empty_cache()
            print("[lora] 分块解码还是显存不足，改到 CPU 上解码（会慢一些）", flush=True)
            dev = latents.device
            self.to("cpu")
            try:
                return orig(self, latents.to("cpu", torch.float32 if self.dtype == torch.float32 else self.dtype)).to(dev)
            finally:
                self.to(dev)

        C.decode_to_pixels = decode_to_pixels

    class _QwenVaeFinder:
        def find_spec(self, name, path, target=None):
            if name != "musubi_tuner.qwen_image.qwen_image_autoencoder_kl":
                return None
            import importlib.util
            _sys5.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return spec
            _exec = spec.loader.exec_module

            def exec_module(module):
                _exec(module)
                try:
                    _patch_qwen_vae(module)
                except Exception as e:
                    print(f"[lora] VAE 解码保护没打上：{e}", flush=True)
            spec.loader.exec_module = exec_module
            return spec

    _sys5.meta_path.insert(0, _QwenVaeFinder())


# ---------------- MOSS-Image 文本编码器（Qwen3.5-9B）分层卸载（8 GB 显卡用） ----------------
# 官方说明里 8 GB 显卡不支持，只是因为缓存文本编码时 int8 的 Qwen3.5-9B 也要 8.3 GB 显存（训练本身 int8 + 块交换只要 5 GB）。
# 有 LORA_TE_OFFLOAD 时：训练脚本把文本编码器整个搬进显存的那一下只搬词表嵌入等小部件，32 层解码层留在内存（锁页），
# 用到哪层才搬进显存、算完切回内存那份。显存约 2 GB。训练中途改了测试提示词、临时加载文本编码器时也用它。
if os.environ.get("LORA_TE_OFFLOAD") or os.environ.get("LORA_TRAINER"):
    import sys as _sys6

    def _offload_text_model(model):
        import torch
        if getattr(model, "_lora_offload", False):
            return model
        model._lora_offload = True
        layers = model.layers
        state = {"dev": None}
        pins = _PinBudget()
        for layer in layers:
            ps = list(layer.parameters()) + list(layer.buffers())
            for p in ps:
                cpu = pins.pin(p.data)
                p._lora_cpu = cpu
                p.data = cpu

            def pre(mod, args, ps=ps):
                if state["dev"] is not None:
                    for p in ps:
                        p.data = p._lora_cpu.to(state["dev"], non_blocking=True)

            def post(mod, args, out, ps=ps):
                for p in ps:
                    p.data = p._lora_cpu
                return out
            layer.register_forward_pre_hook(pre)
            layer.register_forward_hook(post)

        orig_to = model.to

        def to(*args, **kwargs):
            dev = kwargs.get("device", args[0] if args else None)
            if dev is not None and not isinstance(dev, torch.dtype) and torch.device(dev).type == "cuda":
                for name, child in model.named_children():
                    if child is not layers:
                        child.to(*args, **kwargs)
                for name, b in model.named_buffers(recurse=False):
                    setattr(model, name, b.to(dev))
                state["dev"] = torch.device(dev)
                return model
            state["dev"] = None
            return orig_to(*args, **kwargs)
        model.to = to
        print(f"[lora] MOSS 文本编码器分层卸载：解码层留在内存，逐层搬进显存计算（8 GB 显卡模式，锁页 {pins.used / 2**30:.1f} GB）", flush=True)
        return model

    class _MossTEFinder:
        def find_spec(self, name, path, target=None):
            if name != "library.moss_image_utils":
                return None
            import importlib.util
            _sys6.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return spec
            _exec = spec.loader.exec_module

            def exec_module(module):
                _exec(module)
                orig = module.load_text_encoder

                def load_text_encoder(*a, **kw):
                    return _offload_text_model(orig(*a, **kw))
                module.load_text_encoder = load_text_encoder
            spec.loader.exec_module = exec_module
            return spec

    if os.environ.get("LORA_TE_OFFLOAD"):
        _sys6.meta_path.insert(0, _MossTEFinder())


# ---------------- ConvRot INT8：输入行数补齐，避免每种长度都重新调优 ----------------
# int8 矩阵乘的 Triton 内核按 (m, n, k) 自动调优，m 是输入的行数（token 数）。缓存文本编码时每条标签长度都不一样，
# MOSS 这种单流模型训练时图像 token + 标签 token 也随标签长度变，于是几乎每次都要重新调优：
# 实测 Qwen3.5-9B int8 缓存一条标签 3.4 秒，bf16 只要 0.1 秒。这里把行数补齐到 256 的倍数（复制最后一行，
# int8 是逐行量化的，补的行不影响原来的行），算完切掉，调优只需要做几次。
if os.environ.get("LORA_TRAINER"):
    import sys as _sys7

    def _patch_int8_linear(module):
        orig = module.int8_linear

        def int8_linear(x, weight, weight_scale, bias=None, out_dtype=None, convrot=False, convrot_groupsize=256):
            import torch
            kw = {} if out_dtype is None else {"out_dtype": out_dtype}
            x2 = x.reshape(-1, x.shape[-1])
            m = x2.shape[0]
            mp = max(256, (m + 255) // 256 * 256)
            if mp == m:
                return orig(x, weight, weight_scale, bias, convrot=convrot, convrot_groupsize=convrot_groupsize, **kw)
            xp = torch.cat([x2, x2[-1:].expand(mp - m, x2.shape[1])], 0)
            out = orig(xp, weight, weight_scale, bias, convrot=convrot, convrot_groupsize=convrot_groupsize, **kw)
            return out[:m].reshape(*x.shape[:-1], out.shape[-1])
        module.int8_linear = int8_linear

    class _Int8Finder:
        names = {"library.convrot_int8_utils", "musubi_tuner.modules.convrot_int8_utils"}

        def find_spec(self, name, path, target=None):
            if name not in self.names:   # sd-scripts（MOSS / Anima）和 musubi（Krea 2）各有一份同样的内核
                return None
            import importlib.util
            self.names.discard(name)
            if not self.names:
                _sys7.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return spec
            _exec = spec.loader.exec_module

            def exec_module(module):
                _exec(module)
                try:
                    if hasattr(module, "int8_linear"):
                        _patch_int8_linear(module)
                except Exception as e:
                    print(f"[lora] int8 行数补齐没打上：{e}", flush=True)
            spec.loader.exec_module = exec_module
            return spec

    _sys7.meta_path.insert(0, _Int8Finder())


# ---------------- FLUX.2 VAE（MOSS-Image）解码显存不足时改到 CPU ----------------
# 出测试图最后一步解码时 DiT 还在显存里，VAE 又是 fp32，小显存卡可能在这一步爆显存、整个训练失败。
if os.environ.get("LORA_TRAINER"):
    import sys as _sys8

    def _patch_flux2_vae(module):
        C = module.AutoEncoder
        orig = C.decode_from_dit

        def decode_from_dit(self, z):
            import torch
            try:
                return orig(self, z)
            except torch.OutOfMemoryError:
                pass
            torch.cuda.empty_cache()
            print("[lora] VAE 解码显存不足，改到 CPU 上解码（会慢一些）", flush=True)
            dev = z.device
            self.to("cpu")
            try:
                return orig(self, z.to("cpu")).to(dev)
            finally:
                self.to(dev)
        C.decode_from_dit = decode_from_dit

    class _Flux2VaeFinder:
        def find_spec(self, name, path, target=None):
            if name != "library.flux2_vae":
                return None
            import importlib.util
            _sys8.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return spec
            _exec = spec.loader.exec_module

            def exec_module(module):
                _exec(module)
                try:
                    _patch_flux2_vae(module)
                except Exception as e:
                    print(f"[lora] FLUX.2 VAE 解码保护没打上：{e}", flush=True)
            spec.loader.exec_module = exec_module
            return spec

    _sys8.meta_path.insert(0, _Flux2VaeFinder())


# ---------------- 训练中途改测试提示词 ----------------
# 训练器在训练中途改测试提示词时直接重写任务目录里的提示词文件（app/tasks.py 的 update_sample）。
# sd-scripts（MOSS / Anima）：每次出图都会重新读这个文件，但缓存完文本编码后文本编码器就删掉了，没编码过的提示词会被跳过。
#   这里在出每张图之前检查：文件里有还没编码的提示词，就临时加载文本编码器把它们一次编完、加进缓存，再释放。
#   MOSS 的 Qwen3.5-9B 用分层卸载（约 2 GB 显存），DiT 还在显存里也放得下。
# musubi（Krea 2）：开训时就把整套测试参数算好，之后不再读文件。这里在出图前发现文件变了就重新算一遍（文本编码器同样分层卸载）。
# SDXL：文本编码器一直留着，每次出图都重新读文件，本来就支持。
if os.environ.get("LORA_TRAINER"):
    import sys as _sys9

    def _encode_missing(kind, accelerator, args, tokenize_strategy, text_encoding_strategy, cache, prompt_replacement):
        import gc
        import torch
        from library import sampling
        need = []
        for d in sampling.load_prompts(args.sample_prompts):
            for p in (d.get("prompt", ""), d.get("negative_prompt", "")):
                if prompt_replacement is not None:
                    p = p.replace(prompt_replacement[0], prompt_replacement[1])
                if p not in cache and p not in need:
                    need.append(p)
        if not need:
            return
        print(f"[lora] 测试提示词改过了，临时加载文本编码器编码 {len(need)} 条新提示词", flush=True)
        dev = accelerator.device
        if kind == "moss":
            from library import moss_image_utils
            te = moss_image_utils.load_text_encoder(args.text_encoder, dtype=torch.bfloat16, device="cpu",
                                                    disable_mmap=args.disable_mmap_load_safetensors,
                                                    convrot_int8=args.convrot_int8_text_encoder,
                                                    convrot_int8_bwd=args.convrot_int8_bwd, calc_device=dev)
            te = _offload_text_model(te)
            ctx = torch.autocast(device_type=dev.type, enabled=False)   # 和缓存时一样不开 autocast（文本编码器本身是 bf16）
        else:
            from library import anima_utils, accelerator_setup
            te, _ = anima_utils.load_qwen3_text_encoder(args.qwen3, dtype=accelerator_setup.prepare_dtype(args)[0], device="cpu")
            ctx = accelerator.autocast()
        te.eval()
        te.to(dev)
        try:
            with torch.no_grad(), ctx:
                for p in need:
                    cache[p] = text_encoding_strategy.encode_tokens(tokenize_strategy, [te], tokenize_strategy.tokenize(p))
        finally:
            te.to("cpu")
            del te
            gc.collect()
            torch.cuda.empty_cache()

    def _patch_sd_sample(module, kind):
        orig = module._sample_image_inference

        def _sample_image_inference(accelerator, args, dit, text_encoder, vae, tokenize_strategy, text_encoding_strategy,
                                    save_dir, prompt_dict, epoch, steps, sample_prompts_te_outputs, prompt_replacement, *a, **kw):
            if text_encoder is None and sample_prompts_te_outputs is not None:
                try:
                    _encode_missing(kind, accelerator, args, tokenize_strategy, text_encoding_strategy, sample_prompts_te_outputs,
                                    prompt_replacement)
                except Exception as e:
                    print(f"[lora] 新测试提示词编码失败，这几张测试图会跳过：{e}", flush=True)
            return orig(accelerator, args, dit, text_encoder, vae, tokenize_strategy, text_encoding_strategy, save_dir, prompt_dict,
                        epoch, steps, sample_prompts_te_outputs, prompt_replacement, *a, **kw)
        module._sample_image_inference = _sample_image_inference

    def _musubi_refresh(self, accelerator, args, params):
        from musubi_tuner.training.sampling_prompts import load_prompts
        cur = load_prompts(args.sample_prompts)
        last = getattr(self, "_lora_sample", None)
        if last is None:   # 第一次出图：和开训时算好的那套比（那套是提示词字典的副本 + 编码结果）
            if len(cur) == len(params) and all(all(o.get(k) == v for k, v in d.items()) for d, o in zip(cur, params)):
                self._lora_sample = (cur, params)
                return params
        elif last[0] == cur:
            return last[1]
        print("[lora] 测试提示词改过了，重新编码测试提示词", flush=True)
        _RT["te_offload"] = True
        try:
            new = self.process_sample_prompts(args, accelerator, args.sample_prompts)
        finally:
            _RT["te_offload"] = False
        self._lora_sample = (cur, new)
        return new

    def _patch_musubi(module):
        C = module.NetworkTrainer
        orig = C.sample_images

        def sample_images(self, accelerator, args, epoch, steps, sample_resources, transformer, sample_parameters, dit_dtype):
            if sample_parameters is not None and args.sample_prompts and module.should_sample_images(args, steps, epoch):
                try:
                    sample_parameters = _musubi_refresh(self, accelerator, args, sample_parameters)
                except Exception as e:
                    print(f"[lora] 新测试提示词编码失败，这次还用原来的提示词：{e}", flush=True)
            return orig(self, accelerator, args, epoch, steps, sample_resources, transformer, sample_parameters, dit_dtype)
        C.sample_images = sample_images

    class _SamplePromptFinder:
        patches = {"library.moss_image_train_utils": lambda m: _patch_sd_sample(m, "moss"),
                   "library.anima_train_utils": lambda m: _patch_sd_sample(m, "anima"),
                   "musubi_tuner.training.trainer_base": _patch_musubi}

        def find_spec(self, name, path, target=None):
            fn = self.patches.pop(name, None)
            if fn is None:
                return None
            import importlib.util
            if not self.patches:
                _sys9.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return spec
            _exec = spec.loader.exec_module

            def exec_module(module):
                _exec(module)
                try:
                    fn(module)
                except Exception as e:
                    print(f"[lora] 训练中改测试提示词的补丁没打上（改了要下次训练才生效）：{e}", flush=True)
            spec.loader.exec_module = exec_module
            return spec

    _sys9.meta_path.insert(0, _SamplePromptFinder())


# ---------------- MOSS 训练中测试：换主模型出图 / 8 步加速出图 ----------------
# 训练器传来 LORA_SAMPLE_DIT（出图用的主模型，比如用 Base 练、用 SFT 出图）和 LORA_SAMPLE_LORA（8 步加速 LoRA）。
# 换主模型：sd-scripts 的 MOSS 模型参数名和官方权重文件一样，出图前把训练主模型的权重原地换成这个文件的（不占额外显存），
#   出完图再从训练主模型的文件读回来（不在内存里留备份，最多多占约 10 GB 内存）。训练的 LoRA 不受影响。
#   int8 训练（权重已经量化）和开了块交换时不换（参数形状 / 存放位置不一样）。
# 加速 LoRA：出图期间在每个线性层输出上加 scale * up(down(x))（和 WebUI 插件一样不合并进权重：它的增量比 bf16 精度还小），出完摘掉。
if os.environ.get("LORA_TRAINER") and (os.environ.get("LORA_SAMPLE_DIT") or os.environ.get("LORA_SAMPLE_LORA")):
    import sys as _sys10

    def _moss_will_sample(args, epoch, steps):   # 和 moss_image_train_utils.sample_images 开头的判断一样
        if steps == 0:
            return bool(args.sample_at_first)
        if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
            return False
        if args.sample_every_n_epochs is not None:
            return epoch is not None and epoch % args.sample_every_n_epochs == 0
        return steps % args.sample_every_n_steps == 0 and epoch is None

    def _moss_copy_weights(model, path):
        import torch
        from library import moss_image_utils as mu
        from library.lora_utils import load_safetensors_with_lora_and_fp8
        from library.safetensors_utils import WeightTransformHooks
        hooks = WeightTransformHooks()
        hooks.rename_hook = mu._strip_dit_prefix
        hooks.split_hook = mu.split_fused_qkv
        sd = load_safetensors_with_lora_and_fp8(model_files=path, lora_weights_list=None, lora_multipliers=None, fp8_optimization=False,
                                                calc_device=torch.device("cpu"), move_to_device=False, dit_weight_dtype=None,
                                                weight_transform_hooks=hooks)
        sd.pop("rope.bands", None)
        cur = model.state_dict()
        n = 0
        with torch.no_grad():
            for k in list(sd):
                v = sd.pop(k)
                t = cur.get(k)
                if t is not None and tuple(t.shape) == tuple(v.shape):
                    t.copy_(v)
                    n += 1
        return n, len(cur)

    def _moss_lora_hooks(model, path, weight):
        import torch
        import torch.nn.functional as F
        from safetensors.torch import load_file
        sd = load_file(path)
        mods = dict(model.named_modules())
        flat = {name.replace(".", "_"): name for name in mods if name}
        groups = {}
        for key, t in sd.items():
            for suf, kind in ((".lora_down.weight", "down"), (".lora_A.weight", "down"), (".lora_up.weight", "up"),
                              (".lora_B.weight", "up"), (".alpha", "alpha")):
                if key.endswith(suf):
                    name = key[: -len(suf)]
                    for pre in ("diffusion_model.", "transformer.", "lora_unet_"):
                        if name.startswith(pre):
                            name = name[len(pre):]
                    groups.setdefault(name if name in mods else flat.get(name, name), {})[kind] = t
                    break
        handles = []
        for name, g in groups.items():
            if name not in mods or "down" not in g or "up" not in g:
                continue
            down, up = g["down"], g["up"]
            scale = weight * (float(g["alpha"]) if "alpha" in g else down.shape[0]) / down.shape[0]
            cache = {}

            def hook(mod, inputs, out, down=down, up=up, scale=scale, cache=cache):
                x = inputs[0]
                key = (x.device, x.dtype)
                if key not in cache:
                    cache[key] = (down.to(x.device, x.dtype), up.to(x.device, x.dtype))
                d, u = cache[key]
                return out + (F.linear(F.linear(x, d), u) * scale).to(out.dtype)
            handles.append(mods[name].register_forward_hook(hook))
        return handles

    def _patch_moss_sample(module):
        orig = module.sample_images

        def sample_images(accelerator, args, epoch, steps, dit, *a, **kw):
            if not _moss_will_sample(args, epoch, steps):
                return orig(accelerator, args, epoch, steps, dit, *a, **kw)
            from library import convrot_int8_utils
            model = accelerator.unwrap_model(dit)
            alt, swapped, handles = os.environ.get("LORA_SAMPLE_DIT"), False, []
            try:
                if alt:
                    if convrot_int8_utils.is_convrot_int8_model(model) or (getattr(args, "blocks_to_swap", 0) or 0) > 0:
                        print("[lora] int8 训练或开了块交换，测试出图不换主模型", flush=True)
                    else:
                        n, total = _moss_copy_weights(model, alt)
                        swapped = True
                        print(f"[lora] 测试出图换用主模型 {os.path.basename(os.path.dirname(os.path.dirname(alt))) or alt}"
                              f"（换了 {n}/{total} 个张量）", flush=True)
                lora = os.environ.get("LORA_SAMPLE_LORA")
                if lora:
                    handles = _moss_lora_hooks(model, lora, float(os.environ.get("LORA_SAMPLE_LORA_WEIGHT") or 1))
                    print(f"[lora] 测试出图挂上加速 LoRA {os.path.basename(lora)}（{len(handles)} 层）", flush=True)
                return orig(accelerator, args, epoch, steps, dit, *a, **kw)
            finally:
                for h in handles:
                    h.remove()
                if swapped:
                    _moss_copy_weights(model, args.pretrained_model_name_or_path)
                    import gc
                    gc.collect()
                    print("[lora] 测试出图完成，主模型权重已换回训练用的", flush=True)
        module.sample_images = sample_images

    class _MossSampleFinder:
        def find_spec(self, name, path, target=None):
            if name != "library.moss_image_train_utils":
                return None
            import importlib.util
            _sys10.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return spec
            _exec = spec.loader.exec_module

            def exec_module(module):
                _exec(module)
                try:
                    _patch_moss_sample(module)
                except Exception as e:
                    print(f"[lora] 测试出图换主模型 / 加速的补丁没打上：{e}", flush=True)
            spec.loader.exec_module = exec_module
            return spec

    _sys10.meta_path.insert(0, _MossSampleFinder())

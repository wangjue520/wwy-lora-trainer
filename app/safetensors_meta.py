# -*- coding: utf-8 -*-
"""读 safetensors 文件头：判断模型种类、是否量化、文件是否完整（下载中断 / 损坏）。

只读文件开头的 JSON 头，几 GB 的模型也是毫秒级。"""
import json
import os
import struct

VERSION = 3   # 分类规则改了就加 1，扫描缓存里旧版本的结果会重新分类
QUANT_DTYPES = {"I8", "U8", "F8_E4M3", "F8_E5M2", "F8_E8M0", "I4", "U4"}
_PREFIXES = ("model.diffusion_model.", "diffusion_model.")


def read_header(path):
    """返回 (header_dict, error)。error 为中文原因，正常时为 None。"""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return None, "文件不完整（只有几个字节）"
            n = struct.unpack("<Q", raw)[0]
            if n <= 2 or n > 200 * 1024 * 1024 or n + 8 > size:
                return None, "文件头损坏（不是有效的 safetensors 文件）"
            h = json.loads(f.read(n).decode("utf-8"))
    except json.JSONDecodeError:
        return None, "文件头损坏（JSON 解析失败）"
    except OSError as e:
        return None, f"无法读取文件：{e}"
    # 数据区长度和文件大小对不上 = 下载没下完 / 被截断
    end = 0
    for k, v in h.items():
        if k == "__metadata__":
            continue
        try:
            end = max(end, int(v["data_offsets"][1]))
        except Exception:
            return None, "文件头损坏（张量信息不完整）"
    if 8 + n + end > size:
        return None, f"文件不完整：应有 {(8 + n + end) / 1e9:.2f} GB，实际只有 {size / 1e9:.2f} GB（下载中断？）"
    return h, None


def _strip(k):
    for p in _PREFIXES:
        if k.startswith(p):
            return k[len(p):]
    return k


def classify(path):
    """返回 {kind, quant, note, error}。kind 见下方各分支；认不出时为 'unknown'。"""
    h, err = read_header(path)
    if err:
        return {"kind": "broken", "quant": False, "error": err}
    meta = h.pop("__metadata__", None) or {}
    keys = {_strip(k): v for k, v in h.items()}
    quant = any(v.get("dtype") in QUANT_DTYPES and not k.endswith(("comfy_quant", "scale_weight"))
                for k, v in keys.items()) or any(k.endswith("comfy_quant") for k in keys)

    def shape(k):
        v = keys.get(k)
        return v.get("shape") if v else None

    qtype = ""
    if quant:
        dts = {v.get("dtype") for v in keys.values()}
        qtype = "fp8" if any(str(x).startswith("F8") for x in dts) else ("int8" if "I8" in dts else "low-bit")
        if "U8" in dts and "I8" not in dts and not any(str(x).startswith("F8") for x in dts):
            qtype = "fp4"
    info = {"kind": "unknown", "quant": quant, "qtype": qtype, "error": None}

    # LoRA / LyCORIS：不是底模
    if any(".lora_down." in k or ".lora_A." in k or k.startswith("lora_") or ".hada_w1" in k for k in list(keys)[:400]):
        info["kind"] = "lora"
        return info

    # Anima（Cosmos-Predict2 2B 结构 + LLM Adapter）
    if "net.x_embedder.proj.1.weight" in keys or "x_embedder.proj.1.weight" in keys:
        has_adapter = any("llm_adapter." in k for k in keys)
        info["kind"] = "anima_dit" if has_adapter else "cosmos_dit"
        return info

    # Krea 2（single-stream MMDiT）
    if "first.weight" in keys and any(k.startswith("txtfusion.") for k in keys) and "blocks.0.attn.wq.weight" in keys:
        info["kind"] = "krea2_dit"
        return info

    # MOSS-Image DiT（5B 单流 DiT，42 块）：官方 diffusers 分片只有第 1 片带 video_embedder，训练时填第 1 片，其余分片自动加载
    if "layers.0.proj.ffn_up_gate.weight" in keys or "layers.41.proj.ffn_up_gate.weight" in keys:
        info["kind"] = "moss_dit" if "video_embedder.weight" in keys else "moss_dit_shard"
        return info

    # Qwen3.5（MOSS-Image 的文本编码器 Qwen3.5-9B）：带 Gated DeltaNet（linear_attn）层
    if any(".linear_attn.in_proj_qkv." in k for k in list(keys)[:600]) or any(".linear_attn.in_proj_qkv." in k for k in keys):
        emb = shape("language_model.embed_tokens.weight") or shape("model.language_model.embed_tokens.weight") \
            or shape("model.embed_tokens.weight")
        if emb:
            info["kind"] = "qwen35_9b" if emb[1] == 4096 else "qwen35"
        else:
            info["kind"] = "qwen35_shard"   # 官方分片的第 2 片以后：训练时填第 1 片
        return info

    # Qwen3 系列文本编码器
    emb = shape("model.embed_tokens.weight") or shape("model.language_model.embed_tokens.weight") \
        or shape("language_model.model.embed_tokens.weight")
    if emb and len(emb) == 2:
        hidden = emb[1]
        has_visual = any(k.startswith(("model.visual.", "visual.")) for k in keys)
        if has_visual:
            info["kind"] = {2560: "qwen3vl_4b", 4096: "qwen3vl_8b", 2048: "qwen3vl_2b"}.get(hidden, "qwen_vl")
            if hidden == 3584:
                info["kind"] = "qwen25vl_7b"
        else:
            info["kind"] = {1024: "qwen3_06b", 2560: "qwen3_4b", 4096: "qwen3_8b"}.get(hidden, "llm")
        return info

    # Wan2.1 / Qwen-Image VAE（3D 卷积因果 VAE，两者结构相同，靠文件名区分）
    dc = shape("decoder.conv1.weight")
    if dc and len(dc) == 5 and shape("conv2.weight") and shape("conv2.weight")[0] == 16:
        name = os.path.basename(path).lower()
        info["kind"] = "qwen_image_vae" if "qwen" in name else "wan_vae"
        return info
    # SDXL（含 Illustrious / NoobAI / Pony / Animagine 等微调）：两个文本编码器 + 带 label_emb 的 UNet
    if "conditioner.embedders.1.model.ln_final.weight" in keys or "label_emb.0.0.weight" in keys:
        full = any(k.startswith("conditioner.embedders.") for k in keys) and any(k.startswith("first_stage_model.") for k in keys)
        info["kind"] = "sdxl_checkpoint" if full else "sdxl_unet"
        # v-pred 模型（NoobAI v-pred 等）在文件里放了 v_pred / ztsnr 标记，或者写在 modelspec 里
        info["vpred"] = "v_pred" in h or str(meta.get("modelspec.prediction_type", "")).lower() in ("v", "v_prediction")
        return info
    if any(k.startswith("cond_stage_model.") for k in keys) and "input_blocks.0.0.weight" in keys:
        info["kind"] = "sd2_checkpoint" if any(k.startswith("cond_stage_model.model.") for k in keys) else "sd15_checkpoint"
        return info
    vc = shape("decoder.conv_in.weight") or shape("first_stage_model.decoder.conv_in.weight")
    if vc and len(vc) == 4:
        if vc[1] == 4:   # SD1.5 和 SDXL 的 VAE 结构完全一样，只能靠文件名区分
            name = os.path.basename(path).lower()
            info["kind"] = "sdxl_vae" if "xl" in name else "sd_vae"
        elif vc[1] == 16:
            info["kind"] = "flux_vae"
        elif vc[1] == 32:
            info["kind"] = "flux2_vae"   # FLUX.2 VAE（MOSS-Image 用；官方 diffusers 版和 BFL / ComfyUI 版都是 32 通道）
        return info
    if meta.get("modelspec.architecture"):
        info["note"] = meta.get("modelspec.architecture")
    return info

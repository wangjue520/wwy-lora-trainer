# -*- coding: utf-8 -*-
"""标签 / 描述的拼接规则（界面进程和反推子进程共用，只依赖标准库）。"""
import os
import re

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# 本地自然语言反推模型
VLM_MODELS = {
    "toriigate": {"label": "ToriiGate 0.5（二次元专精，可参考 WD14 标签）", "repo": "Minthy/ToriiGate-0.5", "size_gb": 10},
    "joycaption": {"label": "JoyCaption Beta One（通用，描述细致）", "repo": "fancyfeast/llama-joycaption-beta-one-hf-llava", "size_gb": 17},
    "qwen3vl8b": {"label": "Qwen3-VL-8B abliterated（无审查，写实友好）", "repo": "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated", "size_gb": 17.5},
    "qwen3vl4b": {"label": "Qwen3-VL-4B abliterated（显存小）", "repo": "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated", "size_gb": 9},
}

WD14_MODELS = {
    "eva02-large": {"label": "WD EVA02-Large v3（最准，推荐）", "repo": "SmilingWolf/wd-eva02-large-tagger-v3"},
    "vit-large": {"label": "WD ViT-Large v3", "repo": "SmilingWolf/wd-vit-large-tagger-v3"},
    "swinv2": {"label": "WD SwinV2 v3（快）", "repo": "SmilingWolf/wd-swinv2-tagger-v3"},
    "vit": {"label": "WD ViT v3（最快）", "repo": "SmilingWolf/wd-vit-tagger-v3"},
}
# WWY 启动器「WD14 反推」页认识的两个模型（存放目录名 = 仓库名）
LAUNCHER_WD14 = ("wd-vit-tagger-v3", "wd-eva02-large-tagger-v3")

# MOSS 系列预设共用的结尾：官方打标要求（图中文字一字不改 + 描述排版、不写水印）
_MOSS_TAIL = ("Include only information visible in the image. Preserve any visible text exactly as written (do not translate, "
              "paraphrase or romanize it) and describe its layout: line breaks, ordering, relative position, alignment, size, "
              "font style and color. Do not describe watermarks. Use plain natural sentences without markdown or bullet points. "
              "Do not start with phrases like 'This image shows'.")

PROMPT_PRESETS = {
    "detailed": "Write a detailed description of this image for training an image generation model. Describe the subject, "
                "appearance, clothing, pose, expression, composition, background, lighting, colors and art style. "
                "Use plain natural English sentences. Do not use bullet points, markdown or line breaks. "
                "Do not start with phrases like 'This image shows'.",
    "short": "Write a short one or two sentence description of this image for an image generation model. "
             "Mention the main subject, key details and style. Plain English, no line breaks.",
    "style": "Describe this image focusing on the content only (subject, pose, clothing, background). Do NOT describe "
             "the art style, medium, rendering technique or overall color palette (colors of hair, eyes and clothing are fine). "
             "Plain natural English sentences, no line breaks.",
    # 按 MOSS-Image 官方用户指南的打标要求（caption.long）改写成纯文字输出
    "moss": "Write a detailed description of this image for training a text-to-image model. Describe the composition, key "
            "subjects, actions, setting, colors, lighting and artistic style in detail. " + _MOSS_TAIL,
    # MOSS 画风 LoRA：只写内容不写画风，画风留给触发词学（Mika 画风实测就是这样打的标）
    "moss_style": "Write a detailed description of this image for training a text-to-image style LoRA. Describe the composition, "
                  "key subjects, appearance, clothing, pose, expression, actions, setting and lighting in detail. Do NOT describe "
                  "the art style, medium, rendering technique, line work, shading or overall color palette, and do not name any "
                  "artist or style; the colors of individual things such as hair, eyes and clothing are fine. " + _MOSS_TAIL,
    "character": "Describe this image for training a character LoRA. Describe pose, expression, clothing, accessories, "
                 "background and composition, but do NOT describe the character's permanent features (face, hair "
                 "color, eye color, body type). Plain natural English sentences, no line breaks.",
}

TORII_FORMATS = {
    "long": "Make a caption for given image with natural text. Use 2 to 5 paragraphs. Make your description long and vivid, mentioning all the details.\n",
    "short": "The caption for image should be quite short without long purple prose and slop. Cover main objects and details.\n",
}
TORII_SYSTEM = "You are image captioning expert. Describe user's picture according to requested format and instructions."


def torii_query(fmt, tags=None, use_names=True):
    q = "# Captioning format:\n" + TORII_FORMATS.get(fmt, TORII_FORMATS["long"]) + "\n"
    if tags:
        q += "# Booru tags for the image\n[" + ", ".join(tags) + "]\n\n"
    if use_names:
        q += "# Characters on picture:\nTry to recognize the characters in the picture and use their names.\n\n"
    else:
        q += "# Characters on picture:\nAvoid to guess names for characters.\n"
    return q


def split_tags(s):
    return [t.strip() for t in re.split(r",|\n", s or "") if t.strip()]


def clean_text(s):
    s = (s or "").strip()
    s = re.sub(r"^(\*\*)?(caption|description)(\*\*)?\s*[:：]\s*", "", s, flags=re.I)
    s = re.sub(r"[#*`]+", "", s)
    s = re.sub(r"\s*\n+\s*", " ", s)
    return s.strip().strip('"').strip()


def compose(existing, tags=None, text=None, opts=None):
    """拼出最终写入 .txt 的内容。

    opts: trigger 触发词（可逗号分隔多个）/ existing: overwrite 覆盖、append 追加、skip 跳过已有
    混合模式：「触发词, 标签…」后面接自然语言句子。"""
    opts = opts or {}
    triggers = split_tags(opts.get("trigger") or "")
    mode = opts.get("existing") or "overwrite"
    if mode == "skip" and (existing or "").strip():
        return None
    parts = []
    tag_list = list(tags or [])
    if mode == "append" and (existing or "").strip():
        old = existing.strip()
        if text and not tag_list:
            return _with_trigger(old + " " + clean_text(text), triggers)
        tag_list = split_tags(old) + [t for t in tag_list if t not in split_tags(old)]
    tag_list = [t for t in tag_list if t not in triggers]
    head = triggers + tag_list
    if head:
        parts.append(", ".join(head))
    if text:
        t = clean_text(text)
        if t:
            parts.append(t)
    if not parts:
        return ", ".join(triggers) if triggers else ""
    if len(parts) == 2:
        # 只有触发词时用逗号接句子（「k2, A cat …」）；有标签时用句号把标签和句子分开
        return parts[0] + (", " if not tag_list else ". ") + parts[1]
    return parts[0]


def _with_trigger(s, triggers):
    if not triggers:
        return s
    for t in reversed(triggers):
        if not s.lower().startswith(t.lower()):
            s = t + ", " + s
    return s


def list_images(d, recursive=True):
    res = []
    if not d or not os.path.isdir(d):
        return res
    for root, dirs, files in os.walk(d):
        dirs[:] = sorted(x for x in dirs if not x.startswith(("_", ".")))
        for fn in sorted(files):
            if fn.lower().endswith(IMAGE_EXTS):
                res.append(os.path.join(root, fn))
        if not recursive:
            break
    return res


def caption_path(img, ext=".txt"):
    return os.path.splitext(img)[0] + ext


def read_caption(img, ext=".txt"):
    p = caption_path(img, ext)
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def write_caption(img, text, ext=".txt"):
    with open(caption_path(img, ext), "w", encoding="utf-8") as f:
        f.write((text or "").strip() + "\n" if text else "")

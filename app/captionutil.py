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
    # PixAI 官方仓库要登录同意协议，和 WWY 启动器一样用 deepghs 转好的 ONNX（匿名可下）；角色更全更新、会带出作品名，没有分级标签
    "pixai": {"label": "PixAI Tagger v0.9（角色更全更新，附带作品名）", "repo": "deepghs/pixai-tagger-v0.9-onnx",
              "launcher": "pixai-tagger-v0.9", "general_threshold": 0.3},
}
# WWY 启动器「图片反推」页认识的模型：仓库名 → 启动器 wd_tagger_models 下的目录名（WD 系列目录名 = 仓库名）
LAUNCHER_WD14 = {"wd-vit-tagger-v3": "wd-vit-tagger-v3", "wd-eva02-large-tagger-v3": "wd-eva02-large-tagger-v3",
                 "pixai-tagger-v0.9-onnx": "pixai-tagger-v0.9"}
for _v in WD14_MODELS.values():
    _v["launcher"] = LAUNCHER_WD14.get(_v["repo"].split("/")[-1], "")


def launcher_wd14_name(repo):
    """这个 WD14 / PixAI 模型在启动器里的目录名；启动器不认识的返回 ""。"""
    return LAUNCHER_WD14.get((repo or "").split("/")[-1], "")

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


def decode_caption(raw):
    """标签文件的字节 → (文本, 状态)。状态：ok / bom（UTF-8 带 BOM，已去掉）/ bad（不是 UTF-8，文本里有替换字符）。
    不是 UTF-8 的文件不猜编码（GBK 和 Shift-JIS 猜错了写回去就永久乱码），交给标签体检让用户确认后转换。"""
    bom = raw.startswith(b"\xef\xbb\xbf")
    if bom:
        raw = raw[3:]
    try:
        return raw.decode("utf-8"), ("bom" if bom else "ok")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), "bad"


def read_caption_ex(img, ext=".txt"):
    """返回 (文本, 状态)；没有标签文件时 (None, "none")。"""
    try:
        with open(caption_path(img, ext), "rb") as f:
            raw = f.read()
    except OSError:
        return None, "none"
    return decode_caption(raw)


def read_caption(img, ext=".txt"):
    return read_caption_ex(img, ext)[0]


def write_caption(img, text, ext=".txt"):
    """先写临时文件再换过去：批量改到一半出错（磁盘满、被占用）也不会留下写了一半的标签文件。"""
    p = caption_path(img, ext)
    data = ((text or "").strip() + "\n") if (text or "").strip() else ""
    tmp = p + ".wwytmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(data)
        os.replace(tmp, p)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        with open(p, "w", encoding="utf-8", newline="\n") as f:   # 目标被别的程序以不允许替换的方式打开时（Windows），退回直接写
            f.write(data)


# ---------------- 标签编辑用：归一化匹配 + 「标签段 + 自然语言段」拆分 ----------------
SENTENCE_WORDS = 7   # 超过这么多个词的片段当成自然语言句子
_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def norm_tag(t):
    """比较用的标签写法：小写、下划线当空格、去掉转义括号和多余空格。^_^ 这类 3 个字以内的表情保留下划线。"""
    t = (t or "").replace("\ufeff", "").strip().lower().replace("\\(", "(").replace("\\)", ")")
    if len(t) > 3:
        t = t.replace("_", " ")
    return re.sub(r"\s+", " ", t)


def split_input(s):
    """用户输入的一串标签：半角 / 全角逗号、顿号、换行都算分隔。"""
    return [t.strip() for t in re.split(r"[,，、\n]", s or "") if t.strip()]


def is_prose(seg):
    """这一段是不是自然语言（而不是标签）。"""
    s = (seg or "").strip()
    if not s or not (re.search(r"[A-Za-z]{2}", s) or _CJK.search(s)):   # 「...」「!?」这类符号标签不算
        return False
    words = s.split()
    if len(words) >= SENTENCE_WORDS or s.endswith((".", "。")):
        return True
    if len(words) >= 3 and s[0].isupper() and "(" not in s:   # 「A girl stands」这类开头大写的短句；WD14 / D 站标签都是小写
        return True
    return len(_CJK.findall(s)) >= 12   # 中文 / 日文长句（里面没有半角逗号，整段会被当成一个片段）


def parse_caption(cap):
    """拆成 (标签列表, 自然语言段原文, 两者之间的分隔符原文)。
    反推的混合格式是「触发词, 标签…. 自然语言句子」或「触发词, 自然语言句子」：句子里也有逗号，
    不能整条按逗号拆（会把句子拆碎，最后一个标签还会和句子粘在一起）。自然语言段原样保留、不重新拼。"""
    cap = (cap or "").replace("\ufeff", "")
    tags, head_end = [], 0
    for m in re.finditer(r"[^,\n]+", cap):
        seg = m.group(0)
        if not seg.strip():
            continue
        lead = len(seg) - len(seg.lstrip())
        start = m.start() + lead
        # 「mosaic censoring. A girl is standing」：句号前面是最后一个标签，后面是句子
        k = re.search(r"\.\s+(?=[A-Z])", seg)
        if k and len(seg[k.end():].split()) < 2:
            k = None
        if k or is_prose(seg):
            if k and not is_prose(seg[:k.start()]) and seg[:k.start()].strip():
                tags.append(seg[:k.start()].strip())
                head_end = m.start() + k.start()
                start = m.start() + k.end()
            return tags, cap[start:].strip(), cap[head_end:start] if tags else ""
        tags.append(seg.strip())
        head_end = m.end()
    return tags, "", ""


def build_caption(tags, text="", sep=""):
    tags = [t for t in (x.strip() for x in tags) if t]
    text = (text or "").strip()
    if tags and text:
        return ", ".join(tags) + (sep or ". ") + text
    return text or ", ".join(tags)

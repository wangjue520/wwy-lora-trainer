# -*- coding: utf-8 -*-
"""自动判断触发词：主触发词 + 角色特征 + 服装。

- 主触发词：训练集里出现最多的角色标签（D 站字典里分类是「角色」的），或者用户自己填的词。
- 角色特征：发色、发型、瞳色、兽耳、角、尾巴、肤色……（长在角色身上、换衣服也不变的）
- 服装：衣服、鞋袜、帽子、发饰、首饰……
特征和服装有两种选法：
  freq    出现最多：在多少比例以上的图里出现就选上
  typical 典型图片：选一张最能代表角色的图（自动挑，或者用户指定），用它身上的特征和服装
应用时有两种标法：
  全标：每张图的标签重排成「主触发词, 角色特征…, 服装…, 其余标签」，什么都不删
  删标：删掉选中的那组标签（角色特征 / 服装，可以只删一组），被删的特征都记到主触发词上——
        只写主触发词就能出这些特征（人物 LoRA 常用做法）；没删的照样排到前面结果记在 data/triggers.json，送到启动器时写进 LoRA 说明。"""
import collections
import json
import os
import re
import threading
import time

import captionutil as cu
import dataset
import paths

FILE = os.path.join(paths.DATA, "triggers.json")
_lock = threading.Lock()

COLORS = ("aqua black blonde blue brown green grey gray orange pink purple red silver white yellow light_blue dark_blue "
          "light_brown light_purple light_green dark_green dark_red platinum_blonde lavender beige multicolored two-tone "
          "gradient streaked colored_inner colored_tips split-color rainbow").split()
HAIR_LEN = "absurdly_long very_long long medium short very_short".split()
HAIR_STYLE = set("""twintails low_twintails short_twintails ponytail high_ponytail side_ponytail low_ponytail braid twin_braids
side_braid single_braid french_braid crown_braid braided_bangs hair_bun double_bun single_hair_bun drill_hair twin_drills
ringlets bob_cut hime_cut ahoge sidelocks bangs blunt_bangs swept_bangs parted_bangs crossed_bangs hair_between_eyes
hair_over_one_eye hair_over_eyes hair_intakes antenna_hair wavy_hair curly_hair straight_hair messy_hair spiked_hair
hair_flaps one_side_up two_side_up half_updo low-tied_long_hair tied_hair hair_rings asymmetrical_hair
asymmetrical_bangs long_bangs short_bangs pixie_cut undercut mohawk afro dreadlocks buzz_cut quad_tails
heterochromia slit_pupils symbol-shaped_pupils star-shaped_pupils heart-shaped_pupils ringed_eyes sanpaku tsurime tareme
animal_ears cat_ears dog_ears fox_ears wolf_ears rabbit_ears bear_ears horse_ears mouse_ears tiger_ears pointy_ears
elf horns single_horn demon_horns dragon_horns oni_horns tail cat_tail fox_tail dog_tail wolf_tail demon_tail dragon_tail
wings demon_wings angel_wings dragon_wings feathered_wings halo fang fangs skin_fang freckles mole mole_under_eye
mole_under_mouth scar scar_on_face facial_mark tattoo dark-skinned_female dark-skinned_male dark_skin pale_skin tan
tanlines colored_skin blue_skin muscular muscular_female abs flat_chest small_breasts medium_breasts large_breasts
huge_breasts gigantic_breasts thick_eyebrows short_eyebrows long_eyelashes""".split())
NOT_FEATURE = set("""floating_hair wet_hair hair_spread_out hand_in_own_hair hand_in_hair hair_blowing hair_flowing_over
closed_eyes half-closed_eyes one_eye_closed wide-eyed empty_eyes glowing_eyes rolling_eyes covered_eyes looking_at_viewer
fake_animal_ears fake_tail fake_horns""".split())
OUTFIT_WORDS = set("""dress skirt miniskirt shirt t-shirt blouse jacket coat overcoat uniform serafuku gloves glove
thighhighs pantyhose socks kneehighs legwear leggings boots shoes footwear sandals heels loafers sneakers mary_janes slippers
hat cap beret headwear hood hoodie helmet crown tiara veil headband hairband hairclip headdress headpiece ribbon bow bowtie
necktie tie collar sailor_collar neckerchief ascot scarf choker necklace pendant brooch earrings jewelry bracelet wristband
armband armlet sleeves leotard swimsuit bikini one-piece_swimsuit kimono yukata hakama obi sash armor cape capelet cloak
robe belt apron vest shorts pants trousers jeans frills sweater cardigan corset garter bodysuit suit tuxedo overalls parka
bra panties lingerie bloomers glasses eyewear goggles mask epaulettes buttons zipper maid nurse school_uniform
detached_sleeves detached_collar wrist_cuffs thigh_strap hair_ornament hair_ribbon hair_bow hair_flower hair_bobbles
hair_tie scrunchie hairpin kanzashi tabi geta pouch bag backpack""".split())
NOT_OUTFIT_WORDS = set("lifted pull pulled aside removed undone unbuttoned adjusting holding taking tugging torn wet "
                       "no naked nude undressing grabbing biting".split())


def _key(t):
    return t.strip().lower().replace(" ", "_")


def classify(t):
    """feature / outfit / other"""
    k = _key(t)
    if k in NOT_FEATURE or k.startswith("fake_"):
        return "outfit" if k.startswith("fake_") else "other"
    toks = re.split(r"[_\-]", k)
    if any(w in NOT_OUTFIT_WORDS for w in toks):
        return "other"
    if k in HAIR_STYLE:
        return "feature"
    m = re.fullmatch(r"(.+)_(hair|eyes|pupils|skin)", k)
    if m and (m.group(1) in COLORS or m.group(1) in HAIR_LEN):
        return "feature"
    if k in OUTFIT_WORDS or any(w in OUTFIT_WORDS for w in toks) or any(k.endswith("_" + w) for w in OUTFIT_WORDS):
        return "outfit"
    return "other"


def _load_db():
    try:
        with open(FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def get(d):
    return _load_db().get(os.path.normcase(os.path.abspath(d))) if d else None


def _save(d, info):
    with _lock:
        db = _load_db()
        db[os.path.normcase(os.path.abspath(d))] = info
        tmp = FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=1)
        os.replace(tmp, FILE)


def analyze(d, ext=".txt", mode="freq", typical=None, main=None, hints=None, feat_th=0.5, outfit_th=0.35):
    """hints：标签里没有角色名时的备选主触发词（反推页填的触发词、爬图时搜的角色标签），优先用字典里是「角色」的。"""
    import tagdict
    data = dataset.load(d, ext)
    items = [it for it in data["items"] if (it["caption"] or "").strip()]
    n = len(items)
    if not n:
        raise ValueError("这个训练集还没有标签，先去「标签反推」")
    old = get(d) or {}
    old_main = (old.get("main") or "").strip()
    caps = {}
    freq = collections.Counter()
    for it in items:
        tags = [t for t in cu.split_tags(it["caption"]) if not dataset._is_sentence(t)]
        if old_main:   # 上次加的主触发词不参与统计
            tags = [t for t in tags if t != old_main]
        caps[it["path"]] = tags
        freq.update(set(tags))
    info = tagdict.info([_key(t) for t in freq]) if tagdict.load() else {}
    cat = {t: (info.get(_key(t)) or {}).get("category") for t in freq}
    zh = {t: (info.get(_key(t)) or {}).get("zh", "") for t in freq}
    kind = {t: classify(t) if cat[t] in (0, None) else "other" for t in freq}

    # 主触发词：出现最多的角色标签
    chars = [(t, c) for t, c in freq.most_common() if cat[t] == 4]
    cand_main = [{"tag": t, "count": c, "zh": zh[t]} for t, c in chars[:6]]
    if not main:
        main = old_main or (chars[0][0] if chars and chars[0][1] >= n * 0.3 else "")
    if not main and hints:
        hs = [h.strip() for h in hints if h and h.strip()]
        hi = tagdict.info([_key(h) for h in hs]) if hs else {}
        ch = [h for h in hs if (hi.get(_key(h)) or {}).get("category") == 4]
        main = (ch or hs or [""])[0]
        if any(" " in t for t in freq):   # 训练集标签用空格写法时主触发词也用空格
            main = main.replace("_", " ")

    def group(k, th):
        return [t for t, c in freq.most_common() if kind[t] == k and c >= n * th]

    # 典型图片：它身上的特征 / 服装在整个训练集里越常见越典型；用户也可以自己指定
    def typ_score(p):
        return sum(freq[t] for t in caps[p] if kind[t] in ("feature", "outfit")) / n
    if typical and typical in caps:
        typ = typical
    else:
        typ = max(caps, key=typ_score)
    if mode == "typical":
        feats = [t for t in caps[typ] if kind[t] == "feature" and freq[t] >= n * 0.25]
        outfit = [t for t in caps[typ] if kind[t] == "outfit" and freq[t] >= n * 0.15]
        feats.sort(key=lambda t: -freq[t])
        outfit.sort(key=lambda t: -freq[t])
    else:
        feats, outfit = group("feature", feat_th), group("outfit", outfit_th)
        if outfit:   # 训练集里可能有好几套衣服：只要和最常见那件同时出现的（同一套）
            anchor = [p for p in caps if outfit[0] in caps[p]]
            outfit = [t for t in outfit if sum(t in caps[p] for p in anchor) >= freq[t] * 0.6]
    # 人物还是画风：有占三成以上的角色标签，或者大多数图发色、瞳色一致 → 人物；否则当画风（画风不需要特征 / 服装触发词）
    def top_color(part, th):
        return any(freq[t] >= n * th for t in freq if (m := re.fullmatch(r"(.+)_" + part, _key(t))) and m.group(1) in COLORS)
    is_char = bool(chars and chars[0][1] >= n * 0.3) or (top_color("hair", 0.5) and top_color("eyes", 0.4))
    pct = lambda t: round(freq[t] / n * 100)
    row = lambda t: {"tag": t, "pct": pct(t), "zh": zh[t]}
    return {
        "dir": d, "images": n, "mode": mode, "main": main, "main_candidates": cand_main,
        "features": [row(t) for t in feats], "outfit": [row(t) for t in outfit], "typical": typ,
        # 备选：没选上但也比较常见的，界面上可以勾选加进去
        "more_features": [row(t) for t in group("feature", 0.15) if t not in feats][:20],
        "more_outfit": [row(t) for t in group("outfit", 0.1) if t not in outfit][:24],
        "prune": norm_prune(old.get("prune")), "kind": "character" if is_char else "style",
    }


def _norm_tag(t):
    return t.strip().lower().replace("_", " ")


# 提示词开头的质量 / 元信息标签：触发词插在它们后面（官方推荐顺序是 质量词 → 1girl → 角色 → 其他；Pony 的 score 标签必须在最前）
_QUALITY = re.compile(r"^(masterpiece|(best|amazing|good|normal|high) quality|very aesthetic|absurdres|highres|newest|recent|safe|"
                      r"score[_ ]\d+([_ ]up)?|source[_ ]\w+|rating[_ ]\w+|year \d{4}|highly detailed|\d+(girl|boy|other)s?)$")


def sync_prompt(text, info):
    """把一条测试提示词的开头换成「用 LoRA 时写」的那一串（info 是 get() 的结果）。
    全标时要写上人物特征 / 服装，删标时只写触发词——原来写的触发词、特征、服装标签先去掉再放到最前面，
    自己写的场景 / 动作等其余部分原样保留。"""
    want = [t.strip() for t in (info.get("prompt") or "").split(",") if t.strip()]
    if not want:
        return text
    known = {_norm_tag(t) for t in [info.get("main") or ""] + list(info.get("features") or []) + list(info.get("outfit") or [])}
    known |= {_norm_tag(t) for t in want}
    rest = [t.strip() for t in (text or "").split(",") if t.strip() and _norm_tag(t) not in known]
    head = []
    while rest and _QUALITY.match(_norm_tag(rest[0])):
        head.append(rest.pop(0))
    if any(classify(t) == "outfit" for t in rest):   # 提示词里自己写了衣服（换装测试）：原装服装标签不加，免得两套衣服打架
        outfit = {_norm_tag(t) for t in info.get("outfit") or []}
        want = [t for t in want if _norm_tag(t) not in outfit]
    return ", ".join(head + want + rest)


def has_main(text, info):
    return _norm_tag(info.get("main") or "") in {_norm_tag(t) for t in (text or "").split(",")}


def sync_prompts(d, prompts, only_with_main=False):
    """对一组测试提示词（[{prompt, ...}]）做 sync_prompt，返回 (新列表, 改了几条)。训练集没有触发词记录时原样返回。
    only_with_main：只改已经写了主触发词的（自动同步 / 开练前检查用；没写触发词的标准提示词不动）。"""
    info = get(d)
    if not info or not info.get("prompt"):
        return prompts, 0
    out, n = [], 0
    for p in prompts or []:
        q = dict(p)
        if (p.get("prompt") or "").strip() and (not only_with_main or has_main(p["prompt"], info)):
            q["prompt"] = sync_prompt(p["prompt"], info)
            n += q["prompt"] != p["prompt"]
        out.append(q)
    return out, n


def norm_prune(p):
    """删标设置：{"features": 删角色特征, "outfit": 删服装}。旧版本存的是 True/False（只管角色特征）。"""
    if isinstance(p, dict):
        return {"features": bool(p.get("features")), "outfit": bool(p.get("outfit"))}
    return {"features": bool(p), "outfit": False}


def apply(d, main, features, outfit, ext=".txt", prune=None, mode="freq", typical=None):
    """重排每张图的标签：主触发词, 角色特征…, 服装…, 其余。prune 见 norm_prune：删掉选中的那组标签。可在标签编辑里撤销。"""
    prune = norm_prune(prune)
    main = (main or "").strip()
    if not main:
        raise ValueError("请填写主触发词")
    old = get(d) or {}
    old_main = (old.get("main") or "").strip()
    files = cu.list_images(d)
    dataset._backup(d, files, ext, "自动整理触发词")
    feats, outs = list(features or []), list(outfit or [])
    fs, os_ = set(feats), set(outs)
    changed = 0
    for f in files:
        cap = cu.read_caption(f, ext)
        if cap is None:
            continue
        tags = cu.split_tags(cap)
        rest = [t for t in tags if t not in (main, old_main) and t not in fs and t not in os_]
        has = set(tags)
        new = ([main] + ([] if prune["features"] else [t for t in feats if t in has])
               + ([] if prune["outfit"] else [t for t in outs if t in has]) + rest)
        text = ", ".join(new)
        if text != cap:
            cu.write_caption(f, text, ext)
            changed += 1
    # 用 LoRA 时要写的：主触发词 + 没删掉的那几组（删掉的已经记在主触发词上了）
    prompt = ", ".join([main] + ([] if prune["features"] else feats) + ([] if prune["outfit"] else outs))
    info = {"main": main, "features": feats, "outfit": outs, "prune": prune, "mode": mode, "typical": typical,
            "prompt": prompt, "time": time.time()}
    _save(d, info)
    return {"changed": changed, "prompt": prompt, "info": info}

# -*- coding: utf-8 -*-
"""Danbooru 爬图（只用标准库），给训练集准备图片和标签。

基于作者的「Danbooru Pro Crawler v3.5」重写，修掉了两个关键问题（2026-10 用 API 实测确认的规则）：

  1. 收藏数元标签是 favcount:，旧版写成 fav_count: —— 服务器把它当成一个不存在的普通标签，结果永远 0 条。
  2. 标签名额：普通会员和匿名都是 2 个，Gold 6 个，Platinum 以上不限。
       占名额：普通标签、-排除标签、~或标签、order: 排序
       不占名额：score:、favcount:、date:、rating:（还有 id/width/height/filetype 等）
     旧版把分数、日期也算成占名额，登录后先塞满黑名单撞 422，再误判「需要 3 个槽位」直接终止；
     匿名时一开始就按 2 个算、没塞黑名单，反而能跑——所以出现「登录比不登录还差」。

新做法：先查账号等级得到真实名额；名额只留给最能缩小范围的标签（先查各标签图数，最少的优先），
放不下的标签、排序、黑名单全部改为本地过滤 / 本地排序，任何条件组合都不会因为名额失败。
翻页用 page=b<id> 游标，没有 1000 页上限，所以不再需要按月切片。"""
import base64
import fnmatch
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("LORA_DANBOORU_BASE", "https://danbooru.donmai.us").rstrip("/")   # 可用环境变量指向镜像 / 反代
UA = "WWYLoraTrainer/1.0 (dataset-tool)"
PER_PAGE = 200
LEVEL_LIMIT = {"member": 2, "anonymous": 2, "restricted": 2, "gold": 6}   # 其余（platinum/builder/…）不限

ORDERS = {   # 前端选项 → (服务器 order:, 本地排序字段, 降序)
    "new": ("", "id", True), "old": ("order:id_asc", "id", False),
    "score": ("order:score", "score", True), "score_asc": ("order:score_asc", "score", False),
    "fav": ("order:favcount", "fav_count", True), "fav_asc": ("order:favcount_asc", "fav_count", False),
    "random": ("order:random", None, None),
}
MONO = {"monochrome", "greyscale", "grayscale", "spot_color"}
COMIC = {"comic", "4koma", "manga"}
STATIC_EXTS = {"jpg", "jpeg", "png", "webp", "avif"}
BLACKLIST_PRESET = ("lowres jpeg_artifacts scan bad_anatomy watermark signature artist_name web_address username dated "
                    "multiple_views reference_sheet censored photo_(medium) sketch")


class CrawlError(Exception):
    pass


def explain(code, body):
    """Danbooru 返回的不是它自己的 JSON（被 Cloudflare 拦、代理出错）时的中文说明。"""
    low = body.lower()
    if "just a moment" in low or "cf-chl" in low or "attention required" in low or "cloudflare" in low:
        return (f"被 Danbooru 的 Cloudflare 防护拦下了（{code}）。一般是梯子节点的 IP 被它拉黑了："
                "换一个节点（日本 / 美国 / 新加坡等）再试")
    if code == 407:
        return "代理要求用户名密码（407），检查训练器设置里的代理地址"
    text = re.sub(r"<[^>]+>|\s+", " ", body)[:150].strip()
    return f"Danbooru 返回错误 {code}：{text}"


def _is_json(body):
    return body.lstrip()[:1] in ("{", "[")


# ---------------- 连接方式 ----------------
# 和下载模型一样有直连 / 系统代理 / 设置里填的代理 / 本机梯子端口几种走法。以前只走系统代理：
# 梯子没开「系统代理」、或系统代理那个节点被 Danbooru 拦了，就整个爬不了，还被误报成「标签不存在」。
_route = [0.0, None]   # [选定时间, (线路名, 代理)]
_route_lock = threading.Lock()
ROUTE_TTL = 600


def _probe(proxy):
    """用这条线路查一次标签：(通不通, 说明)。"""
    import netpick
    req = urllib.request.Request(f"{BASE}/tags.json?search[name]=solo&only=name&limit=1", headers={"User-Agent": UA})
    try:
        with netpick.opener(proxy).open(req, timeout=10) as r:
            body = r.read().decode("utf-8", "replace")
        if not _is_json(body):
            return False, "返回的不是 Danbooru 的数据（网络被劫持 / 代理出错）"
        return True, ""
    except urllib.error.HTTPError as e:
        return False, explain(e.code, e.read()[:2000].decode("utf-8", "replace"))
    except Exception as e:
        return False, f"连不上（{getattr(e, 'reason', None) or e}）"


def pick_route(refresh=False):
    """选一条能访问 Danbooru 的线路，返回 (线路名, 代理或 None)。都不通就抛 CrawlError，逐条写明原因。"""
    import netpick
    with _route_lock:   # 前端补全会同时发好几个请求，只让一个去测
        if not refresh and _route[1] and time.time() - _route[0] < ROUTE_TTL:
            return _route[1]
        rs = netpick.routes(ttl=0 if refresh else 60)
        with ThreadPoolExecutor(len(rs)) as ex:
            res = list(ex.map(lambda r: _probe(r[1]), rs))
        # 都通时：手动填的代理 > 系统代理 > 直连 > 探测到的本机端口
        prio = ("自定义代理", "系统代理", "直连", "本机代理")
        ok = sorted((r for r, (good, _) in zip(rs, res) if good),
                    key=lambda r: next((i for i, p in enumerate(prio) if r[0].startswith(p)), 9))
        if ok:
            _route[:] = [time.time(), ok[0]]
            return ok[0]
        _route[:] = [0.0, None]
    lines = [f"  {name}：{why}" for (name, _), (_, why) in zip(rs, res)]
    if any("Cloudflare" in why for _, why in res):
        tip = "梯子能连上 Danbooru，但节点被它的防护拦了：换一个节点再试（浏览器能打开 danbooru.donmai.us 的节点一般就行）。"
    else:
        tip = ("国内网络要开梯子。梯子没开「系统代理」模式的话，在「训练器设置 → 网络」里填代理地址"
               "（例如 http://127.0.0.1:7890）。")
    raise CrawlError("所有连接方式都访问不了 Danbooru：\n" + "\n".join(lines) + "\n" + tip)


class Client:
    def __init__(self, job, login="", api_key=""):
        self.job = job
        self.auth = None
        if login and api_key:
            self.auth = "Basic " + base64.b64encode(f"{login}:{api_key}".encode()).decode()
        self._lock = threading.Lock()
        self._last = 0.0
        self.interval = 0.6   # Danbooru 对匿名 / 普通会员有频率限制，搜索接口慢一点更稳
        self.route = None   # (线路名, 代理)，第一次请求时选

    def opener(self, refresh=False):
        import netpick
        if self.route is None or refresh:
            self.route = pick_route(refresh)
        return netpick.opener(self.route[1])

    def _wait(self):
        with self._lock:
            d = time.time() - self._last
            if d < self.interval:
                time.sleep(self.interval - d)
            self._last = time.time()

    def get(self, path, params=None, retries=6, timeout=30):
        url = f"{BASE}/{path}" + ("?" + urllib.parse.urlencode(params) if params else "")
        refreshed = False
        for attempt in range(retries):
            if self.job.stop_flag:
                raise CrawlError("已停止")
            self._wait()
            req = urllib.request.Request(url, headers={"User-Agent": UA, **({"Authorization": self.auth} if self.auth else {})})
            try:
                with self.opener().open(req, timeout=timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read()[:2000].decode("utf-8", "replace")
                if e.code == 401:
                    raise CrawlError("用户名或 API Key 不对（401）。API Key 在 Danbooru 网页「我的账号 → API Key」里生成。")
                if e.code == 429 or e.code >= 500:
                    w = min(5 * (attempt + 1), 30)
                    self.job.write(f"  服务器{'限流' if e.code == 429 else '繁忙'}（{e.code}），{w} 秒后重试")
                    time.sleep(w)
                    continue
                if not _is_json(body):   # 不是 Danbooru 自己的回答（Cloudflare 拦截页、代理报错）：换条线路再试
                    if not refreshed:
                        refreshed = True
                        self.opener(refresh=True)
                        continue
                    raise CrawlError(explain(e.code, body))
                return {"_error": e.code, "_body": body[:400]}
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                if attempt == retries - 1:
                    raise CrawlError(f"连不上 Danbooru：{e}。国内网络需要代理 / 加速器。")
                if attempt >= 1 and not refreshed:   # 连续出错：线路可能断了（梯子关了 / 换了节点），重新选
                    refreshed = True
                    self.opener(refresh=True)
                    continue
                time.sleep(min(3, timeout / 3))
        raise CrawlError("Danbooru 一直限流，稍后再试")

    def download(self, url, dest):
        tmp = dest + ".part"
        for attempt in range(5):
            if self.job.stop_flag:
                return False
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with self.opener().open(req, timeout=60) as r, open(tmp, "wb") as f:
                    while True:
                        b = r.read(1 << 16)
                        if not b:
                            break
                        if self.job.stop_flag:
                            raise CrawlError("已停止")
                        f.write(b)
                if os.path.getsize(tmp) == 0:
                    raise OSError("0 字节")
                os.replace(tmp, dest)
                return True
            except CrawlError:
                break
            except Exception as e:
                code = getattr(e, "code", None)
                time.sleep(min(2 * (attempt + 1), 15) if code == 429 else 1.5 * (attempt + 1))
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


def split_tags(s):
    return [t for t in re.split(r"[\s,]+", (s or "").strip().lower()) if t]


def account_limit(client, login):
    """(等级名, 名额)。名额 None = 不限。"""
    if not client.auth:
        return "匿名", 2
    d = client.get("profile.json")
    if not isinstance(d, dict) or d.get("_error"):
        return "未知（按普通会员算）", 2
    lv = (d.get("level_string") or "Member").lower()
    if "tag_query_limit" in d:   # 账号信息里直接给了名额，比按等级名猜更准（None = 不限）
        lim = d["tag_query_limit"]
        return d.get("level_string") or "Member", (int(lim) if lim else None)
    return d.get("level_string") or "Member", LEVEL_LIMIT.get(lv, None)


def check_tags(client, job, tags):
    """检查标签是否存在，返回 {tag: post_count}；不存在的给出最接近的正确写法。"""
    counts = {}
    for i in range(0, len(tags), 50):
        chunk = tags[i:i + 50]
        d = client.get("tags.json", {"search[name_comma]": ",".join(chunk), "only": "name,post_count", "limit": 100})
        if not isinstance(d, list):   # 查询本身失败，不能当成「标签不存在」
            raise CrawlError("查标签失败：" + (explain(d.get("_error"), d.get("_body", "")) if isinstance(d, dict)
                                              else "Danbooru 返回的数据不对"))
        for t in d:
            counts[t["name"]] = t.get("post_count", 0)
    bad = [t for t in tags if counts.get(t, 0) == 0 and not t.startswith(("*",)) and ":" not in t]
    for t in bad:
        sug = client.get("autocomplete.json", {"search[query]": t, "search[type]": "tag_query", "limit": 5})
        names = [x.get("value") for x in sug if isinstance(x, dict)] if isinstance(sug, list) else []
        hint = f"，是不是：{'、'.join(names[:4])}" if names else ""
        job.write(f"⚠ 标签「{t}」在 Danbooru 上没有图（拼错或不是正式标签名）{hint}")
    return counts, bad


def plan(cfg, limit, counts, job):
    """把用户条件拆成：服务器查询列表 + 本地过滤条件。名额只给最能缩小范围的东西。"""
    pos = split_tags(cfg.get("tags"))
    logic = cfg.get("logic") or "and"
    must = [t for t in split_tags(cfg.get("must_have")) if t not in pos]   # 预设里的「必须包含」（如人物 LoRA 的 solo）
    if logic != "or" or len(pos) <= 1:
        pos, must = pos + must, []   # 「且」：和搜索标签一起按稀有程度分配名额
    bl = split_tags(cfg.get("blacklist"))
    bl_exact = [t for t in bl if "*" not in t and "?" not in t]
    order_tag, sort_field, desc = ORDERS.get(cfg.get("order") or "new", ORDERS["new"])
    free = []   # 不占名额的元标签，一律交给服务器
    if int(cfg.get("min_score") or 0):
        free.append(f"score:>={int(cfg['min_score'])}")
    if int(cfg.get("min_fav") or 0) > 0:
        free.append(f"favcount:>={int(cfg['min_fav'])}")   # 注意是 favcount，不是 fav_count
    if cfg.get("date_from") and cfg.get("date_to"):
        free.append(f"date:{cfg['date_from']}..{cfg['date_to']}")
    elif cfg.get("date_from"):
        free.append(f"date:>={cfg['date_from']}")
    elif cfg.get("date_to"):
        free.append(f"date:<={cfg['date_to']}")
    ratings = cfg.get("ratings") or []
    if ratings and len(ratings) < 4:
        free.append("rating:" + ",".join(ratings))
    if cfg.get("min_side"):   # 训练集常用：短边至少多少像素（width/height 也不占名额）
        free.append(f"width:>={int(cfg['min_side'])}")
        free.append(f"height:>={int(cfg['min_side'])}")
    slots = limit if limit is not None else 999
    queries, local_need = [], []

    if logic == "or" and len(pos) > 1:
        # 或：每条查询放尽量多的 ~标签，放不下就拆成多条查询再合并去重
        per = max(1, slots)
        groups = [pos[i:i + per] for i in range(0, len(pos), per)]
        for g in groups:
            queries.append(" ".join(("~" + t) if len(g) > 1 else t for t in g))
        used = min(len(pos), per) if len(groups) == 1 else per
    else:
        # 且：按图数从少到多排，最稀有的标签最能缩小范围，优先占名额
        ranked = sorted(pos, key=lambda t: counts.get(t, 10 ** 9))
        server_pos = ranked[:slots]
        local_need = ranked[slots:]
        queries.append(" ".join(server_pos))
        used = len(server_pos)
    local_need = local_need + must

    left = slots - used
    server_order = ""
    if order_tag and left > 0 and len(queries) == 1:
        server_order, left = order_tag, left - 1
    ex = [f"-{t}" for t in bl_exact][:max(0, left)]
    queries = [" ".join(x for x in [q, *ex, server_order, *free] if x) for q in queries]

    # ---- 说明 ----
    job.write("搜索方案：")
    for q in queries:
        job.write(f"  服务器查询：{q}")
    if local_need:
        job.write(f"  名额不够，这些标签改在本地筛：{' '.join(local_need)}（会多翻一些页）")
    if order_tag and not server_order:
        job.write("  排序改在本地做：先收集符合条件的图，再按" + {"id": "上传时间", "score": "分数", "fav_count": "收藏数"}.get(sort_field or "", "随机") + "排序")
    if len(ex) < len(bl_exact):
        job.write(f"  黑名单有 {len(bl_exact) - len(ex)} 个放不进服务器查询，在本地过滤")
    return {"queries": queries, "local_need": set(local_need), "sort_field": sort_field, "desc": desc,
            "random": order_tag == "order:random", "server_order": bool(server_order), "bl": bl}


def post_ok(post, cfg, pl, stats):
    tags = set((post.get("tag_string") or "").split())

    def no(reason):
        stats[reason] = stats.get(reason, 0) + 1
        return False

    if not (post.get("file_url") or post.get("large_file_url")):
        return no("没有文件地址（需要 Gold 账号或已被删除）")
    if pl["local_need"] - tags:
        return no("不含全部标签")
    if cfg.get("_anyof") and not tags & set(cfg["_anyof"]):
        return no("不含 ~ 标签里的任意一个")
    ext = (post.get("file_ext") or "").lower()
    if cfg.get("static_only", True) and ext not in STATIC_EXTS:
        return no("非静态图")
    for b in pl["bl"]:
        if ("*" in b or "?" in b) and any(fnmatch.fnmatchcase(t, b) for t in tags) or b in tags:
            return no(f"黑名单 {b}")
    if cfg.get("no_mono", True) and tags & MONO:
        return no("黑白图")
    if cfg.get("no_comic") and tags & COMIC:
        return no("漫画")
    side = int(cfg.get("min_side") or 0)
    if side and min(post.get("image_width") or 0, post.get("image_height") or 0) < side:
        return no("分辨率不够")
    return True


def caption(post, cfg):
    def norm(t):
        return t if len(t) <= 3 else t.replace("_", " ")   # ^_^ 这类表情标签保留下划线
    head = []
    if cfg.get("trigger"):
        head += [x.strip() for x in cfg["trigger"].split(",") if x.strip()]
    if cfg.get("cap_character", False):
        head += [norm(t) for t in (post.get("tag_string_character") or "").split()]
    if cfg.get("cap_copyright", False):
        head += [norm(t) for t in (post.get("tag_string_copyright") or "").split()]
    if cfg.get("cap_artist"):
        head += ["@" + norm(t) for t in (post.get("tag_string_artist") or "").split()]
    body = [norm(t) for t in (post.get("tag_string_general") or "").split()]
    drop = set(split_tags(cfg.get("cap_exclude")))
    out = [t for t in head + body if t.replace(" ", "_") not in drop]
    return ", ".join(dict.fromkeys(out))


def subdir(post, cfg):
    mode = cfg.get("folder") or "none"
    if mode == "none":
        return ""
    def first(field, fallback):
        v = (post.get(field) or "").split()
        return re.sub(r'[\\/:*?"<>|]', "_", v[0])[:60] if v else fallback
    if mode == "character":
        return re.sub(r"_\([^)]*\)", "", first("tag_string_character", "unknown_character"))
    if mode == "copyright":
        return first("tag_string_copyright", "unknown_ip")
    if mode == "artist":
        return first("tag_string_artist", "unknown_artist")
    return ""


def _folder_name(tag):
    """新建文件夹用的名字：字典里有中文名就用中文（「弗洛洛（鸣潮）」→「弗洛洛」），否则用标签去掉作品名（phrolova_(wuthering_waves) → phrolova）。"""
    name = ""
    try:
        import tagdict
        if tagdict.load():
            r = tagdict.resolve(tag) or tag
            name = (tagdict.info([r]).get(r) or {}).get("zh") or ""
    except Exception:
        pass
    name = re.split(r"[（(]", name)[0].strip() or re.sub(r"_\([^)]*\)$", "", tag)
    return re.sub(r'[\\/:*?"<>|]+', "_", name).strip(" ._") or "danbooru"


def plan_dir(out_dir, tags):
    """爬之前检查保存位置：里面已经有别的角色的图（比如上次爬的爱弥斯），这次又爬弗洛洛、也没手动换文件夹，
    就建议新建「弗洛洛」文件夹爬到里面，免得两个角色的图混成一个训练集。
    返回 {ask: False} 或 {ask: True, images, prev, name, suggest}。同一个角色接着爬（补图）不问。"""
    import captionutil as cu
    import triggers
    pos = [t for t in split_tags(tags) if t[0] not in "-~" and ":" not in t]
    if not out_dir or not pos or not os.path.isdir(out_dir):
        return {"ask": False}
    imgs = cu.list_images(out_dir)
    if not imgs:
        return {"ask": False}
    try:
        import tagdict
        if tagdict.load():
            pos = [tagdict.resolve(t) or t for t in pos]
    except Exception:
        pass
    def same(prev):   # 搜的第一个标签（一般就是角色）一样 = 同一个角色接着补图，多加几个过滤标签不算换角色
        return bool(prev) and pos[0].lower() in {t.lower() for t in prev}
    prev = triggers.crawl_tags(out_dir)
    if same(prev):
        return {"ask": False}
    name = _folder_name(pos[0])
    # 保存位置本身就是一个角色的文件夹（图直接放在里面）→ 在它旁边建；只是装着几个子文件夹的大文件夹 → 在它里面建
    direct = any(os.path.dirname(f) == os.path.normpath(out_dir) for f in imgs)
    parent = os.path.dirname(os.path.normpath(out_dir)) if direct else out_dir
    if os.path.basename(os.path.normpath(out_dir)) == name:   # 已经在叫这个名字的文件夹里了
        parent = os.path.dirname(os.path.normpath(out_dir))
    cand, k = os.path.join(parent, name), 0
    while os.path.isdir(cand) and cu.list_images(cand) and not same(triggers.crawl_tags(cand)):
        k += 1
        cand = os.path.join(parent, f"{name}({k})")
    return {"ask": True, "images": len(imgs), "prev": prev, "name": os.path.basename(cand), "suggest": cand}


def run(job, cfg):
    out = cfg.get("out_dir")
    if not out:
        raise CrawlError("请选择保存位置")
    os.makedirs(out, exist_ok=True)
    want = int(cfg.get("count") or 0)
    client = Client(job, cfg.get("login", "").strip(), cfg.get("api_key", "").strip())
    client.opener()
    job.write(f"连接方式：{client.route[0]}")

    level, limit = account_limit(client, cfg.get("login"))
    job.write(f"账号：{level}，每次搜索最多 {limit if limit else '不限'} 个标签"
              "（分数、收藏、日期、分级、分辨率条件不占名额）")
    try:   # 用中英字典把别名换成 D 站正式标签（字典没下载时跳过）
        import tagdict
        fixed = []
        for key in ("tags", "must_have", "blacklist"):
            items = split_tags(cfg.get(key))
            new = []
            for t in items:   # 保留 -（排除）/ ~（任意一个）前缀，只换后面的标签名
                pre = t[0] if t[0] in "-~" and len(t) > 1 else ""
                name = t[len(pre):]
                new.append(pre + ((tagdict.resolve(name) or name) if "*" not in name and "?" not in name else name))
            fixed += [f"{a} → {b}" for a, b in zip(items, new) if a != b]
            cfg[key] = " ".join(new)
        if fixed:
            job.write("别名已换成正式标签：" + "，".join(fixed))
    except Exception:
        pass
    # 搜索标签里的前缀：-标签 = 排除（并入黑名单），~标签 = 这几个里有任意一个就行
    raw = split_tags(cfg.get("tags"))
    neg = [t[1:] for t in raw if t.startswith("-") and len(t) > 1]
    anyof = [t[1:] for t in raw if t.startswith("~") and len(t) > 1]
    pos = [t for t in raw if t[0] not in "-~"]
    if cfg.get("logic") == "or" or not pos:   # 「或」模式下 ~ 没有区别；只写了 ~标签 也当「或」
        pos, anyof = pos + anyof, []
        if len(pos) > 1:
            cfg["logic"] = "or"
    cfg["tags"] = " ".join(pos)
    if neg:
        cfg["blacklist"] = " ".join(split_tags(cfg.get("blacklist")) + neg)
    cfg["_anyof"] = anyof
    if not pos:
        raise CrawlError("请填写至少一个搜索标签（只写排除标签没法搜）")
    extra = [t for t in split_tags(cfg.get("must_have")) + anyof + neg if t not in pos]
    counts, bad = check_tags(client, job, pos + extra)
    bad_extra = [t for t in bad if t not in pos]
    bad = [t for t in bad if t in pos]
    if [t for t in bad_extra if t in anyof] and not [t for t in anyof if t not in bad_extra]:
        raise CrawlError(f"「任意一个」里的标签 {'、'.join(bad_extra)} 在 Danbooru 上都没有图，请按上面的提示改正后再试。")
    if bad and (cfg.get("logic") != "or" or len(bad) == len(pos)):
        raise CrawlError(f"标签 {'、'.join(bad)} 在 Danbooru 上没有图，按「且」搜索一定是空结果。请按上面的提示改正后再试。")
    for t in pos:
        if t in counts:
            job.write(f"  {t}：{counts[t]:,} 张")
    pl = plan(cfg, limit, counts, job)

    # ---- 阶段一：收集符合条件的图（只拿信息，不下载） ----
    cap = want * max(1, int(cfg.get("collect_factor") or 5)) if want else 0
    if want and not pl["server_order"] and (pl["sort_field"] not in (None, "id") or pl["random"]):
        # 排序（或随机）只能在本地做：要拿到真正的前 N 名，得把全部结果都收集过来再排。
        # 先问服务器一共有多少张，不多就全收；太多才按倍率截断，并说明可能漏掉一些
        total, unknown = 0, False
        for q in pl["queries"]:
            c = client.get("counts/posts.json", {"tags": q})
            n = ((c or {}).get("counts") or {}).get("posts") if isinstance(c, dict) else None
            if n is None:   # 登录状态下 Danbooru 对带条件的查询经常不给总数（返回 null）
                unknown = True
            else:
                total += int(n)
        limit_all = int(cfg.get("max_collect") or 20000)
        if unknown:
            cap = limit_all
            job.write(f"服务器没有给出总数：全部收集（最多 {limit_all:,} 张）后再精确排序，翻完就停")
        elif total <= limit_all:
            cap = 0
            job.write(f"服务器上一共 {total:,} 张符合查询，全部收集后再精确排序（约 {total // PER_PAGE + 1} 页）")
        else:
            cap = want * max(20, int(cfg.get("collect_factor") or 5))
            job.write(f"服务器上有 {total:,} 张，太多了：只收集最新的 {cap:,} 张再排序，可能漏掉更早的高分图。"
                      "想更准就加最低分数 / 收藏 / 日期范围把总数降下来。")
    pool, seen, stats = [], set(), {}
    fields = ("id,file_url,large_file_url,file_ext,tag_string,tag_string_general,tag_string_character,"
              "tag_string_copyright,tag_string_artist,score,fav_count,image_width,image_height,rating,created_at")
    for qi, q in enumerate(pl["queries"]):
        cursor, page, scanned = None, 1, 0
        while not job.stop_flag and (not cap or len(pool) < cap):
            params = {"tags": q, "limit": PER_PAGE, "only": fields}
            if pl["server_order"]:
                if page > 1000:
                    job.write("  已翻到 1000 页上限")
                    break
                params["page"] = page
            elif cursor:
                params["page"] = f"b{cursor}"   # 按 id 游标翻页，没有页数上限
            d = client.get("posts.json", params)
            if isinstance(d, dict) and d.get("_error"):
                body = d.get("_body", "")
                if "TagLimitError" in body:
                    raise CrawlError("服务器仍然提示标签太多：" + body[:200])
                raise CrawlError(f"搜索失败（{d['_error']}）：{body[:200]}")
            if not d:
                break
            for p in d:
                if p["id"] in seen:
                    continue
                seen.add(p["id"])
                if post_ok(p, cfg, pl, stats):
                    pool.append(p)
            scanned += len(d)
            cursor = min(p["id"] for p in d)
            job.set_progress(min(0.5, 0.5 * len(pool) / cap) if cap else None,
                             f"收集中：查询 {qi + 1}/{len(pl['queries'])}，翻了 {scanned:,} 张，符合条件 {len(pool):,} 张")
            if page % 5 == 1:
                job.write(f"  已翻 {scanned:,} 张，符合条件 {len(pool):,} 张")
            if len(d) < PER_PAGE:
                break
            page += 1
    if stats:
        job.write("本地过滤掉：" + "，".join(f"{k} {v}" for k, v in sorted(stats.items(), key=lambda x: -x[1])))
    if not pool:
        tips = ["放宽最低分数 / 收藏 / 分辨率", "缩短黑名单", "检查日期范围"]
        if not client.auth:
            tips.append("部分图（如部分画师、敏感标签）只有登录 Gold 账号才能拿到")
        raise CrawlError("没有符合条件的图。可以试试：" + "；".join(tips))

    # ---- 阶段二：排序 ----
    if pl["random"]:
        random.shuffle(pool)
    elif pl["sort_field"]:
        pool.sort(key=lambda p: p.get(pl["sort_field"]) or 0, reverse=pl["desc"])
    job.write(f"收集完成：符合条件 {len(pool):,} 张" + (f"，下载前 {min(want, len(pool))} 张" if want else ""))

    # ---- 阶段三：下载（失败的自动用后面的补上） ----
    target = min(want, len(pool)) if want else len(pool)
    ok = fail = skip = 0
    lock = threading.Lock()
    idx = 0

    def one(p):
        nonlocal ok, fail, skip
        url = p.get("file_url") or p.get("large_file_url")
        ext = (p.get("file_ext") or url.rsplit(".", 1)[-1]).lower()
        d = os.path.join(out, subdir(p, cfg))
        os.makedirs(d, exist_ok=True)
        img = os.path.join(d, f"{p['id']}.{ext}")
        txt = os.path.join(d, f"{p['id']}.txt")
        if os.path.isfile(img) and os.path.getsize(img) > 0:
            res = "skip"
        else:
            res = "ok" if client.download(url, img) else "fail"
        if res != "fail" and cfg.get("save_txt", False) and not os.path.isfile(txt):
            with open(txt, "w", encoding="utf-8") as f:
                f.write(caption(p, cfg))
        with lock:
            if res == "ok":
                ok += 1
            elif res == "skip":
                skip += 1
            else:
                fail += 1
            done = ok + skip
            job.set_progress(0.5 + 0.5 * done / max(1, target), f"下载中：{done} / {target}（失败 {fail}）")
        return res

    with ThreadPoolExecutor(max(1, min(8, int(cfg.get("workers") or 4)))) as ex:
        while not job.stop_flag and ok + skip < target and idx < len(pool):
            need = target - ok - skip
            batch = pool[idx: idx + need]
            idx += len(batch)
            list(ex.map(one, batch))
            if ok + skip < target and idx < len(pool):
                job.write(f"  {target - ok - skip} 张没下成功，用后面的图补上…")
    if ok + skip and (cfg.get("folder") or "none") == "none":   # 分了子文件夹的不记：每个子文件夹的角色不一样
        try:
            import triggers
            triggers.record_crawl(out, split_tags(cfg.get("tags")))
        except Exception:
            pass
    if ok:
        _prep_alpha(job, out)
    job.write(f"完成：新下载 {ok} 张，已存在 {skip} 张，失败 {fail} 张。保存在 {out}")
    return {"ok": ok, "skip": skip, "fail": fail, "dir": out}


def _prep_alpha(job, out):
    """透明底自动改白底（训练前还会再查一次；这里先做好，标签编辑里看到的就是白底）。"""
    import subprocess
    import paths
    import settings
    import tasks
    from jobs import NO_WINDOW, decode_line
    py = paths.env_python()
    if not py:
        return
    try:
        r = subprocess.run([py] + tasks.prep_cmd([out]), cwd=paths.APP, capture_output=True,
                           env=settings.child_env(), timeout=1800, creationflags=NO_WINDOW)
        for line in decode_line(r.stdout).splitlines():
            if line.strip():
                job.write(line)
    except Exception as e:
        job.write(f"[透明底] 检查没完成（训练前会再查）：{e}")


def autocomplete(q, login="", api_key=""):
    """给前端输入框用的标签补全。"""
    class _J:
        stop_flag = False

        def write(self, *_):
            pass
    c = Client(_J(), login, api_key)
    c.interval = 0
    d = c.get("autocomplete.json", {"search[query]": q, "search[type]": "tag_query", "limit": 10}, retries=1, timeout=8)
    if not isinstance(d, list):
        return []
    return [{"name": x.get("value"), "count": x.get("post_count") or (x.get("tag") or {}).get("post_count", 0),
             "category": x.get("category")} for x in d if isinstance(x, dict) and x.get("value")]

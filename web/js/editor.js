/* 06 标签编辑
   - 从「标签反推」过来时，自动打开刚反推的那组训练集
   - 左：整个训练集的标签统计（点击 = 只看有这个标签的，再点 = 只看没有的；可以叠加；悬停可收藏 / 整体改名 / 删除）
   - 中：缩略图（单击查看，Ctrl 多选，Shift 连选），批量操作作用于「已选」或「当前筛选结果」
   - 右：单张 = 大图 + 标签块（拖动排序、× 删除、双击改字、输入自动补全）/ 纯文本视图；
         多选 = 所选图片的合并标签（实心 = 都有，虚线 = 部分有：点一下补到全部所选，× = 从所选里删掉）
   - 标签比较和后端一致：大小写、下划线 / 空格、转义括号不同的写法算同一个标签；
     「标签段 + 自然语言段」的混合标签，批量增删只动标签段，句子原样保留
   - 每张图单独排队保存（快速切图不会丢）；所有批量操作执行前自动备份，可撤销到任意一步 */
(function () {
  const { $, $$, esc, h, api } = App;
  const S = { dir: '', items: [], stats: [], statMap: new Map(), sel: new Set(), cur: -1, filter: 'all', search: '', conds: [], sub: '', only: null,
    view: 'chips', last: -1, trig: null, dictMissing: false, tagQ: '', vis: [], style: ' ' };
  const ZH = {};              // 标签（归一化）→ {zh, cat}；字典里查过、没有 → null
  const TR = {};              // 整条标签原文 → 自然语言译文
  const saving = new Map();   // 图片路径 → {it, t}：每张图单独排队保存
  const CAT = { 0: 'gen', 1: 'art', 3: 'copy', 4: 'char', 5: 'meta' };
  const CAT_NAME = { 1: '画师', 3: '作品', 4: '角色', 5: '元标签' };

  const ext = () => App.params().caption_extension || '.txt';
  // 不能把 App.draft 里的对象存成常量：读取存档会把整个 App.draft 换掉，存下来的引用就失效了（「显示中文」勾了没反应就是这个原因）
  const tr = () => (App.draft.editor_tr = App.draft.editor_tr || { on: false, engine: 'qwen3vl4b' });
  const pins = () => (App.draft.editor_pins = App.draft.editor_pins || []);

  // ---------------- 标签规则（和后端 app/captionutil.py 保持一致） ----------------
  const CJK_G = /[぀-ヿ㐀-鿿가-힯]/g, CJK = /[぀-ヿ㐀-鿿가-힯]/;
  const norm = t => {
    t = String(t || '').replace(/﻿/g, '').trim().toLowerCase().replace(/\\\(/g, '(').replace(/\\\)/g, ')');
    if (t.length > 3) t = t.replace(/_/g, ' ');
    return t.replace(/\s+/g, ' ');
  };
  const splitInput = s => String(s || '').split(/[,，、\n]/).map(x => x.trim()).filter(Boolean);
  const words = s => s.trim().split(/\s+/).filter(Boolean);
  const isProse = seg => {
    const s = String(seg || '').trim();
    if (!s || !(/[A-Za-z]{2}/.test(s) || CJK.test(s))) return false;
    const w = words(s);
    if (w.length >= 7 || /[.。]$/.test(s)) return true;
    if (w.length >= 3 && /^[A-Z]/.test(s) && !s.includes('(')) return true;
    return (s.match(CJK_G) || []).length >= 12;
  };
  function parse(cap) {
    cap = String(cap || '').replace(/﻿/g, '');
    const tags = []; let headEnd = 0, m;
    const re = /[^,\n]+/g;
    while ((m = re.exec(cap))) {
      const seg = m[0];
      if (!seg.trim()) continue;
      let start = m.index + (seg.length - seg.trimStart().length);
      let k = /\.\s+(?=[A-Z])/.exec(seg);
      if (k && words(seg.slice(k.index + k[0].length)).length < 2) k = null;
      if (k || isProse(seg)) {
        const before = k ? seg.slice(0, k.index) : '';
        if (k && before.trim() && !isProse(before)) { tags.push(before.trim()); headEnd = m.index + k.index; start = m.index + k.index + k[0].length; }
        return { tags, text: cap.slice(start).trim(), sep: tags.length ? cap.slice(headEnd, start) : '' };
      }
      tags.push(seg.trim()); headEnd = m.index + seg.length;
    }
    return { tags, text: '', sep: '' };
  }
  const build = (tags, text, sep) => {
    tags = tags.map(t => t.trim()).filter(Boolean); text = (text || '').trim();
    return tags.length && text ? tags.join(', ') + (sep || '. ') + text : (text || tags.join(', '));
  };
  // 每张图的拆分结果缓存起来，标签内容变了才重算
  const pp = it => {
    if (it._pc !== it.caption) { const p = parse(it.caption); p.keys = new Set(p.tags.map(norm)); it._p = p; it._pc = it.caption; }
    return it._p;
  };
  const zhOf = t => { const z = ZH[norm(t)]; return z && z.zh ? z.zh : ''; };
  const catOf = t => { const z = ZH[norm(t)]; return z ? 'c-' + (CAT[z.cat] || 'gen') : ''; };
  const inStyle = t => (S.style === ' ' && t.length > 3 ? t.replace(/_/g, ' ') : t);   // 字典里的写法换成训练集习惯的写法

  // ---------------- 自动补全：训练集里已有的标签 + 中英字典（中文也能搜） ----------------
  function AC(input, { onEnter, onPick, exclude }) {
    const wrap = input.parentElement;
    let box = null, items = [], k = -1, seq = 0, timer = null;
    const close = () => { if (box) box.remove(); box = null; items = []; k = -1; };
    const lastPart = () => input.value.split(/[,，、]/).pop().trim();
    const show = list => {
      close();
      if (!list.length || document.activeElement !== input) return;
      items = list;
      box = h(`<div class="ac">${list.map((x, i) => `<div data-i="${i}"><span>${esc(x.t)}${x.zh ? `<em>${esc(x.zh)}</em>` : ''}</span><span class="muted">${x.n != null ? x.n : '字典'}</span></div>`).join('')}</div>`);
      box.onmousedown = e => { const d = e.target.closest('[data-i]'); if (!d) return; e.preventDefault(); pick(+d.dataset.i); };
      wrap.append(box);
    };
    const pick = i => {
      const parts = input.value.split(/[,，、]/); parts.pop(); parts.push(items[i].t);
      const v = parts.map(x => x.trim()).filter(Boolean).join(', ');
      close();
      if (onPick) onPick(v); else { input.value = v + ', '; input.dispatchEvent(new Event('input')); }
    };
    const query = () => {
      const q = lastPart(); clearTimeout(timer);
      if (!q) { close(); return; }
      const nq = norm(q), ex = exclude ? exclude() : new Set();
      const local = S.stats.filter(s => !ex.has(s.k) && (s.k.includes(nq) || zhOf(s.raw).includes(q)))
        .sort((a, b) => (b.k.startsWith(nq) - a.k.startsWith(nq)) || b.n - a.n).slice(0, 10)
        .map(s => ({ t: s.raw, n: s.n, zh: zhOf(s.raw) }));
      show(local);
      if (S.dictMissing) return;
      const my = ++seq;
      timer = setTimeout(async () => {
        const r = await api('dict.search', { q, limit: 8 }).catch(() => []);
        if (my !== seq) return;
        const have = new Set(local.map(x => norm(x.t)));
        const more = (r || []).filter(x => !have.has(norm(x.name)) && !ex.has(norm(x.name))).map(x => ({ t: inStyle(x.name), n: null, zh: x.zh }));
        if (more.length) show(local.concat(more).slice(0, 14));
      }, 220);
    };
    input.addEventListener('keydown', e => {
      if (e.isComposing || e.keyCode === 229) return;   // 中文输入法选词时的回车 / 方向键不算
      const ds = box ? $$('div[data-i]', box) : [];
      if ((e.key === 'ArrowDown' || e.key === 'ArrowUp') && box) {
        e.preventDefault(); k = (k + (e.key === 'ArrowDown' ? 1 : -1) + ds.length) % ds.length;
        ds.forEach((d, i) => d.classList.toggle('on', i === k)); ds[k].scrollIntoView({ block: 'nearest' });
      } else if (e.key === 'Tab' && box) { e.preventDefault(); pick(k < 0 ? 0 : k); }
      else if (e.key === 'Enter') { e.preventDefault(); if (box && k >= 0) pick(k); else { close(); onEnter(input.value); } }
      else if (e.key === 'Escape' && box) { e.stopPropagation(); close(); }
    });
    input.addEventListener('input', query);
    input.addEventListener('blur', () => setTimeout(close, 150));
    return { close };
  }

  const P = App.pages.editor = {
    init() {
      $('#ed-dir').onchange = e => { const v = e.target.value.trim(); if (v && v !== S.dir) P.open(v); };   // 失焦时浏览器会再触发一次 change
      $('#ed-browse').onclick = async () => { const p = await App.pick({ title: '选择训练集', start: S.dir }); if (p) P.open(p); };
      $('#ed-reload').onclick = () => P.open(S.dir, true);
      $('#ed-search').oninput = e => { S.search = e.target.value.trim().toLowerCase(); clearTimeout(S.srTimer); S.srTimer = setTimeout(() => P.renderThumbs(), 200); };
      $('#ed-sub').onchange = e => { S.sub = e.target.value; P.renderThumbs(); };
      $$('#ed-filter button').forEach(b => b.onclick = () => { S.filter = b.dataset.v; $$('#ed-filter button').forEach(x => x.classList.toggle('on', x === b)); P.renderThumbs(); });
      $('#ed-trigger').onclick = async () => {
        await P.flush();
        const one = S.sel.size === 1 ? S.items[[...S.sel][0]] : null;
        const cur = S.items[S.cur];   // 右边正在看的那张：弹窗里可以一键换成它当典型图片
        App.triggerDialog(S.dir, { typical: one ? one.path : null, cur: cur ? cur.path : null });
      };
      $('#ed-undo').onclick = () => P.undoDialog();
      $('#ed-lint').onclick = () => P.lint();
      $('#ed-selall').onclick = () => { S.vis.forEach(i => S.sel.add(i)); P.selChanged(); };
      $('#ed-selnone').onclick = () => { S.sel.clear(); P.selChanged(); };
      $('#ed-size').onchange = e => { $('#ed-thumbs').style.gridTemplateColumns = `repeat(auto-fill, minmax(${e.target.value}px, 1fr))`; };
      $('#ed-tag-sort').onchange = () => P.renderTags();
      $('#ed-tag-q').oninput = e => { S.tagQ = e.target.value.trim(); clearTimeout(S.tqTimer); S.tqTimer = setTimeout(P.renderTags, 150); };

      // 批量栏
      $$('#ed-batch [data-op]').forEach(b => b.onclick = () => P.batchBar(b.dataset.op));
      $('#ed-tidy').onchange = e => { const v = e.target.value; e.target.value = ''; if (v) P.tidy(v); };
      AC($('#ed-b-tags'), { onEnter: () => P.batchBar('add-back') });
      $('#ed-b-tags').addEventListener('input', () => P.hitPreview());
      $('#ed-b-find').onkeydown = $('#ed-b-rep').onkeydown = e => { if (e.key === 'Enter' && !e.isComposing) P.batchBar('find_replace'); };
      document.addEventListener('mousedown', e => { const fr = $('#ed-fr'); if (fr.open && !fr.contains(e.target)) fr.open = false; });

      // 标签统计：事件委托（几千行不用每行绑一次）
      $('#ed-tags').onclick = e => {
        const row = e.target.closest('[data-k]'); if (!row) return;
        const s = S.statMap.get(row.dataset.k); if (!s) return;
        const a = (e.target.closest('[data-a]') || {}).dataset;
        if (a && a.a === 'pin') return P.togglePin(s.raw);
        if (a && a.a === 'ren') return P.renameTag(s.raw);
        if (a && a.a === 'del') return P.deleteTag(s.raw);
        P.cycleCond(s);
      };
      $('#ed-conds').onclick = e => {
        if (e.target.closest('[data-clear]')) { S.conds = []; S.only = null; }
        else {
          const c = e.target.closest('.cond'); if (!c || !e.target.closest('button')) return;
          if (c.dataset.only) S.only = null; else S.conds.splice(+c.dataset.i, 1);
        }
        P.renderConds(); P.renderTags(); P.renderThumbs();
      };

      // 缩略图：事件委托
      $('#ed-thumbs').onclick = e => {
        const t = e.target.closest('.thumb'); if (!t) return;
        const i = +t.dataset.i;
        if (e.target.closest('.tck')) { S.sel.has(i) ? S.sel.delete(i) : S.sel.add(i); S.last = i; P.selChanged(); return; }
        P.clickThumb(i, e);
      };

      // 右侧
      $$('#ed-view button').forEach(b => b.onclick = () => { S.view = b.dataset.v; $$('#ed-view button').forEach(x => x.classList.toggle('on', x === b)); P.renderDetail(); });
      $('#ed-prev').onclick = () => P.step(-1); $('#ed-next').onclick = () => P.step(1);
      $('#ed-zh-on').onchange = e => { tr().on = e.target.checked; App.saveDraft(); P.zhRefresh(); };
      $('#ed-zh-engine').onchange = e => { tr().engine = e.target.value; App.saveDraft(); };
      $('#ed-zh-one').onclick = () => { const it = S.items[S.cur]; if (it) P.translate([it.caption]); };
      $('#ed-zh-all').onclick = () => P.translate(S.items.filter(it => pp(it).text).map(it => it.caption));
      $('#ed-zh-hint').onclick = e => { if (e.target.closest('[data-dl]')) P.dictDownload(); };
      $('#ed-remove').onclick = () => P.removeCur();
      $('#ed-text').oninput = () => {
        const it = S.items[S.cur]; if (!it || it.enc === 'bad') return;
        it.caption = $('#ed-text').value; P.queueSave(it); P.renderTextZh();
      };
      AC($('#ed-add'), {
        onEnter: v => P.addInput(v), onPick: v => P.addInput(v),
        exclude: () => S.sel.size >= 2 ? new Set() : new Set(S.items[S.cur] ? pp(S.items[S.cur]).keys : []),
      });
      P.bindChips();
      $('#ed-pins').onclick = e => {
        const p = e.target.closest('[data-t]'); if (!p) return;
        if (e.target.closest('button')) return P.togglePin(p.dataset.t);
        P.pinClick(p.dataset.t);
      };

      // 快捷键：焦点不在输入框时方向键直接切图；Alt+方向键在输入框里也能用
      document.addEventListener('keydown', e => {
        if (App.page !== 'editor' || $('.modal-mask') || $('.lightbox')) return;
        const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) || e.target.isContentEditable;
        if (e.altKey && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) { e.preventDefault(); P.step(e.key === 'ArrowLeft' ? -1 : 1); return; }
        if (typing || e.altKey) return;
        if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); P.step(e.key === 'ArrowLeft' ? -1 : 1); }
        else if (e.key === 'Delete' && S.sel.size < 2) { e.preventDefault(); P.removeCur(); }   // 多选时当前图没显示出来，不删
        else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'a') { e.preventDefault(); S.vis.forEach(i => S.sel.add(i)); P.selChanged(); }
        else if (e.key === 'Escape' && S.sel.size) { S.sel.clear(); P.selChanged(); }
      });
      // 关窗口 / 刷新时还没来得及保存的：用 sendBeacon 发出去（普通请求会被浏览器取消）
      // pagehide 比 beforeunload 可靠（关标签页 / 关窗口时 beforeunload 不一定触发）；两个都挂，发过的从队列里拿掉，不会重复保存
      const beacon = () => {
        saving.forEach(s => {
          clearTimeout(s.t);
          navigator.sendBeacon('/api/editor.save', new Blob([JSON.stringify({ dir: S.dir, path: s.it.path, text: s.it.caption || '', base: s.it.base, ext: ext() })], { type: 'text/plain' }));
        });
        saving.clear();
      };
      window.addEventListener('pagehide', beacon);
      window.addEventListener('beforeunload', beacon);
      document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'hidden') P.flush(); });   // 切到别的窗口时顺手存掉
    },
    show() {
      $('#ed-zh-on').checked = !!tr().on; $('#ed-zh-engine').value = tr().engine;   // 读取存档后草稿换了，跟着同步
      const want = App.draft.editor_dir || App.params().dataset_dir || '';
      if (want && want !== S.dir) P.open(want);
      else if (!want) $('#ed-thumbs').innerHTML = '<div class="hint" style="padding:10px">选择一个训练集文件夹开始编辑；反推完点「去编辑这组标签」会自动打开。</div>';
      else { P.renderTags(); P.renderDetail(); }
    },
    hide() { P.flush(); },

    async open(dir, keep) {
      if (!dir) return;
      await P.flush();
      let d;
      try { d = await api('editor.load', { dir, ext: ext() }); } catch (e) { App.toast(e.message, 'bad'); return; }
      const same = keep && dir === S.dir;
      const curPath = same && S.items[S.cur] ? S.items[S.cur].path : null;
      const selPaths = same ? new Set([...S.sel].map(i => S.items[i] && S.items[i].path)) : null;
      S.dir = dir; S.trig = d.trigger;
      S.items = d.items.map(it => Object.assign(it, { base: it.caption }));
      S.sel = new Set();
      if (selPaths) S.items.forEach((it, i) => { if (selPaths.has(it.path)) S.sel.add(i); });
      if (!same) { S.conds = []; S.only = null; S.sub = ''; S.search = ''; $('#ed-search').value = ''; }
      S.cur = curPath ? Math.max(0, S.items.findIndex(i => i.path === curPath)) : (S.items.length ? 0 : -1);
      $('#ed-dir').value = dir;
      App.draft.editor_dir = dir; App.saveDraft(); App.syncDataset(dir, 'editor');
      P.fillSubs(); P.renderConds(); P.recount(); P.renderThumbs(same); P.renderDetail();
      $('#ed-undo').disabled = !d.undo;
      const bad = S.items.filter(i => i.enc === 'bad').length;
      if (bad && !same) App.toast(`${bad} 个标签文件不是 UTF-8 编码：编辑和批量操作会跳过它们，免得写成乱码`, 'bad', '去体检转换', () => P.lint(), 12000);
    },
    fillSubs() {
      const cnt = new Map();
      S.items.forEach(it => { const p = it.rel.split(/[\\/]/); if (p.length > 1) cnt.set(p[0], (cnt.get(p[0]) || 0) + 1); });
      const el = $('#ed-sub');
      el.hidden = !cnt.size;
      if (S.sub && !cnt.has(S.sub)) S.sub = '';
      el.innerHTML = `<option value="">全部文件夹</option>` + [...cnt].map(([k, n]) => `<option value="${esc(k)}">${esc(k)}（${n}）</option>`).join('');
      el.value = S.sub;
    },

    // ---------------- 标签统计 ----------------
    recountSoon() { clearTimeout(S.rcTimer); S.rcTimer = setTimeout(P.recount, 400); },   // 连续改标签时，标签统计最后统一刷新
    recount() {
      const m = new Map(); let sp = 0, us = 0;
      S.items.forEach(it => {
        const seen = new Set();
        pp(it).tags.forEach(t => {
          if (t.includes('\ufffd')) return;   // 编码不对的文件里读坏的标签不计入统计
          const k = norm(t);
          let e = m.get(k); if (!e) m.set(k, e = { k, n: 0, forms: new Map() });
          e.forms.set(t, (e.forms.get(t) || 0) + 1);
          if (!seen.has(k)) { seen.add(k); e.n++; }
          if (t.includes(' ')) sp++; else if (t.length > 3 && t.includes('_')) us++;
        });
      });
      S.style = us > sp ? '_' : ' ';
      S.stats = [...m.values()].map(e => { let raw = '', bn = -1; e.forms.forEach((n, r) => { if (n > bn) { raw = r; bn = n; } }); return { k: e.k, raw, n: e.n, forms: e.forms.size }; });
      S.statMap = new Map(S.stats.map(s => [s.k, s]));
      P.renderTags();
      P.zhRefresh();
    },
    renderTags() {
      const el = $('#ed-tags'), q = S.tagQ, nq = norm(q);
      let arr = S.stats;
      if (q) arr = arr.filter(s => s.k.includes(nq) || zhOf(s.raw).includes(q));
      arr = [...arr].sort($('#ed-tag-sort').value === 'alpha' ? (a, b) => a.k.localeCompare(b.k) : (a, b) => b.n - a.n || a.k.localeCompare(b.k));
      const cond = new Map(S.conds.map(c => [c.k, c.neg ? 'neg' : 'pos'])), pinned = new Set(pins().map(norm)), zhOn = tr().on;
      const LIM = 1500;
      el.innerHTML = arr.slice(0, LIM).map(s => {
        const z = ZH[s.k], st = cond.get(s.k) || '';
        const tip = s.raw + (z && z.zh ? ' · ' + z.zh : '') + (z && CAT_NAME[z.cat] ? ' · ' + CAT_NAME[z.cat] : '') + (s.forms > 1 ? ` · 有 ${s.forms} 种写法（标签体检可以统一）` : '');
        return `<div class="${st}" data-k="${esc(s.k)}" title="${esc(tip)}"><i class="cdot ${z ? 'c-' + (CAT[z.cat] || 'gen') : ''}"></i>`
          + `<span class="tn">${st === 'pos' ? '<b class="cmark">有</b>' : st === 'neg' ? '<b class="cmark">没有</b>' : ''}${esc(s.raw)}${s.forms > 1 ? '<em class="var">*</em>' : ''}${zhOn && z && z.zh ? `<span class="zh">${esc(z.zh)}</span>` : ''}</span>`
          + `<span class="ta"><button class="icon-btn" data-a="pin" title="${pinned.has(s.k) ? '移出常用标签' : '加到常用标签（右边一点就能加上 / 去掉）'}">${pinned.has(s.k) ? '★' : '☆'}</button>`
          + `<button class="icon-btn" data-a="ren" title="整体改名（全部图片）">✎</button><button class="icon-btn" data-a="del" title="从当前范围删除（可撤销）">✕</button></span><span class="tc">${s.n}</span></div>`;
      }).join('');
      if (arr.length > LIM) el.insertAdjacentHTML('beforeend', `<div class="muted" style="padding:8px 10px">还有 ${arr.length - LIM} 个标签没列出来，用上面的搜索框找</div>`);
      if (!arr.length) el.innerHTML = `<div class="muted" style="padding:10px">${q ? '没有匹配的标签' : '还没有标签'}</div>`;
    },
    cycleCond(s) {
      const i = S.conds.findIndex(c => c.k === s.k);
      if (i < 0) S.conds.push({ k: s.k, raw: s.raw, neg: false });
      else if (!S.conds[i].neg) S.conds[i].neg = true;
      else S.conds.splice(i, 1);
      P.renderConds(); P.renderTags(); P.renderThumbs();
    },
    renderConds() {
      const el = $('#ed-conds'), parts = [];
      S.conds.forEach((c, i) => parts.push(`<span class="cond ${c.neg ? 'neg' : ''}" data-i="${i}">${c.neg ? '没有' : '有'} <b>${esc(c.raw)}</b><button title="去掉这个条件">×</button></span>`));
      if (S.only) parts.push(`<span class="cond only" data-only="1">${esc(S.only.label)}（${S.only.set.size} 张）<button title="去掉">×</button></span>`);
      el.hidden = !parts.length;
      el.innerHTML = parts.length ? `<span class="muted">筛选：</span>${parts.join('')}<button class="btn btn-sm" data-clear>清除筛选</button><span class="hint" style="margin:0">批量操作只作用于筛选出来的图</span>` : '';
    },

    // ---------------- 中文 ----------------
    // 字典中文 + 分类：不管开没开「显示中文」都查（分类颜色要用）；查不到的记下来，字典没下载时不记（下载完还要再查）
    async zhRefresh() {
      if (!S.zhBusy) {
        const need = S.stats.filter(s => !(s.k in ZH)).map(s => s.raw);
        if (need.length) {
          S.zhBusy = true;
          try {
            for (let i = 0; i < need.length; i += 3000) {
              const part = need.slice(i, i + 3000);
              const r = await api('dict.zh', { tags: part });
              S.dictMissing = !!r.missing;
              if (r.missing) break;
              part.forEach(t => { ZH[norm(t)] = r.items[t] || null; });
            }
          } catch (e) { /* 后台暂时没响应：下次再查，不当成「字典里没有」 */ }
          S.zhBusy = false;
        }
      }
      if (tr().on) {
        const caps = [...new Set(S.items.filter(it => pp(it).text).map(it => (it.caption || '').trim()).filter(c => !(c in TR)))];
        if (caps.length) Object.assign(TR, await api('translate.get', { texts: caps }).catch(() => ({})));
      }
      P.renderTags(); P.redrawDetail(); P.zhHint();
    },
    zhHint() {
      const el = $('#ed-zh-hint'), it = S.items[S.cur];
      if (!tr().on) { el.innerHTML = ''; return; }
      if (S.dictMissing) { el.innerHTML = '没有中英字典 <button class="btn btn-sm" data-dl>下载（约 17 MB）</button>'; return; }
      const miss = it && S.sel.size < 2 ? pp(it).tags.filter(t => ZH[norm(t)] === null).length : 0;
      el.textContent = miss ? `${miss} 个标签字典里没有` : '';
      el.title = miss ? '字典只收录 Danbooru 上图数 ≥50 的标签；触发词、自己写的标签、拼错的标签都查不到' : '';
    },
    async dictDownload() {
      let id;
      try { id = await api('dict.download'); } catch (e) { App.toast(e.message, 'bad'); return; }
      App.toast('正在下载中英字典…');
      const poll = async () => {
        const j = await api('jobs.get', { id }).catch(() => null);
        if (j && j.status === 'running') { setTimeout(poll, 1500); return; }
        if (j && j.status === 'done') {
          Object.keys(ZH).forEach(k => delete ZH[k]); S.dictMissing = false;
          App.toast('字典已下载，标签能显示中文了', 'ok'); P.zhRefresh();
        } else App.toast('字典下载失败：' + ((j && j.error) || ''), 'bad');
      };
      poll();
    },
    redrawDetail() { if (S.view === 'chips' || S.sel.size >= 2) P.renderDetail(); else { P.renderZh(); P.renderTextZh(); } },   // 文本视图正在打字时不能整个重画（会把光标挪走）
    renderZh() {
      const box = $('#ed-zh'), it = S.items[S.cur];
      box.hidden = !tr().on || !it || S.sel.size >= 2 || !pp(it).text;
      if (box.hidden) return;
      const zh = TR[(it.caption || '').trim()];
      $('#ed-zh-text').textContent = zh || '这张图的自然语言标签还没翻译。';
      $('#ed-zh-text').classList.toggle('muted', !zh);
    },
    renderTextZh() {
      const box = $('#ed-text-zh'), it = S.items[S.cur];
      box.hidden = !tr().on || !it || S.view !== 'text';
      if (box.hidden) return;
      const list = pp(it).tags.map(t => { const z = zhOf(t); return z ? `<span>${esc(t)}<em>${esc(z)}</em></span>` : ''; }).filter(Boolean);
      box.innerHTML = list.length ? list.join('') : '<span class="muted">没有能查到中文的标签</span>';
    },
    async translate(texts) {
      texts = [...new Set(texts.map(c => (c || '').trim()).filter(c => c && !TR[c]))];
      if (!texts.length) { App.toast('都已经翻译过了'); return; }
      const engine = tr().engine;
      const local = engine !== 'api';
      if (local && App.queue && (App.queue.tasks || []).some(t => t.status === 'running') &&
          !(await App.confirm('正在训练', '本地翻译要把 Qwen3-VL 加载到显卡上，和训练抢显存，可能让训练爆显存。<br>建议改用 API，或者等训练完再翻。', '仍然翻译', true))) return;
      const prog = $('#ed-zh-prog');
      let id;
      try {
        id = await api('translate.start', { texts, engine: local ? 'local' : 'api', model: local ? engine : '', api: (App.draft.tag || {}).api });
      } catch (e) { App.toast(e.message, 'bad'); return; }
      $('#ed-zh-one').disabled = $('#ed-zh-all').disabled = true;
      const poll = async () => {
        const j = await api('jobs.get', { id, log: 5 }).catch(() => null);
        if (!j) { setTimeout(poll, 3000); return; }
        prog.textContent = j.status === 'running' ? (j.progress_text || '准备中…') : '';
        const got = await api('translate.get', { texts }).catch(() => ({}));
        if (Object.keys(got).some(k => !TR[k])) { Object.assign(TR, got); P.renderZh(); }
        if (j.status === 'running') { setTimeout(poll, 2000); return; }
        $('#ed-zh-one').disabled = $('#ed-zh-all').disabled = false;
        if (j.status === 'failed') App.toast('翻译失败：' + (j.error || ''), 'bad');
        else App.toast(`翻译完成：${Object.keys(got).length} / ${texts.length} 条`, 'ok');
      };
      poll();
    },

    // ---------------- 缩略图 ----------------
    visible() {
      const out = [], q = S.search, sub = S.sub;
      S.items.forEach((it, i) => {
        const cap = it.caption || '';
        if (S.filter === 'nocap' && cap.trim()) return;
        if (S.filter === 'sel' && !S.sel.has(i)) return;
        if (sub && it.rel.split(/[\\/]/)[0] !== sub) return;
        if (S.only && !S.only.set.has(it.path)) return;
        if (S.conds.length) { const ks = pp(it).keys; if (!S.conds.every(c => ks.has(c.k) !== c.neg)) return; }
        if (q && !cap.toLowerCase().includes(q) && !it.rel.toLowerCase().includes(q)) return;
        out.push(i);
      });
      return out;
    },
    scope() {
      if (S.sel.size) { const idx = [...S.sel]; return { files: idx.map(i => S.items[i].path), idx, label: `已选 ${S.sel.size} 张` }; }
      const v = S.vis;
      if (v.length !== S.items.length) return { files: v.map(i => S.items[i].path), idx: v, label: `当前筛选 ${v.length} 张` };
      return { files: null, idx: v, label: `全部 ${S.items.length} 张` };
    },
    allScope() { return { files: null, idx: S.items.map((_, i) => i), label: `全部 ${S.items.length} 张` }; },
    // 只在图片列表本身变化（打开训练集、筛选、搜索、批量修改）时整体重建；选中 / 当前图 / 单张改标签都是局部更新
    renderThumbs(keepScroll) {
      const el = $('#ed-thumbs'), box = el.parentElement, top = keepScroll ? box.scrollTop : 0;
      const vis = S.vis = P.visible();
      const gen = S.gen = (S.gen || 0) + 1;
      S.els = new Map(); el.innerHTML = '';
      if (!keepScroll) box.scrollTop = 0;
      if (!vis.length) {
        el.innerHTML = `<div class="muted" style="padding:10px">${S.items.length ? '没有符合条件的图片' : (S.dir ? '这个文件夹里没有图片' : '')}</div>`;
        P.marks(); return;
      }
      let k = 0;
      const chunk = () => {   // 分批加进页面：上千张图时不会卡住整个界面
        if (gen !== S.gen) return;
        const frag = document.createDocumentFragment();
        for (const end = Math.min(vis.length, k + 240); k < end; k++) frag.append(P.thumbEl(vis[k]));
        el.append(frag);
        if (k < vis.length) requestAnimationFrame(chunk);
        else if (keepScroll) box.scrollTop = top;
        P.marks();
      };
      chunk();
    },
    thumbEl(i) {
      const it = S.items[i];
      const t = h(`<div class="thumb" data-i="${i}"><img loading="lazy" decoding="async" alt=""><span class="tno"></span><span class="tck" title="勾选（批量编辑）">✓</span><div class="tcap"></div></div>`);
      P.loadThumb(t.querySelector('img'), it.path);
      S.els.set(i, t); P.updateThumb(i);
      return t;
    },
    // 缩略图还没生成好时后台回 503：过一会儿再取，多次取不到才退回原图（以前是后台每张最多等 6 秒，会把别的请求堵住）
    loadThumb(img, path, n = 0) {
      img.onerror = () => {
        if (n < 40) setTimeout(() => { if (img.isConnected) P.loadThumb(img, path, n + 1); }, Math.min(700 + n * 300, 3000));   // 大训练集的缩略图可能要生成一两分钟
        else { img.onerror = null; img.src = '/file?path=' + encodeURIComponent(path); }
      };
      img.src = `/thumb?path=${encodeURIComponent(path)}${n ? '&r=' + n : ''}`;
    },
    updateThumb(i) {
      const t = S.els && S.els.get(i); if (!t) return;
      const it = S.items[i], cap = it.caption || '';
      t.querySelector('.tcap').textContent = cap.slice(0, 120);
      const no = t.querySelector('.tno');
      no.textContent = it.enc === 'bad' ? '编码不对' : '无标签';
      no.hidden = it.enc !== 'bad' && !!cap.trim();
    },
    marks() {
      const one = S.sel.size < 2;
      if (S.els) S.els.forEach((t, i) => { t.classList.toggle('sel', S.sel.has(i)); t.classList.toggle('cur', one && i === S.cur); });
      const vis = S.vis, noCap = S.items.filter(i => !(i.caption || '').trim()).length, bad = S.items.filter(i => i.enc === 'bad').length;
      $('#ed-count').textContent = `图片 ${vis.length}${vis.length !== S.items.length ? ' / ' + S.items.length : ''}${noCap ? ` · ${noCap} 张无标签` : ''}${bad ? ` · ${bad} 张编码不对` : ''}`;
      $('#ed-scope').textContent = '作用于：' + P.scope().label;
      P.hitPreview();
    },
    selChanged() { if (S.filter === 'sel') P.renderThumbs(true); else P.marks(); P.renderDetail(); },
    clickThumb(i, e) {
      if (e.shiftKey && S.last >= 0) {
        const a = S.vis.indexOf(S.last), b = S.vis.indexOf(i);
        if (a >= 0 && b >= 0) S.vis.slice(Math.min(a, b), Math.max(a, b) + 1).forEach(x => S.sel.add(x));
      } else if (e.ctrlKey || e.metaKey) {
        if (!S.sel.size && S.cur >= 0 && S.cur !== i) S.sel.add(S.cur);   // 先单击看了一张再 Ctrl 点别的：正在看的那张也算进去
        S.sel.has(i) ? S.sel.delete(i) : S.sel.add(i);
        S.cur = i;
      } else {
        S.sel.clear(); S.cur = i;   // 普通单击 = 只看这一张（和资源管理器一样）；要多选用 Ctrl / Shift 或右上角的勾
      }
      S.last = i; P.selChanged();
    },
    step(d) {
      const vis = S.vis; if (!vis.length) return;
      let k = vis.indexOf(S.cur); k = k < 0 ? 0 : (k + d + vis.length) % vis.length;
      S.cur = vis[k]; S.sel.clear(); P.marks(); P.renderDetail();
      const el = S.els && S.els.get(S.cur); if (el) el.scrollIntoView({ block: 'nearest' });
    },

    // ---------------- 右侧：单张 / 多选 ----------------
    renderDetail() {
      const multi = S.sel.size >= 2;
      $('#ed-detail').classList.toggle('multi', multi);
      P.renderPins(); P.zhHint();
      if (multi) return P.renderMulti();
      const it = S.items[S.cur];
      const img = $('#ed-img');
      $('#ed-multi').hidden = true; $('#ed-view').hidden = false;
      $('#ed-add').placeholder = '输入标签回车添加（有补全，中文也能搜），可逗号分隔多个';
      if (!it) {
        img.hidden = true; $('#ed-cur-name').textContent = '未选择图片'; $('#ed-chips').innerHTML = '';
        $('#ed-warn').hidden = true; $('#ed-remove-row').hidden = true; $('#ed-add-wrap').hidden = true; $('#ed-text-wrap').hidden = true;
        P.renderZh(); return;
      }
      img.hidden = false; $('#ed-remove-row').hidden = false;
      const src = '/file?path=' + encodeURIComponent(it.path);
      if (img.dataset.src !== src) { img.dataset.src = src; img.src = src; }   // 只改标签时不重新加载大图
      img.onclick = () => App.lightbox(S.vis.map(i => ({ file: S.items[i].path, cap: S.items[i].caption })), Math.max(0, S.vis.indexOf(S.cur)));
      $('#ed-cur-name').textContent = it.rel;
      if (!saving.has(it.path)) $('#ed-saved').textContent = '';
      const bad = it.enc === 'bad';
      $('#ed-warn').hidden = !bad;
      $('#ed-warn').innerHTML = bad ? '这个标签文件不是 UTF-8 编码，直接改会变成乱码，所以现在只能看不能改。<button class="btn btn-sm" data-lint>标签体检 → 转成 UTF-8</button>' : '';
      if (bad) $('#ed-warn [data-lint]').onclick = () => P.lint();
      const chipsMode = S.view === 'chips';
      $('#ed-chips').hidden = !chipsMode; $('#ed-add-wrap').hidden = !chipsMode || bad; $('#ed-text-wrap').hidden = chipsMode;
      P.renderZh();
      if (!chipsMode) {
        const ta = $('#ed-text');
        if (document.activeElement !== ta || ta.value !== (it.caption || '')) ta.value = it.caption || '';
        ta.readOnly = bad; P.renderTextZh(); return;
      }
      const p = pp(it), zhOn = tr().on, trig = S.trig ? norm(S.trig.main) : '';
      const prm = App.params(), keep = prm.shuffle_caption ? (+prm.keep_tokens || 0) : 0;
      let html = '';
      p.tags.forEach((t, k) => {
        const isTrig = trig && norm(t) === trig;
        const z = zhOn && !isTrig ? zhOf(t) : '';   // 触发词是自己起的名字，碰巧和 D 站标签同名时显示的中文没意义
        html += `<span class="chip ${catOf(t)} ${isTrig ? 'ed-trig' : ''}" data-i="${k}" ${bad ? '' : 'draggable="true"'} title="${isTrig ? (S.trig.guessed ? '触发词（推测：几乎每张图的第一个标签都是它）· ' : '触发词 · ') : ''}拖动排序；双击修改"><span>${esc(t)}${z ? `<span class="zh">${esc(z)}</span>` : ''}</span>${bad ? '' : '<button title="删除">×</button>'}</span>`;
        if (keep && k === keep - 1 && k < p.tags.length - 1) html += `<span class="chip-keep" title="开了「打乱标签顺序」：前 ${keep} 个标签训练时固定，后面的每步随机打乱">固定 ┆ 打乱</span>`;
      });
      if (p.text) html += `<span class="chip sentence" data-text="1" title="自然语言段（双击到文本视图里改）"><span>${esc(p.text)}</span>${bad ? '' : '<button title="删除这段句子">×</button>'}</span>`;
      $('#ed-chips').innerHTML = html || '<span class="muted">这张图还没有标签</span>';
    },
    renderMulti() {
      const idx = [...S.sel].filter(i => S.items[i]), N = idx.length;
      $('#ed-img').hidden = true; $('#ed-warn').hidden = true; $('#ed-zh').hidden = true; $('#ed-text-wrap').hidden = true;
      $('#ed-remove-row').hidden = true; $('#ed-view').hidden = true; $('#ed-chips').hidden = false; $('#ed-add-wrap').hidden = false;
      $('#ed-cur-name').textContent = `已选 ${N} 张 · 批量编辑`;
      $('#ed-add').placeholder = `给所选的 ${N} 张添加标签（回车）`;
      const m = $('#ed-multi'); m.hidden = false;
      const bad = idx.filter(i => S.items[i].enc === 'bad').length;
      m.innerHTML = `<div class="ed-mstrip">${idx.slice(0, 16).map(i => `<img data-p="${esc(S.items[i].path)}" alt="">`).join('')}${N > 16 ? `<span>+${N - 16}</span>` : ''}</div>
        <div class="hint" style="margin:6px 0 0">实线 = 所选的都有；虚线 = 部分有，点一下补到全部所选；× = 从所选的 ${N} 张里删掉。${bad ? `<br><b>${bad} 张编码不对，会被跳过。</b>` : ''}</div>
        <div class="row" style="gap:6px;margin-top:6px"><button class="btn btn-sm" data-none>取消选择</button><button class="btn btn-sm" data-only>只看所选</button></div>`;
      $$('img[data-p]', m).forEach(img => P.loadThumb(img, img.dataset.p));
      m.querySelector('[data-none]').onclick = () => { S.sel.clear(); P.selChanged(); };
      m.querySelector('[data-only]').onclick = () => { S.filter = 'sel'; $$('#ed-filter button').forEach(x => x.classList.toggle('on', x.dataset.v === 'sel')); P.renderThumbs(); };
      const cnt = new Map();
      idx.forEach(i => { const seen = new Set(); pp(S.items[i]).tags.forEach(t => { const k = norm(t); if (seen.has(k)) return; seen.add(k); const e = cnt.get(k) || { n: 0, raw: t }; e.n++; cnt.set(k, e); }); });
      const arr = [...cnt].sort((a, b) => (b[1].n === N) - (a[1].n === N) || b[1].n - a[1].n || ((S.statMap.get(b[0]) || {}).n || 0) - ((S.statMap.get(a[0]) || {}).n || 0));
      const zhOn = tr().on, trig = S.trig ? norm(S.trig.main) : '';
      $('#ed-chips').innerHTML = arr.map(([k, e]) => {
        const raw = (S.statMap.get(k) || {}).raw || e.raw, isTrig = trig && k === trig, z = zhOn && !isTrig ? zhOf(raw) : '';
        return `<span class="chip ${catOf(raw)} ${isTrig ? 'ed-trig' : ''} ${e.n < N ? 'part' : ''}" data-raw="${esc(raw)}" title="${e.n < N ? `${e.n} / ${N} 张有，点一下补到全部所选` : `所选的 ${N} 张都有`}"><span>${esc(raw)}${z ? `<span class="zh">${esc(z)}</span>` : ''}</span>${e.n < N ? `<b class="pc">${e.n}/${N}</b>` : ''}<button title="从所选的 ${N} 张里删掉">×</button></span>`;
      }).join('') || '<span class="muted">所选图片都还没有标签</span>';
    },
    bindChips() {
      const el = $('#ed-chips');
      let from = -1;
      el.addEventListener('click', e => {
        const c = e.target.closest('.chip'); if (!c) return;
        const del = e.target.closest('button');
        if (S.sel.size >= 2) {
          if (del) P.run('remove', { tags: c.dataset.raw });
          else if (c.classList.contains('part')) P.run('add', { tags: c.dataset.raw, position: 'back' });
          return;
        }
        if (!del) return;
        if (c.dataset.text) P.editTags(() => ({ text: '' }));
        else { const k = +c.dataset.i; P.editTags(tags => { tags.splice(k, 1); }); }
      });
      el.addEventListener('dblclick', async e => {
        const c = e.target.closest('.chip'); if (!c || S.sel.size >= 2 || e.target.closest('button')) return;
        if (c.dataset.text) { $$('#ed-view button').find(b => b.dataset.v === 'text').click(); return; }
        const k = +c.dataset.i, it = S.items[S.cur]; if (!it || it.enc === 'bad') return;
        const old = pp(it).tags[k];
        const v = await App.prompt('修改标签', '新内容', old);
        if (v === null) return;
        P.editTags(tags => { tags.splice(k, 1, ...splitInput(v)); });
      });
      el.addEventListener('dragstart', e => { const c = e.target.closest('.chip[data-i]'); if (!c) return; from = +c.dataset.i; c.style.opacity = .4; e.dataTransfer.setData('text/plain', String(from)); });
      el.addEventListener('dragend', e => { const c = e.target.closest('.chip'); if (c) c.style.opacity = 1; });
      el.addEventListener('dragover', e => { const c = e.target.closest('.chip[data-i]'); if (!c || from < 0) return; e.preventDefault(); $$('.drag-over', el).forEach(x => x !== c && x.classList.remove('drag-over')); c.classList.add('drag-over'); });
      el.addEventListener('dragleave', e => { const c = e.target.closest('.chip'); if (c && !c.contains(e.relatedTarget)) c.classList.remove('drag-over'); });
      el.addEventListener('drop', e => {
        const c = e.target.closest('.chip[data-i]'); if (!c || from < 0) return;
        e.preventDefault();
        const to = +c.dataset.i, f = from; from = -1;
        if (f === to) { c.classList.remove('drag-over'); return; }
        P.editTags(tags => { const [m] = tags.splice(f, 1); tags.splice(f < to ? to - 1 : to, 0, m); });
      });
    },
    editTags(fn) {
      const it = S.items[S.cur]; if (!it) return;
      if (it.enc === 'bad') { App.toast('这个标签文件不是 UTF-8 编码，先用「标签体检」转换', 'bad'); return; }
      const p = pp(it), tags = p.tags.slice();
      const r = fn(tags) || {};
      P.setCur(build(tags, r.text !== undefined ? r.text : p.text, p.sep));
    },
    setCur(caption) {
      const it = S.items[S.cur]; if (!it) return;
      it.caption = caption; P.queueSave(it); P.renderDetail(); P.updateThumb(S.cur); P.marks(); P.recountSoon();
    },
    addInput(v) {
      const list = splitInput(v);
      $('#ed-add').value = '';
      if (!list.length) return;
      if (S.sel.size >= 2) { P.run('add', { tags: list.join(', '), position: 'back' }); return; }
      P.editTags(tags => { const have = new Set(tags.map(norm)); list.forEach(t => { if (!have.has(norm(t))) { tags.push(t); have.add(norm(t)); } }); });
    },

    // ---------------- 常用标签 ----------------
    togglePin(raw) {
      const list = pins(), k = norm(raw), i = list.findIndex(t => norm(t) === k);
      if (i >= 0) list.splice(i, 1); else list.push(raw);
      App.saveDraft(); P.renderPins(); P.renderTags();
    },
    renderPins() {
      const el = $('#ed-pins'), list = pins();
      const it = S.items[S.cur], multi = S.sel.size >= 2;
      if (!list.length || (!it && !multi)) { el.innerHTML = ''; el.hidden = true; return; }
      el.hidden = false;
      const idx = multi ? [...S.sel] : [];
      el.innerHTML = '<span class="muted">常用</span>' + list.map(t => {
        const k = norm(t);
        let st = '';
        if (multi) { const n = idx.filter(i => pp(S.items[i]).keys.has(k)).length; st = n === idx.length ? 'on' : n ? 'part' : ''; }
        else st = pp(it).keys.has(k) ? 'on' : '';
        return `<span class="pin ${st}" data-t="${esc(t)}" title="${st === 'on' ? '点一下去掉' : '点一下加上'}">${esc(t)}<button title="移出常用标签">×</button></span>`;
      }).join('');
    },
    pinClick(t) {
      const k = norm(t);
      if (S.sel.size >= 2) {
        const all = [...S.sel].every(i => pp(S.items[i]).keys.has(k));
        P.run(all ? 'remove' : 'add', all ? { tags: t } : { tags: t, position: 'back' });
        return;
      }
      P.editTags(tags => { const i = tags.findIndex(x => norm(x) === k); if (i >= 0) tags.splice(i, 1); else tags.push(t); });
    },

    // ---------------- 保存：每张图单独排队，同一张图的保存按顺序发 ----------------
    queueSave(it) {
      if (S.items[S.cur] === it) $('#ed-saved').textContent = '…';
      const old = saving.get(it.path); if (old) clearTimeout(old.t);
      saving.set(it.path, { it, t: setTimeout(() => P.doSave(it), 400) });
    },
    doSave(it) {
      const s = saving.get(it.path); if (s) { clearTimeout(s.t); saving.delete(it.path); }
      it._q = (it._q || Promise.resolve()).then(() => P._save(it));
      return it._q;
    },
    async _save(it) {
      const text = it.caption || '';
      try {
        const r = await api('editor.save', { dir: S.dir, path: it.path, text, base: it.base, ext: ext() });
        if (r && r.conflict) {
          it.caption = it.base = r.current;
          App.toast(`「${it.rel}」的标签刚被别的程序改过（比如反推还在写），已换成文件里的最新内容，你刚才的修改没有保存`, 'bad', null, null, 10000);
          const i = S.items.indexOf(it);
          if (i >= 0) { P.updateThumb(i); P.recountSoon(); if (i === S.cur) { $('#ed-text').blur(); P.renderDetail(); } }
          return;
        } else {
          it.base = text;
          if (S.items[S.cur] === it && !saving.has(it.path)) $('#ed-saved').textContent = '已保存';
        }
      } catch (e) {
        if (S.items[S.cur] === it) $('#ed-saved').textContent = '保存失败';
        App.toast(`保存「${it.rel}」失败：${e.message}`, 'bad', '重试', () => P.queueSave(it), 12000);
        return;
      }
      const i = S.items.indexOf(it);
      if (i >= 0) { P.updateThumb(i); P.recountSoon(); if (i === S.cur && S.view === 'chips') P.renderDetail(); }
    },
    flush() { return Promise.all([...saving.values()].map(s => P.doSave(s.it))); },

    // ---------------- 批量操作 ----------------
    busy(b) { $$('#ed-batch button, #ed-batch select, #ed-batch input').forEach(x => { x.disabled = b; }); $('#ed-batch').classList.toggle('busy', b); },
    async run(op, args, opt = {}) {
      if (!S.dir) return null;
      const sc = opt.scope || P.scope();
      if (sc.files && !sc.files.length) { App.toast('当前筛选一张图都没有', 'bad'); return null; }
      await P.flush();
      P.busy(true);
      let r;
      try { r = await api('editor.batch', { dir: S.dir, op, files: sc.files, args, ext: ext() }); }
      catch (e) { App.toast('批量修改失败：' + e.message, 'bad', null, null, 10000); return null; }
      finally { P.busy(false); }
      P.applyItems(r.items, r.undo);
      const warn = !r.changed && (r.skipped || []).length;
      App.toast(r.message, r.changed ? 'ok' : (warn ? 'bad' : ''), r.changed && r.undo_id ? '撤销' : null, () => P.undoTo(r.undo_id), r.changed ? 8000 : 6000);
      return r;
    },
    // 后台改了哪些文件就只更新哪些：选择、筛选、滚动位置都保留
    applyItems(items, undo) {
      const by = new Map(S.items.map(it => [it.path, it]));
      Object.entries(items || {}).forEach(([p, cap]) => { const it = by.get(p); if (it) { it.caption = it.base = cap; if (cap !== null) it.enc = 'ok'; } });
      if (undo !== undefined) $('#ed-undo').disabled = !undo;
      P.recount(); P.renderThumbs(true); P.renderDetail();
    },
    batchBar(op) {
      const tags = $('#ed-b-tags').value.trim();
      if (op === 'add-front' || op === 'add-back') {
        if (!splitInput(tags).length) return App.toast('先在输入框里填要添加的标签', 'bad');
        return P.run('add', { tags, position: op === 'add-front' ? 'front' : 'back' });
      }
      if (op === 'remove') {
        if (!splitInput(tags).length) return App.toast('先在输入框里填要删除的标签', 'bad');
        return P.run('remove', { tags });
      }
      if (op === 'find_replace') {
        const f = $('#ed-b-find').value;
        if (!f) return App.toast('先填要查找的内容', 'bad');
        return P.run('find_replace', { find: f, replace: $('#ed-b-rep').value, regex: $('#ed-b-re').checked });
      }
    },
    async tidy(v) {
      if (v === 'dedupe') return P.run('dedupe', {});
      if (v === 'sort') return P.run('sort', { keep: 1 });
      if (v === 'clear') {
        const sc = P.scope();
        if (!(await App.confirm('清空标签', `把 <b>${esc(sc.label)}</b> 的标签全部清空（标签文件保留为空）？<br><span class="muted">可以用「撤销…」恢复。</span>`, '清空', true))) return;
        return P.run('clear', {}, { scope: sc });
      }
    },
    // 输入框里的标签在当前范围里命中多少张：点按钮之前就知道会不会「没反应」
    hitPreview() {
      const el = $('#ed-b-hit'), ks = [...new Set(splitInput($('#ed-b-tags').value).map(norm))];
      if (!ks.length || !S.items.length) { el.textContent = ''; return; }
      const idx = P.scope().idx;
      let all = 0, some = 0;
      idx.forEach(i => { const keys = pp(S.items[i]).keys; const c = ks.filter(k => keys.has(k)).length; if (c === ks.length) all++; else if (c) some++; });
      el.textContent = `${idx.length} 张里：${all} 张全有${some ? ` · ${some} 张有一部分` : ''} · ${idx.length - all - some} 张都没有`;
    },
    async renameTag(t) {
      const v = await App.prompt('整体改名', `把全部图片里的「${t}」改成（大小写、下划线写法不同的也一起改）：`, t);
      if (v === null || !v.trim() || v.trim() === t) return;
      P.run('replace', { old: t, new: v.trim() }, { scope: P.allScope() });
    },
    deleteTag(t) { P.run('remove', { tags: t }); },
    async removeCur() {
      const k = S.cur, it = S.items[k]; if (!it) return;
      if (!(await App.confirm('移出训练集', `把 ${esc(it.rel)} 和它的标签移到训练集里的「_已移除」文件夹？训练时不会再用到它。`, '移出', true))) return;
      await P.flush();
      try { await api('editor.remove', { dir: S.dir, path: it.path, ext: ext() }); } catch (e) { App.toast('移出失败：' + e.message, 'bad'); return; }
      S.items.splice(k, 1);
      S.sel = new Set([...S.sel].filter(i => i !== k).map(i => (i > k ? i - 1 : i)));   // 后面的序号都往前挪了一位
      if (S.cur >= S.items.length) S.cur = S.items.length - 1;
      P.recount(); P.renderThumbs(true); P.renderDetail();
    },

    // ---------------- 撤销 ----------------
    async undoDialog() {
      if (!S.dir) return;
      await P.flush();
      let list;
      try { list = await api('editor.history', { dir: S.dir }); } catch (e) { App.toast(e.message, 'bad'); return; }
      if (!list.length) { App.toast('没有可以撤销的批量操作'); $('#ed-undo').disabled = true; return; }
      const body = h(`<div><div class="hint" style="margin:0 0 10px">撤销某一次，会把它之后的批量操作一起撤销（从新到旧依次还原）。单张图的手动修改不在这里。</div><div class="ed-hist"></div></div>`);
      const box = body.querySelector('.ed-hist');
      let m;
      list.forEach((x, i) => {
        const row = h(`<div><span class="muted">${esc(App.fmtTime(x.time))}</span><b>${esc(x.label)}</b><span class="muted">${x.count} 个文件</span><button class="btn btn-sm">${i === 0 ? '撤销这一次' : `撤销到这次之前（共 ${i + 1} 次）`}</button></div>`);
        row.querySelector('button').onclick = () => { m.close(); P.undoTo(x.id); };
        box.append(row);
      });
      m = App.modal({ title: '撤销批量操作', body, wide: true, buttons: [{ label: '关闭' }] });
    },
    async undoTo(id) {
      await P.flush();
      let chk;
      try { chk = await api('editor.undo', { dir: S.dir, ext: ext(), id, dry: true }); } catch (e) { App.toast(e.message, 'bad'); return; }
      if (!chk.ok) { App.toast(chk.message, 'bad'); return; }
      const warn = [];
      if (chk.count > 1) warn.push(`这次之后又做过 <b>${chk.count - 1}</b> 次批量操作，会一起撤销。`);
      if (chk.modified) warn.push(`这些批量操作之后，又手动改过 <b>${chk.modified}</b> 张图的标签，撤销会把这些手动修改一起还原掉。`);
      if (warn.length && !(await App.confirm('确认撤销', warn.join('<br><br>'), '仍然撤销', true))) return;
      let r;
      try { r = await api('editor.undo', { dir: S.dir, ext: ext(), id }); } catch (e) { App.toast(e.message, 'bad'); return; }
      App.toast(r.message, r.ok ? 'ok' : 'bad');
      if (r.ok) P.applyItems(r.items, r.undo);
    },

    // ---------------- 标签体检 ----------------
    async lint() {
      if (!S.dir) { App.toast('先选择训练集', 'bad'); return; }
      await P.flush();
      const p = App.params();
      const limit = +(p.max_token_length || p.qwen3_max_token_length || p.moss_max_token_length || 0) || null;
      const body = h('<div class="lint"><div class="muted">正在检查…</div></div>');
      const m = App.modal({ title: '标签体检', body, wide: true, buttons: [{ label: '关闭' }] });
      const load = async () => {
        let r;
        try { r = await api('editor.lint', { dir: S.dir, ext: ext(), limit }); } catch (e) { body.innerHTML = `<div class="issue"><b>检查失败</b>${esc(e.message)}</div>`; return; }
        if (body.isConnected) P.renderLint(body, r, m, load);
      };
      load();
    },
    renderLint(body, r, m, reload) {
      if (!r.issues.length) { body.innerHTML = `<div class="lint-ok">✓ ${r.count} 张图的标签没发现问题</div>`; return; }
      body.innerHTML = `<div class="hint" style="margin:0 0 10px">共 ${r.count} 张图。自动修复和批量操作一样会先备份，可以用「撤销…」恢复。</div>`;
      const fixBtn = (label, fx, cls = 'btn-primary') => { const b = h(`<button class="btn btn-sm ${cls}">${esc(label)}</button>`); b.onclick = () => P.lintFix(fx, reload, b); return b; };
      r.issues.forEach(it => {
        const card = h(`<div class="lint-card ${it.level}"><div class="lint-h"><b>${esc(it.title)}</b><span class="lint-act"></span></div><div class="lint-d">${esc(it.desc)}</div><div class="lint-rows"></div></div>`);
        const act = card.querySelector('.lint-act'), rows = card.querySelector('.lint-rows');
        if ((it.files || []).length) {
          const b = h('<button class="btn btn-sm">筛选这些图</button>');
          b.onclick = () => { S.only = { label: it.title, set: new Set(it.files) }; S.sel.clear(); m.close(); P.renderConds(); P.renderThumbs(); P.renderDetail(); };
          act.append(b);
        }
        (it.fixes || []).forEach(fx => act.append(fixBtn(fx.label, fx)));
        const code = t => `<code>${esc(t)}</code>`;
        if (it.code === 'variants') rows.innerHTML = it.rows.slice(0, 60).map(x => `<div>${x.forms.map(([f, n]) => `${code(f)}<span class="muted">×${n}</span>`).join(' ')} → <b>${esc(x.to)}</b></div>`).join('') + (it.rows.length > 60 ? `<div class="muted">……还有 ${it.rows.length - 60} 个</div>` : '');
        else if (it.code === 'alias') rows.innerHTML = it.rows.slice(0, 60).map(x => `<div>${code(x.tag)} → <b>${esc(x.to)}</b> <span class="muted">${x.count} 张</span></div>`).join('');
        else if (it.code === 'encoding') rows.innerHTML = Object.values(it.encodings || {}).map(v => `<div><b>${esc(v.name)}</b> 读出来是：${v.sample ? code(v.sample) : '<span class="muted">（没有中日文字，哪种都一样）</span>'}</div>`).join('');
        else if (it.code === 'unknown' || it.code === 'frequent') {
          it.rows.forEach(x => {
            const row = h(`<div>${code(x.tag)} <span class="muted">${x.count} 张</span> </div>`);
            (x.suggest || []).forEach(s => row.append(fixBtn('改成 ' + s, { id: 'rename', data: { map: { [x.tag]: s } } }, '')));
            row.append(fixBtn('删掉', { id: 'remove', data: { tags: x.tag } }, 'btn-danger'));
            rows.append(row);
          });
        } else if (it.code === 'once') rows.innerHTML = `<div class="lint-once">${it.rows.map(x => code(x.tag)).join(' ')}</div>`;
        body.append(card);
      });
    },
    async lintFix(fx, reload, btn) {
      if (fx.id === 'dict_download') { P.dictDownload(); return; }
      if (btn) btn.disabled = true;
      try {
        const r = await api('editor.lint_fix', { dir: S.dir, ext: ext(), fix: fx.id, data: fx.data });
        P.applyItems(r.items, r.undo);
        App.toast(r.message, r.changed ? 'ok' : '', r.changed && r.undo_id ? '撤销' : null, () => P.undoTo(r.undo_id), 8000);
      } catch (e) { App.toast(e.message, 'bad'); if (btn) btn.disabled = false; return; }
      reload();
    },
  };
})();

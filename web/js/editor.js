/* 05 标签编辑
   - 从「标签反推」过来时，自动打开刚反推的那组训练集
   - 左：整个训练集的标签统计（点击筛选；悬停可整体改名 / 删除）
   - 中：缩略图（单击查看，Ctrl 多选，Shift 连选），批量操作作用于「已选」或「当前筛选结果」
   - 右：大图 + 标签块（拖动排序、× 删除、双击改字、输入自动补全）/ 纯文本视图，改动自动保存
   - 所有批量操作执行前自动备份，可撤销 */
(function () {
  const { $, $$, esc, h, api } = App;
  const S = { dir: '', items: [], tags: [], sel: new Set(), cur: -1, filter: 'all', search: '', tagFilter: null, view: 'chips', last: -1 };
  let saveTimer = null;

  const split = s => (s || '').split(/,|\n/).map(x => x.trim()).filter(Boolean);
  const join = a => a.join(', ');
  const isSentence = t => t.split(/\s+/).length >= 7 || t.endsWith('.');
  const hasSentence = cap => split(cap).some(isSentence);
  const ZH = {}, TR = {};   // 标签 → 字典中文；整条标签原文 → 译文
  const ext = () => App.params().caption_extension || '.txt';

  const P = App.pages.editor = {
    init() {
      $('#ed-dir').onchange = e => P.open(e.target.value.trim());
      $('#ed-browse').onclick = async () => { const p = await App.pick({ title: '选择训练集', start: S.dir }); if (p) P.open(p); };
      $('#ed-reload').onclick = () => P.open(S.dir, true);
      $('#ed-search').oninput = e => { S.search = e.target.value.trim().toLowerCase(); clearTimeout(S.srTimer); S.srTimer = setTimeout(P.renderThumbs, 200); };
      $$('#ed-filter button').forEach(b => b.onclick = () => { S.filter = b.dataset.v; $$('#ed-filter button').forEach(x => x.classList.toggle('on', x === b)); P.renderThumbs(); });
      $('#ed-trigger').onclick = () => {
        const one = S.sel.size === 1 ? S.items[[...S.sel][0]] : null;
        const cur = S.items[S.cur];   // 右边正在看的那张：弹窗里可以一键换成它当典型图片
        App.triggerDialog(S.dir, { typical: one ? one.path : null, cur: cur ? cur.path : null });
      };
      $('#ed-undo').onclick = async () => { const r = await api('editor.undo', { dir: S.dir, ext: ext() }); App.toast(r.message, r.ok ? 'ok' : 'bad'); if (r.ok) P.open(S.dir, true); };
      $('#ed-selall').onclick = () => { P.visible().forEach(i => S.sel.add(i)); P.selChanged(); };
      $('#ed-selnone').onclick = () => { S.sel.clear(); P.selChanged(); };
      $('#ed-size').onchange = e => { $('#ed-thumbs').style.gridTemplateColumns = `repeat(auto-fill, minmax(${e.target.value}px, 1fr))`; };
      $('#ed-tag-sort').onchange = () => P.renderTags();
      $$('#ed-batch [data-op]').forEach(b => b.onclick = () => P.batch(b.dataset.op));
      $$('#ed-view button').forEach(b => b.onclick = () => { S.view = b.dataset.v; $$('#ed-view button').forEach(x => x.classList.toggle('on', x === b)); P.renderDetail(); });
      $('#ed-prev').onclick = () => P.step(-1); $('#ed-next').onclick = () => P.step(1);
      // 显示中文：D 站标签查字典；自然语言长句用本地模型或 API 翻（只看，不改标签文件）
      const tr = App.draft.editor_tr = App.draft.editor_tr || { on: false, engine: 'qwen3vl4b' };
      $('#ed-zh-on').checked = !!tr.on; $('#ed-zh-engine').value = tr.engine;
      $('#ed-zh-on').onchange = e => { tr.on = e.target.checked; App.saveDraft(); P.zhRefresh(); };
      $('#ed-zh-engine').onchange = e => { tr.engine = e.target.value; App.saveDraft(); };
      $('#ed-zh-one').onclick = () => { const it = S.items[S.cur]; if (it) P.translate([it.caption]); };
      $('#ed-zh-all').onclick = () => P.translate(S.items.map(it => it.caption).filter(hasSentence));
      $('#ed-remove').onclick = P.removeCur;
      $('#ed-text').oninput = () => { const it = S.items[S.cur]; if (!it) return; it.caption = $('#ed-text').value; P.queueSave(it); };
      const add = $('#ed-add');
      add.onkeydown = e => {
        const ac = $('.ac');
        if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { if (ac) { e.preventDefault(); P.acMove(e.key === 'ArrowDown' ? 1 : -1); } return; }
        if (e.key === 'Tab' && ac) { e.preventDefault(); P.acPick(); return; }
        if (e.key === 'Enter') { e.preventDefault(); if (ac && ac.querySelector('.on')) P.acPick(); else P.addTags(add.value); }
        if (e.key === 'Escape') P.acClose();
      };
      add.oninput = () => P.acShow();
      add.onblur = () => setTimeout(P.acClose, 150);
      document.addEventListener('keydown', e => {
        if (App.page !== 'editor' || !e.altKey) return;
        if (e.key === 'ArrowLeft') { e.preventDefault(); P.step(-1); }
        if (e.key === 'ArrowRight') { e.preventDefault(); P.step(1); }
      });
    },
    show() {
      const want = App.draft.editor_dir || App.params().dataset_dir || '';
      if (want && want !== S.dir) P.open(want);
      else if (!want) $('#ed-thumbs').innerHTML = '<div class="hint" style="padding:10px">选择一个训练集文件夹开始编辑；反推完点「去编辑这组标签」会自动打开。</div>';
    },
    async open(dir, keepCur) {
      if (!dir) return;
      let d;
      try { d = await api('editor.load', { dir, ext: ext() }); } catch (e) { App.toast(e.message, 'bad'); return; }
      const curPath = keepCur && S.items[S.cur] ? S.items[S.cur].path : null;
      S.dir = dir; S.items = d.items; S.sel.clear(); S.tagFilter = null;
      S.cur = curPath ? Math.max(0, S.items.findIndex(i => i.path === curPath)) : (S.items.length ? 0 : -1);
      $('#ed-dir').value = dir;
      App.draft.editor_dir = dir; App.saveDraft(); App.syncDataset(dir, 'editor');
      P.recount(); P.renderThumbs(); P.renderDetail();
      $('#ed-undo').disabled = !d.undo;
    },
    recountSoon() { clearTimeout(S.rcTimer); S.rcTimer = setTimeout(P.recount, 400); },   // 连续改标签时，标签统计最后统一刷新
    recount() {
      const f = new Map();
      S.items.forEach(it => split(it.caption).forEach(t => { if (!isSentence(t)) f.set(t, (f.get(t) || 0) + 1); }));
      S.tags = [...f.entries()];
      P.renderTags();
      P.zhRefresh();
    },
    // 取还没有的字典中文 / 已翻好的译文，取到了再重画
    async zhRefresh() {
      if (!App.draft.editor_tr.on) { P.renderTags(); P.redrawDetail(); return; }
      const tags = S.tags.map(x => x[0]).filter(t => !(t in ZH));
      const caps = [...new Set(S.items.map(it => (it.caption || '').trim()).filter(c => c && hasSentence(c) && !(c in TR)))];
      const [z, t] = await Promise.all([tags.length ? api('dict.zh', { tags }).catch(() => ({})) : {}, caps.length ? api('translate.get', { texts: caps }).catch(() => ({})) : {}]);
      tags.forEach(x => { ZH[x] = z[x] || ''; });
      Object.assign(TR, t);
      P.renderTags(); P.redrawDetail();
    },
    // 文本视图正在打字时不能整个重画（会把光标挪走），只更新译文框
    redrawDetail() { if (S.view === 'chips') P.renderDetail(); else P.renderZh(); },
    renderZh() {
      const box = $('#ed-zh'), it = S.items[S.cur], on = App.draft.editor_tr.on;
      box.hidden = !on || !it || !hasSentence(it.caption);
      if (box.hidden) return;
      const zh = TR[(it.caption || '').trim()];
      $('#ed-zh-text').textContent = zh || '这张图的自然语言标签还没翻译。';
      $('#ed-zh-text').classList.toggle('muted', !zh);
    },
    async translate(texts) {
      texts = [...new Set(texts.map(c => (c || '').trim()).filter(c => c && !TR[c]))];
      if (!texts.length) { App.toast('都已经翻译过了'); return; }
      const engine = App.draft.editor_tr.engine;
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
        if (!j) return;
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
    renderTags() {
      const el = $('#ed-tags');
      const arr = [...S.tags].sort($('#ed-tag-sort').value === 'alpha' ? (a, b) => a[0].localeCompare(b[0]) : (a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
      el.innerHTML = '';
      arr.slice(0, 1500).forEach(([t, n]) => {
        const r = h(`<div class="${S.tagFilter === t ? 'on' : ''}" title="${esc(t)}"><span class="tn">${esc(t)}${App.draft.editor_tr && App.draft.editor_tr.on && ZH[t] ? `<span class="zh">${esc(ZH[t])}</span>` : ''}</span><span class="ta"><button class="icon-btn" data-a="ren" title="整体改名">✎</button><button class="icon-btn" data-a="del" title="从所有图片删除">✕</button></span><span class="tc">${n}</span></div>`);
        r.onclick = e => {
          const a = e.target.dataset.a;
          if (a === 'ren') return P.renameTag(t);
          if (a === 'del') return P.deleteTag(t);
          S.tagFilter = S.tagFilter === t ? null : t; P.renderTags(); P.renderThumbs();
        };
        el.append(r);
      });
      if (!arr.length) el.innerHTML = '<div class="muted" style="padding:10px">还没有标签</div>';
    },
    visible() {
      const out = [];
      S.items.forEach((it, i) => {
        const cap = it.caption || '';
        if (S.filter === 'nocap' && cap.trim()) return;
        if (S.filter === 'sel' && !S.sel.has(i)) return;
        if (S.tagFilter && !split(cap).includes(S.tagFilter)) return;
        if (S.search && !cap.toLowerCase().includes(S.search) && !it.rel.toLowerCase().includes(S.search)) return;
        out.push(i);
      });
      return out;
    },
    scope() {
      if (S.sel.size) return { files: [...S.sel].map(i => S.items[i].path), label: `已选 ${S.sel.size} 张` };
      const v = P.visible();
      if (v.length !== S.items.length) return { files: v.map(i => S.items[i].path), label: `当前筛选 ${v.length} 张` };
      return { files: null, label: `全部 ${S.items.length} 张` };
    },
    // 只在图片列表本身变化（打开训练集、筛选、搜索）时整体重建；选中 / 当前图 / 标签变化都是局部更新
    renderThumbs() {
      const el = $('#ed-thumbs'), vis = P.visible();
      S.els = new Map();
      el.innerHTML = '';
      const frag = document.createDocumentFragment();
      vis.forEach(i => {
        const it = S.items[i];
        const t = h(`<div class="thumb" data-i="${i}"><img loading="lazy" decoding="async" src="/thumb?path=${encodeURIComponent(it.path)}">
          <span class="tno">无标签</span><span class="tck">✓</span><div class="tcap"></div></div>`);
        // 训练器还是旧版本（没有 /thumb 接口）或缩略图生成失败时，退回原图
        t.querySelector('img').onerror = function () { if (!this.dataset.fb) { this.dataset.fb = 1; this.src = '/file?path=' + encodeURIComponent(it.path); } };
        t.onclick = e => P.clickThumb(i, e);
        t.querySelector('.tck').onclick = e => { e.stopPropagation(); S.sel.has(i) ? S.sel.delete(i) : S.sel.add(i); P.selChanged(); };
        S.els.set(i, t);
        P.updateThumb(i);
        frag.append(t);
      });
      el.append(frag);
      if (!vis.length) el.innerHTML = '<div class="muted" style="padding:10px">没有符合条件的图片</div>';
      P.marks();
    },
    updateThumb(i) {
      const t = S.els && S.els.get(i); if (!t) return;
      const cap = S.items[i].caption || '';
      t.querySelector('.tcap').textContent = cap.slice(0, 120);
      t.querySelector('.tno').hidden = !!cap.trim();
    },
    marks() {
      if (S.els) S.els.forEach((t, i) => { t.classList.toggle('sel', S.sel.has(i)); t.classList.toggle('cur', i === S.cur); });
      const vis = P.visible();
      const noCap = S.items.filter(i => !(i.caption || '').trim()).length;
      $('#ed-count').textContent = `图片 ${vis.length}${vis.length !== S.items.length ? ' / ' + S.items.length : ''}${noCap ? ` · ${noCap} 张无标签` : ''}`;
      $('#ed-scope').textContent = '批量作用于：' + P.scope().label;
    },
    selChanged() { if (S.filter === 'sel') P.renderThumbs(); else P.marks(); },
    clickThumb(i, e) {
      if (e.shiftKey && S.last >= 0) {
        const vis = P.visible(); const a = vis.indexOf(S.last), b = vis.indexOf(i);
        vis.slice(Math.min(a, b), Math.max(a, b) + 1).forEach(x => S.sel.add(x));
      } else if (e.ctrlKey || e.metaKey) {
        S.sel.has(i) ? S.sel.delete(i) : S.sel.add(i);
      } else {
        S.cur = i; P.renderDetail();
      }
      S.last = i; P.selChanged();
    },
    step(d) {
      const vis = P.visible(); if (!vis.length) return;
      let k = vis.indexOf(S.cur); k = k < 0 ? 0 : (k + d + vis.length) % vis.length;
      S.cur = vis[k]; P.renderDetail(); P.marks();
      const el = $(`#ed-thumbs .thumb[data-i="${S.cur}"]`); if (el) el.scrollIntoView({ block: 'nearest' });
    },
    renderDetail() {
      const it = S.items[S.cur];
      const img = $('#ed-img');
      if (!it) { img.hidden = true; $('#ed-cur-name').textContent = '未选择图片'; $('#ed-chips').innerHTML = ''; P.renderZh(); return; }
      img.hidden = false;
      const src = '/file?path=' + encodeURIComponent(it.path);
      if (img.dataset.src !== src) { img.dataset.src = src; img.src = src; }   // 只改标签时不重新加载大图
      img.onclick = () => App.lightbox(P.visible().map(i => ({ file: S.items[i].path, cap: S.items[i].caption })), Math.max(0, P.visible().indexOf(S.cur)));
      $('#ed-cur-name').textContent = it.rel;
      $('#ed-saved').textContent = '';
      const chipsMode = S.view === 'chips';
      $('#ed-chips').hidden = !chipsMode; $('#ed-add-wrap').hidden = !chipsMode; $('#ed-text-wrap').hidden = chipsMode;
      P.renderZh();
      if (!chipsMode) { $('#ed-text').value = it.caption || ''; return; }
      const zhOn = App.draft.editor_tr.on;
      const el = $('#ed-chips'); el.innerHTML = '';
      const tags = split(it.caption);
      tags.forEach((t, k) => {
        const c = h(`<span class="chip ${k === 0 ? 'first' : ''} ${isSentence(t) ? 'sentence' : ''}" draggable="true" title="拖动排序；双击修改"><span>${esc(t)}${zhOn && ZH[t] ? `<span class="zh">${esc(ZH[t])}</span>` : ''}</span><button title="删除">×</button></span>`);
        c.querySelector('button').onclick = () => { tags.splice(k, 1); P.setCur(join(tags)); };
        c.ondblclick = async () => { const v = await App.prompt('修改标签', '新内容', t); if (v === null) return; tags[k] = v.trim(); P.setCur(join(tags.filter(Boolean))); };
        c.ondragstart = e => { e.dataTransfer.setData('text/plain', String(k)); c.style.opacity = .4; };
        c.ondragend = () => { c.style.opacity = 1; };
        c.ondragover = e => { e.preventDefault(); c.classList.add('drag-over'); };
        c.ondragleave = () => c.classList.remove('drag-over');
        c.ondrop = e => { e.preventDefault(); const from = +e.dataTransfer.getData('text/plain'); const [m] = tags.splice(from, 1); tags.splice(from < k ? k - 1 : k, 0, m); P.setCur(join(tags)); };
        el.append(c);
      });
      if (!tags.length) el.innerHTML = '<span class="muted">这张图还没有标签</span>';
    },
    setCur(caption) {
      const it = S.items[S.cur]; if (!it) return;
      it.caption = caption; P.queueSave(it); P.renderDetail(); P.updateThumb(S.cur); P.marks(); P.recountSoon();
    },
    queueSave(it) {
      clearTimeout(saveTimer); $('#ed-saved').textContent = '…';
      saveTimer = setTimeout(async () => {
        try { await api('editor.save', { dir: S.dir, path: it.path, text: it.caption, ext: ext() }); $('#ed-saved').textContent = '已保存'; }
        catch (e) { $('#ed-saved').textContent = '保存失败：' + e.message; }
        if (S.view === 'text') { P.updateThumb(S.items.indexOf(it)); P.recountSoon(); }
      }, 350);
    },
    addTags(v) {
      const it = S.items[S.cur]; if (!it || !v.trim()) return;
      const tags = split(it.caption); split(v).forEach(t => { if (!tags.includes(t)) tags.push(t); });
      $('#ed-add').value = ''; P.acClose(); P.setCur(join(tags));
    },
    // 自动补全：训练集里已有的标签，按出现次数排序
    acShow() {
      P.acClose();
      const v = $('#ed-add').value.split(',').pop().trim().toLowerCase(); if (!v) return;
      const cur = new Set(split((S.items[S.cur] || {}).caption));
      const m = S.tags.filter(([t]) => t.toLowerCase().includes(v) && !cur.has(t)).sort((a, b) => (b[0].toLowerCase().startsWith(v) - a[0].toLowerCase().startsWith(v)) || b[1] - a[1]).slice(0, 12);
      if (!m.length) return;
      const ac = h(`<div class="ac">${m.map(([t, n], i) => `<div class="${i === 0 ? 'on' : ''}" data-t="${esc(t)}"><span>${esc(t)}</span><span class="muted">${n}</span></div>`).join('')}</div>`);
      ac.style.left = '10px'; ac.style.right = '10px'; ac.style.top = '36px';
      $$('div', ac).forEach(d => d.onmousedown = e => { e.preventDefault(); $$('div', ac).forEach(x => x.classList.remove('on')); d.classList.add('on'); P.acPick(); });
      $('#ed-add-wrap').append(ac);
    },
    acMove(d) { const ds = $$('.ac > div'); let k = ds.findIndex(x => x.classList.contains('on')); ds.forEach(x => x.classList.remove('on')); k = (k + d + ds.length) % ds.length; ds[k].classList.add('on'); ds[k].scrollIntoView({ block: 'nearest' }); },
    acPick() {
      const on = $('.ac > div.on'); if (!on) return;
      const parts = $('#ed-add').value.split(','); parts.pop();
      P.addTags([...parts, on.dataset.t].join(','));
    },
    acClose() { const a = $('.ac'); if (a) a.remove(); },

    async batch(op) {
      if (!S.dir) return;
      const sc = P.scope(); let args = {}, opname = op, desc = '';
      const tags = $('#ed-b-tags').value.trim();
      if (op === 'add-front' || op === 'add-back') { if (!tags) return App.toast('先在左边输入框里填要添加的标签', 'bad'); opname = 'add'; args = { tags, position: op === 'add-front' ? 'front' : 'back' }; desc = `把「${tags}」加到${op === 'add-front' ? '最前' : '最后'}`; }
      else if (op === 'remove') { if (!tags) return App.toast('先填要删除的标签', 'bad'); args = { tags }; desc = `删除标签「${tags}」`; }
      else if (op === 'find_replace') { const f = $('#ed-b-find').value; if (!f) return App.toast('先填要查找的内容', 'bad'); args = { find: f, replace: $('#ed-b-rep').value }; desc = `把「${f}」替换成「${args.replace}」`; }
      else if (op === 'dedupe') desc = '去掉重复的标签';
      else if (op === 'sort') { args = { keep: 1 }; desc = '第一个标签（触发词）不动，其余标签按全集出现次数从多到少排序，自然语言句子放最后'; }
      else if (op === 'trigger') { const t = await App.prompt('设置触发词', '触发词（会放到每个标签文件最前面；多个用逗号分隔）', App.draft.tag && App.draft.tag.trigger || ''); if (!t) return; args = { trigger: t }; desc = `把触发词「${t}」放到最前面`; }
      else if (op === 'clear') desc = '清空标签内容（标签文件保留为空）';
      if (!(await App.confirm('批量修改标签', `${esc(desc)}<br><br>作用于：<b>${esc(sc.label)}</b><br><span class="muted">执行前会自动备份，可以点「撤销批量」恢复。</span>`, '执行', op === 'clear'))) return;
      const r = await api('editor.batch', { dir: S.dir, op: opname, files: sc.files, args, ext: ext() });
      App.toast(`已修改 ${r.changed} 个标签文件`, 'ok');
      P.open(S.dir, true);
    },
    async renameTag(t) {
      const v = await App.prompt('整体改名', `把所有图片里的「${t}」改成：`, t);
      if (v === null || v.trim() === t) return;
      const r = await api('editor.batch', { dir: S.dir, op: 'replace', args: { old: t, new: v.trim() }, ext: ext() });
      App.toast(`已修改 ${r.changed} 个文件`, 'ok'); P.open(S.dir, true);
    },
    async deleteTag(t) {
      const sc = P.scope();
      if (!(await App.confirm('删除标签', `从${esc(sc.label)}里删除「${esc(t)}」？（可撤销）`, '删除', true))) return;
      const r = await api('editor.batch', { dir: S.dir, op: 'remove', files: sc.files, args: { tags: t }, ext: ext() });
      App.toast(`已修改 ${r.changed} 个文件`, 'ok'); P.open(S.dir, true);
    },
    async removeCur() {
      const it = S.items[S.cur]; if (!it) return;
      if (!(await App.confirm('移出训练集', `把 ${esc(it.rel)} 和它的标签移到训练集里的「_已移除」文件夹？训练时不会再用到它。`, '移出', true))) return;
      await api('editor.remove', { dir: S.dir, path: it.path, ext: ext() });
      S.items.splice(S.cur, 1); S.sel.clear(); if (S.cur >= S.items.length) S.cur = S.items.length - 1;
      P.recount(); P.renderThumbs(); P.renderDetail();
    },
  };
})();

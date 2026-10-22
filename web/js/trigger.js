/* 触发词整理：主触发词 + 角色特征 + 服装（出现最多 / 典型图片两种选法）
   App.triggerDialog(dir, { typical })：标签编辑页「触发词」按钮
   App.autoTrigger(dir)：反推完自动整理（无感），结果用 toast 告诉用户，可点开修改 */
(function () {
  const { esc, h, api } = App;

  const sameDir = (a, b) => (a || '').replace(/[\\/]+$/, '').toLowerCase() === (b || '').replace(/[\\/]+$/, '').toLowerCase();
  // 主触发词的备选：反推页填的触发词、爬图时搜的角色标签（标签里没有角色名时用；爬图标签只在爬到的就是这个训练集时才算）
  const hints = dir => {
    const out = [];
    const tr = (App.draft.tag || {}).trigger;
    if (tr) out.push(...tr.split(/[,，]/));
    const c = App.draft.crawl || {};
    if (!dir || sameDir(c.out_dir, dir)) out.push(...String(c.tags || '').split(/\s+/).filter(t => t && !/^[-~]/.test(t) && !t.includes(':')));
    return out;
  };
  const ext = () => ((App.draft.tag || {}).ext) || '.txt';
  // 删标设置：上次在弹窗里选的（反推完自动整理也照这个来）；null = 全标
  const lastPrune = () => (App.draft.tag || {}).trigger_prune;
  const NOPRUNE = { features: false, outfit: false };

  // 触发词定下来后，用这个训练集的各底模配置里的测试提示词开头都换成「用 LoRA 时写」的那一串
  // （全标要写上特征 / 服装，删标只写触发词；测试提示词和标法对不上，测试图就没法看）
  App.syncSamplePrompts = async function (dir) {
    const same = d => sameDir(d, dir);
    let n = 0;
    for (const c of Object.values(App.draft.by_model || {})) {
      if (!c.sample || !(c.sample.prompts || []).length || !same((c.params || {}).dataset_dir || App.params().dataset_dir)) continue;
      // 只改已经写了触发词的；没写触发词的标准提示词不动（要加就点测试页的「在提示词前加上触发词」）
      const r = await api('trigger.sync_prompts', { dir, prompts: c.sample.prompts, only_with_main: true }).catch(() => null);
      if (r && r.changed) { c.sample.prompts = r.prompts; n += r.changed; }
    }
    if (n) { App.saveDraft(); if (App.page === 'sample') App.pages.sample.renderPrompts(); }
    return n;
  };

  App.autoTrigger = async function (dir) {
    if (!dir) return;
    let r;
    // auto：不沿用这个训练集上次记下的触发词（标签刚重新反推过）
    try { r = await api('trigger.analyze', { dir, ext: ext(), mode: (App.draft.tag || {}).trigger_mode || 'freq', hints: hints(dir), auto: true }); } catch (e) { return; }
    if (!r.main) {
      // 没填触发词、标签里也没有角色名：不加触发词；标签是覆盖重写的，旧的触发词记录也一起作废
      if (((App.draft.tag || {}).existing || 'overwrite') === 'overwrite') await api('trigger.clear', { dir }).catch(() => null);
      App.toast('没填触发词，标签里也没有角色名：没有加触发词', '', '设置触发词', () => App.triggerDialog(dir), 8000);
      return;
    }
    const style = r.kind === 'style';   // 画风训练集只要主触发词
    const res = await api('trigger.apply', { dir, ext: ext(), main: r.main, features: style ? [] : r.features.map(x => x.tag), outfit: style ? [] : r.outfit.map(x => x.tag),
      prune: lastPrune() || r.prune, mode: r.mode, typical: r.typical }).catch(() => null);
    if (res) {
      const n = await App.syncSamplePrompts(dir);
      App.toast(`已自动整理触发词：${res.prompt.length > 60 ? res.prompt.slice(0, 60) + '…' : res.prompt}${n ? `；${n} 条测试提示词已同步` : ''}`, 'ok', '查看 / 修改', () => App.triggerDialog(dir), 12000);
    }
  };

  App.triggerDialog = async function (dir, opt = {}) {
    if (!dir) { App.toast('先选择训练集', 'bad'); return; }
    let st = { mode: (App.draft.tag || {}).trigger_mode || 'freq', typical: opt.typical || null, main: '', sel: { features: new Set(), outfit: new Set() },
      prune: lastPrune() ? Object.assign({}, lastPrune()) : null };
    let r = null;
    const body = h(`<div class="trig">
      <div class="trig-row"><label>选法</label><div class="seg" data-mode><button data-v="freq">出现最多（按比例自动选）</button><button data-v="typical">一张典型图片</button></div></div>
      <div class="hint" data-kind style="margin:0"></div>
      <div class="trig-row"><label>主触发词</label><div><input type="text" data-main style="width:100%" placeholder="角色名，例如 hatsune miku"><div class="trig-cands" data-cands></div></div></div>
      <div class="trig-row" data-typ-row><label>典型图片</label><div class="row" style="align-items:flex-start;gap:10px"><img data-typ class="trig-typ"><div class="hint" style="margin:0">自动挑的是特征和服装最有代表性的一张。<br>想换一张：在标签编辑里选中那张图，再点上面的「触发词」按钮。</div></div></div>
      <div class="trig-row"><label>角色特征</label><div class="trig-chips" data-g="features"></div></div>
      <div class="trig-row"><label>服装</label><div class="trig-chips" data-g="outfit"></div></div>
      <div class="trig-row"><label>标法</label><div>
        <div class="seg" data-tm><button data-v="all">全标（都保留）</button><button data-v="prune">删标（删掉的记到主触发词上）</button></div>
        <div class="row" data-prune-row style="gap:8px;margin-top:8px;flex-wrap:wrap">
          <label class="sw"><input type="checkbox" data-pf><i></i><span>删除人物特征（发色、瞳色、兽耳……只写主触发词就能出）</span></label>
          <label class="sw"><input type="checkbox" data-po><i></i><span>删除服装（只写主触发词就穿原装；换衣服会难一些）</span></label>
        </div>
        <div class="hint" data-tm-hint style="margin:6px 0 0"></div></div></div>
      <div class="trig-row"><label>用 LoRA 时写</label><div class="code-box" data-prompt style="margin:0;white-space:normal"></div></div>
      <div class="hint">点标签切换选中 / 不选；百分比是在多少比例的图里出现。应用后每张图最前面是主触发词，其余标签保持原来的顺序（删标时删掉的那组不出现），可以在标签编辑里撤销。</div>
    </div>`);
    const $b = s => body.querySelector(s);
    const pr = () => st.prune || NOPRUNE;
    const prompt = () => {
      const f = r ? r.features.concat(r.more_features).filter(x => st.sel.features.has(x.tag)).map(x => x.tag) : [];
      const o = r ? r.outfit.concat(r.more_outfit).filter(x => st.sel.outfit.has(x.tag)).map(x => x.tag) : [];
      // 删掉的那组已经记在主触发词上，用的时候不用再写
      $b('[data-prompt]').textContent = [st.main, ...(pr().features ? [] : f), ...(pr().outfit ? [] : o)].filter(Boolean).join(', ') || '—';
      return { f, o };
    };
    const renderTm = () => {
      const on = !!st.prune;
      App.$$('[data-tm] button', body).forEach(b => b.classList.toggle('on', b.dataset.v === (on ? 'prune' : 'all')));
      $b('[data-prune-row]').hidden = !on;
      $b('[data-pf]').checked = pr().features; $b('[data-po]').checked = pr().outfit;
      $b('[data-tm-hint]').textContent = !on ? '所有标签都留着，顺序不变，只把主触发词放在最前面。用 LoRA 时要把特征词、服装一起写上。'
        : pr().features && pr().outfit ? '特征和服装都删掉：只写主触发词就是原装的完整造型；换衣服时要写清楚新衣服。'
        : pr().features ? '只删特征（人物 LoRA 最常用）：只写主触发词就有发色、兽耳等；衣服照常靠服装标签控制，换装容易。'
        : pr().outfit ? '只删服装：只写主触发词就穿原装，特征还要写出来。' : '两项都没勾，等于全标。';
      App.$$('[data-g]', body).forEach(el => el.classList.toggle('pruned', !!pr()[el.dataset.g]));
      prompt();
    };
    const render = () => {
      App.$$('[data-mode] button', body).forEach(b => b.classList.toggle('on', b.dataset.v === st.mode));
      $b('[data-typ-row]').hidden = st.mode !== 'typical';
      if (!r) return;
      $b('[data-main]').value = st.main;
      $b('[data-cands]').innerHTML = r.main_candidates.map(c => `<a data-c="${esc(c.tag)}">${esc(c.tag)}${c.zh ? ' ' + esc(c.zh) : ''} · ${c.count}</a>`).join('');
      App.$$('[data-c]', body).forEach(a => a.onclick = () => { st.main = a.dataset.c; render(); });
      $b('[data-typ]').src = r.typical ? '/thumb?path=' + encodeURIComponent(r.typical) + '&size=256' : '';
      for (const g of ['features', 'outfit']) {
        const box = $b(`[data-g=${g}]`); box.innerHTML = '';
        const all = r[g].concat(r['more_' + g]);
        if (!all.length) box.innerHTML = '<span class="muted">没有找到</span>';
        all.forEach(x => {
          const c = h(`<span class="chip trig-chip ${st.sel[g].has(x.tag) ? 'on' : ''}" title="${esc(x.zh || '')}"><span>${esc(x.tag)}</span><em>${x.pct}%</em></span>`);
          c.onclick = () => { st.sel[g].has(x.tag) ? st.sel[g].delete(x.tag) : st.sel[g].add(x.tag); render(); };
          box.append(c);
        });
      }
      renderTm();
    };
    const load = async () => {
      try { r = await api('trigger.analyze', { dir, ext: ext(), mode: st.mode, typical: st.typical, main: st.main || null, hints: hints(dir) }); }
      catch (e) { App.toast(e.message, 'bad'); return; }
      st.main = st.main || r.main; st.typical = r.typical;
      if (!st.prune && (r.prune.features || r.prune.outfit)) st.prune = Object.assign({}, r.prune);   // 这个训练集上次是删标
      const style = r.kind === 'style' && !opt.typical && st.mode === 'freq';
      st.sel = { features: new Set(style ? [] : r.features.map(x => x.tag)), outfit: new Set(style ? [] : r.outfit.map(x => x.tag)) };
      $b('[data-kind]').textContent = r.kind === 'style' ? '看起来是画风训练集（没有统一的角色），默认只用主触发词；需要的话可以点下面的标签加上' : '看起来是人物训练集';
      render();
    };
    App.$$('[data-mode] button', body).forEach(b => b.onclick = () => {
      st.mode = b.dataset.v; App.draft.tag = App.draft.tag || {}; App.draft.tag.trigger_mode = st.mode; App.saveDraft(); load();
    });
    $b('[data-main]').oninput = e => { st.main = e.target.value.trim(); prompt(); };
    App.$$('[data-tm] button', body).forEach(b => b.onclick = () => {
      st.prune = b.dataset.v === 'prune' ? (st.prune || { features: true, outfit: false }) : null; renderTm();
    });
    $b('[data-pf]').onchange = e => { st.prune = Object.assign({}, pr(), { features: e.target.checked }); renderTm(); };
    $b('[data-po]').onchange = e => { st.prune = Object.assign({}, pr(), { outfit: e.target.checked }); renderTm(); };
    App.modal({ title: '触发词 · ' + App.base(dir), body, wide: true, buttons: [
      { label: '取消' },
      { label: '应用到全部标签', cls: 'btn-primary', fn: async () => {
        if (!st.main) { App.toast('请填写主触发词', 'bad'); return false; }
        const { f, o } = prompt();
        try {
          const prune = pr();
          const res = await api('trigger.apply', { dir, ext: ext(), main: st.main, features: f, outfit: o, prune, mode: st.mode, typical: st.typical });
          App.draft.tag = App.draft.tag || {};   // 记住这次的标法，反推完自动整理也照这个来
          App.draft.tag.trigger_prune = prune.features || prune.outfit ? prune : null; App.saveDraft();
          const n = await App.syncSamplePrompts(dir);
          App.toast(`已整理 ${res.changed} 个标签文件。用 LoRA 时写：${res.prompt}${n ? `（${n} 条测试提示词已同步）` : ''}`, 'ok', null, null, 10000);
          if (App.pages.editor && App.page === 'editor') App.pages.editor.open(dir, true);
        } catch (e) { App.toast(e.message, 'bad'); return false; }
      } },
    ] });
    render();
    load();
  };
})();

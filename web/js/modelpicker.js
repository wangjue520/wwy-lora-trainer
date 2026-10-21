/* 大模型选择器：模型多（比如以后的 SDXL，WebUI / ComfyUI 里动辄上百个微调版）时用。
   点开是一个带搜索框的列表：按「官方 / 微调 / Turbo」或按文件夹分组，有预览图（.preview.png）、Civitai 名称、底模类型。
   用法：const mp = ModelPicker(容器, { items, value, allowNone, groups:[[key,label]...], onChange(path) }); mp.set(items, value) */
(function () {
  const { esc, h } = App;
  let openPicker = null;
  document.addEventListener('mousedown', e => { if (openPicker && !openPicker.contains(e.target)) { openPicker.classList.remove('open'); openPicker = null; } });

  const label = x => x.civitai_name ? `${x.civitai_name}${x.civitai_version ? ' · ' + x.civitai_version : ''}` : x.name;

  window.ModelPicker = function (box, opt) {
    let items = opt.items || [], value = opt.value || '', groupBy = 'variant', q = '', hi = -1, flat = [];
    box.classList.add('mpick');
    box.innerHTML = `<button type="button" class="mp-btn"><img class="mp-thumb" hidden><span class="mp-cur"></span><span class="mp-caret">▾</span></button>
      <div class="mp-panel"><div class="mp-bar"><input type="text" class="mp-q" placeholder="搜索：文件名 / Civitai 名称 / 底模类型 / 文件夹">
        <div class="seg mp-gb"><button data-v="variant">按类型</button><button data-v="folder">按文件夹</button></div></div>
        <div class="mp-list"></div></div>`;
    const btn = box.querySelector('.mp-btn'), cur = box.querySelector('.mp-cur'), thumb = box.querySelector('.mp-thumb');
    const qi = box.querySelector('.mp-q'), list = box.querySelector('.mp-list');

    const meta = x => [App.fmtSize(x.size), x.base_model, x.quant ? (x.qtype ? x.qtype + ' 量化' : '量化') : '', x.source.split(' · ')[0] + (x.folder ? ' / ' + x.folder : '')].filter(Boolean).join(' · ');
    function renderBtn() {
      const it = items.find(x => x.path === value);
      cur.innerHTML = it ? `<b>${esc(label(it))}</b>${it.civitai_name ? ` <span class="muted">${esc(it.name)}</span>` : ''}${it === items[0] ? ' <span class="badge ok">★推荐</span>' : ''}<br><span class="muted">${esc(meta(it))}</span>`
        : value ? `<b>自选：</b>${esc(value)}` : `<span class="muted">${opt.allowNone ? '（不使用）' : items.length ? '请选择' : '没有找到匹配的文件'}</span>`;
      thumb.hidden = !(it && it.preview);
      if (it && it.preview) thumb.src = '/thumb?path=' + encodeURIComponent(it.preview);
    }
    function match(x) {
      if (!q) return true;
      const s = [x.name, x.civitai_name, x.civitai_version, x.base_model, x.folder, x.source].join(' ').toLowerCase();
      return q.toLowerCase().split(/\s+/).every(w => s.includes(w));
    }
    function renderList() {
      const xs = items.filter(match);
      const groups = [];
      if (groupBy === 'folder') {
        const m = new Map();
        xs.forEach(x => { const k = x.source.split(' · ')[0] + ' / ' + (x.folder || ''); if (!m.has(k)) m.set(k, []); m.get(k).push(x); });
        [...m.keys()].sort().forEach(k => groups.push([k, m.get(k)]));
      } else {
        (opt.groups || [['', '']]).forEach(([g, lab]) => { const ys = xs.filter(x => (x.variant || '') === g); if (ys.length) groups.push([lab, ys]); });
        const rest = xs.filter(x => !(opt.groups || []).some(([g]) => (x.variant || '') === g));
        if (rest.length && opt.groups) groups.push(['其它', rest]);
      }
      flat = [];
      let html = opt.allowNone ? `<div class="mp-row ${!value ? 'on' : ''}" data-i="-1"><div class="mp-name">（不使用）</div></div>` : '';
      groups.forEach(([lab, ys]) => {
        if (lab) html += `<div class="mp-group">${esc(lab)} · ${ys.length} 个</div>`;
        ys.forEach(x => {
          const i = flat.push(x) - 1;
          html += `<div class="mp-row ${x.path === value ? 'on' : ''}" data-i="${i}">
            ${x.preview ? `<img loading="lazy" src="/thumb?path=${encodeURIComponent(x.preview)}">` : '<i class="mp-noimg">无预览</i>'}
            <div><div class="mp-name">${esc(label(x))}${x === items[0] ? ' <span class="badge ok">★推荐</span>' : ''}</div>
            <div class="muted">${x.civitai_name ? esc(x.name) + ' · ' : ''}${esc(meta(x))}</div></div></div>`;
        });
      });
      if (!flat.length && !opt.allowNone) html = `<div class="muted" style="padding:14px">没有匹配「${esc(q)}」的模型</div>`;
      list.innerHTML = html;
      hi = -1;
      App.$$('.mp-row', list).forEach(r => r.onmousedown = e => { e.preventDefault(); pick(+r.dataset.i); });
      App.$$('.mp-gb button', box).forEach(b => b.classList.toggle('on', b.dataset.v === groupBy));
    }
    function pick(i) {
      value = i < 0 ? '' : flat[i].path;
      box.classList.remove('open'); openPicker = null;
      renderBtn(); opt.onChange && opt.onChange(value);
    }
    btn.onclick = () => {
      const open = !box.classList.contains('open');
      if (openPicker && openPicker !== box) openPicker.classList.remove('open');
      box.classList.toggle('open', open); openPicker = open ? box : null;
      if (open) { q = qi.value = ''; renderList(); setTimeout(() => qi.focus(), 0); const on = list.querySelector('.mp-row.on'); if (on) on.scrollIntoView({ block: 'nearest' }); }
    };
    qi.oninput = () => { q = qi.value.trim(); renderList(); };
    qi.onkeydown = e => {
      const rows = App.$$('.mp-row', list);
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault(); hi = Math.max(0, Math.min(rows.length - 1, hi + (e.key === 'ArrowDown' ? 1 : -1)));
        rows.forEach((r, k) => r.classList.toggle('hi', k === hi)); rows[hi] && rows[hi].scrollIntoView({ block: 'nearest' });
      } else if (e.key === 'Enter' && rows.length) { e.preventDefault(); pick(+rows[Math.max(0, hi)].dataset.i); }
      else if (e.key === 'Escape') { box.classList.remove('open'); openPicker = null; }
    };
    App.$$('.mp-gb button', box).forEach(b => b.onmousedown = e => { e.preventDefault(); groupBy = b.dataset.v; renderList(); });
    renderBtn();
    return { set(newItems, v) { items = newItems || []; value = v || ''; renderBtn(); }, get: () => value };
  };
})();

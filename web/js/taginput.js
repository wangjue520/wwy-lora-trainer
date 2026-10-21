/* D 站标签输入框：只能放 Danbooru 正式标签。
   输入中文 / 英文 / 别名 / 日文 → 候选（本地中英字典，没下载字典时用 Danbooru 在线补全）；
   回车、空格、逗号或点候选把标签加成「标签块」（显示中文名和图数）；
   不是 D 站标签的不会加进去，并提示最接近的正确写法；别名自动换成正式名。
   用法：const ti = TagInput(容器, { wild: 允许 * 通配符, prefix: 允许 -排除 / ~任意 前缀, placeholder, onChange(list) }); ti.set(list) */
(function () {
  const { esc, h, api } = App;
  const CAT_COLOR = { 0: 'var(--chart)', 1: '#e05c5c', 3: '#b07cff', 4: '#3ecf8e', 5: '#f5a742' };
  const cache = {};   // tag → {zh, count, category, ok}

  window.TagInput = function (box, opt = {}) {
    let list = [];
    box.classList.add('tagbox');
    box.innerHTML = `<span class="tb-chips"></span><input type="text" autocomplete="off" placeholder="${esc(opt.placeholder || '输入中文或英文，回车添加')}"><div class="ac" hidden></div><div class="tb-err" hidden></div>`;
    const chips = box.querySelector('.tb-chips'), inp = box.querySelector('input'), ac = box.querySelector('.ac'), err = box.querySelector('.tb-err');
    let timer = null, seq = 0;

    const emit = () => opt.onChange && opt.onChange(list.slice());
    function render() {
      chips.innerHTML = '';
      list.forEach((t, i) => {
        const pre = (t.match(/^[-~]/) || [''])[0], c = cache[t.slice(pre.length)] || {};
        const el = h(`<span class="chip tb-chip ${pre === '-' ? 'neg' : pre === '~' ? 'any' : ''}" title="${pre === '-' ? '排除 · ' : pre === '~' ? '任意一个 · ' : ''}"${esc((c.zh || '') + (c.count ? ' · ' + c.count.toLocaleString() + ' 张' : '') + (c.rare ? ' · 冷门标签（字典未收录）' : '') + (c.unverified ? ' · 没联网，开爬前再确认' : ''))}">
          <i class="tb-dot" style="background:${CAT_COLOR[c.category] || 'var(--line-strong)'}"></i><span>${esc(t)}</span>${c.zh ? `<em>${esc(c.zh)}</em>` : ''}<button title="删除">×</button></span>`);
        el.querySelector('button').onclick = e => { e.stopPropagation(); list.splice(i, 1); render(); emit(); };
        chips.append(el);
      });
    }
    async function validate(tags) {
      const need = tags.filter(t => !cache[t]);
      if (need.length) {
        const res = await api('dict.validate', { tags: need }).catch(() => []);
        res.forEach(r => { if (r.unverified) return; cache[r.tag] = r; if (r.input !== r.tag) cache[r.input.toLowerCase()] = r; });
        res.filter(r => r.unverified).forEach(r => { tmp[r.tag] = r; });
      }
      return tags.map(t => cache[t] || tmp[t] || { input: t, tag: t, ok: false });
    }
    const tmp = {};
    async function add(raw) {
      const words = raw.split(/[\s,，、]+/).map(w => w.trim()).filter(Boolean);
      if (!words.length) return;
      const pres = words.map(w => opt.prefix ? (w.match(/^[-~]/) || [''])[0] : '');
      const res = await validate(words.map((w, i) => w.slice(pres[i].length).toLowerCase()).map(w => w.replace(/^[-~]+/, '')));
      const bad = [];
      res.forEach((r, i) => {
        if (r.ok && (!r.wild || opt.wild)) {
          const t = pres[i] + r.tag;
          if (!list.includes(t)) list.push(t);
          if (r.renamed) App.toast(`「${r.input}」是别名，已换成正式标签 ${r.tag}`, 'ok');
        } else bad.push(r);
      });
      inp.value = bad.map(b => b.input).join(' ');
      err.hidden = !bad.length;
      err.innerHTML = bad.map(b => b.wild ? `「${esc(b.input)}」：通配符只能用在黑名单里` : `「${esc(b.input)}」不是 D 站标签${b.suggest && b.suggest.length ? '，是不是：' + b.suggest.map(s => `<a data-s="${esc(s)}">${esc(s)}</a>`).join('、') : ''}`).join('<br>');
      App.$$('a[data-s]', err).forEach(a => a.onclick = () => { inp.value = ''; err.hidden = true; add(a.dataset.s); });
      ac.hidden = true;
      render(); emit();
    }
    async function suggest() {
      const q = inp.value.trim().split(/[\s,，、]+/).pop().replace(/^[-~]/, '');
      err.hidden = true;
      if (!q || (opt.wild && /[*?]/.test(q))) { ac.hidden = true; return; }
      const my = ++seq;
      let res = await api('dict.search', { q, limit: 10 }).catch(() => []);
      if (!res.length) res = (await api('crawl.autocomplete', { q }).catch(() => [])).map(x => ({ name: x.name, count: x.count, category: x.category, zh: '' }));
      if (my !== seq) return;
      if (!res.length) { ac.hidden = true; return; }
      res.forEach(r => { cache[r.name] = cache[r.name] || { tag: r.name, ok: true, zh: r.zh, count: r.count, category: r.category }; });
      ac.innerHTML = res.map((r, i) => `<div class="${i ? '' : 'on'}" data-v="${esc(r.name)}"><span><i class="tb-dot" style="background:${CAT_COLOR[r.category] || 'var(--line-strong)'}"></i>${esc(r.name)} <em class="muted">${esc(r.zh || '')}</em></span><span class="muted">${(r.count || 0).toLocaleString()}</span></div>`).join('');
      ac.hidden = false;
      App.$$('div', ac).forEach(d => d.onmousedown = e => { e.preventDefault(); pick(d.dataset.v); });
    }
    function pick(tag) {
      const parts = inp.value.trim().split(/[\s,，、]+/), last = parts.pop() || '';
      const pre = opt.prefix ? (last.match(/^[-~]/) || [''])[0] : '';
      inp.value = '';
      add([...parts, pre + tag].join(' '));
      inp.focus();
    }
    inp.oninput = () => { clearTimeout(timer); timer = setTimeout(suggest, 160); };
    inp.onkeydown = e => {
      const ds = App.$$('div', ac); let k = ds.findIndex(x => x.classList.contains('on'));
      if (!ac.hidden && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
        e.preventDefault(); ds.forEach(x => x.classList.remove('on'));
        k = (k + (e.key === 'ArrowDown' ? 1 : -1) + ds.length) % ds.length; ds[k].classList.add('on'); ds[k].scrollIntoView({ block: 'nearest' });
        return;
      }
      if (e.key === 'Enter' || e.key === 'Tab' || e.key === ',' || e.key === '，') {
        if (!inp.value.trim()) return;
        e.preventDefault();
        if (!ac.hidden && k >= 0) pick(ds[k].dataset.v); else add(inp.value);
      } else if (e.key === 'Backspace' && !inp.value && list.length) {
        list.pop(); render(); emit();
      } else if (e.key === 'Escape') ac.hidden = true;
    };
    inp.onpaste = e => {
      const t = (e.clipboardData || window.clipboardData).getData('text');
      if (/[\s,，]/.test(t.trim())) { e.preventDefault(); add(t); }
    };
    inp.onblur = () => setTimeout(() => { ac.hidden = true; }, 150);
    box.onclick = e => { if (e.target === box || e.target === chips) inp.focus(); };

    return {
      set(tags) {
        list = (Array.isArray(tags) ? tags : String(tags || '').split(/\s+/)).filter(Boolean);
        render();
        validate(list).then(render);   // 补上中文名
      },
      get() { return list.slice(); },
    };
  };
})();

/* 核心：接口调用、当前配置（草稿）、页面切换、弹窗 / 提示 / 文件选择、报错修复弹窗、后台作业浮窗 */
(function () {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const h = (html) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild; };

  async function api(name, args = {}) {
    let r;
    try {
      r = await fetch('/api/' + name, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(args) });
    } catch (e) {
      throw new Error('连不上训练器后台（训练器窗口被关掉了？）');
    }
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || '未知错误');
    return d.data;
  }

  const PAGES = {
    train: ['01 // TRAIN', '启动训练'], model: ['02 // MODEL', '模型设置'], params: ['03 // PARAMS', '高级参数'],
    crawl: ['04 // CRAWL', '图片爬取'], tag: ['05 // CAPTION', '标签反推'], editor: ['06 // EDITOR', '标签编辑'],
    sample: ['07 // PREVIEW', '训练中测试'], settings: ['08 // SYSTEM', '训练器设置'],
  };

  const App = {
    $, $$, esc, h, api, pages: {}, state: {}, draft: {}, page: 'train',

    // ---------------- 启动 ----------------
    async boot() {
      try {
        this.state = await api('boot');
      } catch (e) {
        document.body.innerHTML = `<div style="padding:40px;font-size:15px">训练器后台没有响应：${esc(e.message)}<br>请重新运行「启动LoRA训练器.bat」。</div>`;
        return;
      }
      this.applyTheme(this.state.settings.theme);
      $('#app-version').textContent = 'V' + (this.state.version.version || '0.1.0');
      this.draft = this.state.draft || {};
      this.normalizeDraft();
      $$('.nav-item').forEach(b => b.onclick = () => this.go(b.dataset.page));
      // 参数说明的悬停提示（卡片有切角裁剪，所以提示框挂在 body 上）
      const tip = h('<div id="tip" hidden></div>'); document.body.append(tip);
      document.addEventListener('mouseover', e => {
        const t = e.target.closest && e.target.closest('[data-tip]');
        if (!t) { tip.hidden = true; return; }
        tip.textContent = t.dataset.tip; tip.hidden = false;
        const r = t.getBoundingClientRect();
        tip.style.left = Math.min(r.left, innerWidth - tip.offsetWidth - 12) + 'px';
        tip.style.top = (r.bottom + 6 + tip.offsetHeight > innerHeight ? r.top - tip.offsetHeight - 6 : r.bottom + 6) + 'px';
      });
      $('#global-status').onclick = () => this.go('train');
      $('#restart-btn').onclick = () => this.restart();
      Object.values(this.pages).forEach(p => p.init && p.init());
      const hash = location.hash.replace('#', '');
      this.go(PAGES[hash] ? hash : (this.draft.model_type ? 'train' : 'model'));
      this.tick();
      setInterval(() => this.tick(), 1500);
      setInterval(() => { const d = new Date(); $('#topbar-clock').textContent = d.toTimeString().slice(0, 8); }, 1000);
      if (!this.state.env.env) {
        this.toast('训练环境还没安装，训练和本地反推前需要先安装（约 4 GB）', 'warn', '去安装', () => this.go('settings'), 12000);
      }
    },

    applyTheme(t) { document.documentElement.dataset.theme = t === 'dark' ? 'dark' : 'light'; },

    go(page) {
      if (!PAGES[page]) return;
      const prev = this.pages[this.page];
      if (prev && prev.hide) prev.hide();
      this.page = page;
      $$('.nav-item').forEach(b => b.classList.toggle('active', b.dataset.page === page));
      $$('.page').forEach(p => p.classList.toggle('active', p.id === 'page-' + page));
      $('#page-code').textContent = PAGES[page][0];
      $('#page-title').textContent = PAGES[page][1];
      history.replaceState(null, '', '#' + page);
      const p = this.pages[page];
      if (p && p.show) p.show();
    },

    // ---------------- 当前配置（草稿） ----------------
    spec(id) { return (this.state.specs || []).find(s => s.id === (id || this.draft.model_type)); },
    normalizeDraft() {
      const d = this.draft;
      d.by_model = d.by_model || {};
      d.output_dir = d.output_dir || this.state.settings.output_dir;
      if (d.subdir_per_task === undefined) d.subdir_per_task = true;
      d.tag = d.tag || {};
      if (d.model_type && !this.spec(d.model_type)) d.model_type = '';
      if (d.model_type) this.cur();
    },
    cur() {
      const d = this.draft, spec = this.spec();
      if (!spec) return null;
      const c = d.by_model[spec.id] = d.by_model[spec.id] || {};
      c.models = c.models || {};
      c.params = c.params || {};
      if (!c.sample) {
        // 默认用底模的标准正向 / 负向提示词、正常步数出图（不用 Turbo 加速模型）；auto_default：还没改过，换主模型时跟着换成对应系列的标准写法
        c.sample = { enabled: true, follow_save: true, every_n_steps: 200, every_n_epochs: 0, at_first: false, use_turbo: false,
          auto_default: true, prompts: [Object.assign({}, spec.sample_defaults)] };
      }
      return c;
    },
    defaults(spec) { const o = {}; (spec || this.spec()).params.forEach(f => { if (f.default !== null && f.default !== undefined) o[f.key] = f.default; }); return o; },
    params() { const spec = this.spec(); if (!spec) return {}; return Object.assign(this.defaults(spec), this.cur().params); },
    setParam(k, v) { const c = this.cur(); if (!c) return; c.params[k] = v; this.saveDraft(); this.emit('params'); if (k === 'dataset_dir') this.syncDataset(v, 'params'); },
    // 训练集文件夹同步（训练器设置里的开关）：爬图保存位置 / 反推 / 标签编辑 / 高级参数，改一处其它都跟着变
    syncDataset(dir, from) {
      dir = (dir || '').trim();
      if (!dir || this.state.settings.sync_dataset_dir === false) return;
      const d = this.draft;
      if (from !== 'crawl' && d.crawl) d.crawl.out_dir = dir;
      if (from !== 'tag') { d.tag = d.tag || {}; d.tag.dir = dir; }
      if (from !== 'editor') d.editor_dir = dir;
      if (from !== 'params') {
        Object.values(d.by_model || {}).forEach(c => { c.params = c.params || {}; c.params.dataset_dir = dir; });
        const c = this.cur(); if (c) { c.params.dataset_dir = dir; this.emit('params'); }
      }
      this.saveDraft();
    },
    setModelType(id) {
      const old = this.draft.model_type ? this.params() : {};
      this.draft.model_type = id;
      const c = this.cur();
      // 换底模时沿用训练集和保存名称
      // 保存名称是默认的（mylora_旧模型_v1）就换成新底模的默认名，自己改过的才沿用
      if (/^mylora_.*_v1$|^my_lora$/.test(old.output_name || '')) delete old.output_name;
      ['dataset_dir', 'output_name'].forEach(k => { if (old[k] && c.params[k] === undefined) c.params[k] = old[k]; });
      this.saveDraft(true);
      this.emit('model_type');
    },
    _saveTimer: null,
    saveDraft(now) {
      clearTimeout(this._saveTimer);
      const f = () => api('draft.save', { draft: this.draft }).catch(e => this.toast('保存配置失败：' + e.message, 'bad'));
      if (now) f(); else this._saveTimer = setTimeout(f, 400);
    },
    taskDraft() {
      const c = this.cur();
      return { model_type: this.draft.model_type, models: c.models, params: this.params(), sample: c.sample,
        output_dir: this.draft.output_dir, subdir_per_task: this.draft.subdir_per_task };
    },
    _ls: {},
    on(ev, fn) { (this._ls[ev] = this._ls[ev] || []).push(fn); },
    emit(ev, d) { (this._ls[ev] || []).forEach(f => { try { f(d); } catch (e) { console.error(e); } }); },

    // ---------------- 轮询 ----------------
    async tick() {
      try {
        const [q, jobs, g] = await Promise.all([api('queue.list'), api('jobs.list'), api('gpu')]);
        this.queue = q; this.jobs = jobs; this.gpu = g;
        this.renderGpu(g); this.renderStatus(q); this.renderJobs(jobs); this.renderRestart(q, jobs);
        this.emit('tick', { q, jobs, g });
      } catch (e) { /* 后台暂时没响应，下次再试 */ }
    },
    // 后端代码改过（更新 / 手动改）还没重启：顶部提示 + 重启按钮；训练中 / 有后台作业时按钮不能点
    renderRestart(q, jobs) {
      const bar = $('#restart-bar'); bar.hidden = !q.restart || this._restarting;
      if (bar.hidden) return;
      const busy = q.tasks.some(t => t.status === 'running' || t.status === 'checking') ? '正在训练' : (jobs.some(j => j.status === 'running') ? '有后台作业在进行' : '');
      const b = $('#restart-btn'); b.disabled = !!busy; b.title = busy ? busy + '，结束后再重启' : '';
    },
    async restart() {
      try { await api('app.restart'); } catch (e) { this.toast(e.message, 'bad'); return; }
      this._restarting = true; $('#restart-bar').hidden = true;
      this.toast('正在重启训练器…', '', null, null, 30000);
      await new Promise(r => setTimeout(r, 2500));
      for (let i = 0; i < 60; i++) {   // 等新的训练器起来再刷新页面
        try { const r = await fetch('/api/queue.list', { method: 'POST', body: '{}' }); if (r.ok) { location.reload(); return; } } catch (e) { /* 还没起来 */ }
        await new Promise(r => setTimeout(r, 1000));
      }
      this.toast('训练器没有自动起来，请手动双击「启动LoRA训练器.bat」', 'bad', null, null, 30000);
    },
    renderGpu(g) {
      const el = $('#gpu-mon');
      if (!g || !g.name) { el.innerHTML = '<div class="gm-name">未检测到 NVIDIA 显卡</div>'; return; }
      const pct = g.total_gb ? Math.round(g.used_gb / g.total_gb * 100) : 0;
      el.innerHTML = `<div class="gm-name">${esc(g.name.replace('NVIDIA GeForce ', ''))}</div>
        <div class="gm-row"><span>显存</span><span>${g.used_gb} / ${g.total_gb} GB</span></div><div class="gm-bar"><i style="width:${pct}%"></i></div>
        <div class="gm-row"><span>占用</span><span>${Math.round(g.util)}%</span></div><div class="gm-bar"><i style="width:${g.util}%"></i></div>
        <div class="gm-row"><span>温度</span><span>${Math.round(g.temp)}°C</span></div>`;
    },
    renderStatus(q) {
      const run = q.tasks.find(t => t.status === 'running' || t.status === 'checking');
      const waiting = q.tasks.filter(t => t.status === 'queued').length;
      const alert = q.tasks.filter(t => t.status === 'failed' || t.status === 'blocked').length;
      let st = 'idle', txt = '队列空闲';
      if (run) {
        st = 'running';
        const p = run.progress || {};
        txt = p.total ? `训练中 ${Math.round(p.step / p.total * 100)}%` : (run.message || '训练中');
        if (waiting) txt += ` · 排队 ${waiting}`;
      } else if (alert) { st = 'error'; txt = `${alert} 个任务需要处理`; }
      else if (waiting) { txt = `排队 ${waiting} 个（未开始）`; }
      for (const id of ['#global-status', '#train-status']) {
        const el = $(id); if (!el) continue;
        el.dataset.state = st; el.querySelector('span:last-child').textContent = txt;
      }
      const navTrain = $('.nav-item[data-page="train"]');
      let dot = navTrain.querySelector('.nav-dot');
      if (alert && !dot) navTrain.append(h('<i class="nav-dot"></i>'));
      if (!alert && dot) dot.remove();
      // 任务刚失败 / 被拦下：弹提示
      this._seen = this._seen || {};
      for (const t of q.tasks) {
        const key = t.id + ':' + t.status;
        if ((t.status === 'failed' || t.status === 'blocked') && this._seen[t.id] && this._seen[t.id] !== key) {
          this.toast(`「${t.name}」${t.status === 'failed' ? '训练失败' : '需要确认'}：${t.message || ''}`, 'bad', '查看并处理', () => this.showTaskIssue(t.id), 15000);
        }
        if (t.status === 'done' && this._seen[t.id] && this._seen[t.id] !== key) this.toast(`「${t.name}」训练完成`, 'ok', '打开输出', () => api('fs.open', { path: t.output_dir }));
        this._seen[t.id] = key;
      }
    },
    renderJobs(jobs) {
      const dock = $('#jobdock');
      const show = jobs.filter(j => j.status === 'running' || (j.finished && Date.now() / 1000 - j.finished < 8));
      dock.innerHTML = '';
      show.slice(0, 4).forEach(j => {
        const st = { running: '', done: '完成', failed: '失败', stopped: '已停止' }[j.status];
        const el = h(`<div class="jd"><div class="jd-t"><b>${esc(j.title)}</b><span>${j.progress != null ? Math.round(j.progress * 100) + '%' : ''} ${st}</span></div>
          <div class="prog ${j.progress == null && j.status === 'running' ? 'indet' : ''}"><i style="width:${(j.progress || 0) * 100}%"></i></div>
          <div class="jd-s">${esc(j.error || j.progress_text || '')}</div></div>`);
        el.onclick = () => { this.go('settings'); this.pages.settings.showJob(j.id); };
        dock.append(el);
      });
    },

    // ---------------- 提示 / 弹窗 ----------------
    toast(msg, kind = '', btn = null, fn = null, ms = 4500) {
      const el = h(`<div class="toast ${kind}"><span>${esc(msg)}</span></div>`);
      if (btn) { const b = h(`<button class="btn btn-sm">${esc(btn)}</button>`); b.onclick = () => { fn && fn(); el.remove(); }; el.append(b); }
      el.addEventListener('click', e => { if (e.target === el || e.target.tagName === 'SPAN') el.remove(); });
      $('#toasts').append(el);
      setTimeout(() => el.remove(), ms);
    },
    modal({ title, body, buttons = [], wide = false, bad = false, onclose }) {
      const m = h(`<div class="modal-mask"><div class="modal ${wide ? 'wide' : ''}"><div class="modal-h ${bad ? 'bad' : ''}"><i class="mk"></i><b>${esc(title)}</b><button class="icon-btn x">✕</button></div>
        <div class="modal-b"></div><div class="modal-f"></div></div></div>`);
      const b = m.querySelector('.modal-b');
      if (typeof body === 'string') b.innerHTML = body; else if (body) b.append(body);
      const close = () => { m.remove(); onclose && onclose(); };
      m.querySelector('.x').onclick = close;
      m.addEventListener('mousedown', e => { if (e.target === m) close(); });
      buttons.forEach(bt => {
        const el = h(`<button class="btn ${bt.cls || ''}">${esc(bt.label)}</button>`);
        el.onclick = async () => { if (bt.fn) { const r = await bt.fn(m); if (r === false) return; } close(); };
        m.querySelector('.modal-f').append(el);
      });
      document.body.append(m);
      return { el: m, close, body: b };
    },
    confirm(title, html, okLabel = '确定', danger = false) {
      return new Promise(res => {
        this.modal({ title, body: html, onclose: () => res(false), buttons: [
          { label: '取消', fn: () => { res(false); } },
          { label: okLabel, cls: danger ? 'btn-danger' : 'btn-primary', fn: () => { res(true); } },
        ] });
      });
    },
    prompt(title, label, value = '') {
      return new Promise(res => {
        const body = h(`<div><div style="margin-bottom:8px">${esc(label)}</div><input type="text" style="width:100%"></div>`);
        const inp = body.querySelector('input'); inp.value = value;
        let done = false;
        const m = this.modal({ title, body, onclose: () => { if (!done) res(null); }, buttons: [
          { label: '取消' }, { label: '确定', cls: 'btn-primary', fn: () => { done = true; res(inp.value); } }] });
        inp.focus(); inp.select();
        inp.onkeydown = e => { if (e.key === 'Enter') { done = true; res(inp.value); m.close(); } };
      });
    },
    lightbox(list, i = 0) {
      const lb = h('<div class="lightbox"><img><div class="lb-cap"></div></div>');
      const show = () => { lb.querySelector('img').src = '/file?path=' + encodeURIComponent(list[i].file); lb.querySelector('.lb-cap').textContent = list[i].cap || ''; };
      const key = e => {
        if (e.key === 'Escape') close();
        if (e.key === 'ArrowRight') { i = (i + 1) % list.length; show(); }
        if (e.key === 'ArrowLeft') { i = (i - 1 + list.length) % list.length; show(); }
      };
      const close = () => { lb.remove(); document.removeEventListener('keydown', key); };
      lb.onclick = close;
      document.addEventListener('keydown', key);
      document.body.append(lb); show();
    },

    // 文件 / 文件夹选择（在训练器后台列目录，浏览器本身拿不到本地路径）
    pick({ title = '选择文件夹', start = '', exts = null, dir = true } = {}) {
      return new Promise(res => {
        const body = h(`<div><div class="picker-path"><button class="btn btn-sm" data-up>上一级</button><input type="text" style="flex:1"><button class="btn btn-sm" data-goto>转到</button></div>
          <div class="picker-list"></div><div class="hint" data-info></div></div>`);
        const inp = body.querySelector('input'), list = body.querySelector('.picker-list'), info = body.querySelector('[data-info]');
        let cur = '', parent = '', chosen = null, done = false;
        const load = async p => {
          try {
            const d = await api('fs.list', { path: p, exts });
            cur = d.path; parent = d.parent; inp.value = cur; chosen = dir ? cur : null;
            list.innerHTML = '';
            d.dirs.forEach(x => {
              const el = h(`<div><span class="pk-i">▸</span>${esc(x.name)}</div>`);
              el.ondblclick = () => load(x.path);
              el.onclick = () => { $$('div', list).forEach(z => z.classList.remove('on')); el.classList.add('on'); if (dir) chosen = x.path; };
              list.append(el);
            });
            (d.files || []).forEach(x => {
              const el = h(`<div><span class="pk-i">◇</span>${esc(x.name)}<span class="muted" style="margin-left:auto">${(x.size / 1e9).toFixed(2)} GB</span></div>`);
              el.onclick = () => { $$('div', list).forEach(z => z.classList.remove('on')); el.classList.add('on'); chosen = x.path; };
              el.ondblclick = () => { done = true; res(x.path); m.close(); };
              list.append(el);
            });
            info.textContent = cur ? (dir ? `当前文件夹有 ${d.images} 张图片；双击进入子文件夹` : '双击文件直接选择') : '选择一个磁盘';
          } catch (e) { info.textContent = e.message; }
        };
        body.querySelector('[data-up]').onclick = () => load(parent);
        body.querySelector('[data-goto]').onclick = () => load(inp.value);
        inp.onkeydown = e => { if (e.key === 'Enter') load(inp.value); };
        const m = this.modal({ title, body, wide: true, onclose: () => { if (!done) res(null); }, buttons: [
          { label: '取消' },
          { label: dir ? '选择这个文件夹' : '选择文件', cls: 'btn-primary', fn: () => { if (!chosen) { info.textContent = dir ? '请先进入或选中一个文件夹' : '请先选中一个文件'; return false; } done = true; res(chosen); } }] });
        let st = start || '';
        if (st && !dir) st = st.replace(/[\\/][^\\/]*$/, '');
        load(st);
      });
    },

    // ---------------- 报错 / 体检结果 + 自动修复（必须用户确认） ----------------
    async showTaskIssue(tid) {
      const t = await api('queue.task', { id: tid });
      if (!t) return;
      if (t.status === 'blocked') return this.showIssues(t.issues || [], tid, t.name);
      if (t.status === 'failed' && t.error) return this.showError(t.error, tid, t.name);
      this.go('train');
    },
    showError(err, tid, name) {
      const body = h(`<div>
        <div class="issue"><b>${esc(err.title)}</b>${esc(err.reason)}</div>
        <div data-fixes></div>
        <details style="margin-top:12px"><summary class="muted" style="cursor:pointer">原始报错（最后几十行日志）</summary><div class="code-box" style="max-height:260px;margin-top:6px">${esc(err.detail || '')}</div></details>
      </div>`);
      this._renderFixes(body.querySelector('[data-fixes]'), err.fixes || [], tid, () => m.close(), err.code);
      if (!(err.fixes || []).length) body.querySelector('[data-fixes]').innerHTML = '<div class="hint">这个问题没法自动解决，请按上面的说明处理后点「重试」。</div>';
      const m = this.modal({ title: `训练失败${name ? '：' + name : ''}`, body, wide: true, bad: true, buttons: [
        { label: '关闭' },
        ...(tid ? [{ label: '不修改，直接重试', fn: async () => { await api('queue.requeue', { id: tid, start: true }); this.toast('已重新加入队列'); } }] : []),
      ] });
    },
    showIssues(issues, tid, name) {
      const body = h('<div></div>');
      issues.forEach(it => {
        const box = h(`<div style="margin-bottom:16px"><div class="issue ${it.level === 'warn' ? 'warn' : ''}"><b>${esc(it.title)}</b>${esc(it.reason)}</div><div data-fixes></div></div>`);
        if (it.files && it.files.length) box.append(h(`<details><summary class="muted" style="cursor:pointer">查看文件（${it.files.length}）</summary><div class="fix-list">${it.files.map(f => `<div>${esc(f)}</div>`).join('')}</div></details>`));
        body.append(box);
        if (it.fixes.length) this._renderFixes(box.querySelector('[data-fixes]'), it.fixes, tid, () => m.close(), it.code);
        else box.querySelector('[data-fixes]').innerHTML = '<div class="hint">需要你手动处理。</div>';
      });
      const m = this.modal({ title: tid ? `训练前检查发现问题${name ? '：' + name : ''}` : '训练集检查结果', body, wide: true, bad: true, buttons: [
        { label: tid ? '关闭（任务保持暂停）' : '关闭' },
        ...(tid ? [{ label: '全部忽略，直接训练', cls: 'btn-danger', fn: async () => {
          await api('fix.apply', { task: tid, fix: 'ignore', data: { codes: issues.map(i => i.code) } });
          this.toast('已忽略检查结果，开始训练');
        } }] : []),
      ] });
    },
    // 每个修复动作：先展示「会做什么」（要删的文件 / 要结束的进程），用户勾选确认后才执行
    _renderFixes(el, fixes, tid, closeAll, code) {
      fixes.forEach(fx => {
        const box = h(`<div class="fix-opt"><div style="flex:1"><b>${esc(fx.label)}</b><div class="fd">${esc(fx.desc || '')}</div><div data-x></div>
          <div class="row" style="margin-top:8px"><button class="btn btn-sm ${fx.danger ? 'btn-danger' : 'btn-primary'}" data-go>确认执行</button></div></div></div>`);
        const x = box.querySelector('[data-x]');
        let getData = () => Object.assign({}, fx.data || {}, fx.id === 'ignore' ? { codes: [code] } : {});
        if (fx.id === 'kill_gpu') {
          const procs = (fx.data.procs || []);
          x.innerHTML = `<div class="fix-list">${procs.map(p => `<label><input type="checkbox" value="${p.pid}" ${p.suggest ? 'checked' : ''}>
            <b>${esc(p.label || p.name)}</b> <span class="muted">${esc(p.name)} · PID ${p.pid}${p.used_mb ? ' · ' + p.used_mb + ' MB' : ''}</span>
            <span class="muted" style="margin-left:auto;max-width:330px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(p.cmd)}">${esc(p.cmd)}</span></label>`).join('') || '<div>没有找到可以结束的进程</div>'}</div>
            <div class="hint">默认只勾选 WebUI / ComfyUI / Python 程序；系统进程和浏览器不会出现在这里。</div>`;
          getData = () => ({ procs: procs.filter(p => x.querySelector(`input[value="${p.pid}"]`).checked) });
        }
        if (fx.id === 'clean_dataset') {
          const files = fx.data.files || [];
          x.innerHTML = `<div class="fix-list">${files.map(f => `<div><span class="badge ${f.what.includes('损坏') ? 'bad' : ''}">${esc(f.what)}</span>${esc(f.path)}</div>`).join('')}</div>`;
        }
        box.querySelector('[data-go]').onclick = async (ev) => {
          const data = getData();
          if (fx.id === 'kill_gpu' && !data.procs.length) { this.toast('没有勾选任何进程', 'bad'); return; }
          if (fx.danger && !(await this.confirm('再次确认', `<b>${esc(fx.label)}</b><br>${esc(fx.desc || '')}<br><br>确定要执行吗？`, '确定执行', true))) return;
          ev.target.disabled = true;
          try {
            const r = await api('fix.apply', { task: tid, fix: fx.id, data });
            this.toast(r.message || '已执行', r.ok ? 'ok' : 'bad');
            if (r.ok) closeAll();
          } catch (e) { this.toast(e.message, 'bad'); ev.target.disabled = false; }
        };
        el.append(box);
      });
    },

    fmtSize(b) { return b > 1e9 ? (b / 1e9).toFixed(2) + ' GB' : (b / 1e6).toFixed(0) + ' MB'; },
    fmtTime(ts) { if (!ts) return ''; const d = new Date(ts * 1000); return d.toLocaleString('zh-CN', { hour12: false }).replace(/\//g, '-'); },
    base(p) { return String(p || '').split(/[\\/]/).pop(); },
  };
  window.App = App;
})();

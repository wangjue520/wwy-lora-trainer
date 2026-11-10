/* 报错诊断面板（后端 app/diag_ai.py：本地规则 + AI；AI 配置见 app/aiconf.py，和 WWY 启动器同步）
   入口：顶栏「诊断」（有新问题时带角标）、训练失败弹窗里的「诊断面板」、Ctrl+Shift+D。右侧抽屉，不挡操作。
   照启动器 web/js/diagnose.js：规则结论 + 修复按钮、AI 分析（执行修复按钮）、AI 接管、底部常驻输入框（追问 / 记住）。
   所有修复执行前都要用户确认。 */
(function () {
  const { $, esc, api } = App;
  let target = null;      // {type: 'task', id} / {type: 'job', id, kind} / {type: 'trainer'}
  let report = null;
  let job = '', aiSteps = [], aiResult = null, aiError = '', aiFix = false;
  let tk = null;          // AI 接管 {job, id, steps: [], done: null}
  let since = 0, tkSince = 0, pollTimer = null;
  let memo = '';
  const SEEN_KEY = 'wwy-diag-seen';
  let known = null;       // 已经提示过的问题（第一次轮询的不弹提示）

  const sameTarget = (a, b) => a && b && a.type === b.type && (a.type === 'trainer' || a.id === b.id);
  const drawer = () => $('#diag-drawer');
  const seen = () => { try { return new Set(JSON.parse(localStorage.getItem(SEEN_KEY) || '[]')); } catch (e) { return new Set(); } };
  const saveSeen = s => localStorage.setItem(SEEN_KEY, JSON.stringify([...s].slice(-300)));

  /* ---------- 角标：有新问题就标出来（规则认不出的也算） ---------- */
  let problems = [];
  async function refreshProblems() {
    let r;
    try { r = await api('diag.problems'); } catch (e) { return; }
    problems = r.problems || [];
    const s = seen();
    if (!drawer().hidden) {
      if (problems.some(p => !s.has(p.key))) fillTargets();   // 面板开着时又出了新问题：下拉框里加上
      problems.forEach(p => s.add(p.key)); saveSeen(s);
    }
    const fresh = problems.filter(p => !s.has(p.key));
    const b = $('#diag-badge');
    b.hidden = !fresh.length; b.textContent = fresh.length;
    if (known) {
      const nw = fresh.filter(p => !known.has(p.key));
      if (nw.length && drawer().hidden) App.toast(nw[0].name + ' 出错了', 'bad', '诊断', () => open(nw[0].target), 10000);
    }
    known = new Set(problems.map(p => p.key));
  }

  /* ---------- 抽屉 ---------- */
  async function fillTargets() {
    let info = null;
    try { info = await api('diag.info'); } catch (e) { /* 拿不到就只放当前对象 */ }
    const list = (info && info.targets) || [target];
    if (!list.some(t => sameTarget(t, target))) list.unshift(Object.assign({ name: report ? report.name : '' }, target));
    const sel = $('#dg-target');
    sel.innerHTML = list.map((t, i) => `<option value="${i}">${esc(t.name || '')}</option>`).join('');
    sel._targets = list;
    sel.value = String(Math.max(0, list.findIndex(t => sameTarget(t, target))));
  }

  async function open(t) {
    if (t) target = t;
    if (!target) target = problems.length ? problems[0].target : { type: 'trainer' };
    drawer().hidden = false;
    document.body.classList.add('diag-open');
    const s = seen(); problems.forEach(p => s.add(p.key)); saveSeen(s); $('#diag-badge').hidden = true;
    await run();
    fillTargets();
  }
  function close() { drawer().hidden = true; document.body.classList.remove('diag-open'); }

  async function run() {
    report = null;
    if (!job) { aiSteps = []; aiResult = null; aiError = ''; }
    $('#dg-findings').innerHTML = '<div class="dg-empty">正在分析日志…</div>';
    try { report = await api('diag.run', { target }); } catch (e) { report = { ok: false, error: e.message }; }
    if (report && report.takeover && !(tk && tk.job === report.takeover)) {
      tk = { job: report.takeover, id: target.id, steps: [], done: null }; tkSince = 0; startPoll();
    }
    render();
  }

  /* ---------- 渲染 ---------- */
  function render() {
    const box = $('#dg-findings');
    if (!report || report.ok === false) {
      box.innerHTML = `<div class="dg-empty dg-bad">${esc((report && report.error) || '诊断失败')}</div>`;
      renderAi(); return;
    }
    $('#dg-sub').textContent = report.lines ? `看了 ${report.lines} 行日志` : '';
    renderMemo(report.note || '');
    const fs = report.findings || [];
    box.innerHTML = fs.length ? '' : `<div class="dg-empty">${target.type === 'trainer' ? '训练器本身没有发现报错。' : '这里没有发现报错。'}可以让 AI 看看，或者在下面写清楚遇到的问题。</div>`;
    fs.forEach(f => box.append(findingCard(f)));
    renderAi();
  }

  function findingCard(f) {
    const el = App.h(`<div class="dg-card" data-level="${esc(f.level || 'error')}"><div class="dg-card-title">${esc(f.title)}</div>
      ${f.text ? `<div class="dg-card-text">${esc(f.text)}</div>` : ''}${f.evidence ? `<div class="dg-evidence">${esc(f.evidence)}</div>` : ''}
      <div class="dg-actions"></div></div>`);
    const acts = $('.dg-actions', el);
    (f.actions || []).forEach(a => acts.append(actionEl(a)));
    if (!(f.actions || []).length) acts.remove();
    return el;
  }

  function actionEl(a) {
    const wrap = App.h('<span class="dg-act"></span>');
    const btn = App.h(`<button class="btn btn-sm ${a.risk === 'danger' ? 'btn-danger' : a.risk === 'ui' ? '' : 'btn-primary'}">${esc(a.label)}</button>`);
    if (a.desc) btn.dataset.tip = a.desc;
    btn.onclick = () => doAction(a, btn, wrap);
    wrap.append(btn);
    return wrap;
  }

  function undoBtn(token, wrap) {
    const u = App.h('<button class="btn btn-sm">撤销</button>');
    u.onclick = async () => {
      u.disabled = true;
      let r; try { r = await api('diag.undo', { token }); } catch (e) { r = { ok: false, error: e.message }; }
      if (r.ok) { App.toast(r.text || '已撤销', 'ok'); syncParams(r); wrap.innerHTML = '<span class="dg-applied dg-undone">已撤销</span>'; }
      else { App.toast(r.error || '撤销失败', 'bad'); u.disabled = false; }
    };
    return u;
  }

  // 修复改了任务参数：当前训练配置是同一个底模的话也跟着改（和训练失败弹窗里的修复一样）
  function syncParams(r) {
    if (r && r.params && r.model_type === App.draft.model_type && App.cur()) {
      Object.assign(App.cur().params, r.params); App.saveDraft(true); App.emit('params');
    }
  }

  function uiAction(a) {
    const p = a.params || {};
    if (a.aid === 'open_page') { close(); App.go(p.page); }
  }

  // 结束进程要让用户自己勾；其余动作确认一次
  async function confirmAction(a) {
    if (a.aid === 'kill_gpu') {
      const procs = (a.params && a.params.procs) || [];
      if (!procs.length) { App.toast('没有找到可以结束的进程', 'bad'); return null; }
      const body = App.h(`<div><div class="hint" style="margin:0 0 8px">勾选要结束的程序，未保存的内容会丢失：</div><div class="fix-list">${procs.map(p =>
        `<label><input type="checkbox" value="${p.pid}" ${p.suggest ? 'checked' : ''}><b>${esc(p.label || p.name)}</b> <span class="muted">${esc(p.name)} · PID ${p.pid}${p.used_mb ? ' · ' + p.used_mb + ' MB' : ''}</span></label>`).join('')}</div></div>`);
      const ok = await new Promise(res => App.modal({ title: '结束占用显存的程序', body, onclose: () => res(false), buttons: [
        { label: '取消', fn: () => res(false) }, { label: '结束并重试', cls: 'btn-danger', fn: () => res(true) }] }));
      if (!ok) return null;
      const chosen = procs.filter(p => body.querySelector(`input[value="${p.pid}"]`).checked);
      if (!chosen.length) { App.toast('没有勾选任何进程', 'bad'); return null; }
      return { procs: chosen };
    }
    const ok = await App.confirm('确认修复', `<b>${esc(a.label)}</b>${a.desc ? '<br>' + esc(a.desc) : ''}`, '执行', a.risk === 'danger');
    return ok ? (a.params || {}) : null;
  }

  async function exec(a, params) {
    let r;
    try { r = await api('diag.fix', { target, aid: a.aid, params, rule: !!a.rule }); } catch (e) { r = { ok: false, error: e.message }; }
    if (r && r.ui) { uiAction(Object.assign({}, a, { params: r.params })); r = { ok: true, text: '已打开' }; }
    if (r && r.ok) syncParams(r);
    return r || { ok: false, error: '执行失败' };
  }

  async function doAction(a, btn, wrap) {
    if (a.risk === 'ui') { uiAction(a); return; }
    const params = await confirmAction(a);
    if (!params) return;
    btn.disabled = true;
    const r = await exec(a, params);
    if (!r.ok) { btn.disabled = false; App.toast(r.error || '执行失败', 'bad', null, null, 6000); return; }
    App.toast(r.text || '已完成', 'ok');
    wrap.innerHTML = `<span class="dg-applied">✓ ${esc(r.text || '已完成')}</span>`;
    if (r.undo) wrap.append(undoBtn(r.undo, wrap));
  }

  /* ---------- AI ---------- */
  function renderAi() {
    const box = $('#dg-ai');
    const ready = report ? report.ai_ready : true;
    $('#dg-ai-model').textContent = report && report.ai_model ? ' · ' + report.ai_model : '';
    if (!ready && !job && !aiResult) {
      box.innerHTML = `<div class="dg-empty">还没设置 AI：到「训练器设置 → 报错诊断」选服务商、填自己的 API Key（在 WWY 启动器里填过的会自动同步过来）。</div>
        <div class="dg-actions"><button class="btn btn-sm btn-primary" id="dg-ai-setup">去设置</button></div>`;
      $('#dg-ai-setup').onclick = () => { close(); App.go('settings'); setTimeout(() => $('#st-diag-card').scrollIntoView({ behavior: 'smooth', block: 'start' }), 100); };
      renderTk(); return;
    }
    let html = '';
    if (job || aiSteps.length) html += `<ol class="dg-steps">${aiSteps.map(s => `<li>${esc(s)}</li>`).join('')}${job ? '<li class="dg-running">思考中…</li>' : ''}</ol>`;
    if (aiError) html += `<div class="dg-empty dg-bad">${esc(aiError)}</div>`;
    box.innerHTML = html;
    if (aiResult) box.append(aiCard(aiResult));
    const row = App.h('<div class="dg-actions"></div>');
    if (job) {
      row.innerHTML = '<button class="btn btn-sm" id="dg-ai-stop">停止</button>';
      $('#dg-ai-stop', row).onclick = () => api('diag.cancel', { job });
    } else {
      const has = report && (report.findings || []).length;
      const canTk = report && report.can_takeover && !(tk && !tk.done);
      row.innerHTML = `<button class="btn btn-sm btn-primary" id="dg-ai-go">${aiResult ? '让 AI 重新分析' : has ? '让 AI 再深入看看' : '让 AI 分析'}</button>
        ${canTk ? '<button class="btn btn-sm btn-primary" id="dg-tk-go">AI 接管修复</button>' : ''}
        <span class="hint">发给 AI 的日志已脱敏。「分析」只读不改，「接管」会直接动手修</span>`;
      $('#dg-ai-go', row).onclick = async () => { const t = await takeInput(); startAi({ note: t, said: t }); };
      const go = $('#dg-tk-go', row);
      if (go) go.onclick = startTakeover;
    }
    box.append(row);
    renderTk();
  }

  const runnable = res => (res.actions || []).filter(a => a.risk !== 'ui' && a.risk !== 'danger');

  function aiCard(res) {
    const conf = { high: '把握较大', medium: '有一定把握', low: '把握不大' }[res.confidence] || '';
    const el = App.h(`<div class="dg-card dg-ai-card"><div class="dg-card-title">${esc(res.summary || 'AI 诊断结果')}${conf ? `<em class="dg-conf">${conf}</em>` : ''}</div>
      ${res.cause ? `<div class="dg-card-text">${esc(res.cause)}</div>` : ''}
      ${(res.steps || []).length ? `<ol class="dg-todo">${res.steps.map(s => `<li>${esc(s)}</li>`).join('')}</ol>` : ''}
      <div class="dg-actions"></div>
      ${(res.dropped || []).length ? `<div class="hint">AI 还推荐了几个训练器做不了 / 不安全的操作，已忽略：${esc(res.dropped.join('；'))}</div>` : ''}</div>`);
    const acts = $('.dg-actions', el);
    if (res.execLog) {
      const ol = App.h('<ol class="dg-tk-steps"></ol>');
      res.execLog.forEach(x => {
        const li = App.h(`<li class="${x.ok ? 'dg-tk-act' : 'dg-tk-warn'}"><span class="dg-tk-ico">${x.ok ? '✓' : '!'}</span><span>${esc(x.label)} → ${esc(x.text)}</span></li>`);
        if (x.undo) { const w = App.h('<span class="dg-act"></span>'); w.append(undoBtn(x.undo, w)); li.append(w); }
        ol.append(li);
      });
      if (res.executing) ol.insertAdjacentHTML('beforeend', '<li class="dg-running">执行中…</li>');
      acts.replaceWith(ol);
      return el;
    }
    (res.actions || []).forEach(a => acts.append(actionEl(a)));
    const fix = App.h('<button class="btn btn-sm btn-primary dg-fix-btn">执行修复</button>');
    fix.dataset.tip = runnable(res).length ? '把上面的修复一次做完（会先确认）' : '让 AI 把方案变成能直接执行的操作，确认后执行';
    fix.onclick = () => execFix(res, false);
    acts.prepend(fix);
    return el;
  }

  // 执行修复：AI 给了动作就确认一次全做；没给就让 AI 把方案变成能执行的动作
  async function execFix(res, fromAi) {
    const acts = runnable(res);
    if (!acts.length) {
      if (fromAi) { App.toast('训练器没法替你做，照 AI 给的步骤手动处理', 'bad', null, null, 6000); return; }
      const t = await takeInput();
      startAi({ fix: true, followup: pickResult(res), note: t, said: '执行修复' + (t ? '：' + t : '') });
      return;
    }
    const ok = await App.confirm('执行修复', '将依次执行：<br>' + acts.map((a, i) => `${i + 1}. ${esc(a.label)}`).join('<br>') +
      '<br><br>改过的参数可以撤销。', '执行');
    if (!ok) return;
    res.execLog = []; res.executing = true; renderAi();
    for (const a of acts) {
      const r = await exec(a, a.params || {});
      res.execLog.push({ label: a.label, ok: !!r.ok, text: (r.ok ? r.text : r.error) || '', undo: r.undo || '' });
      renderAi();
    }
    res.executing = false; renderAi();
    const bad = res.execLog.filter(x => !x.ok).length;
    App.toast(bad ? `执行完了，有 ${bad} 项没成功（原因见诊断面板）` : '修复已执行', bad ? 'bad' : 'ok', null, null, 6000);
  }

  function pickResult(res) {
    return res ? { summary: res.summary, cause: res.cause, steps: res.steps || [], actions: (res.actions || []).map(a => ({ aid: a.aid, params: a.params })) } : null;
  }

  async function startAi(opts) {
    if (job) { App.toast('AI 还在分析，等它说完或点「停止」', 'bad'); return; }
    aiSteps = opts.said ? ['你：' + opts.said] : []; aiResult = null; aiError = ''; aiFix = !!opts.fix;
    let r;
    try { r = await api('diag.ai', { target, note: opts.note || '', followup: opts.followup || null, fix: !!opts.fix }); }
    catch (e) { aiError = e.message; renderAi(); return; }
    job = r.job; since = 0; renderAi(); startPoll();
  }

  /* ---------- AI 接管 ---------- */
  const TK_ICON = { round: '▶', act: '✓', result: '◆', warn: '!', ai: '·', info: '·' };
  function renderTk() {
    const box = $('#dg-tk');
    if (!tk || !target || target.type !== 'task' || tk.id !== target.id) { box.innerHTML = ''; return; }
    const running = !tk.done;
    box.innerHTML = `<div class="dg-section">AI 接管${running ? ' · 进行中' : ''}</div><ol class="dg-tk-steps"></ol><div class="dg-actions"></div>`;
    const ol = $('.dg-tk-steps', box);
    tk.steps.forEach(s => {
      const li = App.h(`<li class="dg-tk-${esc(s.kind || 'info')}"><span class="dg-tk-ico">${TK_ICON[s.kind] || '·'}</span><span>${esc(s.text)}</span></li>`);
      if (s.undo) { const w = App.h('<span class="dg-act"></span>'); w.append(undoBtn(s.undo, w)); li.append(w); }
      ol.append(li);
    });
    if (running) ol.insertAdjacentHTML('beforeend', '<li class="dg-running">处理中…</li>');
    const row = $('.dg-actions', box);
    if (running) {
      row.innerHTML = '<button class="btn btn-sm" id="dg-tk-stop">停止接管</button>';
      $('#dg-tk-stop', row).onclick = e => { e.target.disabled = true; api('diag.cancel', { job: tk.job }); };
    } else {
      const card = App.h(`<div class="dg-card dg-ai-card" data-level="${tk.done.ok ? 'ok' : 'error'}"><div class="dg-card-title">${tk.done.ok ? '✓ ' : ''}${esc(tk.done.text)}</div>
        ${(tk.done.steps || []).length ? `<ol class="dg-todo">${tk.done.steps.map(s => `<li>${esc(s)}</li>`).join('')}</ol>` : ''}</div>`);
      row.replaceWith(card);
    }
  }

  async function startTakeover() {
    const ok = await App.confirm('让 AI 接管修复',
      'AI 会自己决定并直接执行修复（改训练参数、清缓存、装依赖、重下坏掉的模型、切换网络），每轮修完把任务重新排队训练，' +
      '开始正常出步数就算修好；还失败就接着修，最多 5 轮。<br><br>结束别的程序不会自动做；改过的参数可以在步骤里撤销；随时可以停止。', '开始接管');
    if (!ok) return;
    const note = await takeInput();
    let r;
    try { r = await api('diag.takeover', { target, note }); } catch (e) { App.toast(e.message, 'bad', null, null, 6000); return; }
    tk = { job: r.job, id: target.id, steps: [], done: null }; tkSince = 0;
    renderAi(); startPoll();
  }

  /* ---------- 轮询 AI / 接管的进度 ---------- */
  function startPoll() {
    if (pollTimer) return;
    pollTimer = setInterval(async () => {
      let busy = false;
      if (job) {
        busy = true;
        try {
          const r = await api('diag.poll', { job, since });
          aiSteps.push(...(r.steps || [])); since = r.n || since;
          if (r.done) {
            job = '';
            if (r.error) aiError = r.error; else { aiResult = r.result; if (aiFix && aiResult && !drawer().hidden) execFix(aiResult, true); }
          }
          if (!drawer().hidden) renderAi();
        } catch (e) { /* 下次再试 */ }
      }
      if (tk && !tk.done) {
        busy = true;
        try {
          const r = await api('diag.poll', { job: tk.job, tk_since: tkSince });
          tk.steps.push(...(r.tk_steps || [])); tkSince = r.tk_n || tkSince;
          if (r.done) {
            tk.done = r.tk_done || { ok: false, text: r.error || '接管结束', steps: [] };
            App.toast('AI 接管结束：' + tk.done.text, tk.done.ok ? 'ok' : 'bad', null, null, 8000);
            if (!drawer().hidden) run();
          }
          if (!drawer().hidden) renderTk();
        } catch (e) { /* 下次再试 */ }
      }
      if (!busy) { clearInterval(pollTimer); pollTimer = null; }
    }, 1000);
  }

  /* ---------- 底部输入框：随时跟 AI 说话；勾「记住」的存成长期说明 ---------- */
  function renderMemo(text) {
    memo = text || '';
    $('#dg-memo').hidden = !memo;
    $('#dg-memo-text').textContent = memo;
  }
  async function saveMemo(text) {
    try { const r = await api('diag.note_set', { target, text }); renderMemo(r.note); } catch (e) { App.toast(e.message, 'bad'); }
  }
  async function takeInput() {
    const inp = $('#dg-input'), text = (inp.value || '').trim(), rem = $('#dg-remember');
    if (text && rem.checked) { await saveMemo(memo ? memo + '\n' + text : text); rem.checked = false; }
    inp.value = '';
    return text;
  }
  async function send() {
    if (job) { App.toast('AI 还在分析，等它说完或点「停止」', 'bad'); return; }
    const text = await takeInput();
    if (!text && aiResult) { App.toast('先在输入框里写点什么', 'bad'); return; }
    startAi({ note: text, said: text, followup: aiResult ? pickResult(aiResult) : null });
  }

  function reportText() {
    const out = [`# WWY LoRA 训练器诊断报告（${report ? report.name : ''}）`];
    ((report && report.findings) || []).forEach(f => { out.push(`\n## ${f.title}\n${f.text || ''}`); if (f.evidence) out.push('证据：' + f.evidence); });
    if (aiResult) { out.push(`\n## AI：${aiResult.summary}\n${aiResult.cause || ''}`); (aiResult.steps || []).forEach((s, i) => out.push(`${i + 1}. ${s}`)); }
    return out.join('\n');
  }

  /* ---------- 设置页的「报错诊断」卡片：和启动器共用一套 ---------- */
  let providers = [];
  function refreshProviderHints() {
    const pid = $('#dg-provider').value, p = providers.find(x => x.id === pid) || {};
    $('#dg-model').placeholder = p.model ? `留空用默认：${p.model}` : '必填，例如 deepseek-v4-flash';
    $('#dg-base').placeholder = p.base ? `留空用默认：${p.base}` : '必填，例如 https://api.xxx.com/v1';
    $('#dg-format').hidden = $('#dg-format-label').hidden = pid !== 'custom';
    $('#dg-key-link').hidden = !p.key_url;
  }
  async function loadSettings() {
    let c;
    try { c = await api('diag.config'); } catch (e) { return; }
    providers = c.providers || [];
    const sel = $('#dg-provider');
    sel.innerHTML = providers.map(p => `<option value="${esc(p.id)}">${esc(p.label)}</option>`).join('');
    sel.value = c.provider || 'deepseek';
    $('#dg-key').value = c.api_key || ''; $('#dg-model').value = c.model || ''; $('#dg-base').value = c.base_url || '';
    $('#dg-format').value = c.format || 'openai';
    $('#dg-sync-hint').textContent = (c.synced_from === 'launcher' ? '当前设置来自 WWY 启动器。' : '') + '和 WWY 启动器自动同步：' + c.shared_file;
    refreshProviderHints();
  }
  const formValues = () => ({ diag_llm_provider: $('#dg-provider').value, diag_llm_api_key: $('#dg-key').value.trim(),
    diag_llm_model: $('#dg-model').value.trim(), diag_llm_base_url: $('#dg-base').value.trim(), diag_llm_format: $('#dg-format').value });
  function initSettings() {
    const save = async () => { try { await api('diag.config_save', { values: formValues() }); App.toast('已保存，WWY 启动器也会用这套设置', 'ok'); } catch (e) { App.toast(e.message, 'bad'); } };
    $('#dg-provider').onchange = () => { refreshProviderHints(); save(); };
    ['#dg-key', '#dg-model', '#dg-base', '#dg-format'].forEach(s => $(s).onchange = save);
    $('#dg-key-link').onclick = () => { const p = providers.find(x => x.id === $('#dg-provider').value); if (p && p.key_url) window.open(p.key_url, '_blank'); };
    $('#dg-test').onclick = async () => {
      const btn = $('#dg-test'), out = $('#dg-test-out');
      btn.disabled = true; out.textContent = '正在测试…'; out.className = 'hint';
      try { const r = await api('diag.test', { values: formValues() }); out.textContent = r.text; out.className = 'hint ' + (r.ok ? 'ok' : 'bad'); }
      catch (e) { out.textContent = e.message; out.className = 'hint bad'; }
      btn.disabled = false;
    };
  }

  /* ---------- 界面脚本报错也算训练器的问题 ---------- */
  let lastReport = 0;
  function reportJsError(msg) {
    const now = Date.now();
    if (now - lastReport < 10000) return;
    lastReport = now;
    api('diag.report_error', { text: String(msg).slice(0, 3000) }).catch(() => {});
  }
  window.addEventListener('error', e => {
    if (!e.filename || !/\/js\//.test(e.filename)) return;
    reportJsError(`${e.message}\n  at ${e.filename.split('/').pop()}:${e.lineno}:${e.colno}\n${(e.error && e.error.stack) || ''}`);
  });
  window.addEventListener('unhandledrejection', e => {
    const r = e.reason;
    if (r && /连不上训练器后台/.test(String(r.message || r))) return;
    reportJsError('未处理的 Promise 异常：' + ((r && (r.stack || r.message)) || r));
  });

  App.diagnose = { open, close, run, loadSettings };
  App.pages.diag = {
    init() {
      $('#diag-btn').onclick = () => drawer().hidden ? open() : close();
      $('#dg-close').onclick = close;
      $('#dg-rerun').onclick = run;
      $('#dg-copy').onclick = async () => { try { await navigator.clipboard.writeText(reportText()); App.toast('诊断报告已复制', 'ok'); } catch (e) { App.toast('复制失败：' + e.message, 'bad'); } };
      $('#dg-send').onclick = send;
      $('#dg-input').onkeydown = e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); send(); } };
      $('#dg-memo-clear').onclick = () => saveMemo('');
      $('#dg-target').onchange = e => {
        const t = (e.target._targets || [])[+e.target.value];
        if (t) { target = t; if (!job) { aiSteps = []; aiResult = null; aiError = ''; } run(); }
      };
      document.addEventListener('keydown', e => {
        if (e.ctrlKey && e.shiftKey && (e.key === 'D' || e.key === 'd')) { e.preventDefault(); drawer().hidden ? open() : close(); }
        else if (e.key === 'Escape' && !drawer().hidden && !document.querySelector('.modal-mask')) close();
      });
      initSettings(); loadSettings();
      refreshProblems();
      setInterval(refreshProblems, 5000);
    },
  };
})();

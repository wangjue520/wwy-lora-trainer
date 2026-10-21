/* 01 启动训练：当前配置摘要、保存位置、开始 / 排队 / 停止、队列、日志与 loss 曲线 */
(function () {
  const { $, esc, h, api } = App;
  const ST = { queued: ['排队中', ''], checking: ['检查中', 'y'], blocked: ['需要确认', 'warn'], running: ['训练中', 'k'],
    done: ['完成', 'ok'], failed: ['失败', 'bad'], stopped: ['已停止', ''], paused: ['已冻结', 'y'] };
  let sel = null, logTimer = null, lastLogLen = -1, estimate = null;

  const P = App.pages.train = {
    init() {
      $('#train-outdir').onchange = e => { App.draft.output_dir = e.target.value.trim(); App.saveDraft(); P.outHint(); };
      $('#train-outdir-browse').onclick = async () => { const p = await App.pick({ title: '选择 LoRA 保存位置', start: App.draft.output_dir }); if (p) { $('#train-outdir').value = p; $('#train-outdir').onchange({ target: { value: p } }); } };
      $('#train-outdir-open').onclick = async () => { await api('fs.mkdir', { path: App.draft.output_dir }); api('fs.open', { path: App.draft.output_dir }); };
      $('#train-subdir').onchange = e => { App.draft.subdir_per_task = e.target.checked; App.saveDraft(); P.outHint(); };
      $('#train-start').onclick = () => P.add(true);
      $('#train-add').onclick = () => P.add(false);
      $('#train-stop').onclick = async () => {
        if (await App.confirm('停止训练', '确定停止当前正在训练的任务吗？已经保存的中间 LoRA 会保留。', '停止', true)) { await api('queue.stop'); App.toast('已发送停止'); }
      };
      $('#train-freeze').onclick = async () => {
        try { await api('queue.freeze'); App.toast('正在保存训练进度，这一步跑完就停下并释放显存', 'ok'); App.tick(); } catch (e) { App.toast(e.message, 'bad'); }
      };
      $('#train-pause').onclick = async () => {
        const st = App.queue && App.queue.state;
        if (st && st.paused) { await api('queue.start'); App.toast('队列继续'); } else { await api('queue.pause'); App.toast('当前任务训完后队列暂停'); }
      };
      $('#queue-clear').onclick = () => api('queue.clear').then(() => App.tick());
      $('#td-copy').onclick = () => { navigator.clipboard.writeText($('#log-box').innerText); App.toast('日志已复制'); };
      $('#td-open').onclick = () => { const t = P.selTask(); if (t) api('fs.open', { path: t.output_dir }); };
      $('#td-cmd').onclick = async () => {
        const t = P.selTask(); if (!t) return;
        const log = await api('queue.log', { id: t.id, tail: 4000 });
        const cmds = log.filter(l => l.startsWith('$ ')).map(l => l.slice(2));
        App.modal({ title: '训练命令', wide: true, body: `<div class="hint" style="margin:0 0 8px">任务目录：${esc(t.id)}（data/tasks 里有 commands.txt 和 dataset.toml）</div><div class="code-box" style="max-height:420px;user-select:text">${esc(cmds.join('\n\n') || '还没开始运行')}</div>`, buttons: [{ label: '关闭' }] });
      };
      $('#preset-save').onclick = async () => {
        const n = await App.prompt('存档当前配置', '存档名称（同名会覆盖）', App.params().output_name || '');
        if (!n) return;
        App.state.presets = await api('presets.save', { name: n, draft: JSON.parse(JSON.stringify(App.draft)) });
        P.renderPresets(); App.toast('已存档：' + n, 'ok');
      };
      $('#preset-load').onclick = async () => {
        const n = $('#preset-select').value; if (!n) return;
        if (!(await App.confirm('读取存档', `用存档「${esc(n)}」覆盖当前的模型设置、参数和测试设置？`, '读取'))) return;
        App.draft = JSON.parse(JSON.stringify(App.state.presets[n])); App.normalizeDraft(); App.saveDraft(true);
        App.emit('model_type'); P.show(); App.toast('已读取存档：' + n, 'ok');
      };
      $('#preset-del').onclick = async () => {
        const n = $('#preset-select').value; if (!n) return;
        if (!(await App.confirm('删除存档', `删除存档「${esc(n)}」？`, '删除', true))) return;
        App.state.presets = await api('presets.delete', { name: n }); P.renderPresets();
      };
      App.on('tick', ({ q }) => { if (App.page === 'train') P.renderQueue(q); });

    },
    show() {
      $('#train-outdir').value = App.draft.output_dir || '';
      $('#train-subdir').checked = App.draft.subdir_per_task !== false;
      P.outHint(); P.renderSummary(); P.renderPresets();
      if (App.queue) P.renderQueue(App.queue);
      clearInterval(logTimer); logTimer = setInterval(P.refreshLog, 1500); P.refreshLog();
    },
    hide() { clearInterval(logTimer); },
    outHint() {
      const p = App.params();
      const name = p.output_name || 'lora';
      const sp = App.sep(App.draft.output_dir);
      const dir = App.draft.output_dir + (App.draft.subdir_per_task !== false ? sp + name : '');
      $('#train-outdir-hint').textContent = `将保存为：${dir}${sp}${name}.safetensors（中间轮次：${name}-000001.safetensors …）`;
    },
    renderPresets() {
      const s = $('#preset-select'); const names = Object.keys(App.state.presets || {});
      s.innerHTML = '<option value="">配置存档…</option>' + names.map(n => `<option>${esc(n)}</option>`).join('');
    },
    async renderSummary() {
      const spec = App.spec(), el = $('#train-summary');
      if (!spec) { el.innerHTML = `<div data-go="model"><div class="k">MODEL</div><div class="v bad">未选择底模</div><div class="s">先到「模型设置」选择 Anima 或 Krea 2</div></div>`; P.bindSummary(); return; }
      const c = App.cur(), p = App.params();
      const missing = spec.slots.filter(s => s.required && !c.models[s.key]).map(s => s.label);
      try { estimate = (await api('dataset.stats', { params: p })).estimate; } catch (e) { estimate = null; }
      const lr = Number(p.learning_rate);
      const cell = (go, k, v, s, bad) => `<div data-go="${go}"><div class="k">${k}</div><div class="v ${bad ? 'bad' : ''}" title="${esc(v)}">${esc(v)}</div><div class="s" title="${esc(s)}">${esc(s)}</div></div>`;
      el.innerHTML =
        cell('model', 'MODEL · 底模', spec.label, spec.engine) +
        cell('model', 'CHECKPOINT · 主模型', c.models.dit ? App.base(c.models.dit) : '未选择', missing.length ? '缺少：' + missing.join('、') : '模型文件齐全', missing.length > 0) +
        cell('params', 'DATASET · 训练集', estimate && estimate.images ? `${estimate.images} 张` : '未设置', p.dataset_dir || '到高级参数里选择', !(estimate && estimate.images)) +
        cell('params', 'STEPS · 总步数', estimate ? String(estimate.total_steps) : '—', estimate ? `${estimate.epochs} 轮 × ${estimate.steps_per_epoch} 步` : '') +
        cell('params', 'NETWORK', `dim ${p.network_dim} / α ${p.network_alpha}`, `批量 ${p.batch_size} · 分辨率 ${p.resolution}`) +
        cell('params', 'LR · 学习率', isFinite(lr) ? lr.toExponential(1).replace('e-', 'e-') : String(p.learning_rate), `${p.optimizer_type} · ${p.lr_scheduler}`) +
        cell('sample', 'PREVIEW · 测试', !c.sample.enabled ? '关闭' : c.sample.follow_save !== false ? (p.save_every_n_epochs > 0 ? `每 ${p.save_every_n_epochs} 轮` : '训练结束时')
          : (c.sample.every_n_epochs > 0 ? `每 ${c.sample.every_n_epochs} 轮` : `每 ${c.sample.every_n_steps || '—'} 步`), `${(c.sample.prompts || []).length} 条提示词`) +
        cell('params', 'NAME · 保存名称', p.output_name || '—', p.gradient_checkpointing ? '梯度检查点：开' : '梯度检查点：关');
      P.bindSummary();
    },
    bindSummary() { App.$$('#train-summary > div').forEach(d => d.onclick = () => App.go(d.dataset.go)); },

    async add(start) {
      const spec = App.spec();
      if (!spec) { App.toast('先在「模型设置」选择底模', 'bad'); App.go('model'); return; }
      const c = App.cur(), p = App.params();
      const missing = spec.slots.filter(s => s.required && !c.models[s.key]);
      if (missing.length) { App.toast('缺少模型文件：' + missing.map(s => s.label).join('、'), 'bad', '去设置', () => App.go('model')); return; }
      if (!p.dataset_dir) { App.toast('还没设置训练集位置', 'bad', '去设置', () => App.go('params')); return; }
      if (!App.state.env.env) {
        const st = await api('env.status'); App.state.env = st;
        if (!st.env) { App.toast('训练环境还没安装', 'bad', '去安装', () => App.go('settings'), 10000); return; }
      }
      const same = (App.queue ? App.queue.tasks : []).find(t => t.status !== 'done' && t.params.output_name === p.output_name);
      if (same && !(await App.confirm('保存名称重复', `队列里已经有一个叫「${esc(p.output_name)}」的任务，两个任务会写到同一个位置、互相覆盖。仍然加入吗？`, '仍然加入'))) return;
      try {
        const t = await api('queue.add', { draft: App.taskDraft(), start });
        sel = t.id;
        App.toast(start ? `「${t.name}」已开始（前面有任务会先排队）` : `「${t.name}」已加入队列`, 'ok');
        App.tick();
      } catch (e) { App.toast(e.message, 'bad'); }
    },
    selTask() { return App.queue && App.queue.tasks.find(t => t.id === sel); },

    renderQueue(q) {
      const el = $('#queue-list');
      const running = q.tasks.find(t => t.status === 'running' || t.status === 'checking');
      $('#train-stop').disabled = !running;
      const rs = running && App.spec(running.model_type);
      $('#train-freeze').disabled = !(rs && rs.resume);
      $('#train-freeze').title = running && !(rs && rs.resume) ? `${rs ? rs.label : ''} 暂不支持冻结` : '';
      $('#train-pause').textContent = q.state.paused ? '继续队列' : '训完当前就暂停队列';
      if (!sel || !q.tasks.find(t => t.id === sel)) sel = (running || q.tasks[q.tasks.length - 1] || {}).id || null;
      if (!q.tasks.length) { el.innerHTML = '<div class="q-empty">队列是空的。设置好模型和参数后点「开始训练」或「加入队列」。</div>'; P.renderDetail(); return; }
      el.innerHTML = '';
      q.tasks.forEach((t, i) => {
        const [stt, cls] = ST[t.status] || [t.status, ''];
        const p = t.progress || {};
        const pct = p.total ? p.step / p.total * 100 : (t.status === 'done' ? 100 : 0);
        const spec = App.spec(t.model_type);
        const nums = t.status === 'running'
          ? (p.total ? `<span>${p.step}/${p.total} 步${p.epoch ? ' · 第 ' + p.epoch + ' 轮' : ''}</span><span>${p.loss != null ? 'loss ' + p.loss.toFixed(4) : ''}</span><span>${p.elapsed || ''} / 剩余 ${p.eta || '?'}</span>` : `<span>${esc(p.cache || t.message || '准备中…')}</span>`)
          : `<span>${esc(t.message || '')}</span><span>${t.finished ? App.fmtTime(t.finished) : ''}</span>`;
        const row = h(`<div class="q-row ${t.id === sel ? 'sel' : ''}">
          <div class="q-idx">${String(i + 1).padStart(2, '0')}</div>
          <div><div class="q-name">${esc(t.name)} <span class="badge ${cls}">${stt}</span></div>
            <div class="q-sub">${esc(spec ? spec.label : t.model_type)} · ${esc(App.base(t.params.dataset_dir))} · dim ${t.params.network_dim} · lr ${t.params.learning_rate}${t.gpus && t.gpus.length > 1 ? ` · ${t.gpus.length} 卡（${t.gpus.join(',')}）` : ''}</div></div>
          <div class="q-prog"><div class="prog ${t.status === 'checking' || (t.status === 'running' && !p.total) ? 'indet' : ''}"><i style="width:${pct}%"></i></div><div class="q-nums">${nums}</div></div>
          <div class="q-acts"></div></div>`);
        const acts = row.querySelector('.q-acts');
        const btn = (label, title, fn, dis) => { const b = h(`<button class="icon-btn" title="${title}">${label}</button>`); b.disabled = !!dis; b.onclick = e => { e.stopPropagation(); fn(); }; acts.append(b); };
        btn('↑', '上移', () => api('queue.move', { id: t.id, delta: -1 }).then(App.tick.bind(App)), i === 0);
        btn('↓', '下移', () => api('queue.move', { id: t.id, delta: 1 }).then(App.tick.bind(App)), i === q.tasks.length - 1);
        if (t.can_resume) btn('▶', (t.status === 'paused' ? '继续：' : '一键续训：') + t.can_resume, () => api('queue.resume', { id: t.id })
          .then(() => { App.toast(t.can_resume, 'ok'); App.tick(); }).catch(e => App.toast(e.message, 'bad')));
        if (['failed', 'stopped', 'done', 'blocked', 'paused'].includes(t.status)) btn('↻', '从头重新训练', () => api('queue.requeue', { id: t.id, start: true }).then(() => { App.toast('已重新加入队列'); App.tick(); }));
        if (['done', 'stopped', 'paused', 'failed'].includes(t.status)) btn('🚀', '把训练好的 LoRA（连同示例图、触发词）送到 WWY 启动器', () => App.sendToLauncher(t.id));
        btn('⇪', '把这个任务的设置载入为当前配置', () => P.loadTask(t));
        btn('📁', '打开输出文件夹', () => api('fs.mkdir', { path: t.output_dir }).then(() => api('fs.open', { path: t.output_dir })));
        if (!['running', 'checking'].includes(t.status)) btn('✕', '删除任务', async () => {
          if (await App.confirm('删除任务', `从队列删除「${esc(t.name)}」？已经训练好的 LoRA 文件不会被删除。`, '删除', true)) { await api('queue.remove', { id: t.id }); App.tick(); }
        });
        row.onclick = () => { sel = t.id; lastLogLen = -1; P.renderQueue(App.queue); P.refreshLog(); };
        el.append(row);
        if (t.status === 'paused' && t.can_resume) {
          const al = h(`<div class="q-row" style="cursor:default;padding-top:0"><div></div><div class="q-alert warn">
            <span>❄ ${esc(t.message)}</span><button class="btn btn-sm btn-primary">▶ 继续训练</button></div></div>`);
          al.querySelector('button').onclick = () => api('queue.resume', { id: t.id }).then(() => { App.toast('继续训练：' + t.can_resume, 'ok'); App.tick(); }).catch(e => App.toast(e.message, 'bad'));
          el.append(al);
        }
        if (t.status === 'failed' || t.status === 'blocked') {
          const al = h(`<div class="q-row" style="cursor:default;padding-top:0"><div></div><div class="q-alert ${t.status === 'blocked' ? 'warn' : ''}">
            <span>${t.status === 'failed' ? '✕ ' + esc((t.error || {}).title || t.message) + '：' + esc(((t.error || {}).reason || '').slice(0, 90)) : '! ' + esc(t.message)}</span>
            <button class="btn btn-sm ${t.status === 'failed' ? 'btn-danger' : 'btn-primary'}">${(t.error && t.error.fixes && t.error.fixes.length) || t.status === 'blocked' ? '查看原因并自动修复' : '查看原因'}</button></div></div>`);
          al.querySelector('button').onclick = () => App.showTaskIssue(t.id);
          el.append(al);
        }
      });
      P.renderDetail();
    },
    loadTask(t) {
      App.confirm('载入任务设置', `用任务「${esc(t.name)}」的模型、参数和测试设置覆盖当前配置？`, '载入').then(ok => {
        if (!ok) return;
        App.draft.model_type = t.model_type;
        const c = App.cur();
        c.models = Object.assign({}, t.models); c.params = Object.assign({}, t.params); c.sample = JSON.parse(JSON.stringify(t.sample || c.sample));
        App.saveDraft(true); App.emit('model_type'); P.renderSummary(); App.toast('已载入，可以修改后再加入队列', 'ok');
      });
    },
    renderDetail() {
      const t = P.selTask();
      $('#td-title').textContent = t ? `训练曲线 · ${t.name}` : '训练曲线';
      $('#td-sub').textContent = t ? `${(ST[t.status] || [t.status])[0]}${t.step_names ? ' · 步骤：' + t.step_names.join(' → ') : ''}`
        + (t.progress && t.progress.ready_stop ? ` · EmoSens 提示可以停止 ${t.progress.ready_stop} 次（最近在第 ${t.progress.ready_stop_step} 步附近，优先测试那附近保存的 LoRA）` : '') : '';
      const p = (t && t.progress) || {};
      const stat = (k, v) => `<div><div class="k">${k}</div><div class="v">${esc(v)}</div></div>`;
      $('#td-stats').innerHTML = stat('STEP · 步数', p.total ? `${p.step} / ${p.total}` : '—') + stat('EPOCH · 轮', p.epoch || '—') +
        stat('LOSS', p.loss != null ? p.loss.toFixed(4) : '—') + stat('ELAPSED · 已用', p.elapsed || '—') + stat('ETA · 剩余', p.eta || '—');
      $('#td-prog i').style.width = (p.total ? p.step / p.total * 100 : (t && t.status === 'done' ? 100 : 0)) + '%';
      P.loadMetrics();
    },
    // 2×2 图表：loss/average、loss/current、loss/epoch_average、学习率；有验证集时再加验证 loss
    async loadMetrics(force) {
      const t = P.selTask(), box = $('#td-charts');
      if (!t) { box.innerHTML = ''; P._mFor = null; return; }
      const live = t.status === 'running';
      if (!force && P._mFor === t.id && !live && P._mDone) return;
      if (live && Date.now() - (P._mAt || 0) < 4000 && P._mFor === t.id) return;   // 训练中每 4 秒刷新一次
      P._mAt = Date.now();
      let d; try { d = await api('queue.metrics', { id: t.id }); } catch (e) { return; }
      d = d || {};
      const pick = (...names) => names.find(n => d[n]);
      // 学习率：自适应优化器（Automagic / EmoSens）用实际学习率，否则用引擎记录的调度器学习率
      const lrTag = d['lr/actual'] ? 'lr/actual' : Object.keys(d).filter(k => k.startsWith('lr/') && k !== 'lr/epoch').sort((a, b) => a.length - b.length)[0];
      const want = [pick('loss/average'), pick('loss/current'), pick('loss/epoch_average', 'loss/epoch'), 'lr/epoch', lrTag || 'lr/unet',
        ...Object.keys(d).filter(k => k.startsWith('loss/validation/'))].filter(Boolean);
      if (P._mFor !== t.id || P._mTags !== want.join('|')) {
        box.innerHTML = '';
        P._cards = {};
        const sub = `logs${App.sep(t.output_dir)}${t.id}`;
        const NAMES = { 'loss/average': '平均 loss', 'loss/current': '当前步 loss', 'loss/epoch_average': '每轮平均 loss', 'loss/epoch': '每轮平均 loss',
          'lr/epoch': '每轮学习率（横轴是轮数）', 'lr/actual': '实际学习率（优化器自动调整后）', 'lr/unet': '学习率（调度器）' };
        want.forEach(tag => {
          const c = Charts.card(tag, (NAMES[tag] || (tag.startsWith('loss/validation') ? '验证集 loss' : tag.startsWith('lr/') ? '学习率' : '')) + ' · ' + sub);
          P._cards[tag] = c; box.append(c);
        });
        P._mFor = t.id; P._mTags = want.join('|');
      }
      want.forEach(tag => P._cards[tag].set(d[tag] || []));
      P._mDone = !live;
    },
    async refreshLog() {
      const t = P.selTask(); const box = $('#log-box');
      if (!t) { box.textContent = ''; return; }
      if (t.status !== 'running' && t.status !== 'checking' && lastLogLen >= 0 && P._logFor === t.id + t.status) return;
      let lines;
      try { lines = await api('queue.log', { id: t.id, tail: 500 }); } catch (e) { return; }
      P._logFor = t.id + t.status;
      if (lines.length === lastLogLen && t.status !== 'running') return;
      lastLogLen = lines.length;
      // 进度条（steps / Sampling / Loading / Caching 的百分比行）每秒刷几十行，进度在上面的数字和图表里看，这里不显示
      lines = lines.filter(l => !/(\d+%\|.*\|\s*\d+\/\d+|it\/s\]|s\/it\]|s\/it,)/.test(l) || /error|Traceback/i.test(l));
      box.innerHTML = lines.map(l => {
        let c = '';
        if (/^\$ /.test(l)) c = 'l-cmd'; else if (/error|Traceback|失败|Exception/i.test(l)) c = 'l-err';
        else if (/warn/i.test(l)) c = 'l-warn'; else if (/=====|完成|saved|保存/i.test(l)) c = 'l-ok'; else if (/^\[(提示|检查)\]/.test(l)) c = 'l-tip';
        return `<div class="${c}">${esc(l)}</div>`;
      }).join('') || '<span class="muted">还没有日志</span>';
      if ($('#log-follow').checked) box.scrollTop = box.scrollHeight;
    },
  };
  App.on('params', () => { if (App.page === 'train') P.renderSummary(); });
  App.on('model_type', () => { if (App.page === 'train') P.renderSummary(); });
})();

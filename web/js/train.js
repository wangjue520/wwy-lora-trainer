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
      $('#preset-import').onclick = () => P.importDialog();
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
      clearInterval(logTimer); logTimer = setInterval(P._pollLog || (P._pollLog = App.poll(P.refreshLog)), 1500); P.refreshLog();
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

    // ---------------- 导入外部配置：选文件 / 拖进来 / 粘贴 → 后台换算 → 逐项勾选后应用 ----------------
    importDialog() {
      const body = h(`<div>
        <div class="imp-drop"><b>点这里选择配置文件，或者把文件拖进来</b><span class="muted">.toml / .json / .yaml / .sh / .bat / .ps1 / .txt / 练好的 LoRA（.safetensors）</span><input type="file" hidden></div>
        <div class="hint">认得：kohya sd-scripts / musubi-tuner 的 TOML（含数据集配置）、kohya_ss GUI 的 JSON、秋叶 lora-scripts 的 TOML 和 train.sh / train.ps1、OneTrainer、ai-toolkit、SimpleTuner、
          训练命令行（accelerate launch … --xxx）、用 sd-scripts 系练出来的 LoRA 文件（读里面记的训练参数）、本训练器的配置存档（data/presets.json）。导入前会列出每一项让你确认，不会直接覆盖。</div>
        <div style="margin:12px 0 6px">或者把配置内容 / 训练命令直接粘贴在这里：</div>
        <textarea rows="6" style="width:100%" placeholder="accelerate launch sdxl_train_network.py --pretrained_model_name_or_path=… --network_dim 16 …&#10;或者 TOML / JSON / YAML 的内容"></textarea></div>`);
      const inp = body.querySelector('input'), drop = body.querySelector('.imp-drop'), ta = body.querySelector('textarea');
      const go = async (text, name) => {
        try { const r = await api('config.import', { text, name, current: App.draft.model_type || '' }); m.close(); P.importPreview(r, text, name); }
        catch (e) { App.toast(e.message, 'bad', null, null, 8000); }
      };
      const read = f => P.readCfgFile(f).then(x => go(x.text, x.name)).catch(e => App.toast(e.message, 'bad', null, null, 8000));
      drop.onclick = () => inp.click();
      inp.onchange = () => { if (inp.files[0]) read(inp.files[0]); inp.value = ''; };
      drop.ondragover = e => { e.preventDefault(); drop.classList.add('on'); };
      drop.ondragleave = () => drop.classList.remove('on');
      drop.ondrop = e => { e.preventDefault(); drop.classList.remove('on'); if (e.dataTransfer.files[0]) read(e.dataTransfer.files[0]); };
      const m = App.modal({ title: '导入外部配置', body, wide: true, buttons: [{ label: '取消' },
        { label: '解析粘贴的内容', cls: 'btn-primary', fn: () => { if (!ta.value.trim()) App.toast('先粘贴内容，或者选一个文件', 'bad'); else go(ta.value, ''); return false; } }] });
    },
    // LoRA 文件只读文件头里的元数据（几 KB～几 MB），不用把整个文件传给后台
    async readCfgFile(f) {
      if (/\.safetensors$/i.test(f.name)) {
        const n = Number(new DataView(await f.slice(0, 8).arrayBuffer()).getBigUint64(0, true));
        if (!(n > 2 && n < 200 * 1024 * 1024 && n + 8 <= f.size)) throw new Error('不是有效的 safetensors 文件');
        const meta = JSON.parse(new TextDecoder().decode(await f.slice(8, 8 + n).arrayBuffer())).__metadata__;
        if (!meta) throw new Error('这个 LoRA 文件里没有记录训练参数（训练时关了元数据，或者被合并 / 转换过）');
        return { text: JSON.stringify(meta), name: f.name };
      }
      if (f.size > 20 * 1024 * 1024) throw new Error('文件太大，不像是配置文件');
      return { text: await f.text(), name: f.name };
    },
    async importPreview(r, text, name) {
      if (r.presets) {
        const names = Object.keys(r.presets), dup = names.filter(n => (App.state.presets || {})[n]);
        if (!(await App.confirm('导入配置存档', `文件里有 ${names.length} 份配置存档：${esc(names.join('、'))}。<br>全部加到这台电脑的配置存档里？${dup.length ? `<br><span class="hint warn">同名的会被覆盖：${esc(dup.join('、'))}</span>` : ''}`, '全部导入'))) return;
        App.state.presets = await api('presets.import', { presets: r.presets }); P.renderPresets(); App.toast(`已导入 ${names.length} 份配置存档，在右边的下拉框里读取`, 'ok');
        return;
      }
      const spec = App.spec(r.target), c = (App.draft.by_model || {})[r.target] || {};
      const cur = Object.assign(App.defaults(spec), c.params || {}), curM = c.models || {};
      const fields = Object.fromEntries(spec.params.map(f => [f.key, f]));
      const show = (k, v) => {
        if (v === undefined || v === null || v === '') return '（空）';
        if (typeof v === 'boolean') return v ? '开' : '关';
        const f = fields[k], o = f && f.type === 'select' && (f.options || []).find(o => String(Array.isArray(o) ? o[0] : o) === String(v));
        return o && Array.isArray(o) ? String(o[1]).split(' · ')[0].split('（')[0] : String(v);
      };
      const same = (a, b) => String(a ?? '') === String(b ?? '') || (a !== '' && b !== '' && a != null && b != null && !isNaN(a) && !isNaN(b) && Number(a) === Number(b));
      const rows = r.rows.filter(x => !same(x.value, cur[x.key])), nSame = r.rows.length - rows.length;
      const items = [];   // {el, apply()}
      const row = (label, sub, o, n, on, apply, off, tag) => {
        const el = h(`<label class="imp-row ${off ? 'off' : ''}"><input type="checkbox" ${on && !off ? 'checked' : ''} ${off ? 'disabled' : ''}>
          <span class="k">${esc(label)}${sub ? `<small>${esc(sub)}</small>` : ''}</span><span class="o" title="${esc(o)}">${esc(o)}</span><span class="a">→</span>
          <span class="n" title="${esc(n)}">${tag ? `<span class="badge bad">${esc(tag)}</span> ` : ''}${esc(n)}</span></label>`);
        items.push({ el, apply }); return el;
      };
      const fams = {}; App.state.specs.forEach(s => { fams[s.family] = (fams[s.family] || 0) + 1; });
      const body = h(`<div>
        <div class="imp-head"><span>识别为 <b>${esc(r.format)}</b>${r.family ? ` · 原底模 <b>${esc(r.family === r.detected ? spec.label : (App.spec(r.family) || {}).label || r.family)}</b>` : ''}</span>
          <span class="row" style="margin-left:auto">套用到<select data-target style="width:220px">${App.state.specs.map(s => `<option value="${s.id}" ${s.id === r.target ? 'selected' : ''}>${esc(s.label + (fams[s.family] > 1 ? ' · ' + s.engine_label : ''))}</option>`).join('')}</select></span></div>
        ${r.notes.length ? `<div class="issue warn imp-notes">${r.notes.map(n => `<div>${esc(n)}</div>`).join('')}</div>` : ''}
        <div data-list></div></div>`);
      const list = body.querySelector('[data-list]');
      const sec = (t, hint) => list.append(h(`<div class="imp-sec">${esc(t)}${hint ? `<span class="muted">${esc(hint)}</span>` : ''}</div>`));
      const box = () => { const b = h('<div class="imp-list"></div>'); list.append(b); return b; };
      if (r.models.length || r.output_dir) {
        sec('模型文件 / 保存位置', '别的电脑上的路径会按文件名在本机模型库里找');
        const b = box();
        r.models.forEach(x => b.append(x.path
          ? row(x.label, '', curM[x.key] ? App.base(curM[x.key]) : '（未选）', App.base(x.path), x.path !== curM[x.key], cc => { cc.models[x.key] = x.path; }, x.path === curM[x.key])
          : row(x.label, '', curM[x.key] ? App.base(curM[x.key]) : '（未选）', App.base(x.src), false, null, true, '本机没找到')));
        if (r.output_dir) b.append(row('LoRA 保存位置', '', App.draft.output_dir || '', r.output_dir, r.output_dir_ok && r.output_dir !== App.draft.output_dir,
          () => { App.draft.output_dir = r.output_dir; }, !r.output_dir_ok || r.output_dir === App.draft.output_dir, r.output_dir_ok ? '' : '本机没有'));
      }
      sec(`参数（${rows.length} 项要改）`, nSame ? `另有 ${nSame} 项和当前一样，不用改` : '');
      const pb = box();
      rows.forEach(x => pb.append(row(x.label, x.group, show(x.key, cur[x.key]), show(x.key, x.value), !x.missing, cc => { cc.params[x.key] = x.value; }, false, x.missing ? '本机找不到' : '')));
      if (!rows.length) pb.append(h('<div class="imp-row off"><span></span><span class="muted">没有要改的参数</span></div>'));
      if (r.sample) {
        const s = r.sample, ps = s.prompts || [];
        sec('训练中测试');
        const b = box(), cs = c.sample || {};
        const iv = s.every_n_steps ? `每 ${s.every_n_steps} 步` : s.every_n_epochs ? `每 ${s.every_n_epochs} 轮` : '';
        if (ps.length) b.append(row('测试提示词', '', `${(cs.prompts || []).length} 条`, `${ps.length} 条：${ps.map(p => p.prompt).join(' / ')}`, true,
          cc => { cc.sample.prompts = ps.map(p => Object.assign({}, p)); cc.sample.auto_default = false; cc.sample.enabled = true; }));
        if (iv) b.append(row('出图间隔', '', cs.follow_save !== false ? '跟着保存' : cs.every_n_steps ? `每 ${cs.every_n_steps} 步` : `每 ${cs.every_n_epochs || 0} 轮`, iv, true,
          cc => { Object.assign(cc.sample, { follow_save: false, every_n_steps: s.every_n_steps || 0, every_n_epochs: s.every_n_steps ? 0 : (s.every_n_epochs || 0), enabled: true }); }));
        if (s.at_first) b.append(row('训练前先出一次图', '', cs.at_first ? '开' : '关', '开', !cs.at_first, cc => { cc.sample.at_first = true; }));
      }
      const unk = Object.entries(r.unknown || {});
      if (r.skipped.length || unk.length) list.append(h(`<details class="imp-more"><summary class="muted">没有导入的设置（${r.skipped.length + unk.length}）</summary><div class="fix-list">
        ${r.skipped.map(x => `<div><b>${esc(x.key)}</b> = ${esc(x.value)}<span class="muted" style="margin-left:auto">${esc(x.reason)}</span></div>`).join('')}
        ${unk.map(([k, v]) => `<div><b>${esc(k)}</b> = ${esc(v)}<span class="muted" style="margin-left:auto">这个训练器没有对应的设置</span></div>`).join('')}</div>
        <div class="hint">需要的话可以写进「高级参数 → 额外命令行参数」。</div></details>`));
      const m = App.modal({ title: '导入外部配置 · 确认要改的设置', body, wide: true, buttons: [{ label: '取消' },
        { label: '应用到当前配置', cls: 'btn-primary', fn: () => P.importApply(r, items) }] });
      body.querySelector('[data-target]').onchange = async e => {
        try { const r2 = await api('config.import', { text, name, target: e.target.value, current: App.draft.model_type || '' }); m.close(); P.importPreview(r2, text, name); }
        catch (err) { App.toast(err.message, 'bad'); }
      };
    },
    importApply(r, items) {
      const todo = items.filter(x => x.apply && x.el.querySelector('input').checked);
      if (!todo.length) { App.toast('没有勾选任何一项', 'bad'); return false; }
      if (r.target !== App.draft.model_type) App.setModelType(r.target);
      const c = App.cur(), ds = c.params.dataset_dir;
      todo.forEach(x => x.apply(c));
      App.saveDraft(true);
      if (c.params.dataset_dir && c.params.dataset_dir !== ds) App.syncDataset(c.params.dataset_dir, 'params');
      App.emit('model_type'); App.emit('params');
      P.show();
      App.toast(`已导入 ${todo.length} 项设置。想留一份可以点「存档当前配置」`, 'ok', null, null, 8000);
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
      try {
        const t = await api('queue.add', { draft: App.taskDraft(), start });
        sel = t.id;
        // 保存名称撞上了别的任务 / 上次训练的结果：后台自动加了 (1)，不会覆盖；配置里的名字也跟着改，下一次接着往后排
        if (t.renamed_from) {
          App.setParam('output_name', t.params.output_name); P.outHint(); P.renderSummary();
          App.toast(`「${t.renamed_from}」已经有训练结果了，这次保存为「${t.params.output_name}」，不会覆盖`, 'ok', null, null, 8000);
        }
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
      if (P._mBusy && !force) return;   // 上一次还没返回
      P._mBusy = true;
      try { await P._loadMetrics(force); } finally { P._mBusy = false; }
    },
    async _loadMetrics(force) {
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

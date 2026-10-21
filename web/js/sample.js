/* 06 训练中测试：每 N 步用固定提示词 / 种子出一组图，表格对比不同步数的效果 */
(function () {
  const { $, esc, h, api } = App;
  let timer = null, curTask = null;

  const P = App.pages.sample = {
    init() {
      const bind = (id, key, conv) => { $(id).onchange = e => { const s = App.cur().sample; s[key] = conv(e.target); App.saveDraft(); P.hint(); }; };
      bind('#s-enabled', 'enabled', t => t.checked);
      bind('#s-steps', 'every_n_steps', t => parseInt(t.value || '0', 10));
      bind('#s-epochs', 'every_n_epochs', t => parseInt(t.value || '0', 10));
      bind('#s-first', 'at_first', t => t.checked);
      bind('#s-clear-old', 'clear_old', t => t.checked);
      // 训练中途改提示词：把这里的提示词写进正在训练的任务，下一次出测试图起生效
      $('#s-apply').onclick = async () => {
        const t = P.running(); if (!t) return;
        try {
          await api('samples.update_prompts', { id: t.id, prompts: App.cur().sample.prompts });
          App.toast(`已用到「${t.name}」，下一次出测试图起按新提示词出`, 'ok');
        } catch (e) { App.toast(e.message, 'bad'); }
        App.tick();
      };
      $('#s-clear').onclick = async () => {
        const t = P.task(); if (!t) return;
        if (!(await App.confirm('清空测试图', `删除「${esc(t.name)}」保存位置里的全部测试图：<br>${esc(t.output_dir)}\\sample<br><br>LoRA 文件不受影响。`, '删除', true))) return;
        try { const n = await api('samples.clear', { id: t.id }); App.toast(`已删除 ${n} 张测试图`, 'ok'); } catch (e) { App.toast(e.message, 'bad'); }
        P._gridSig = null; P.loadGrid();
      };
      App.$$('#s-mode button').forEach(b => b.onclick = () => {
        App.cur().sample.follow_save = b.dataset.v === 'follow'; App.saveDraft(); P.renderMode(); P.hint();
      });
      $('#s-turbo').onchange = e => {
        const s = App.cur().sample; s.use_turbo = e.target.checked;
        const spec = App.spec(); const d = e.target.checked ? { steps: 8, cfg: 1 } : { steps: spec.sample_defaults.steps, cfg: spec.sample_defaults.cfg };
        s.prompts.forEach(p => Object.assign(p, d)); App.saveDraft(); P.renderPrompts(); P.hint();
      };
      $('#s-add').onclick = () => { const s = App.cur().sample; const last = s.prompts[s.prompts.length - 1] || App.spec().sample_defaults; s.prompts.push(Object.assign({}, last, { prompt: '' })); s.auto_default = false; App.saveDraft(); P.renderPrompts(); };
      $('#s-defaults').onclick = () => P.loadDefaults(true);
      $('#s-trigger').onclick = async () => {
        // 训练集已经整理过触发词：按标法加上「用 LoRA 时写」的那一串（全标带特征 / 服装，删标只有触发词）
        const dir = App.params().dataset_dir;
        const r = dir ? await api('trigger.sync_prompts', { dir, prompts: App.cur().sample.prompts }).catch(() => null) : null;
        if (r && r.prompt) {
          App.cur().sample.prompts = r.prompts; App.cur().sample.auto_default = false; App.saveDraft(); P.renderPrompts();
          App.toast(r.changed ? `已按训练集的标法加上：${r.prompt}` : '测试提示词已经和训练集的标法一致', 'ok', null, null, 8000);
          return;
        }
        const t = await App.prompt('加上触发词', '触发词', (App.draft.tag || {}).trigger || ''); if (!t) return;
        App.cur().sample.prompts.forEach(p => { if (!p.prompt.toLowerCase().startsWith(t.toLowerCase())) p.prompt = t + ', ' + p.prompt; });
        App.cur().sample.auto_default = false; App.saveDraft(); P.renderPrompts();
      };
      $('#s-task').onchange = e => { curTask = e.target.value; P.loadGrid(); };
      $('#s-refresh').onclick = () => P.loadGrid();
      $('#s-open').onclick = () => { const t = (App.queue.tasks || []).find(t => t.id === curTask); if (t) api('fs.mkdir', { path: t.output_dir + '\\sample' }).then(() => api('fs.open', { path: t.output_dir + '\\sample' })); };
      App.on('model_type', () => { if (App.page === 'sample') P.show(); });
    },
    show() {
      const spec = App.spec();
      if (!spec) { $('#s-prompts').innerHTML = '<div class="hint">先到「模型设置」选择底模。</div>'; return; }
      const s = App.cur().sample;
      $('#s-enabled').checked = !!s.enabled; $('#s-steps').value = s.every_n_steps || 0; $('#s-epochs').value = s.every_n_epochs || 0;
      $('#s-first').checked = !!s.at_first; $('#s-clear-old').checked = !!s.clear_old;
      P.renderMode();
      const hasTurbo = spec.slots.some(x => x.key === 'turbo');
      $('#s-turbo-wrap').hidden = !hasTurbo; $('#s-turbo').checked = !!s.use_turbo;
      const hasAlt = spec.slots.some(x => x.key === 'sample_dit');
      $('#s-model-wrap').hidden = !hasAlt;
      if (hasAlt) P.loadSampleModels();
      P.renderPrompts(); P.hint(); P.renderTasks();
      // 旧版本存下的默认提示词（没有 auto_default 标记）：内容还是当时的默认值就当作没改过，换成新的标准写法
      const OLD = ['a photo of a young woman standing on a city street at dusk, natural light', 'masterpiece, best quality, 1girl, solo, looking at viewer, upper body'];
      if (s.auto_default === undefined && s.prompts.length === 1 && OLD.includes((s.prompts[0].prompt || '').trim())) s.auto_default = true;
      if (s.auto_default) P.loadDefaults(false);
      clearInterval(timer); timer = setInterval(() => { const t = P.task(); if (t && t.status === 'running') P.loadGrid(); }, 5000);
    },
    hide() { clearInterval(timer); },
    // 测试出图用的主模型（MOSS：比如用 Base 练、用 SFT 出图）；和「模型设置」里的「测试出图用的主模型」是同一个设置
    async loadSampleModels() {
      const spec = App.spec(), c = App.cur(), sel = $('#s-model');
      const cur = c.models.sample_dit || '';
      sel.innerHTML = `<option value="">和训练用的主模型一样</option>${cur ? `<option value="${esc(cur)}">${esc(App.base(cur))}</option>` : ''}`;
      sel.value = cur;
      sel.onchange = () => { c.models.sample_dit = sel.value; App.saveDraft(); P.hint(); };
      let res; try { res = await api('models.list', { model_type: spec.id }); } catch (e) { return; }
      if (App.spec() !== spec) return;
      const cands = ((res.slots && res.slots.sample_dit) || []).filter(x => !x.quant);   // int8 单文件不能用来换
      sel.innerHTML = '<option value="">和训练用的主模型一样</option>' + cands.map(x => `<option value="${esc(x.path)}">${esc(x.name)}</option>`).join('')
        + (cur && !cands.some(x => x.path === cur) ? `<option value="${esc(cur)}">自选：${esc(cur)}</option>` : '');
      sel.value = c.models.sample_dit || '';
    },
    // 标准正向 / 负向提示词：按底模和选的主模型系列（SDXL 的 Illustrious / NoobAI / Pony 写法不同）向后端要
    async loadDefaults(force) {
      const spec = App.spec(), c = App.cur(); if (!spec) return;
      const d = await api('sample.defaults', { model_type: spec.id, models: c.models }).catch(() => spec.sample_defaults);
      const s = c.sample;
      if (!force && !s.auto_default) return;   // 等结果期间用户改过了
      const keep = s.use_turbo ? { steps: 8, cfg: 1 } : {};
      const next = [Object.assign({}, d, keep)];
      if (!force && JSON.stringify(next) === JSON.stringify(s.prompts)) return;
      s.prompts = next; s.auto_default = true; App.saveDraft();
      if (App.page === 'sample') P.renderPrompts();
    },
    // 默认跟随「每几轮保存一次」：每存一个 LoRA 出一组图，测试图和保存的文件一一对应
    renderMode() {
      const s = App.cur().sample, p = App.params(), follow = s.follow_save !== false;
      App.$$('#s-mode button').forEach(b => b.classList.toggle('on', (b.dataset.v === 'follow') === follow));
      $('#s-custom').hidden = follow;
      const n = parseInt(p.save_every_n_epochs || 0, 10);
      $('#s-follow-text').hidden = !follow;
      $('#s-follow-text').innerHTML = n > 0
        ? `每 <b>${n}</b> 轮出一组，和保存的 LoRA 一一对应<span class="muted">（在「高级参数 → 每几轮保存一次」里改）</span>`
        : '「每几轮保存一次」是 0（只保存最终结果），训练结束时出一组';
    },
    hint() {
      const spec = App.spec(), s = App.cur().sample, c = App.cur(), p = App.params();
      const msgs = [];
      if (spec.id === 'krea2') {
        if (s.use_turbo && !c.models.turbo) msgs.push('没有选择 Turbo 模型，会改用 RAW 出图（到「模型设置」可以选或下载 Turbo）。');
        if (s.use_turbo && c.models.turbo && (p.blocks_to_swap > 0)) msgs.push('开了块交换，Turbo 出图不能和块交换同时用，会自动改用 RAW 出图。');
        if (!s.use_turbo) msgs.push('RAW 出图需要 CFG（约 5.5）和 28 步左右，比较慢；不开 CFG 会糊。');
      } else if (spec.id === 'moss') {
        if (s.use_turbo) msgs.push('加速出图用 8 步加速 LoRA（' + (c.models.turbo ? '模型设置里选的那个' : '没选就自动在模型库的 loras 文件夹里找 MOSS-SFT-8step-DMD') + '），8 步、CFG 1。');
        if (c.models.sample_dit) msgs.push('测试图用「模型设置 → 测试出图用的主模型」里选的主模型出（只在 bf16、不开 int8 时生效）。');
        if (!s.use_turbo) msgs.push('MOSS-Image 测试图按官方方式出：CFG 3、出图时跳过第 22 层，反向提示词用官方推荐的负面 tag；官方默认 50 步，测试用 40 步（28 步细节不够）。提示词用英文自然语言长句效果最好（也支持中文和 tag 串）。');
      } else if (spec.id === 'moss_native') {
        msgs.push('MOSS 专用引擎：每次只用第 1 条提示词出 1 张，固定 50 步；只能按轮出（自定义间隔填「每几轮」）。');
      } else if (spec.id === 'sdxl') {
        msgs.push('SDXL 测试图用训练中的大模型 + 当前 LoRA 生成，28 步、CFG 5–7 即可；采样器在「高级参数 → 其他」里改。');
        if (/pony/i.test(c.models.dit || '') && !/score_\d/.test(s.prompts.map(x => x.prompt).join(' ')))
          msgs.push('Pony 模型的提示词建议在前面加 score_9, score_8_up, score_7_up。');
      } else msgs.push('Anima 测试图用训练中的模型 + 当前 LoRA 直接生成，30 步左右即可。');
      if (s.enabled && s.follow_save === false && !(s.every_n_steps > 0) && !(s.every_n_epochs > 0)) msgs.push('⚠ 没设置间隔，只会在训练开始前（如果勾了）出图。');
      $('#s-hint').textContent = msgs.join(' ');
    },
    renderPrompts() {
      const s = App.cur().sample, el = $('#s-prompts'); el.innerHTML = '';
      s.prompts.forEach((p, i) => {
        const r = h(`<div class="prompt-row"><div class="pi">${i + 1}</div>
          <div><textarea data-k="prompt" placeholder="提示词"></textarea><input type="text" data-k="negative" placeholder="反向提示词（可选）" style="margin-top:4px"></div>
          <input type="number" data-k="width" step="64"><input type="number" data-k="height" step="64"><input type="number" data-k="steps"><input type="number" data-k="cfg" step="0.5"><input type="number" data-k="seed">
          <button class="icon-btn" title="删除">✕</button></div>`);
        App.$$('[data-k]', r).forEach(inp => {
          const k = inp.dataset.k; inp.value = p[k] == null ? '' : p[k];
          inp.onchange = () => { p[k] = inp.type === 'number' ? (inp.value === '' ? '' : Number(inp.value)) : inp.value; s.auto_default = false; App.saveDraft(); P.renderApply(); };
        });
        r.querySelector('.icon-btn').onclick = () => { s.prompts.splice(i, 1); s.auto_default = false; App.saveDraft(); P.renderPrompts(); };
        el.append(r);
      });
      if (!s.prompts.length) el.innerHTML = '<div class="hint">没有提示词，点「添加一条」。</div>';
      P.renderApply();
    },
    task() { return ((App.queue && App.queue.tasks) || []).find(t => t.id === curTask); },
    // 正在训练、开了测试出图、和当前选的底模一样的任务：这里改的提示词可以用到它上面
    running() {
      const spec = App.spec();
      return ((App.queue && App.queue.tasks) || []).find(t => t.status === 'running' && t.sample && t.sample.enabled && spec && t.model_type === spec.id);
    },
    renderApply() {
      const t = P.running(), b = $('#s-apply'), hint = $('#s-apply-hint');
      b.hidden = hint.hidden = !t;
      if (!t) return;
      const same = JSON.stringify(t.sample.prompts || []) === JSON.stringify(App.cur().sample.prompts);
      b.disabled = same;
      hint.textContent = same ? `和正在训练的「${t.name}」一致` : `和正在训练的「${t.name}」不一样，点按钮后从下一次出图起生效`;
    },
    renderTasks() {
      const ts = ((App.queue && App.queue.tasks) || []).filter(t => t.sample && t.sample.enabled);
      const sel = $('#s-task');
      if (!curTask || !ts.find(t => t.id === curTask)) curTask = ((ts.find(t => t.status === 'running') || ts[ts.length - 1]) || {}).id;
      sel.innerHTML = ts.map(t => `<option value="${t.id}">${esc(t.name)} · ${{ running: '训练中', done: '完成', failed: '失败', queued: '排队', stopped: '已停止', blocked: '待确认', checking: '检查中' }[t.status] || t.status}</option>`).join('') || '<option value="">（还没有开了测试出图的任务）</option>';
      if (curTask) sel.value = curTask;
      P.loadGrid();
    },
    async loadGrid() {
      const el = $('#s-grid'), t = P.task();
      $('#s-clear').disabled = !t || ['running', 'checking'].includes(t.status);
      if (!t) { el.innerHTML = '<div class="q-empty">开了测试出图的任务训练后，测试图会显示在这里。</div>'; return; }
      let list; try { list = await api('samples.list', { id: t.id }); } catch (e) { return; }
      if (!list.length) { el.innerHTML = `<div class="q-empty">${t.status === 'running' ? '训练到第一个测试点后会出现测试图…' : '这个任务还没有测试图'}</div>`; return; }
      const sig = t.id + ':' + list.map(x => x.name).join('|') + '#' + (t.sample.prompts || []).map(p => p.prompt).join('|');   // 中途改了提示词也要重画行标题
      if (sig === P._gridSig && el.querySelector('table')) return;   // 没有新图就不重画，避免闪烁
      P._gridSig = sig;
      // 列 = 步数 / 轮数，行 = 提示词序号
      const cols = [...new Set(list.map(x => (x.epoch ? 'e' : '') + x.num))].sort((a, b) => parseInt(a.replace('e', '')) - parseInt(b.replace('e', '')));
      const rows = [...new Set(list.map(x => x.prompt))].sort((a, b) => a - b);
      const cell = {}; list.forEach(x => cell[(x.epoch ? 'e' : '') + x.num + ':' + x.prompt] = x);
      const prompts = (t.sample.prompts || []).filter(p => (p.prompt || '').trim());
      const flat = [];
      const lora = c => t.output_dir + '\\' + (t.params.output_name || 'lora') + '-' + String(parseInt(c.slice(1))).padStart(6, '0') + '.safetensors';
      let html = '<table><tr><th></th>' + cols.map(c => `<th>${c.startsWith('e') ? '第 ' + parseInt(c.slice(1)) + ' 轮' + (parseInt(c.slice(1)) > 0 ? ` <button class="btn btn-sm" data-send="${esc(lora(c))}" title="满意这一轮？把这一轮的 LoRA 连同示例图、触发词送到 WWY 启动器">送到启动器</button>` : '') : (parseInt(c) === 0 ? '训练前' : parseInt(c) + ' 步')}</th>`).join('') + '</tr>';
      rows.forEach(r => {
        html += `<tr><td class="pl" title="${esc((prompts[r] || {}).prompt || '')}">#${r + 1} ${esc(((prompts[r] || {}).prompt || '').slice(0, 80))}</td>`;
        cols.forEach(c => {
          const x = cell[c + ':' + r];
          if (x) { flat.push({ file: x.file, cap: `${c.startsWith('e') ? '第 ' + parseInt(c.slice(1)) + ' 轮' : parseInt(c) + ' 步'} · #${r + 1} ${(prompts[r] || {}).prompt || ''}` }); }
          html += `<td>${x ? `<img loading="lazy" data-f="${flat.length - 1}" src="/file?path=${encodeURIComponent(x.file)}">` : ''}</td>`;
        });
        html += '</tr>';
      });
      el.innerHTML = html + '</table>';
      App.$$('img', el).forEach(img => img.onclick = () => App.lightbox(flat, +img.dataset.f));
      App.$$('[data-send]', el).forEach(b => b.onclick = () => App.sendToLauncher(t.id, b.dataset.send));
    },
  };
  // 任务列表（增删、状态变化）变了才刷新下拉框和表格；训练中的新图由上面的 5 秒定时器加载
  App.on('tick', ({ q }) => {
    if (App.page !== 'sample') return;
    const key = q.tasks.filter(t => t.sample && t.sample.enabled).map(t => t.id + t.status).join('|');
    if (key !== P._taskKey) { P._taskKey = key; P.renderTasks(); }
    P.renderApply();
  });
})();

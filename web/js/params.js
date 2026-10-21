/* 03 高级参数：按显存一键套用 → 梯度检查点 → 基础参数（实时算总步数）→ 学习率 → 其他高级设置
   表单完全按底模注册表（app/models/*.py 的 params()）生成，新增底模不用改这里 */
(function () {
  const { $, esc, h, api } = App;
  let statsTimer = null, lastStats = null;

  function visible(f, p) {
    if (!f.show_if) return true;
    return Object.entries(f.show_if).every(([k, v]) => Array.isArray(v) ? v.includes(p[k]) : p[k] === v);
  }
  function fmtNum(v, f) {
    if (v === '' || v === null || v === undefined) return '';
    if (f.sci && Number(v) !== 0 && Math.abs(Number(v)) < 0.001) return Number(v).toExponential().replace(/\.?0+e/, 'e');
    return String(v);
  }

  // 这几项在「训练长度」卡的总步数栏里直接改，下面的表单不再重复
  const CALC_KEYS = ['repeats', 'epochs', 'max_train_steps', 'batch_size', 'steps_mode', 'use_folder_repeats', 'gradient_checkpointing'];

  // 开关类参数：紧凑的勾选块，排成一行放在输入框网格下面
  function switchEl(f, p, defaults) {
    const el = h(`<label class="sw"><input type="checkbox"><i></i><span>${esc(f.label)}</span>${f.help ? `<em class="q" data-tip="${esc(f.help)}">?</em>` : ''}</label>`);
    const cb = el.querySelector('input'); cb.checked = !!p[f.key];
    el.classList.toggle('changed', p[f.key] !== undefined && !!p[f.key] !== !!defaults[f.key]);
    cb.onchange = () => {
      App.setParam(f.key, cb.checked);
      el.classList.toggle('changed', cb.checked !== !!defaults[f.key]);
      if (App.spec().params.some(x => x.show_if && f.key in x.show_if)) P.render();
      P.scheduleStats();
    };
    return el;
  }

  function fieldEl(f, p, defaults) {
    const v = p[f.key];
    const wide = f.type === 'textarea';                     // 多行文本占满一行
    const span2 = f.type === 'dir' || f.type === 'file';    // 路径框占两格
    const wrap = h(`<div class="field ${wide ? 'wide' : ''} ${span2 ? 'span2' : ''}" data-key="${f.key}"><label><span>${esc(f.label)}</span>${f.help ? `<i class="q" data-tip="${esc(f.help)}">?</i>` : ''}</label></div>`);
    let input;
    const set = val => {
      App.setParam(f.key, val);
      wrap.classList.toggle('changed', JSON.stringify(val) !== JSON.stringify(defaults[f.key]));
      if (App.state.specs && App.spec().params.some(x => x.show_if && f.key in x.show_if)) P.render();
      P.scheduleStats();
    };
    if (f.type === 'bool') {
      input = h(`<label class="chk-row"><input type="checkbox"><i></i><span>${v ? '开启' : '关闭'}</span></label>`);
      const cb = input.querySelector('input'); cb.checked = !!v;
      cb.onchange = () => { input.querySelector('span').textContent = cb.checked ? '开启' : '关闭'; set(cb.checked); };
      wrap.querySelector('label').after(input);
    } else if (f.type === 'select') {
      input = h(`<select>${f.options.map(o => `<option value="${esc(o[0])}">${esc(o[1])}</option>`).join('')}</select>`);
      input.value = String(v);
      input.onchange = () => { const o = f.options.find(o => String(o[0]) === input.value); set(o ? o[0] : input.value); };
      wrap.querySelector('label').after(input);
    } else if (f.type === 'dir' || f.type === 'file') {
      input = h(`<div class="row"><input type="text" class="grow"><button class="btn">浏览…</button>${f.type === 'dir' ? '<button class="btn" data-open>打开</button>' : ''}</div>`);
      const t = input.querySelector('input'); t.value = v || '';
      t.onchange = () => set(t.value.trim());
      input.querySelector('.btn').onclick = async () => {
        const r = await App.pick({ title: f.label, dir: f.type === 'dir', exts: f.type === 'file' ? ['.safetensors'] : null, start: t.value });
        if (r) { t.value = r; set(r); }
      };
      const ob = input.querySelector('[data-open]'); if (ob) ob.onclick = () => t.value && api('fs.open', { path: t.value });
      wrap.querySelector('label').after(input);
    } else if (f.type === 'textarea') {
      input = h('<textarea rows="2"></textarea>'); input.value = v || '';
      input.onchange = () => set(input.value);
      wrap.querySelector('label').after(input);
    } else {
      const isNum = f.type === 'int' || f.type === 'float';
      input = h(`<input type="${isNum && !f.sci ? 'number' : 'text'}">`);
      if (f.min !== undefined) input.min = f.min; if (f.max !== undefined) input.max = f.max;
      if (f.step) input.step = f.step; else if (f.type === 'int') input.step = 1; else if (f.type === 'float') input.step = 'any';
      input.value = isNum ? fmtNum(v, f) : (v || '');
      const commit = () => {
        let val = input.value.trim();
        if (isNum) {
          if (val === '') val = defaults[f.key];
          else { val = f.type === 'int' ? parseInt(val, 10) : parseFloat(val); if (!isFinite(val)) { App.toast(`「${f.label}」不是有效的数字`, 'bad'); input.value = fmtNum(p[f.key], f); return; } }
          if (f.min !== undefined && val < f.min) val = f.min;
          if (f.max !== undefined && val > f.max) val = f.max;
          input.value = fmtNum(val, f);
        }
        set(val);
      };
      input.onchange = commit;
      if (isNum) input.oninput = () => { if (['repeats', 'epochs', 'max_train_steps', 'batch_size', 'grad_accum'].includes(f.key)) { const n = parseFloat(input.value); if (isFinite(n)) { App.cur().params[f.key] = f.type === 'int' ? Math.round(n) : n; P.scheduleStats(); } } };
      wrap.querySelector('label').after(input);
    }
    if (JSON.stringify(v) !== JSON.stringify(defaults[f.key]) && v !== undefined) wrap.classList.add('changed');
    return wrap;
  }

  const P = App.pages.params = {
    init() {
      $('#p-reset').onclick = async () => {
        if (!(await App.confirm('恢复默认值', '把这个底模的所有参数恢复成推荐默认值？（训练集位置和保存名称保留）', '恢复'))) return;
        const p = App.params(); const keep = { dataset_dir: p.dataset_dir, output_name: p.output_name };
        App.cur().params = keep; App.saveDraft(true); P.render(); P.scheduleStats();
      };
      $('#p-check-ds').onclick = () => P.checkDataset(true);
      App.on('model_type', () => { if (App.page === 'params') P.show(); });
    },
    show() {
      if (!App.spec()) { $('#p-basic').innerHTML = '<div class="hint">先到「模型设置」选择底模。</div>'; return; }
      P.renderVram(); P.render(); P.scheduleStats(0);
    },
    render() {
      const spec = App.spec(), p = App.params(), defaults = App.defaults(spec);
      const changed = f => p[f.key] !== undefined && JSON.stringify(p[f.key]) !== JSON.stringify(defaults[f.key]);
      const by = sec => spec.params.filter(f => f.section === sec && visible(f, p) && !CALC_KEYS.includes(f.key));
      // 一组字段 → 输入框网格 + 下面一排勾选开关
      const fill = (grid, sw, fs) => {
        grid.innerHTML = ''; sw.innerHTML = '';
        fs.filter(f => f.type !== 'bool').forEach(f => grid.append(fieldEl(f, p, defaults)));
        fs.filter(f => f.type === 'bool').forEach(f => sw.append(switchEl(f, p, defaults)));
        sw.hidden = !sw.children.length;
        grid.hidden = !grid.children.length;
        sw.classList.toggle('solo', grid.hidden);   // 这一组只有开关时不画分隔线
      };
      // 最上面：梯度检查点
      const top = $('#p-top'); top.innerHTML = '';
      spec.params.filter(f => f.section === 'top').forEach(f => {
        const el = h(`<label class="chk-row big"><input type="checkbox"><i></i></label>`);
        const cb = el.querySelector('input'); cb.checked = !!p[f.key];
        cb.onchange = () => { App.setParam(f.key, cb.checked); P.renderVram(); };
        top.append(el, h(`<div class="bt-text"><b>${esc(f.label)}</b><div class="hint">${esc(f.help || '')}</div></div>`));
      });
      // 训练长度：按轮数 / 按总步数、沿用文件夹重复次数
      App.$$('#p-mode button').forEach(b => {
        b.classList.toggle('on', b.dataset.v === (p.steps_mode || 'epochs'));
        b.onclick = () => { App.setParam('steps_mode', b.dataset.v); P.render(); P.scheduleStats(0); };
      });
      const calcSw = $('#p-calc-sw'); calcSw.innerHTML = '';
      spec.params.filter(f => f.key === 'use_folder_repeats').forEach(f => calcSw.append(switchEl(f, p, defaults)));
      for (const [sec, id] of [['basic', '#p-basic'], ['lr', '#p-lr']]) {
        const grid = $(id);
        let sw = grid.nextElementSibling;
        if (!sw || !sw.classList.contains('switches')) { sw = h('<div class="switches"></div>'); grid.after(sw); }
        fill(grid, sw, by(sec));
      }
      // 其他高级设置：分页标签
      const groups = {};
      by('adv').forEach(f => (groups[f.group || '其他'] = groups[f.group || '其他'] || []).push(f));
      const order = spec.adv_order || [];
      const rank = g => (order.indexOf(g) + 1) || 99;
      const names = Object.keys(groups).sort((a, b) => rank(a) - rank(b));
      if (!names.includes(P._tab)) P._tab = names[0];
      const tabs = $('#p-adv-tabs'); tabs.innerHTML = '';
      names.forEach(g => {
        const n = spec.params.filter(f => f.section === 'adv' && (f.group || '其他') === g && changed(f)).length;
        const b = h(`<button class="${g === P._tab ? 'on' : ''}">${esc(g)}${n ? `<i class="dot" title="${n} 项已改"></i>` : ''}</button>`);
        b.onclick = () => { P._tab = g; P.render(); };
        tabs.append(b);
      });
      const adv = $('#p-adv'); adv.innerHTML = '';
      const grid = h('<div class="fields"></div>'), sw = h('<div class="switches"></div>');
      adv.append(grid, sw);
      fill(grid, sw, groups[P._tab] || []);
      const total = spec.params.filter(f => f.section === 'adv' && changed(f)).length;
      $('#p-adv-hint').textContent = total ? `共改了 ${total} 项；改过的项左边有黄色标记` : '一般保持默认即可；改过的项左边有黄色标记';
    },
    renderVram() {
      const spec = App.spec(), g = App.gpu || App.state.gpu || {};
      $('#pv-gpu').textContent = g.name ? `${g.name.replace('NVIDIA GeForce ', '')} · ${g.total_gb} GB` : '没有检测到 NVIDIA 显卡';
      const rec = spec.presets.reduce((a, pr) => (g.total_gb || 0) + 0.5 >= pr.min_gb ? pr : a, spec.presets[0]);
      $('#pv-gpu-sub').textContent = rec ? `推荐「${rec.label}」档：${rec.note}` : '';
      const p = App.params(), el = $('#pv-presets'); el.innerHTML = '';
      spec.presets.forEach(pr => {
        const on = Object.entries(pr.values).every(([k, v]) => p[k] === v);
        const b = h(`<button class="preset ${pr === rec ? 'rec' : ''} ${on ? 'on' : ''}" title="${esc(Object.entries(pr.values).map(([k, v]) => k + '=' + v).join('  '))}"><b>${esc(pr.label)}</b><span>${esc(pr.note)}</span></button>`);
        b.onclick = () => {
          const c = App.cur(); Object.assign(c.params, pr.values); App.saveDraft(true);
          App.toast(`已套用 ${pr.label} 显存档：${pr.note}`, 'ok'); P.render(); P.renderVram(); P.scheduleStats(0); App.emit('params');
        };
        el.append(b);
      });
    },
    scheduleStats(ms = 300) { clearTimeout(statsTimer); statsTimer = setTimeout(P.stats, ms); },
    async stats() {
      const p = App.params(); let r;
      try { r = await api('dataset.stats', { params: p }); } catch (e) { return; }
      lastStats = r; const e = r.estimate;
      const ga = p.grad_accum || 1, byEpoch = p.steps_mode !== 'steps';
      const multi = r.subsets.length > 1;
      $('#steps-sub').textContent = (r.num_gpus > 1 ? `${r.num_gpus} 张卡并行：每步共处理 ${p.batch_size || 1}×${r.num_gpus} 张图，每轮步数已按卡数平分。` : '') + (multi ? '多个子集：' + r.subsets.map(s => `${App.base(s.dir)} ${s.images} 张×${s.repeats}`).join('，') + '。' : '') +
        (byEpoch ? '' : '按总步数训练：轮数由总步数换算。') + (ga > 1 ? `梯度累积 ${ga} 步，每轮步数已按累积换算。` : '') +
        (p.dataset_dir ? '' : '还没设置训练集位置，图片数按 0 计算。');
      P.buildCalc(byEpoch);
      const set = (k, v) => { const el = $(`#steps-calc [data-c="${k}"]`); if (!el) return; if (el.tagName === 'INPUT') { if (document.activeElement !== el) el.value = v; } else el.textContent = v; };
      set('images', e.images);
      // 所有子集都按文件夹名（「5_xxx」）定重复次数时，「每张重复」不起作用：锁住，显示文件夹里的次数
      const fromFolder = r.subsets.length > 0 && r.subsets.every(s => s.from_folder);
      const someFolder = r.subsets.some(s => s.from_folder);
      const rep = $('#steps-calc [data-c="repeats"]'), repField = $('#page-params .field[data-key="repeats"] input');
      const reps = [...new Set(r.subsets.map(s => s.repeats))];
      if (rep) {
        rep.disabled = fromFolder; rep.parentElement.classList.toggle('edit', !fromFolder);
        rep.title = fromFolder ? '按文件夹名里的重复次数；要改请改文件夹名，或关掉「沿用文件夹名里的重复次数」' : '直接修改';
        if (fromFolder) { rep.value = reps.length === 1 ? reps[0] : ''; rep.placeholder = reps.join('/'); } else { rep.placeholder = ''; set('repeats', p.repeats); }
      }
      if (repField) { repField.disabled = fromFolder; repField.title = rep ? rep.title : ''; }
      if (someFolder && !fromFolder) $('#steps-sub').textContent += '「每张重复」只对文件夹名没有数字前缀的图片生效。';
      if (p.use_folder_repeats !== false && !someFolder && r.subsets.length)
        $('#steps-sub').textContent += '训练集里没有「数字_名字」格式的子文件夹（训练集文件夹本身的名字不算），「沿用文件夹名里的重复次数」不起作用，按「每张重复」算。';
      set('batch_size', p.batch_size || 1); set('per_epoch', e.steps_per_epoch);
      set('epochs', byEpoch ? (p.epochs || 1) : e.epochs); set('max_train_steps', byEpoch ? e.total_steps : (p.max_train_steps || 0));
      $('#p-basic-hint').textContent = p.dataset_dir ? (e.images ? `训练集共 ${e.images} 张图` : '训练集里没有图片！') : '';
    },
    // 总步数栏：重复次数、单批数量、轮数（按总步数模式时是总步数）直接在栏里改
    buildCalc(byEpoch) {
      const el = $('#steps-calc');
      if (el.dataset.mode === String(byEpoch)) return;
      el.dataset.mode = String(byEpoch);
      const cell = (cn, en, key, edit, cls = '') => `<div class="${cls} ${edit ? 'edit' : ''}"><div class="k">${cn}</div><div class="en">${en}</div>${edit
        ? `<input class="v" type="number" min="1" data-c="${key}" title="直接修改">` : `<div class="v" data-c="${key}">—</div>`}</div>`;
      const op = s => `<div class="op">${s}</div>`;
      el.innerHTML = cell('图片', 'IMAGES', 'images', false) + op('×') + cell('每张重复', 'REPEATS', 'repeats', true) + op('÷') +
        cell('单批数量', 'BATCH', 'batch_size', true) + op('=') + cell('每轮步数', 'STEPS / EPOCH', 'per_epoch', false) + op('×') +
        cell('训练轮数', 'EPOCHS', 'epochs', byEpoch) + op('=') + cell('总步数', 'TOTAL STEPS', 'max_train_steps', !byEpoch, 'total');
      App.$$('input', el).forEach(inp => {
        const key = inp.dataset.c;
        const commit = () => {
          let v = parseInt(inp.value, 10);
          if (!isFinite(v) || v < 1) return;
          if (key === 'batch_size') v = Math.min(v, 64);
          App.setParam(key, v);
          const f = $(`#page-params .field[data-key="${key}"] input`); if (f) f.value = v;   // 同步下面表单里的同一项
          P.scheduleStats(150);
          if (key === 'batch_size') P.renderVram();
        };
        inp.oninput = commit;
        inp.onchange = () => { commit(); P.scheduleStats(0); };
        inp.onkeydown = ev => { if (ev.key === 'Enter') inp.blur(); };
      });
    },
    async checkDataset(manual) {
      const p = App.params();
      if (!p.dataset_dir) { App.toast('先设置训练集位置', 'bad'); return; }
      const r = await api('dataset.check', { dir: p.dataset_dir, ext: p.caption_extension });
      if (!r.issues.length) { App.toast(`训练集没有问题：${r.stats.images} 张图，都有标签`, 'ok'); return; }
      // 手动检查时也走同一个确认弹窗；清理动作需要传训练集目录
      r.issues.forEach(i => i.fixes.forEach(f => { if (f.id === 'clean_dataset') f.data.dataset_dir = p.dataset_dir; }));
      const issues = r.issues.map(i => Object.assign({}, i, { fixes: i.fixes.filter(f => f.id !== 'ignore') }));
      App.showIssues(issues, null, '');
    },
  };
  App.on('tick', () => { if (App.page === 'params' && App.spec() && !P._vramOnce) { P._vramOnce = true; P.renderVram(); } });
})();

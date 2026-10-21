/* 02 模型设置：选底模 → 自动在 WebUI / ComfyUI / 共享模型库里找文件 → 每个槽位优先选 base 模型，也可自选 */
(function () {
  const { $, esc, h, api } = App;
  let listing = null, scanTimer = null;

  const KIND_LABEL = { anima_dit: 'Anima 主模型', krea2_dit: 'Krea 2 主模型', qwen3_06b: 'Qwen3-0.6B', qwen3vl_4b: 'Qwen3-VL-4B',
    qwen_image_vae: 'Qwen-Image VAE', wan_vae: 'Wan / Qwen VAE', sdxl_checkpoint: 'SDXL 大模型', sdxl_vae: 'SDXL VAE', sd_vae: 'SD VAE',
    moss_dit: 'MOSS-Image 主模型', qwen35_9b: 'Qwen3.5-9B', flux2_vae: 'FLUX.2 VAE' };

  const P = App.pages.model = {
    init() {
      $('#scan-start').onclick = async () => { await api('scan.start'); P.pollScan(); };
      $('#scan-add').onclick = async () => {
        const d = await App.pick({ title: '添加模型搜索目录（会递归查找里面的 safetensors）' });
        if (!d) return;
        const s = App.state.settings; const dirs = Array.from(new Set([...(s.extra_search_dirs || []), d]));
        App.state.settings = await api('settings.save', { patch: { extra_search_dirs: dirs } });
        P.renderInstalls(); P.loadModels(true);
      };
      $('#slots-refresh').onclick = () => P.loadModels(true);
      App.on('model_type', () => { if (App.page === 'model') { P.renderSpecs(); P.loadModels(); } });
    },
    show() { P.renderSpecs(); P.renderInstalls(); P.pollScan(); if (App.draft.model_type) P.loadModels(); else $('#slots').innerHTML = '<div class="hint">先在上面选择一个底模。</div>'; },
    hide() { clearTimeout(scanTimer); },

    renderSpecs() {
      const el = $('#spec-cards'); el.innerHTML = '';
      App.state.specs.forEach(s => {
        const c = h(`<div class="spec-card ${App.draft.model_type === s.id ? 'on' : ''}"><div class="sc-eng">${esc(s.engine.toUpperCase())}</div>
          <div class="sc-name">${esc(s.label)}</div><div class="sc-sub">${esc(s.subtitle)}</div><i class="sc-mark"></i></div>`);
        c.onclick = () => { if (App.draft.model_type !== s.id) App.setModelType(s.id); };
        el.append(c);
      });
      el.append(h('<div class="spec-card more">更多底模 · 敬请期待</div>'));
    },

    async pollScan() {
      clearTimeout(scanTimer);
      const st = await api('scan.state');
      App.state.scan = st;
      P.renderInstalls();
      if (st.running) scanTimer = setTimeout(P.pollScan, 1200);
      else if (P._wasScanning) { P._wasScanning = false; P.loadModels(true); }
      if (st.running) P._wasScanning = true;
    },
    renderInstalls() {
      const st = App.state.scan || {}, el = $('#installs');
      const names = { webui: 'WebUI', comfyui: 'ComfyUI', launcher: 'WWY 启动器', library: '共享模型库' };
      $('#scan-hint').textContent = st.running ? `正在搜索… ${st.current || ''}` : `找到 ${(st.installs || []).length} 处；搜索结果会缓存，装了新的 WebUI / ComfyUI 后点「重新搜索电脑」`;
      const extra = (App.state.settings.extra_search_dirs || []).map(d => ({ type: 'custom', path: d }));
      const rows = [...(st.installs || []), ...extra];
      el.innerHTML = rows.map(x => `<div><span class="badge ${x.type === 'library' ? 'y' : ''}">${x.type === 'custom' ? '自定义' : names[x.type] || x.type}</span><code>${esc(x.path)}</code>
        ${x.type === 'custom' ? `<button class="icon-btn" data-rm="${esc(x.path)}" title="移除">✕</button>` : ''}</div>`).join('') || (st.running ? '' : '<div class="muted">没有找到 WebUI / ComfyUI。可以「添加搜索目录」手动指定模型所在文件夹，或在下面一键下载官方模型。</div>');
      App.$$('[data-rm]', el).forEach(b => b.onclick = async () => {
        const dirs = (App.state.settings.extra_search_dirs || []).filter(d => d !== b.dataset.rm);
        App.state.settings = await api('settings.save', { patch: { extra_search_dirs: dirs } }); P.renderInstalls(); P.loadModels(true);
      });
    },

    async loadModels(force) {
      const spec = App.spec(); if (!spec) return;
      $('#slots').innerHTML = '<div class="hint">正在读取模型文件…（第一次要读每个文件的文件头，稍等几秒）</div>';
      let res;
      try { res = await api('models.list', { model_type: spec.id, force: !!force }); }
      catch (e) { $('#slots').innerHTML = `<div class="hint bad">${esc(e.message)}</div>`; return; }
      if (App.draft.model_type !== spec.id) return;   // 等结果期间换了底模，这份结果作废
      listing = res;
      // 自动选择：没选过的槽位选分数最高的；已选的文件不在候选里时，按内容再校验一次，不匹配就换掉
      const c = App.draft.by_model[spec.id]; let changed = false;
      for (const s of spec.slots) {
        const cands = listing.slots[s.key] || [];
        const curp = c.models[s.key];
        if (curp && !cands.some(x => x.path === curp)) {
          const r = await api('models.check', { model_type: spec.id, slot: s.key, path: curp }).catch(() => ({ ok: false, message: '' }));
          if (!r.ok) { App.toast(`${s.label}：之前选的 ${App.base(curp)} 不能用（${r.message}），已自动更换`, 'warn', null, null, 8000); c.models[s.key] = ''; changed = true; }
        }
        if (!c.models[s.key] && cands.length && (s.required || cands[0].score > 0)) { c.models[s.key] = cands[0].path; changed = true; }
      }
      if (App.draft.model_type !== spec.id) return;
      if (changed) App.saveDraft(true);
      P.renderSlots();
    },
    renderSlots() {
      const spec = App.spec(), c = App.cur(), el = $('#slots');
      el.innerHTML = '';
      spec.slots.forEach(s => {
        const cands = (listing && listing.slots[s.key]) || [];
        const curPath = c.models[s.key] || '';
        const inList = cands.some(x => x.path === curPath);
        const best = cands[0];
        const opt = x => {
          const tags = [];
          if (x === best) tags.push('★推荐');
          if (x.quant) tags.push(x.qtype ? x.qtype + ' 量化' : '量化');
          return `<option value="${esc(x.path)}" ${x.path === curPath ? 'selected' : ''}>${esc(x.name)} · ${App.fmtSize(x.size)}${tags.length ? ' · ' + tags.join(' · ') : ''} · ${esc(x.source.split(' · ')[0])}</option>`;
        };
        // 主模型按底模自己声明的分组（官方底模 / 各系列微调 / 加速版……）显示，电脑里有很多微调版时好挑
        const VG = spec.variant_groups || [], GROUPS = VG.map(g => [g[0], g[1]]), NOTE = Object.fromEntries(VG.map(g => [g[0], g[2] || '']));
        const grouped = cands.some(x => x.variant);
        const optsHtml = grouped
          ? GROUPS.map(([g, label]) => { const xs = cands.filter(x => x.variant === g); return xs.length ? `<optgroup label="${esc(label)} · ${xs.length} 个">${xs.map(opt).join('')}</optgroup>` : ''; }).join('')
            + cands.filter(x => !x.variant).map(opt).join('')
          : cands.map(opt).join('');
        const box = h(`<div class="slot"><div class="slot-head"><b>${esc(s.label)}</b>${s.required ? '' : '<span class="badge">可选</span>'}${cands.length > 1 ? `<span class="badge">找到 ${cands.length} 个可用的</span>` : ''}
          <span class="muted" style="font-size:11.5px">${esc(s.help || '')}</span><span class="slot-state"></span></div>
          <div class="row">${grouped ? '<div class="grow" data-mp></div>' : `<select class="grow">${s.required ? '' : '<option value="">（不使用）</option>'}${!inList && curPath ? `<option value="${esc(curPath)}" selected>自选：${esc(curPath)}</option>` : ''}${optsHtml}${!cands.length && !curPath ? '<option value="">没有找到匹配的文件</option>' : ''}</select>`}
          <button class="btn" data-pick>自选文件…</button><button class="btn" data-open title="在资源管理器中显示">📁</button></div>
          <div class="hint" data-path></div><div class="row" data-dl style="margin-top:8px;flex-wrap:wrap"></div></div>`);
        const sel = box.querySelector('select'), state = box.querySelector('.slot-state'), ph = box.querySelector('[data-path]');
        const upd = () => {
          const p = c.models[s.key] || '';
          const it = cands.find(x => x.path === p);
          ph.textContent = p ? p + (it ? `（${it.source}）` : '') : '';
          if (it && NOTE[it.variant]) ph.textContent += '\n' + NOTE[it.variant];
          if (it && it.vpred && it.variant !== 'vpred') ph.textContent += '\n这是 v-pred 模型，训练时会自动开启 v-parameterization 和 zero terminal SNR。';
          ph.style.whiteSpace = 'pre-line';
          state.innerHTML = p ? (it ? `<span class="badge ok">✓ 已识别 · ${esc(KIND_LABEL[it.kind] || it.kind)}</span>` : '<span class="badge warn">自选文件</span>')
            : (s.required ? '<span class="badge bad">未选择</span>' : '<span class="badge">未使用</span>');
        };
        const setModel = v => { c.models[s.key] = v; App.saveDraft(); upd(); App.emit('params'); };
        if (grouped)   // 主模型用可搜索的选择器（带预览图、Civitai 名称），模型多的时候好挑
          ModelPicker(box.querySelector('[data-mp]'), { items: cands, value: curPath, allowNone: !s.required, groups: GROUPS, onChange: setModel });
        else sel.onchange = () => setModel(sel.value);
        box.querySelector('[data-open]').onclick = () => c.models[s.key] && api('fs.open', { path: c.models[s.key] });
        box.querySelector('[data-pick]').onclick = async () => {
          const f = await App.pick({ title: `选择${s.label}`, dir: false, exts: ['.safetensors', '.sft'], start: c.models[s.key] || '' });
          if (!f) return;
          const r = await api('models.check', { model_type: spec.id, slot: s.key, path: f });
          if (!r.ok) { App.toast(r.message, 'bad', null, null, 8000); return; }
          c.models[s.key] = f; App.saveDraft(true); P.renderSlots(); App.toast('已选择：' + App.base(f), 'ok');
        };
        // 一键下载官方文件
        const dl = box.querySelector('[data-dl]');
        if (s.downloads && s.downloads.length) {
          if (!cands.length) dl.append(h(`<span class="hint warn" style="margin:0 6px 0 0">电脑里没找到，可以直接下载（${App.state.network === 'cn' ? '国内镜像 hf-mirror' : 'HuggingFace'}）：</span>`));
          s.downloads.forEach(d => {
            const b = h(`<button class="btn btn-sm">${cands.length ? '另外下载 ' : '下载 '}${esc(d.label)}（${d.size_gb} GB）</button>`);
            b.onclick = async () => {
              if (!(await App.confirm('下载模型', `下载 <b>${esc(d.label)}</b>（${d.size_gb} GB）<br>来源：${esc(d.repo)} / ${esc(d.file)}<br>保存到训练器的 models\\base 文件夹，下完自动选上。`, '开始下载'))) return;
              await api('download.start', Object.assign({ slot: s.key, model_type: spec.id }, d));
              App.toast('开始下载，进度在左下角', 'ok');
              P.watchDownload();
            };
            dl.append(b);
          });
        }
        upd();
        el.append(box);
      });
      if (listing && listing.broken && listing.broken.length) {
        el.append(h(`<div class="hint bad">发现 ${listing.broken.length} 个损坏 / 没下载完的模型文件：${listing.broken.slice(0, 5).map(b => esc(b.name) + '（' + esc(b.error) + '）').join('；')}</div>`));
      }
    },
    watchDownload() {
      const t = setInterval(async () => {
        const run = (App.jobs || []).filter(j => j.kind === 'download' && j.status === 'running');
        if (!run.length) {
          clearInterval(t);
          const d = await api('boot'); App.draft.by_model = d.draft.by_model || App.draft.by_model;
          if (App.page === 'model') P.loadModels(true);
        }
      }, 3000);
    },
  };
})();

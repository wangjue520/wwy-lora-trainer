/* 04 标签反推：WD14 / 本地自然语言（ToriiGate、JoyCaption、Qwen3-VL abliterated）/ WD14+自然语言 / API；触发词 */
(function () {
  const { $, esc, h, api } = App;
  let jobId = null, timer = null;

  const MODES = [
    { id: 'wd14', name: 'WD14 标签', desc: 'danbooru 标签，速度快，适合二次元。' },
    { id: 'vlm', name: '本地自然语言', desc: 'ToriiGate / JoyCaption / Qwen3-VL，本机显卡运行，不上传图片。' },
    { id: 'hybrid', name: 'WD14 + 自然语言', desc: '先打标签，再把标签交给自然语言模型参考写描述，两者合在一起。' },
    { id: 'api', name: 'API 自然语言', desc: 'OpenAI / Gemini / 通义 / 硅基流动等，不占显卡，按量计费。' },
  ];
  // 给自然语言模型的提示词模板（文字在 captionutil.PROMPT_PRESETS）。练画风不写画风、练人物不写人物固有特征，留给触发词去学
  const PRESETS = [
    ['moss', 'MOSS 官方风格（通用 / 练人物，保留图中文字）'], ['moss_style', 'MOSS · 练画风（不描述画风）'],
    ['detailed', '详细描述（通用）'], ['short', '简短描述'], ['style', '练画风（不描述画风）'], ['character', '练人物（不描述人物固有特征）'], ['custom', '自定义'],
  ];

  const T = () => App.draft.tag;
  // 底模推荐的反推方式。没手动选过时每次打开本页都跟着当前底模走（第一次打开训练器时还没选底模，不能把那时的默认值存死）
  const specMode = () => ({ natural: 'vlm', tags: 'wd14' })[(App.spec() || {}).caption_style] || 'hybrid';
  const defaults = () => ({
    mode: specMode(),
    wd14: { model: 'eva02-large', general_threshold: 0.35, character_threshold: 0.85, replace_underscore: true, include_rating: false, exclude: '' },
    vlm: { model: ['krea2', 'moss'].includes((App.spec() || {}).id) ? 'qwen3vl8b' : 'toriigate', quant: 'auto',
      preset: (App.spec() || {}).id === 'moss' ? 'moss' : 'detailed', prompt: App.state.tagger.prompts[(App.spec() || {}).id === 'moss' ? 'moss' : 'detailed'],
      max_tokens: 512, torii_format: 'long', feed_tags: true, use_names: true, temperature: 0.6 },
    api: { preset: 'Google Gemini', base_url: App.state.tagger.api_presets[1].base_url, model: App.state.tagger.api_presets[1].model, api_key: '', concurrency: 4,
      fallback_local: true, fallback_model: 'qwen3vl8b' },
    hybrid_keep_tags: true, trigger: '', existing: 'overwrite', ext: '.txt',
  });

  const P = App.pages.tag = {
    init() {
      const d = App.draft.tag = Object.assign(defaults(), App.draft.tag || {});
      ['wd14', 'vlm', 'api'].forEach(k => d[k] = Object.assign(defaults()[k], d[k] || {}));
      $('#tag-dir').onchange = e => { d.dir = e.target.value.trim(); App.saveDraft(); P.dirHint(); App.syncDataset(d.dir, 'tag'); };
      $('#tag-dir-browse').onclick = async () => { const p = await App.pick({ title: '选择要反推的训练集', start: $('#tag-dir').value }); if (p) { $('#tag-dir').value = p; $('#tag-dir').onchange({ target: { value: p } }); } };
      $('#tag-dir-sync').onclick = () => { const p = App.params().dataset_dir; if (p) { $('#tag-dir').value = p; $('#tag-dir').onchange({ target: { value: p } }); } };
      $('#tag-trigger').onchange = e => { T().trigger = e.target.value; App.saveDraft(); };
      $('#tag-autotrig').onchange = e => { T().auto_trigger = e.target.checked; App.saveDraft(); };
      App.$$('#tag-trigmode button').forEach(b => b.onclick = () => { T().trigger_mode = b.dataset.v; App.saveDraft(); P.renderSeg(); });
      $('#tag-ext').onchange = e => { T().ext = e.target.value.trim() || '.txt'; App.saveDraft(); };
      App.$$('#tag-existing button').forEach(b => b.onclick = () => { T().existing = b.dataset.v; App.saveDraft(); P.renderSeg(); });
      $('#tag-start').onclick = P.start;
      $('#tag-stop').onclick = () => jobId && api('jobs.stop', { id: jobId });
      $('#tag-goedit').onclick = () => P.goEdit();
      $('#tag-resume').onclick = P.resume;
    },
    show() {
      const d = T();
      if (!d.mode_chosen) d.mode = specMode();
      if (!d.dir) d.dir = App.params().dataset_dir || App.state.tagger.last_dir || '';
      $('#tag-dir').value = d.dir || ''; $('#tag-trigger').value = d.trigger || ''; $('#tag-ext').value = d.ext || '.txt';
      P.dirHint(); P.renderModes(); P.renderOpts(); P.renderSeg();
      const run = (App.jobs || []).find(j => j.kind === 'tag' && j.status === 'running');
      if (run) { jobId = run.id; P.watch(); } else P.refreshResume();
    },
    // 上次停止 / 出错时没做完的图：显示「继续反推」，只做剩下的，设置沿用上次的
    async refreshResume() {
      const r = await api('tag.resume_info').catch(() => null), b = $('#tag-resume');
      b.style.display = r ? '' : 'none';
      if (!r) return;
      b.textContent = `继续反推（还剩 ${r.left} 张）`;
      b.title = `${r.dataset_dir}
上次做到 ${r.done} / ${r.total}，用上次的设置接着做剩下的图`;
      if (!$('#tag-prog-text').textContent) $('#tag-prog-text').textContent = `上次停在 ${r.done} / ${r.total}：${r.dataset_dir}`;
    },
    async resume() {
      const training = (App.queue ? App.queue.tasks : []).some(t => t.status === 'running');
      if (training && !(await App.confirm('正在训练', '现在有任务在训练，本地反推会和训练抢显存，可能导致训练显存不足报错。仍然继续吗？', '仍然继续', true))) return;
      try { jobId = await api('tag.resume'); } catch (e) { App.toast(e.message, 'bad'); return; }
      $('#tag-log').innerHTML = ''; P.watch();
    },
    hide() { },
    renderSeg() {
      App.$$('#tag-existing button').forEach(b => b.classList.toggle('on', b.dataset.v === T().existing));
      App.$$('#tag-trigmode button').forEach(b => b.classList.toggle('on', b.dataset.v === (T().trigger_mode || 'freq')));
      $('#tag-autotrig').checked = T().auto_trigger === true;
    },
    async dirHint() {
      const d = T().dir; const el = $('#tag-dir-hint');
      if (!d) { el.textContent = ''; return; }
      try { const r = await api('fs.list', { path: d }); el.textContent = `文件夹里有 ${r.images} 张图片（包括子文件夹的会一起处理）`; el.className = 'hint'; }
      catch (e) { el.textContent = e.message; el.className = 'hint bad'; }
    },
    renderModes() {
      const el = $('#tag-modes'), spec = App.spec(); el.innerHTML = '';
      $('#tag-mode-hint').textContent = spec ? ({ natural: `${spec.label} 用自然语言描述效果最好`,
        tags: `${spec.label} 用 D 站标签（WD14）效果最好` + (spec.id === 'sdxl' ? '，Illustrious / NoobAI 就是用 D 站标签训练的' : '') }[spec.caption_style]
        || `${spec.label} 推荐「标签 + 自然语言」混合打标`) : '';
      MODES.forEach(m => {
        const c = h(`<div class="mode-card ${T().mode === m.id ? 'on' : ''}"><b>${m.name}</b><span>${m.desc}</span></div>`);
        c.onclick = () => { T().mode = m.id; T().mode_chosen = true; App.saveDraft(); P.renderModes(); P.renderOpts(); };
        el.append(c);
      });
    },
    renderOpts() {
      const d = T(), el = $('#tag-opts'), mode = d.mode; el.innerHTML = '';
      $('#tag-opt-title').textContent = { wd14: 'WD14 选项', vlm: '自然语言模型', hybrid: 'WD14 + 自然语言选项', api: 'API 设置' }[mode];
      const tg = App.state.tagger;
      const grid = h('<div class="form-grid"></div>'); el.append(grid);
      const row = (label, node, help) => { grid.append(h(`<label>${esc(label)}</label>`)); grid.append(node); if (help) grid.append(h(`<div class="fg-help">${esc(help)}</div>`)); };
      const sel = (obj, key, opts, onch) => { const s = h(`<select>${opts.map(o => `<option value="${esc(o[0])}">${esc(o[1])}</option>`).join('')}</select>`); s.value = obj[key]; s.onchange = () => { obj[key] = s.value; App.saveDraft(); onch && onch(); }; return s; };
      const num = (obj, key, step = 'any') => { const i = h(`<input type="number" step="${step}" style="width:110px">`); i.value = obj[key]; i.onchange = () => { obj[key] = parseFloat(i.value); App.saveDraft(); }; return i; };
      const txt = (obj, key, ph = '', type = 'text') => { const i = h(`<input type="${type}" placeholder="${esc(ph)}">`); i.value = obj[key] || ''; i.onchange = () => { obj[key] = i.value; App.saveDraft(); }; return i; };
      const chk = (obj, key, label) => { const c = h(`<label class="chk-row" style="justify-self:start"><input type="checkbox"><i></i><span>${esc(label)}</span></label>`); const cb = c.querySelector('input'); cb.checked = !!obj[key]; cb.onchange = () => { obj[key] = cb.checked; App.saveDraft(); }; return c; };

      if (mode === 'wd14' || mode === 'hybrid') {
        const w = d.wd14;
        // 启动器「WD14 反推」已经下载过的模型直接用，不重复下载
        const lw = tg.launcher_wd14 || {};
        const inLauncher = k => lw[(tg.wd14[k] || {}).repo.split('/').pop()];
        row('WD14 模型', sel(w, 'model', Object.entries(tg.wd14).map(([k, v]) => [k, v.label + (inLauncher(k) ? ' · 启动器已下载' : '')]), () => P.renderOpts()),
          inLauncher(w.model) ? `直接用 WWY 启动器里下好的模型：${inLauncher(w.model)}`
            : (tg.launcher_wd14_dir && ['vit', 'eva02-large'].includes(w.model) ? `第一次使用会下载到 WWY 启动器的模型目录，启动器也能直接用：${tg.launcher_wd14_dir}`
              : '第一次使用会自动下载到训练器的 models/taggers'));
        row('通用标签阈值', num(w, 'general_threshold', 0.01), '越低标签越多（也越容易有错的），0.3–0.4 常用');
        row('角色标签阈值', num(w, 'character_threshold', 0.01));
        row('排除标签', txt(w, 'exclude', '逗号分隔，例如 simple background, white background'));
        row('', chk(w, 'replace_underscore', '下划线换成空格（Anima 用空格格式）'));
        row('', chk(w, 'include_rating', '加上分级标签（general / sensitive …）'));
      }
      // 选的是预设（不是自定义）就始终用当前版本的预设文字：草稿里存的可能是旧版本或者加载失败时留下的空指令
      const vp = d.vlm;
      if (vp && vp.preset !== 'custom' && tg.prompts[vp.preset] && vp.prompt !== tg.prompts[vp.preset]) { vp.prompt = tg.prompts[vp.preset]; App.saveDraft(); }
      if (mode === 'vlm' || mode === 'hybrid') {
        const v = d.vlm;
        if (mode === 'hybrid') grid.append(h('<div style="grid-column:1/-1;border-top:1px solid var(--line);margin:4px 0"></div>'));
        row('自然语言模型', sel(v, 'model', Object.entries(tg.vlm).map(([k, x]) => [k, `${x.label} · 约 ${x.size_gb} GB`]), () => P.renderOpts()));
        row('精度', sel(v, 'quant', [['auto', '自动（按剩余显存选，推荐）'], ['bf16', 'bf16（质量最好，显存够就选它）'], ['8bit', '8bit（显存减半）'], ['4bit', '4bit（显存最小，略降质量）']]),
          '第一次使用会自动下载模型到 models/taggers；训练时不要同时反推，会抢显存');
        if (v.model === 'toriigate') {
          row('描述格式', sel(v, 'torii_format', [['long', '详细（2–5 段）'], ['short', '简短']]));
          row('', chk(v, 'use_names', '识别并写出角色名'));
        } else {
          const pre = sel(v, 'preset', PRESETS,
            () => { if (v.preset !== 'custom') { v.prompt = tg.prompts[v.preset]; App.saveDraft(); } P.renderOpts(); });
          row('提示词模板', pre);
          const ta = h('<textarea rows="4"></textarea>'); ta.value = v.prompt || ''; ta.onchange = () => { v.prompt = ta.value; v.preset = 'custom'; App.saveDraft(); };
          row('给模型的指令', ta);
        }
        row('最长输出', num(v, 'max_tokens', 1), 'token 数，详细描述 400–600 足够');
        if (mode === 'hybrid') {
          row('', chk(v, 'feed_tags', '把 WD14 标签交给自然语言模型参考（描述更准）'));
          row('', chk(d, 'hybrid_keep_tags', '标签文件里保留 WD14 标签（关掉则只写自然语言）'));
        }
      }
      if (mode === 'api') {
        const a = d.api;
        const pre = h(`<select>${tg.api_presets.map(x => `<option>${esc(x.name)}</option>`).join('')}<option>自定义</option></select>`);
        pre.value = a.preset || '自定义';
        pre.onchange = () => { const x = tg.api_presets.find(x => x.name === pre.value); a.preset = pre.value; if (x) { a.base_url = x.base_url; a.model = x.model; } App.saveDraft(); P.renderOpts(); };
        row('服务商', pre);
        row('接口地址', txt(a, 'base_url', 'https://…/v1'), 'OpenAI 兼容格式（…/v1/chat/completions）');
        row('API Key', txt(a, 'api_key', 'sk-…', 'password'), '只保存在本机 data/draft.json');
        row('模型名', txt(a, 'model', '必须是支持图片输入的模型'));
        row('同时请求数', num(a, 'concurrency', 1));
        row('', chk(a, 'fallback_local', 'API 拒答时用本地模型补上'));
        row('补拒答的本地模型', sel(a, 'fallback_model', Object.entries(tg.vlm).map(([k, x]) => [k, `${x.label} · 约 ${x.size_gb} GB`])),
          '拒答的图在 API 跑完后统一用它反推，用上面同一段指令（ToriiGate 用它自己的格式）；需要训练环境，第一次用会下载模型');
        const v = d.vlm;
        const pre2 = sel(v, 'preset', PRESETS,
          () => { if (v.preset !== 'custom') { v.prompt = tg.prompts[v.preset]; App.saveDraft(); } P.renderOpts(); });
        row('提示词模板', pre2);
        const ta = h('<textarea rows="4"></textarea>'); ta.value = v.prompt || ''; ta.onchange = () => { v.prompt = ta.value; v.preset = 'custom'; App.saveDraft(); };
        row('给模型的指令', ta);
        const tb = h('<div class="row"><button class="btn btn-sm">测试连接</button><span class="hint" style="margin:0"></span></div>');
        tb.querySelector('button').onclick = async () => {
          tb.querySelector('span').textContent = '测试中…';
          const r = await api('tag.test_api', { api: a });
          tb.querySelector('span').textContent = r.message; tb.querySelector('span').className = 'hint ' + (r.ok ? 'ok' : 'bad');
        };
        row('', tb);
      }
    },
    async start() {
      const d = T();
      if (!d.mode_chosen) d.mode = specMode();
      if (!d.dir) { App.toast('先选择训练集文件夹', 'bad'); return; }
      if (d.mode !== 'api' && !App.state.env.env) {
        const st = await api('env.status'); App.state.env = st;
        if (!st.env) { App.toast('本地反推需要先安装训练环境；API 反推不需要', 'bad', '去安装', () => App.go('settings'), 10000); return; }
      }
      const training = (App.queue ? App.queue.tasks : []).some(t => t.status === 'running');
      if (training && d.mode !== 'api' && !(await App.confirm('正在训练', '现在有任务在训练，本地反推会和训练抢显存，可能导致训练显存不足报错。仍然开始吗？', '仍然开始', true))) return;
      if (d.existing === 'overwrite' && !(await App.confirm('覆盖已有标签', '已有的标签文件会被覆盖（不会备份）。确定开始吗？', '开始反推'))) return;
      const cfg = { dataset_dir: d.dir, mode: d.mode, wd14: d.wd14, vlm: d.vlm, api: d.api, hybrid_keep_tags: d.hybrid_keep_tags,
        out: { trigger: d.trigger, existing: d.existing, ext: d.ext } };
      try { jobId = await api('tag.start', { cfg }); } catch (e) { App.toast(e.message, 'bad'); return; }
      $('#tag-log').innerHTML = ''; P.watch();
    },
    watch() {
      clearInterval(timer);
      $('#tag-start').disabled = true; $('#tag-stop').disabled = false; $('#tag-resume').style.display = 'none';
      timer = setInterval(App.poll(async () => {
        const j = await api('jobs.get', { id: jobId, log: 300 }).catch(() => null);
        if (!j) return;
        $('#tag-prog i').style.width = (j.progress || 0) * 100 + '%';
        $('#tag-prog').classList.toggle('indet', j.status === 'running' && !j.progress);
        $('#tag-prog-text').textContent = j.progress_text || '';
        const box = $('#tag-log'); box.innerHTML = (j.log || []).map(l => `<div class="${/失败|出错|Error/.test(l) ? 'l-err' : ''}">${esc(l)}</div>`).join(''); box.scrollTop = box.scrollHeight;
        if (j.status !== 'running') {
          clearInterval(timer); $('#tag-start').disabled = false; $('#tag-stop').disabled = true;
          P.refreshResume();
          if (j.status === 'done') {
            $('#tag-prog-text').textContent = `完成：写入 ${j.result.count} 个标签文件${j.result.refused ? `（${j.result.refused} 张被 API 拒答，本地模型补上 ${j.result.local} 张）` : ''}${j.result.failed ? `，${j.result.failed} 张失败` : ''}`;
            App.toast('反推完成', 'ok', '去编辑标签', () => P.goEdit(), 10000);
            // 用这次反推的文件夹和触发词（反推途中界面上可能已经换到别的文件夹了）
            const dir = j.result.dataset_dir || T().dir;
            App.draft.editor_dir = dir; App.saveDraft();
            if (T().auto_trigger === true) App.autoTrigger(dir, j.result.trigger);
          } else if (j.status === 'failed') {
            App.modal({ title: '反推失败', bad: true, body: `<div class="issue"><b>${esc(j.error)}</b></div><div class="code-box" style="max-height:240px">${esc((j.log || []).slice(-25).join('\n'))}</div>`, buttons: [{ label: '关闭' }] });
          }
        }
      }), 1000);
    },
    goEdit() { App.draft.editor_dir = T().dir; App.saveDraft(true); App.go('editor'); },
  };
})();

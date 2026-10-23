/* 04 图片爬取（Danbooru）：搜索条件 → 过滤 → 保存 → 爬完一键设为训练集去编辑标签 */
(function () {
  const { $, esc, h, api } = App;
  let jobId = null, timer = null, tiTags, tiMust, tiBl;

  const RATINGS = [['g', '全年龄'], ['s', '轻微'], ['q', '擦边'], ['e', '成人']];
  const SW1 = [['no_mono', '排除黑白 / 灰度图'], ['no_comic', '排除漫画 / 四格'], ['static_only', '只要静态图（跳过动图、视频）']];
  const SW2 = [['save_txt', '同时生成 .txt 标签'], ['cap_character', '标签里加角色名'], ['cap_copyright', '标签里加作品名'],
    ['cap_artist', '标签里加画师（@画师名，Anima 写法）'], ['use_as_dataset', '爬完自动设为训练集']];
  const DEF = { tags: '', logic: 'and', order: 'score', count: 100, min_score: 0, min_fav: 0, min_side: 768, date_from: '', date_to: '',
    ratings: [], blacklist: '', must_have: '', no_mono: true, no_comic: true, static_only: true, out_dir: '', folder: 'none', trigger: '', cap_exclude: '',
    save_txt: false, cap_character: false, cap_copyright: false, cap_artist: false, use_as_dataset: true, workers: 4 };

  // 过滤预设：标签都在 Danbooru 中英字典里核对过
  const COMMON_BL = 'lowres jpeg_artifacts scan bad_anatomy watermark signature artist_name web_address username dated multiple_views reference_sheet censored photo_(medium) sketch';
  const BUILTIN = [
    { id: 'char', name: '人物 LoRA', hint: '只要单人图', must: 'solo',
      bl: 'solo_focus multiple_girls multiple_boys 2girls 2boys comic 4koma multiple_views reference_sheet chibi photo_(medium) 3d cosplay monochrome greyscale lowres jpeg_artifacts watermark text_focus ai-generated ai-assisted sketch',
      sw: { no_mono: true, no_comic: true, static_only: true }, min_side: 768 },
    { id: 'style', name: '画风 LoRA', hint: '去掉草图、线稿、截图等', must: '',
      bl: 'sketch lineart unfinished monochrome greyscale spot_color comic 4koma multiple_views reference_sheet photo_(medium) 3d realistic anime_screenshot pixel_art oekaki ai-generated ai-assisted lowres jpeg_artifacts watermark',
      sw: { no_mono: true, no_comic: true, static_only: true }, min_side: 768 },
    { id: 'clean', name: '通用去噪', hint: '只往黑名单里追加', merge: true, must: '', bl: COMMON_BL },
  ];
  const clone = o => JSON.parse(JSON.stringify(o));
  // 预设可以改：改过后整份存到训练器设置（不跟配置存档走），内置的可一键恢复默认
  const presets = () => { const p = App.state.settings.crawl_presets; return p && p.length ? p : clone(BUILTIN); };
  const savePresets = async list => { App.state.settings = await api('settings.save', { patch: { crawl_presets: list } }); P.renderPresets(); };

  const C = () => App.draft.crawl;
  const save = () => App.saveDraft();

  const P = App.pages.crawl = {
    init() {
      App.draft.crawl = Object.assign({}, DEF, App.draft.crawl || {});
      const bind = (id, key, conv = v => v) => { $(id).onchange = e => { C()[key] = conv(e.target.value); save(); }; };
      bind('#cr-order', 'order'); bind('#cr-folder', 'folder');
      bind('#cr-count', 'count', v => Math.max(0, parseInt(v || '0', 10))); bind('#cr-score', 'min_score', v => parseInt(v || '0', 10));
      bind('#cr-fav', 'min_fav', v => Math.max(0, parseInt(v || '0', 10))); bind('#cr-side', 'min_side', v => Math.max(0, parseInt(v || '0', 10)));
      bind('#cr-from', 'date_from'); bind('#cr-to', 'date_to');
      $('#cr-out').onchange = e => { C().out_dir = e.target.value.trim(); save(); App.syncDataset(C().out_dir, 'crawl'); }; bind('#cr-trigger', 'trigger', v => v.trim()); bind('#cr-capex', 'cap_exclude', v => v.trim());
      App.$$('#cr-logic button').forEach(b => b.onclick = () => { C().logic = b.dataset.v; save(); P.render(); });
      // 标签框：只能放 D 站正式标签（字典 + 在线确认）
      tiTags = TagInput($('#cr-tags'), { prefix: true, placeholder: '输入中文或英文，例如 初音未来 / hatsune_miku，回车添加', onChange: l => { C().tags = l.join(' '); save(); } });
      tiMust = TagInput($('#cr-must'), { placeholder: '例如 单人 / solo', onChange: l => { C().must_have = l.join(' '); save(); } });
      tiBl = TagInput($('#cr-bl'), { wild: true, placeholder: '含有这些标签的图跳过，支持 *_focus 这类通配符', onChange: l => { C().blacklist = l.join(' '); save(); } });
      P.renderPresets();
      $('#cr-out-browse').onclick = async () => { const p = await App.pick({ title: '选择保存位置', start: C().out_dir }); if (p) { C().out_dir = p; save(); P.render(); App.syncDataset(p, 'crawl'); } };
      $('#cr-out-open').onclick = async () => { if (C().out_dir) { await api('fs.mkdir', { path: C().out_dir }); api('fs.open', { path: C().out_dir }); } };
      // 账号存在训练器设置里（不跟配置存档走）
      const saveAcc = () => api('settings.save', { patch: { danbooru_login: $('#cr-login').value.trim(), danbooru_api_key: $('#cr-key').value.trim() } })
        .then(s => App.state.settings = s);
      $('#cr-login').onchange = saveAcc; $('#cr-key').onchange = saveAcc;
      $('#cr-check').onclick = async () => {
        $('#cr-acc').textContent = '检查中…';
        const r = await api('crawl.account', { login: $('#cr-login').value.trim(), api_key: $('#cr-key').value.trim() });
        $('#cr-acc').textContent = r.ok ? `${r.level}：每次搜索最多 ${r.limit || '不限'} 个标签` : r.message;
        $('#cr-acc').className = 'hint ' + (r.ok ? 'ok' : 'bad');
      };
      $('#cr-start').onclick = P.start;
      $('#cr-stop').onclick = () => jobId && api('jobs.stop', { id: jobId });
      $('#cr-use').onclick = () => P.useAsDataset(true);
      $('#cr-dict-dl').onclick = P.downloadDict;
    },
    renderPresets() {
      const pb = $('#cr-presets'); pb.innerHTML = '';
      presets().forEach(p => {
        const g = h(`<span class="preset-grp"><button class="btn preset-btn" title="${esc((p.must ? '必须包含：' + p.must + ' ｜ ' : '') + '黑名单：' + (p.bl || '（空）'))}"><b>${esc(p.name)}</b><small>${esc(p.hint || '')}</small></button><button class="btn preset-edit" title="编辑这个预设">✎</button></span>`);
        g.children[0].onclick = () => P.applyPreset(p);
        g.children[1].onclick = () => P.editPreset(p);
        pb.append(g);
      });
      const add = h('<button class="btn" title="用当前的必须包含 / 黑名单 / 开关新建一个预设">＋ 新建预设</button>');
      add.onclick = () => { const c = C(); P.editPreset(null, { name: '', hint: '', must: c.must_have, bl: c.blacklist, merge: false,
        sw: { no_mono: c.no_mono, no_comic: c.no_comic, static_only: c.static_only }, min_side: c.min_side }); };
      pb.append(add);
    },
    editPreset(orig, seed) {
      const p = clone(orig || seed), builtin = orig && BUILTIN.find(b => b.id === orig.id);
      p.sw = p.sw || {};
      const body = h(`<div class="fields" style="grid-template-columns:1fr 1fr">
        <div class="field"><label><span>名称</span></label><input type="text" data-k="name"></div>
        <div class="field"><label><span>说明</span></label><input type="text" data-k="hint"></div>
        <div class="field span2"><label><span>套用方式</span></label><div class="seg" data-seg><button data-v="0">替换（重新填必须包含 / 黑名单 / 开关）</button><button data-v="1">追加到黑名单</button></div></div>
        <div class="field span2" data-only="0"><label><span>必须包含</span></label><div data-ti="must"></div></div>
        <div class="field span2"><label><span>黑名单</span><i class="q" data-tip="支持 *_focus 这类通配符">?</i></label><div data-ti="bl"></div></div>
        <div class="field span2" data-only="0"><label><span>过滤开关</span></label><div class="switches" data-sw></div></div>
        <div class="field" data-only="0"><label><span>短边至少（像素）</span><i class="q" data-tip="套用时只会调高，不会调低">?</i></label><input type="number" min="0" step="64" data-k="min_side"></div>
      </div>`);
      App.$$('input[data-k]', body).forEach(i => { i.value = p[i.dataset.k] ?? ''; i.onchange = () => { p[i.dataset.k] = i.type === 'number' ? parseInt(i.value || '0', 10) : i.value.trim(); }; });
      const tm = TagInput(body.querySelector('[data-ti=must]'), { placeholder: '例如 单人 / solo', onChange: l => p.must = l.join(' ') });
      const tb = TagInput(body.querySelector('[data-ti=bl]'), { wild: true, placeholder: '输入中文或英文，回车添加', onChange: l => p.bl = l.join(' ') });
      tm.set(p.must || ''); tb.set(p.bl || '');
      const swBox = body.querySelector('[data-sw]');
      SW1.forEach(([k, label]) => {
        const sw = h(`<label class="sw"><input type="checkbox"><i></i><span>${esc(label)}</span></label>`);
        const cb = sw.querySelector('input'); cb.checked = !!p.sw[k]; cb.onchange = () => { p.sw[k] = cb.checked; };
        swBox.append(sw);
      });
      const seg = () => {
        App.$$('[data-seg] button', body).forEach(b => b.classList.toggle('on', b.dataset.v === (p.merge ? '1' : '0')));
        App.$$('[data-only]', body).forEach(el => el.hidden = !!p.merge);
      };
      App.$$('[data-seg] button', body).forEach(b => b.onclick = () => { p.merge = b.dataset.v === '1'; seg(); });
      seg();
      const buttons = [];
      if (orig) buttons.push(builtin
        ? { label: '恢复默认', fn: () => savePresets(presets().map(x => x.id === orig.id ? clone(builtin) : x)) }
        : { label: '删除', cls: 'btn-danger', fn: async () => { if (!(await App.confirm('删除预设', `删除「${esc(orig.name)}」？`, '删除', true))) return false; await savePresets(presets().filter(x => x.id !== orig.id)); } });
      buttons.push({ label: '填入当前过滤', fn: () => { const c = C(); p.must = c.must_have; p.bl = c.blacklist; p.min_side = c.min_side;
        p.sw = { no_mono: c.no_mono, no_comic: c.no_comic, static_only: c.static_only }; tm.set(p.must); tb.set(p.bl);
        App.$$('[data-sw] input', body).forEach((cb, i) => cb.checked = !!p.sw[SW1[i][0]]); body.querySelector('[data-k=min_side]').value = p.min_side; return false; } });
      buttons.push({ label: '取消' });
      buttons.push({ label: '保存', cls: 'btn-primary', fn: async () => {
        if (!p.name) { App.toast('请填写预设名称', 'bad'); return false; }
        if (!p.must && !p.bl) { App.toast('必须包含和黑名单至少填一个', 'bad'); return false; }
        if (orig) await savePresets(presets().map(x => x.id === orig.id ? p : x));
        else { p.id = 'u' + Date.now().toString(36); await savePresets([...presets(), p]); }
        App.toast(`预设「${p.name}」已保存`, 'ok');
      } });
      App.modal({ title: orig ? `编辑预设：${orig.name}` : '新建过滤预设', body, wide: true, buttons });
    },
    applyPreset(p) {
      const c = C(), mine = new Set(c.tags.split(/\s+/).filter(Boolean));
      const bl = (p.bl || '').split(/\s+/).filter(t => t && !mine.has(t));   // 搜的就是 comic 时不能把 comic 拉黑
      if (p.merge) c.blacklist = [...new Set([...c.blacklist.split(/\s+/).filter(Boolean), ...bl])].join(' ');
      else {
        c.must_have = p.must || ''; c.blacklist = bl.join(' ');
        Object.assign(c, p.sw || {});
        if (p.min_side && (c.min_side || 0) < p.min_side) c.min_side = p.min_side;
      }
      save(); P.render();
      App.toast(`已套用「${p.name}」预设`, 'ok');
    },
    async dictStatus() {
      const st = await api('dict.status').catch(() => null);
      const miss = st && !st.file;
      $('#cr-dict-dl').hidden = !miss;
      $('#cr-dict-hint').textContent = miss ? '还没下载中英字典：现在只能输英文（在线补全），下载后中文也能搜（约 17 MB）'
        : 'Danbooru · 输入中文或英文都能找到 D 站标签，不是 D 站标签的加不进去';
    },
    async downloadDict() {
      let id;
      try { id = await api('dict.download'); } catch (e) { App.toast(e.message, 'bad'); return; }
      $('#cr-dict-dl').disabled = true; $('#cr-dict-dl').textContent = '下载中…';
      const t = setInterval(App.poll(async () => {
        const j = await api('jobs.get', { id }).catch(() => null);
        if (!j || j.status === 'running') { if (j && j.progress_text) $('#cr-dict-dl').textContent = '下载中 ' + j.progress_text; return; }
        clearInterval(t); $('#cr-dict-dl').disabled = false; $('#cr-dict-dl').textContent = '下载中英字典';
        if (j.status === 'done') App.toast('字典已下载，中文也能搜标签了', 'ok'); else App.toast('字典下载失败：' + (j.error || ''), 'bad');
        P.dictStatus();
      }), 800);
    },
    show() {
      const c = C();
      if (!c.out_dir && App.state.settings.sync_dataset_dir !== false) c.out_dir = App.params().dataset_dir || '';
      if (!c.out_dir) c.out_dir = (App.state.settings.output_dir || '').replace(/[\\/]output$/i, '') + '\\datasets\\' + (c.tags.split(/\s+/)[0] || 'danbooru');
      P.render(); P.dictStatus();
      $('#cr-login').value = App.state.settings.danbooru_login || ''; $('#cr-key').value = App.state.settings.danbooru_api_key || '';
      const run = (App.jobs || []).find(j => j.kind === 'crawl' && j.status === 'running');
      if (run) { jobId = run.id; P.watch(); }
    },
    render() {
      const c = C();
      tiTags.set(c.tags); tiMust.set(c.must_have); tiBl.set(c.blacklist);
      $('#cr-order').value = c.order; $('#cr-count').value = c.count; $('#cr-score').value = c.min_score;
      $('#cr-fav').value = c.min_fav; $('#cr-side').value = c.min_side; $('#cr-from').value = c.date_from; $('#cr-to').value = c.date_to;
      $('#cr-out').value = c.out_dir; $('#cr-folder').value = c.folder; $('#cr-trigger').value = c.trigger;
      $('#cr-capex').value = c.cap_exclude;
      App.$$('#cr-logic button').forEach(b => b.classList.toggle('on', b.dataset.v === c.logic));
      const rt = $('#cr-ratings'); rt.innerHTML = '';
      RATINGS.forEach(([v, label]) => {
        const sw = h(`<label class="sw"><input type="checkbox"><i></i><span>${label}</span></label>`);
        const cb = sw.querySelector('input'); cb.checked = c.ratings.includes(v);
        cb.onchange = () => { c.ratings = RATINGS.map(r => r[0]).filter(x => x === v ? cb.checked : c.ratings.includes(x)); save(); };
        rt.append(sw);
      });
      for (const [id, list] of [['#cr-sw', SW1], ['#cr-sw2', SW2]]) {
        const el = $(id); el.innerHTML = '';
        list.forEach(([k, label]) => {
          const sw = h(`<label class="sw"><input type="checkbox"><i></i><span>${esc(label)}</span></label>`);
          const cb = sw.querySelector('input'); cb.checked = !!c[k];
          cb.onchange = () => { c[k] = cb.checked; save(); };
          el.append(sw);
        });
      }
    },
    async start() {
      const c = C();
      if (!c.tags.trim()) { App.toast('请添加搜索标签（输完回车，变成标签块才算加上）', 'bad'); return; }
      if (!c.out_dir) { App.toast('请选择保存位置', 'bad'); return; }
      // 保存位置里已经有别的角色的图（上次爬的），这次又换了角色：问要不要新建一个以这次角色命名的文件夹
      const plan = await api('crawl.plan_dir', { out_dir: c.out_dir, tags: c.tags }).catch(() => null);
      if (plan && plan.ask) {
        const was = plan.prev && plan.prev.length ? `（上次爬的是 ${plan.prev.join(' ')}）` : '';
        const pick = await new Promise(res => App.modal({ title: '保存位置里已经有图片了', onclose: () => res(null),
          body: `<div>「${esc(c.out_dir)}」里已经有 ${plan.images} 张图${esc(was)}。</div>
            <div style="margin-top:8px">这次爬的图放进去会和它们混成同一个训练集。要新建文件夹「<b>${esc(plan.name)}</b>」，爬到里面吗？</div>
            <div class="hint" style="margin-top:6px">会建在：${esc(plan.suggest)}</div>`,
          buttons: [{ label: '取消', fn: () => res(null) },
            { label: '仍然放在这里', fn: () => res('keep') },
            { label: `新建「${plan.name}」并爬到里面`, cls: 'btn-primary', fn: () => res('new') }] }));
        if (!pick) return;
        if (pick === 'new') {
          c.out_dir = plan.suggest; $('#cr-out').value = c.out_dir;
          App.syncDataset(c.out_dir, 'crawl');
        }
      }
      App.saveDraft(true);
      try { jobId = await api('crawl.start', { cfg: Object.assign({}, c) }); } catch (e) { App.toast(e.message, 'bad'); return; }
      $('#cr-log').innerHTML = ''; $('#cr-use').hidden = true; P.watch();
    },
    watch() {
      clearInterval(timer);
      $('#cr-start').disabled = true; $('#cr-stop').disabled = false;
      timer = setInterval(App.poll(async () => {
        const j = await api('jobs.get', { id: jobId, log: 400 }).catch(() => null);
        if (!j) return;
        $('#cr-prog i').style.width = (j.progress || 0) * 100 + '%';
        $('#cr-prog').classList.toggle('indet', j.status === 'running' && j.progress == null);
        $('#cr-prog-text').textContent = j.progress_text || '';
        const box = $('#cr-log'), atEnd = box.scrollTop + box.clientHeight > box.scrollHeight - 30;
        box.innerHTML = (j.log || []).map(l => `<div class="${/⚠|出错|失败|没有符合/.test(l) ? 'l-err' : /服务器查询|搜索方案/.test(l) ? 'l-cmd' : /完成/.test(l) ? 'l-ok' : /本地/.test(l) ? 'l-tip' : ''}">${esc(l)}</div>`).join('');
        if (atEnd) box.scrollTop = box.scrollHeight;
        if (j.status !== 'running') {
          clearInterval(timer); $('#cr-start').disabled = false; $('#cr-stop').disabled = true;
          if (j.status === 'done') {
            $('#cr-use').hidden = false;
            App.toast(`爬取完成：新下载 ${j.result.ok} 张`, 'ok', '去编辑标签', () => P.useAsDataset(true), 10000);
            if (C().use_as_dataset) P.useAsDataset(false);
          } else if (j.status === 'failed') {
            App.modal({ title: '爬取没有完成', bad: true, body: `<div class="issue"><b>${esc(j.error)}</b></div>`, buttons: [{ label: '关闭' }] });
          }
        }
      }), 1000);
    },
    useAsDataset(go) {
      const dir = C().out_dir;
      if (App.spec()) App.setParam('dataset_dir', dir);
      App.draft.editor_dir = dir;
      if (App.draft.tag) App.draft.tag.dir = dir;
      App.saveDraft(true);
      if (go) App.go('editor'); else App.toast('已设为训练集位置', 'ok');
    },
  };
})();

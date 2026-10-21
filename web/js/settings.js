/* 07 训练器设置：一键更新、训练环境、国内 / 国外网络、端口、界面、显卡、后台作业 */
(function () {
  const { $, esc, h, api } = App;
  let jobId = null, jobTimer = null;

  const P = App.pages.settings = {
    init() {
      const save = async patch => { App.state.settings = await api('settings.save', { patch }); return App.state.settings; };
      $('#st-update').onclick = async () => {
        if (!(await App.confirm('一键更新', '更新训练器本体，并把 sd-scripts、musubi-tuner 切到新版训练器测试过的版本、自动装好对应的依赖（三样一起更新，版本对应）。<br>正在训练时请不要更新。', '开始更新'))) return;
        P.run('update');
      };
      $('#st-update-self').onclick = async () => {
        if (!(await App.confirm('只更新训练器', '只更新训练器本体（修 bug 用），sd-scripts、musubi-tuner 和训练环境都不动。<br>如果新版训练器要求了别的引擎版本或新依赖，重启训练器后会自动同步。<br>正在训练时请不要更新。', '开始更新'))) return;
        P.run('self');
      };
      $('#st-install').onclick = () => P.run('install');
      $('#st-repair').onclick = () => P.run('repair');
      $('#st-torch').onclick = async () => { if (await App.confirm('重装 PyTorch', `按「${$('#st-torchv').selectedOptions[0].text}」重新下载安装 PyTorch（约 3 GB）。`, '重装')) P.run('torch'); };
      $('#st-torchv').onchange = e => save({ torch_variant: e.target.value });
      $('#st-repo').onchange = e => save({ update_repo: e.target.value.trim() });
      App.$$('#st-net button').forEach(b => b.onclick = async () => { await save({ network: b.dataset.v }); if (b.dataset.v === 'auto') await api('net.detect'); P.show(); });
      $('#st-ghproxy').onchange = e => save({ github_proxy: e.target.value.trim() });
      $('#st-proxy').onchange = e => save({ proxy_url: e.target.value.trim() });
      $('#st-nettest').onclick = async () => {
        const out = $('#st-nettest-out'); $('#st-nettest').disabled = true; out.textContent = '测速中（每条线路试下一小段，约 10 秒）…';
        try {
          const r = await api('net.test');
          const fmt = xs => xs.length ? xs.map((x, i) => `${i ? '' : '★ '}${x.name} ${x.speed.toFixed(1)} MB/s`).join('　') : '全部连不上';
          out.textContent = `连接方式：${r.routes.join('、')}\n模型（HuggingFace）：${fmt(r.hf)}\nGitHub：${fmt(r.github)}\npip：${fmt(r.pip)}\n★ = 下载时优先用的线路；下载中出错会自动换下一条。`;
        } catch (e) { out.textContent = '测速失败：' + e.message; }
        $('#st-nettest').disabled = false;
      };
      $('#st-hftoken').onchange = e => save({ hf_token: e.target.value.trim() });
      $('#st-port').onchange = e => { const v = parseInt(e.target.value, 10); if (v > 1023 && v < 65536) save({ port: v }).then(() => App.toast(`端口改为 ${v}，重启训练器后生效`, 'ok')); };
      App.$$('#st-theme button').forEach(b => b.onclick = async () => {
        App.applyTheme(b.dataset.v); App.$$('#st-theme button').forEach(x => x.classList.toggle('on', x === b));
        await save({ theme: b.dataset.v });
      });
      $('#st-openbrowser').onchange = e => save({ open_browser: e.target.checked });
      $('#st-outdir').onchange = e => save({ output_dir: e.target.value.trim() });
      $('#st-outdir-browse').onclick = async () => { const p = await App.pick({ title: '默认保存位置', start: $('#st-outdir').value }); if (p) { $('#st-outdir').value = p; save({ output_dir: p }); } };
      $('#st-check').onchange = e => save({ auto_check_before_train: e.target.checked });
      $('#st-shutdown').onchange = e => save({ shutdown_after_queue: e.target.checked });
      $('#st-syncds').onchange = async e => {
        await save({ sync_dataset_dir: e.target.checked });
        if (e.target.checked) { const d = App.params().dataset_dir; if (d) { App.syncDataset(d, 'params'); App.toast('已把各页训练集位置统一成：' + d, 'ok'); } }
      };
      $('#st-job-stop').onclick = () => jobId && api('jobs.stop', { id: jobId });
      App.on('tick', ({ jobs }) => { if (App.page === 'settings') P.renderJobs(jobs); });
    },
    async show() {
      const s = App.state.settings;
      const [env, boot] = await Promise.all([api('env.status'), api('boot')]);
      App.state.env = env; App.state.network = boot.network;
      $('#st-versions').innerHTML = `<tr><td>训练器</td><td>V${esc(boot.version.version || '0.1.0')}</td></tr>` +
        Object.entries(env.engines).map(([k, e]) => `<tr><td>${esc(k)}</td><td>${e.installed ? esc(e.commit || '已下载')
          + (e.pinned ? (e.match ? ' <span class="badge ok">和训练器匹配</span>' : ` <span class="badge warn">训练器要求 ${esc(e.pinned.slice(0, 7))}</span>`) : '')
          : '<span class="badge bad">未下载</span>'}</td></tr>`).join('')
        + (env.sync_reason ? `<tr><td>同步</td><td><span class="badge warn">需要同步</span> ${esc(env.sync_reason)}（启动时会自动同步，也可以点「一键更新」）</td></tr>` : '');
      $('#st-env').innerHTML = env.env
        ? `<tr><td>状态</td><td><span class="badge ok">已安装</span></td></tr><tr><td>PyTorch</td><td>${esc(env.torch || '?')} · CUDA ${esc(env.cuda || '?')}</td></tr><tr><td>位置</td><td><code>${esc(env.python)}</code></td></tr>`
        : '<tr><td>状态</td><td><span class="badge bad">未安装</span> 训练和本地反推前需要安装（下载约 4 GB，装好约 8 GB）</td></tr>';
      $('#st-install').textContent = env.env ? '重新安装 / 检查' : '安装训练环境';
      $('#st-torchv').value = s.torch_variant || 'cu128';
      $('#st-repo').value = s.update_repo || '';
      App.$$('#st-net button').forEach(b => b.classList.toggle('on', b.dataset.v === s.network));
      $('#st-net-hint').textContent = (s.network === 'auto' ? `自动检测结果：${boot.network === 'cn' ? '国内网络' : '国外网络'}。` : '') +
        '每次下载前都会测速：镜像 / 官方源 × 直连 / VPN 代理，挑最快的；下载中出错自动换下一条线路。' +
        (boot.network === 'cn' ? '国内网络：速度差不多时优先镜像（hf-mirror、清华源、GitHub 加速）。' : '国外网络：速度差不多时优先官方源。');
      $('#st-ghproxy').value = s.github_proxy || ''; $('#st-proxy').value = s.proxy_url || ''; $('#st-hftoken').value = s.hf_token || '';
      $('#st-port').value = s.port; App.$$('#st-theme button').forEach(b => b.classList.toggle('on', b.dataset.v === (s.theme || 'light')));
      $('#st-openbrowser').checked = s.open_browser !== false;
      const g = App.gpu || boot.gpu || {};
      P.renderGpus(g.all && g.all.length ? g.all : [{ index: '0', name: '显卡 0', total_gb: '?' }]);
      $('#st-outdir').value = s.output_dir || '';
      $('#st-check').checked = s.auto_check_before_train !== false; $('#st-shutdown').checked = !!s.shutdown_after_queue; $('#st-syncds').checked = s.sync_dataset_dir !== false;
      P.renderJobs(App.jobs || []);
    },
    // 显卡多选：勾两张及以上就是多卡训练（每张卡一个进程，各跑「单批数量」张图）
    renderGpus(all) {
      const s = App.state.settings;
      const chosen = new Set((s.gpu_ids && s.gpu_ids.length ? s.gpu_ids : [s.gpu_id || '0']).map(String));
      const el = $('#st-gpu'); el.innerHTML = '';
      all.forEach(x => {
        const sw = h(`<label class="sw"><input type="checkbox"><i></i><span>${esc(x.index)}: ${esc(String(x.name).replace('NVIDIA GeForce ', ''))} · ${x.total_gb} GB</span></label>`);
        const cb = sw.querySelector('input'); cb.checked = chosen.has(String(x.index));
        cb.onchange = async () => {
          const ids = App.$$('input', el).map((c, i) => c.checked ? String(all[i].index) : null).filter(v => v !== null);
          if (!ids.length) { cb.checked = true; App.toast('至少要选一张显卡', 'bad'); return; }
          App.state.settings = await api('settings.save', { patch: { gpu_ids: ids, gpu_id: ids[0] } });
          P.gpuHint(ids, all.length);
        };
        el.append(sw);
      });
      P.gpuHint([...chosen], all.length);
    },
    gpuHint(ids, n) {
      $('#st-gpu-hint').textContent = ids.length > 1
        ? `多卡训练：${ids.length} 张卡同时训练，每步处理「单批数量 × ${ids.length}」张图，总步数相应减少。` +
          (navigator.userAgent.includes('Windows') ? 'Windows 下卡间通信走 gloo，比 Linux 慢一些。' : '') + '只对之后加入队列的任务生效。'
        : (n > 1 ? '勾选两张及以上显卡即可多卡训练。' : '');
    },
    async run(mode) {
      try { jobId = await api('env.install', { mode }); } catch (e) { App.toast(e.message, 'bad'); return; }
      P.showJob(jobId);
    },
    renderJobs(jobs) {
      const el = $('#st-jobs');
      el.innerHTML = jobs.slice(0, 10).map(j => `<div class="q-row" data-j="${j.id}" style="grid-template-columns:minmax(0,1fr) 120px auto">
        <div><div class="q-name">${esc(j.title)}</div><div class="q-sub">${esc(j.error || j.progress_text || '')}</div></div>
        <div class="prog ${j.status === 'running' && j.progress == null ? 'indet' : ''}"><i style="width:${(j.progress || (j.status === 'done' ? 1 : 0)) * 100}%"></i></div>
        <span class="badge ${{ done: 'ok', failed: 'bad', running: 'k' }[j.status] || ''}">${{ running: '进行中', done: '完成', failed: '失败', stopped: '已停止' }[j.status]}</span></div>`).join('') || '<div class="muted">没有后台作业</div>';
      App.$$('[data-j]', el).forEach(r => r.onclick = () => P.showJob(r.dataset.j));
    },
    showJob(id) {
      jobId = id; $('#st-job-log-card').hidden = false;
      clearInterval(jobTimer);
      const f = async () => {
        const j = await api('jobs.get', { id, log: 400 }).catch(() => null);
        if (!j) return;
        $('#st-job-title').firstChild.textContent = '作业日志 · ' + j.title + ' ';
        $('#st-job-prog i').style.width = (j.progress || 0) * 100 + '%';
        $('#st-job-prog').classList.toggle('indet', j.status === 'running' && j.progress == null);
        $('#st-job-stop').disabled = j.status !== 'running';
        const box = $('#st-job-log'); const atEnd = box.scrollTop + box.clientHeight > box.scrollHeight - 30;
        box.innerHTML = (j.log || []).map(l => `<div class="${/出错|失败|ERROR|Error/.test(l) ? 'l-err' : /^\S+ \$ /.test(l) ? 'l-cmd' : /完成|通过|Successfully/.test(l) ? 'l-ok' : ''}">${esc(l)}</div>`).join('');
        if (atEnd) box.scrollTop = box.scrollHeight;
        if (j.status !== 'running') {
          clearInterval(jobTimer);
          if (j.kind === 'env') { if (j.status === 'done') App.toast(j.title + '完成', 'ok'); P.show(); }
        }
      };
      f(); jobTimer = setInterval(App.poll(f), 1200);
      $('#st-job-log-card').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    },
    hide() { },
  };
})();

/* 把满意的 LoRA 送到 WWY 启动器（共享模型库 / 某个实例），示例图当预览图、触发词写进模型信息 */
(function () {
  const { esc, h, api } = App;

  App.sendToLauncher = async function (taskId, path) {
    let d;
    try { d = await api('launcher.send_info', { id: taskId, path }); } catch (e) { App.toast(e.message, 'bad'); return; }
    if (!d) return;
    if (!d.outputs || !d.outputs.length) { App.toast('这个任务还没有保存出 LoRA 文件', 'bad'); return; }
    if (!d.targets.length) { App.toast('没找到 WWY 启动器（到「模型设置」点重新扫描）', 'bad'); return; }
    const st = { path: d.path, type: d.type || '', trigger: d.trigger || '', preview: d.preview, sel: new Set() };
    const lib = d.targets.find(x => x.kind === 'library');
    st.sel.add((lib || d.targets[0]).id);
    const body = h(`<div class="trig">
      <div class="trig-row"><label>LoRA 文件</label><select data-lora></select></div>
      <div class="trig-row"><label>预览图</label><div class="row" style="align-items:flex-start;gap:10px"><img data-prev class="trig-typ"><div class="hint" style="margin:0" data-prev-hint></div></div></div>
      <div class="trig-row"><label>类型</label><select data-type></select></div>
      <div class="trig-row"><label>触发词</label><input type="text" data-trig placeholder="用 LoRA 时写在提示词里的词（写进模型信息，启动器里点 LoRA 会自动带上）"></div>
      <div class="trig-row"><label>送到</label><div data-tg style="display:grid;gap:6px"></div></div>
      <div class="hint" data-where></div>
    </div>`);
    const $b = s => body.querySelector(s);
    const sel = $b('[data-lora]');
    d.outputs.slice().reverse().forEach(o => sel.append(h(`<option value="${esc(o.path)}">${esc(o.name)}</option>`)));
    sel.value = st.path;
    d.types.forEach(t => $b('[data-type]').append(h(`<option value="${esc(t)}">${esc(t)}</option>`)));
    $b('[data-type]').value = st.type || '画风';
    st.type = $b('[data-type]').value;
    $b('[data-trig]').value = st.trigger;
    const where = () => {
      const t = d.targets.filter(x => st.sel.has(x.id));
      $b('[data-where]').textContent = t.length ? '会放到：' + t.map(x => x.dir + (x.organize ? `\\${d.base}\\${st.type}` : '')).join('；') : '';
    };
    const prev = () => {
      $b('[data-prev]').hidden = !st.preview;
      if (st.preview) $b('[data-prev]').src = '/thumb?path=' + encodeURIComponent(st.preview);
      $b('[data-prev-hint]').textContent = st.preview ? '训练中测试出的这一轮的示例图，会存成 .preview.png，启动器和 WebUI 里当封面' : '这一轮没有示例图（训练时没开「训练中测试」），送过去就没有封面';
    };
    d.targets.forEach(x => {
      const sw = h(`<label class="sw"><input type="checkbox"><i></i><span>${esc(x.label)}${x.kind === 'library' ? '（所有实例共用，推荐）' : ''}</span></label>`);
      const cb = sw.querySelector('input'); cb.checked = st.sel.has(x.id);
      cb.onchange = () => { cb.checked ? st.sel.add(x.id) : st.sel.delete(x.id); where(); };
      $b('[data-tg]').append(sw);
    });
    sel.onchange = async () => {
      st.path = sel.value;
      const r = await api('launcher.send_info', { id: taskId, path: st.path }).catch(() => null);
      st.preview = r && r.preview; prev();
    };
    $b('[data-type]').onchange = e => { st.type = e.target.value; where(); };
    $b('[data-trig]').oninput = e => { st.trigger = e.target.value; };
    prev(); where();
    const go = async overwrite => {
      const r = await api('launcher.send', { id: taskId, path: st.path, targets: [...st.sel], type: st.type, trigger: st.trigger, preview: st.preview, overwrite });
      if (r.exists.length) {
        if (await App.confirm('同名文件已存在', `这些位置已经有同名但内容不同的 LoRA：<br>${r.exists.map(esc).join('<br>')}<br><br>覆盖它们吗？`, '覆盖', true)) return go(true);
        return false;
      }
      App.toast(`已送到启动器：${r.done.map(x => x.target.split(' · ')[0]).join('、')}`, 'ok', '打开文件夹', () => api('fs.open', { path: r.done[0].path.replace(/[\\/][^\\/]+$/, '') }), 10000);
    };
    App.modal({ title: '送到 WWY 启动器', body, wide: true, buttons: [
      { label: '取消' },
      { label: '送过去', cls: 'btn-primary', fn: async () => {
        if (!st.sel.size) { App.toast('请选择要送到哪里', 'bad'); return false; }
        try { return await go(false); } catch (e) { App.toast(e.message, 'bad'); return false; }
      } },
    ] });
  };
})();

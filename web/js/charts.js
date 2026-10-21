/* 训练曲线图表（TensorBoard 风格）：标题、来源、最新值 / 最小值，网格坐标轴，折线 + 面积填充，鼠标悬停看具体数值 */
(function () {
  const NICE = [1, 2, 2.5, 5, 10];

  function niceStep(range, count) {
    const raw = range / Math.max(1, count);
    const mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
    for (const n of NICE) if (n * mag >= raw) return n * mag;
    return 10 * mag;
  }
  // 数值显示：很小的数（学习率）用科学计数，其余 4 位小数
  function fmt(v) {
    if (v === 0) return '0.00e+0';
    const a = Math.abs(v);
    if (a < 1e-3 || a >= 1e5) return v.toExponential(2);
    return v.toFixed(4);
  }
  function fmtStep(v) { return v >= 1000 ? v.toLocaleString('en-US') : String(Math.round(v)); }

  function draw(cv, pts, hover) {
    const w = cv.clientWidth, h = cv.clientHeight, dpr = devicePixelRatio || 1;
    if (!w || !h) return;
    cv.width = w * dpr; cv.height = h * dpr;
    const g = cv.getContext('2d'); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
    const css = getComputedStyle(document.documentElement);
    const axis = css.getPropertyValue('--text-dim').trim(), grid = css.getPropertyValue('--line').trim();
    const line = css.getPropertyValue('--chart').trim() || '#22c3c9';
    g.font = '12px "WWY Inter", "Microsoft YaHei UI", sans-serif';
    if (!pts || pts.length < 1) { g.fillStyle = axis; g.fillText('训练开始后显示', 12, 24); return null; }
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    let xMin = Math.min(...xs), xMax = Math.max(...xs);
    let yMin = Math.min(...ys), yMax = Math.max(...ys);
    if (xMax === xMin) xMax = xMin + 1;
    if (yMax - yMin < Math.abs(yMax) * 1e-6 + 1e-12) { yMax += Math.abs(yMax) * 0.05 || 1e-6; yMin -= Math.abs(yMin) * 0.05; }
    // 坐标轴刻度
    const yStep = niceStep(yMax - yMin, 6);
    const yLo = Math.floor(yMin / yStep) * yStep, yHi = Math.ceil(yMax / yStep) * yStep;
    // x 轴从 0 开始（起点离 0 很近时），终点取下一个整刻度，和 TensorBoard 一样右边留一点空
    const x0 = xMin >= 0 && xMin <= (xMax - xMin) * 0.2 ? 0 : xMin;
    const xStep = niceStep(xMax - x0, 7);
    const xLo = Math.floor(x0 / xStep) * xStep;
    const xHi = Math.max(Math.ceil(xMax / xStep) * xStep, xLo + xStep);
    const yLabels = []; for (let v = yLo; v <= yHi + yStep * 1e-6; v += yStep) yLabels.push(v);
    const L = 14 + Math.max(...yLabels.map(v => g.measureText(fmt(v)).width)), R = 16, T = 10, B = 26;
    const X = v => L + (v - xLo) / (xHi - xLo) * (w - L - R);
    const Y = v => T + (1 - (v - yLo) / (yHi - yLo)) * (h - T - B);
    g.strokeStyle = grid; g.lineWidth = 1; g.fillStyle = axis; g.textAlign = 'right'; g.textBaseline = 'middle';
    yLabels.forEach(v => { const y = Math.round(Y(v)) + 0.5; g.beginPath(); g.moveTo(L, y); g.lineTo(w - R, y); g.stroke(); g.fillText(fmt(v), L - 6, y); });
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (let v = xLo; v <= xHi + xStep * 1e-6; v += xStep) {
      const x = Math.round(X(v)) + 0.5; g.beginPath(); g.moveTo(x, T); g.lineTo(x, h - B); g.stroke(); g.fillText(fmtStep(v), x, h - B + 6);
    }
    // 面积 + 折线
    g.beginPath(); pts.forEach((p, i) => i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1])));
    g.lineTo(X(pts[pts.length - 1][0]), h - B); g.lineTo(X(pts[0][0]), h - B); g.closePath();
    const grad = g.createLinearGradient(0, T, 0, h - B); grad.addColorStop(0, hexA(line, .22)); grad.addColorStop(1, hexA(line, .03));
    g.fillStyle = grad; g.fill();
    g.beginPath(); pts.forEach((p, i) => i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1])));
    g.strokeStyle = line; g.lineWidth = 1.6; g.lineJoin = 'round'; g.stroke();
    if (pts.length === 1) { g.fillStyle = line; g.beginPath(); g.arc(X(pts[0][0]), Y(pts[0][1]), 3, 0, 7); g.fill(); }
    // 悬停：竖线 + 数值
    if (hover != null) {
      let best = 0, bd = 1e9;
      pts.forEach((p, i) => { const d = Math.abs(X(p[0]) - hover); if (d < bd) { bd = d; best = i; } });
      const p = pts[best], x = X(p[0]), y = Y(p[1]);
      g.strokeStyle = axis; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(x, T); g.lineTo(x, h - B); g.stroke(); g.setLineDash([]);
      g.fillStyle = line; g.beginPath(); g.arc(x, y, 3.5, 0, 7); g.fill();
      const txt = `第 ${fmtStep(p[0])} ${el_unit(cv)}  ${fmt(p[1])}`;
      g.font = '12px "WWY Mono", Consolas, monospace';
      const tw = g.measureText(txt).width + 14, tx = Math.min(Math.max(x + 8, L), w - R - tw), ty = Math.max(T, y - 30);
      g.fillStyle = css.getPropertyValue('--ink').trim() || '#111'; g.fillRect(tx, ty, tw, 22);
      g.fillStyle = '#f1f2ee'; g.textAlign = 'left'; g.textBaseline = 'middle'; g.fillText(txt, tx + 7, ty + 11);
    }
    return { X, xLo, xHi };
  }

  function el_unit(cv) { return cv.dataset.unit || '步'; }
  function hexA(hex, a) {
    const m = hex.replace('#', '').match(/^([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
    if (!m) return `rgba(34,195,201,${a})`;
    return `rgba(${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)},${a})`;
  }

  // 卡片：<div class="chart-card"> 标题 / 来源 / 最新 / 最小 / canvas
  function card(title, sub) {
    const el = document.createElement('div');
    el.className = 'chart-card';
    el.innerHTML = `<div class="cc-head"><div><b class="cc-title"></b><div class="cc-sub"></div></div><div class="cc-num"><div>latest <span data-l>—</span></div><div>min <span data-m>—</span></div></div></div><canvas></canvas>`;
    el.querySelector('.cc-title').textContent = title;
    if (/epoch/.test(title)) el.querySelector('canvas').dataset.unit = '轮';   // 每轮的图，悬停显示「第 N 轮」
    el.querySelector('.cc-sub').textContent = sub || '';
    const cv = el.querySelector('canvas');
    el._pts = [];
    el.set = pts => {
      el._pts = pts || [];
      el.querySelector('[data-l]').textContent = el._pts.length ? fmt(el._pts[el._pts.length - 1][1]) : '—';
      el.querySelector('[data-m]').textContent = el._pts.length ? fmt(Math.min(...el._pts.map(p => p[1]))) : '—';
      draw(cv, el._pts, el._hover);
    };
    cv.onmousemove = e => { el._hover = e.offsetX; draw(cv, el._pts, el._hover); };
    cv.onmouseleave = () => { el._hover = null; draw(cv, el._pts, null); };
    new ResizeObserver(() => draw(cv, el._pts, el._hover)).observe(cv);
    return el;
  }

  window.Charts = { card, fmt };
})();

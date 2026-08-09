/* ==========================================================================
   NSE Quant — Dashboard frontend (vanilla JS, no build step)
   Consumes the FastAPI JSON contract:
     GET /api/health        -> {status, db_ok, market_open, candles_total}
     GET /api/symbols       -> {symbols: [{symbol, name, instrument_type}]}
     GET /api/candles?symbol=&timeframe=&days=&from=&to=
                             -> {symbol, timeframe, candles:[{ts, open, high,
                                low, close, volume}]} (ts ASC; from/to
                                YYYY-MM-DD inclusive, wins over days)
     GET /api/indicators?symbol=&timeframe=&days=&from=&to=
                             -> same candles + sma20, sma50, ema12, ema26,
                                rsi14, macd, macd_signal, macd_hist,
                                bb_upper, bb_mid, bb_lower, vol_sma20
                                (indicator fields are null during warmup)
     GET /api/movers?n=     -> {gainers:[...], losers:[...]}
     GET /api/portfolio     -> {summary:{...}, positions:[]}
     GET /api/sentiment?symbol= -> {rows:[{title, url, source,
                               published_at, sentiment_compound, ...}]}
   ========================================================================== */
'use strict';

/* ---------- Constants ---------- */
var UP = '#3fb950';      // green — Indian convention: close >= open
var DOWN = '#f85149';    // red
var FLAT = '#8b949e';    // gray

var C_SMA20 = '#d29922';
var C_SMA50 = '#bc8cff';
var C_EMA12 = '#58a6ff';
var C_EMA26 = '#f0883e';
var C_BB    = '#39c5cf';

var API = {
  health: '/api/health',
  symbols: '/api/symbols',
  candles: '/api/candles',
  indicators: '/api/indicators',
  movers: '/api/movers',
  portfolio: '/api/portfolio',
  sentiment: '/api/sentiment'
};

var DEFAULT_SYMBOL = 'TCS';
var DEFAULT_TIMEFRAME = '1d';
var REFRESH_MS = 60 * 1000;      // silent auto-refresh of movers + status chip
var MAX_POINTS = 2500;           // downsample above this (bucketing)

// Date-range presets (months back from today, local tz). MAX = everything
// the backend has for the timeframe (requested via limit=5000, no from/to).
var PRESETS = { '1M': 1, '3M': 3, '6M': 6 };
var PRESET_KEYS = ['1M', '3M', '6M', 'MAX'];
var DEFAULT_PRESET_BY_TF = { '1d': '6M', '15m': '3M', '5m': '1M' };
var MAX_FETCH_LIMIT = 5000;      // backend cap for the MAX preset
var TF_LABEL = { '1d': '1D', '15m': '15M', '5m': '5M' };

// Indicator fields carried on /api/indicators candles.
var IND_FIELDS = [
  'sma20', 'sma50', 'ema12', 'ema26', 'rsi14',
  'macd', 'macd_signal', 'macd_hist',
  'bb_upper', 'bb_mid', 'bb_lower'
];

var TOOLTIP_STYLE = {
  backgroundColor: '#1c2128',
  borderColor: '#30363d',
  borderWidth: 1,
  titleColor: '#e6edf3',
  bodyColor: '#c9d1d9',
  padding: 8,
  boxPadding: 4
};

/* ---------- State ---------- */
var chart = null;
var volumeChart = null;
var rsiChart = null;
var macdChart = null;

var symbols = [];
var currentSymbol = DEFAULT_SYMBOL;
var currentTimeframe = DEFAULT_TIMEFRAME;
var currentItems = [];       // last rendered (possibly downsampled) items
var currentLabels = [];      // per-candle x-axis labels (category scale, index-aligned)
var hasIndicators = false;   // true when the indicators feed is available
var indicatorsState = { sma20: true, sma50: true, ema: false, bb: false };

// Active chart date range. preset is '1M'|'3M'|'6M'|'MAX' or null for custom.
var currentRange = { from: null, to: null };   // resolved YYYY-MM-DD bounds
var activePreset = null;                       // preset key currently in effect
var presetTouched = false;                     // true once user picks preset/custom

var loadSeq = 0;             // guards against out-of-order candle responses
var sentimentSeq = 0;        // guards out-of-order sentiment responses
var statusTimer = null;
var lastHoverIndex = -1;

/* ---------- DOM ---------- */
function $(sel) { return document.querySelector(sel); }

var el = {
  marketChip: $('#market-chip'),
  candlesTotal: $('#candles-total'),
  lastUpdated: $('#last-updated'),
  symbolSelect: $('#symbol-select'),
  chartTitle: $('#chart-title'),
  chartRange: $('#chart-range'),
  chartCanvas: $('#candle-chart'),
  volumeCanvas: $('#volume-chart'),
  rsiCanvas: $('#rsi-chart'),
  macdCanvas: $('#macd-chart'),
  statusMsg: $('#status-msg'),
  gainers: $('#gainers-list'),
  losers: $('#losers-list'),
  portfolioBody: $('#portfolio-body'),
  sentimentBody: $('#sentiment-body'),
  sentimentSymbol: $('#sentiment-symbol'),
  tfButtons: Array.prototype.slice.call(document.querySelectorAll('.tf-btn')),
  indChips: Array.prototype.slice.call(document.querySelectorAll('.ind-chip')),
  presetBtns: Array.prototype.slice.call(document.querySelectorAll('.preset-btn')),
  fromDate: $('#from-date'),
  toDate: $('#to-date'),
  applyRangeBtn: $('#apply-range')
};

/* ---------- Chart.js setup ---------- */
function setupChartDefaults() {
  if (!window.Chart) { return; }
  if (typeof ChartFinancial !== 'undefined') {
    Chart.register(ChartFinancial);
  }
  // Fix chartjs-chart-financial 0.2.1 + Chart.js 4.4 tooltip crash:
  // core Tooltip._getLabel calls controller.getLabelAndValue(element) but the
  // plugin expects a NUMBER index — getParsed(element) returns undefined and
  // every hover throws (tooltip silently dead). Resolve the element's index.
  var FinCtl = Chart.registry.getController('candlestick');
  if (FinCtl && !FinCtl.prototype.__candleTooltipFix) {
    FinCtl.prototype.getLabelAndValue = function (elOrIdx) {
      var idx = (typeof elOrIdx === 'object' && elOrIdx !== null) ? elOrIdx.index : elOrIdx;
      var parsed = this.getParsed(idx);
      if (!parsed) { return { label: '', value: '' }; }
      return {
        label: String(idx + 1),
        value: 'O: ' + parsed.o + '  H: ' + parsed.h + '  L: ' + parsed.l + '  C: ' + parsed.c
      };
    };
    FinCtl.prototype.__candleTooltipFix = true;
  }
  // chartjs-adapter-luxon auto-registers on load; explicit register is a
  // no-op if already registered but guards against CDN variants that don't.
  if (window.chartjsAdapterLuxon) {
    Chart.register(window.chartjsAdapterLuxon);
  }
  Chart.defaults.color = '#8b949e';
  Chart.defaults.borderColor = 'rgba(139,148,158,0.12)';
  Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', sans-serif";
  Chart.defaults.font.size = 11;
}

/* ---------- Small helpers ---------- */
function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, function (m) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m];
  });
}

// DOM builder — textContent everywhere, no unescaped innerHTML.
function mkEl(tag, className, text) {
  var e = document.createElement(tag);
  if (className) { e.className = className; }
  if (text !== undefined && text !== null) { e.textContent = text; }
  return e;
}

// chartjs-chart-financial sets parsing:false on the whole chart, so every
// dataset item is used as-is ({x, o, h, l, c} / {x, y}); x is the candle's
// ARRAY INDEX (category scale) while the epoch-ms timestamp lives on the
// parallel currentItems[i].x (tooltips / range meta resolve through it).
function toMs(ts) {
  if (typeof ts === 'number') { return ts; }
  if (window.luxon && luxon.DateTime) {
    var dt = luxon.DateTime.fromISO(ts);
    if (dt.isValid) { return dt.toMillis(); }
  }
  return new Date(ts).getTime();
}

function toDateTime(ts) {
  if (window.luxon && luxon.DateTime) {
    var dt = typeof ts === 'number'
      ? luxon.DateTime.fromMillis(ts)
      : luxon.DateTime.fromISO(ts);
    if (dt.isValid) { return dt; }
    return null;
  }
  var d = new Date(ts);
  return isNaN(d.getTime()) ? null : d;
}

function fmtTs(ts) {
  var dt = toDateTime(ts);
  if (!dt) { return String(ts); }
  return dt.toFormat ? dt.toFormat('dd MMM yyyy HH:mm') : dt.toLocaleString('en-IN');
}

function fmtDay(ts) {
  var dt = toDateTime(ts);
  if (!dt) { return String(ts); }
  return dt.toFormat
    ? dt.toFormat('MMM d')
    : dt.toLocaleDateString('en-IN', { month: 'short', day: 'numeric' });
}

function fmtRange(a, b) {
  var sa = fmtDay(a), sb = fmtDay(b);
  var ya = new Date(a).getFullYear(), yb = new Date(b).getFullYear();
  if (!isNaN(ya) && !isNaN(yb) && ya !== yb) { sb += ' ' + yb; }
  return sa + ' – ' + sb;
}

function fmtPct(p) {
  var n = Number(p);
  if (!isFinite(n)) { return '—'; }
  return (n >= 0 ? '+' : '') + n.toFixed(2) + '%';
}

function fmtInt(n) {
  n = Number(n);
  if (!isFinite(n)) { return '—'; }
  return n.toLocaleString('en-IN');
}

function fmtMoney(v) {
  var n = Number(v);
  if (!isFinite(n)) { return '—'; }
  var sign = n >= 0 ? '+' : '-';
  return sign + '₹' + Math.abs(n).toLocaleString('en-IN', {
    maximumFractionDigits: 2, minimumFractionDigits: 2
  });
}

function fmtWinRate(v) {
  if (v === null || v === undefined) { return '—'; }
  var n = Number(v);
  if (!isFinite(n)) { return '—'; }
  if (n >= -1 && n <= 1) { n = n * 100; } // tolerate 0..1 fractions
  return n.toFixed(1) + '%';
}

function touchUpdated() {
  el.lastUpdated.textContent = 'Updated ' + new Date().toLocaleTimeString('en-IN');
}

function showToast(msg, isError) {
  el.statusMsg.textContent = msg;
  el.statusMsg.className = isError ? 'show error' : 'show';
  clearTimeout(statusTimer);
  statusTimer = setTimeout(function () {
    el.statusMsg.classList.remove('show');
  }, isError ? 6000 : 2500);
}

/* ---------- Date range ---------- */
function pad2(n) { return (n < 10 ? '0' : '') + n; }

// Local-timezone YYYY-MM-DD (never toISOString: that is UTC).
function toDateStr(d) {
  return d.getFullYear() + '-' + pad2(d.getMonth() + 1) + '-' + pad2(d.getDate());
}

// Resolved from/to (YYYY-MM-DD) for a preset key, relative to today (local tz).
function presetDates(key) {
  if (key === 'MAX') { return { from: null, to: null }; }
  var today = new Date();
  today.setHours(0, 0, 0, 0);
  var from = new Date(today);
  from.setMonth(from.getMonth() - PRESETS[key]);
  return { from: toDateStr(from), to: toDateStr(today) };
}

// Set the active preset without reloading (keeps the UI in sync).
function setPreset(key) {
  activePreset = key;
  var d = presetDates(key);
  currentRange = { from: d.from, to: d.to };
  syncRangeUI();
}

function applyPreset(key) {
  setPreset(key);
  presetTouched = true;
  loadChart();
}

function applyCustomRange() {
  var from = el.fromDate.value || null;
  var to = el.toDate.value || null;
  if (!from && !to) {
    showToast('Pick at least one of From / To', true);
    return;
  }
  if (from && to && from > to) {
    var t = from; from = to; to = t;
    el.fromDate.value = from;
    el.toDate.value = to;
    showToast('From was after To — swapped', false);
  }
  activePreset = null;
  currentRange = { from: from, to: to };
  presetTouched = true;
  syncRangeUI();
  loadChart();
}

function syncRangeUI() {
  el.presetBtns.forEach(function (b) {
    b.classList.toggle('active', b.getAttribute('data-preset') === activePreset);
  });
  el.fromDate.value = currentRange.from || '';
  el.toDate.value = currentRange.to || '';
}

// Query params for the current range. Presets (except MAX) and custom ranges
// use from/to; MAX asks the backend for its full limit without from/to.
function rangeParams() {
  if (activePreset === 'MAX') { return 'limit=' + MAX_FETCH_LIMIT; }
  var p = [];
  if (currentRange.from) { p.push('from=' + encodeURIComponent(currentRange.from)); }
  if (currentRange.to) { p.push('to=' + encodeURIComponent(currentRange.to)); }
  return p.join('&');
}

/* ---------- API ---------- */
function fetchJson(url) {
  return fetch(url).then(function (res) {
    if (!res.ok) { throw new Error('HTTP ' + res.status + ' for ' + url); }
    return res.json();
  });
}

/* ---------- Health / status chip ---------- */
function loadHealth() {
  fetchJson(API.health).then(function (h) {
    setMarketChip(h.market_open);
    var total = Number(h.candles_total);
    el.candlesTotal.textContent = (isFinite(total) ? total.toLocaleString('en-IN') : '?') + ' candles';
    touchUpdated();
  }).catch(function (err) {
    setMarketChip(null);
    showToast('Health check failed: ' + err.message, true);
  });
}

function setMarketChip(open) {
  var chip = el.marketChip;
  var label = open === true ? 'OPEN' : (open === false ? 'CLOSED' : '?');
  chip.className = 'chip' + (open === true ? ' open' : '');
  chip.textContent = '';
  var dot = document.createElement('span');
  dot.className = 'dot';
  chip.appendChild(dot);
  chip.appendChild(document.createTextNode(label));
}

/* ---------- Symbols ---------- */
function loadSymbols() {
  return fetchJson(API.symbols).then(function (data) {
    symbols = Array.isArray(data.symbols) ? data.symbols : [];
    var eq = symbols.filter(function (s) { return s.instrument_type !== 'INDEX'; });
    var idx = symbols.filter(function (s) { return s.instrument_type === 'INDEX'; });

    var html = '';
    html += buildGroup('Equities', eq);
    html += buildGroup('Indices', idx);
    el.symbolSelect.innerHTML = html || '<option value="">No symbols</option>';

    // Default to TCS (an EQ); fall back to first equity if absent.
    var hasDefault = eq.some(function (s) { return s.symbol === DEFAULT_SYMBOL; });
    if (hasDefault) {
      currentSymbol = DEFAULT_SYMBOL;
    } else if (eq.length) {
      currentSymbol = eq[0].symbol;
    } else if (idx.length) {
      currentSymbol = idx[0].symbol;
    }
    el.symbolSelect.value = currentSymbol;
  }).catch(function (err) {
    el.symbolSelect.innerHTML = '<option value="' + esc(currentSymbol) + '">' + esc(currentSymbol) + '</option>';
    showToast('Failed to load symbols: ' + err.message, true);
  });
}

function buildGroup(label, list) {
  if (!list.length) { return ''; }
  return '<optgroup label="' + esc(label) + ' (' + list.length + ')">' +
    list.map(function (s) {
      return '<option value="' + esc(s.symbol) + '">' + esc(s.symbol) + (s.name ? ' — ' + esc(s.name) : '') + '</option>';
    }).join('') +
    '</optgroup>';
}

/* ---------- Chart data loading ---------- */
// Fetch one series feed for the active symbol/timeframe/range. MAX expands
// past the backend's 5000-row cap: when the first call returns a full batch,
// fetch the older tail via from=<earliest ts day> and merge (deduped by ts).
function fetchSeries(path, sym, tf) {
  var base = path + '?symbol=' + encodeURIComponent(sym) +
             '&timeframe=' + encodeURIComponent(tf);
  var q = rangeParams();
  var url = base + (q ? '&' + q : '');
  return fetchJson(url).then(function (data) {
    if (activePreset !== 'MAX') { return data; }
    var candles = Array.isArray(data.candles) ? data.candles : [];
    if (!candles.length || candles.length < MAX_FETCH_LIMIT) { return data; }
    var earliest = candles[0].ts;
    var fromDate = (typeof earliest === 'string' && earliest.length >= 10)
      ? earliest.slice(0, 10) : null;
    if (!fromDate) { return data; }
    return fetchJson(base + '&from=' + encodeURIComponent(fromDate)).then(function (older) {
      var olderCandles = Array.isArray(older.candles) ? older.candles : [];
      var seen = {}, merged = [];
      olderCandles.concat(candles).forEach(function (c) {
        var k = c.ts;
        if (!seen[k]) { seen[k] = 1; merged.push(c); }
      });
      return { symbol: data.symbol, timeframe: data.timeframe, candles: merged };
    });
  });
}

function loadChart() {
  var seq = ++loadSeq;
  var sym = currentSymbol, tf = currentTimeframe;

  showToast('Loading ' + sym + ' ' + TF_LABEL[tf] + '…');

  // Prefer the indicators feed (it carries OHLCV + indicator columns); fall
  // back to plain candles when it is unavailable (older backend).
  fetchSeries(API.indicators, sym, tf).then(function (data) {
    if (seq !== loadSeq) { return; }
    renderAll(data, true);
  }).catch(function () {
    if (seq !== loadSeq) { return; }
    fetchSeries(API.candles, sym, tf).then(function (data) {
      if (seq !== loadSeq) { return; }
      renderAll(data, false);
    }).catch(function (err) {
      if (seq !== loadSeq) { return; }
      destroyAllCharts();
      setChipsEnabled(false);
      el.chartTitle.textContent = sym + ' · ' + TF_LABEL[tf] + ' · unavailable';
      el.chartRange.textContent = '';
      showToast('Failed to load chart data: ' + err.message, true);
    });
  });
}

function renderAll(data, withIndicators) {
  var candles = Array.isArray(data.candles) ? data.candles : [];
  if (!candles.length) {
    destroyAllCharts();
    setChipsEnabled(false);
    el.chartTitle.textContent = currentSymbol + ' · ' + TF_LABEL[currentTimeframe] + ' · no data';
    el.chartRange.textContent = '';
    showToast('No candle data for ' + currentSymbol + ' ' + TF_LABEL[currentTimeframe], true);
    return;
  }

  currentItems = candles.map(function (c) {
    var it = {
      x: toMs(c.ts),
      o: Number(c.open), h: Number(c.high),
      l: Number(c.low), c: Number(c.close),
      v: Number(c.volume) || 0
    };
    IND_FIELDS.forEach(function (f) {
      it[f] = (c[f] === undefined || c[f] === null) ? null : Number(c[f]);
    });
    return it;
  });
  currentItems = downsample(currentItems, MAX_POINTS);
  currentLabels = buildLabels();
  hasIndicators = withIndicators;
  setChipsEnabled(hasIndicators);

  renderMainChart();
  renderVolumePanel();
  renderRsiPanel();
  renderMacdPanel();
  updateChartMeta();
}

/* ---------- Downsampling (bucket aggregation for big series) ---------- */
function downsample(items, maxPoints) {
  if (items.length <= maxPoints) { return items; }
  var bucket = Math.ceil(items.length / maxPoints);
  var out = [];
  for (var i = 0; i < items.length; i += bucket) {
    var slice = items.slice(i, i + bucket);
    var first = slice[0], last = slice[slice.length - 1];
    var b = {
      x: last.x,
      o: first.o, h: first.h, l: first.l, c: last.c,
      v: 0
    };
    for (var j = 0; j < slice.length; j++) {
      var it = slice[j];
      b.h = Math.max(b.h, it.h);
      b.l = Math.min(b.l, it.l);
      b.v += it.v;
    }
    IND_FIELDS.forEach(function (f) {
      b[f] = null;
      for (var k = slice.length - 1; k >= 0; k--) {
        if (slice[k][f] !== null) { b[f] = slice[k][f]; break; }
      }
    });
    out.push(b);
  }
  return out;
}

/* ---------- Series builders ---------- */
// Per-candle x-axis labels for the category scale: 1d → '06 Aug', intraday →
// 'HH:mm' (the ticks callback surfaces the date at day boundaries; tooltips
// always show the full date via fmtTs, never these short labels).
function buildLabels() {
  return currentItems.map(function (it) {
    var dt = toDateTime(it.x);
    if (!dt) { return String(it.x); }
    if (currentTimeframe === '1d') {
      return dt.toFormat ? dt.toFormat('dd MMM') : fmtDay(it.x);
    }
    return dt.toFormat ? dt.toFormat('HH:mm') : fmtDay(it.x);
  });
}

// Tooltip title for every chart: on a category scale the data index is the
// position — resolve the candle's epoch-ms from the parallel currentItems
// array to always show the full date+time. Uses dataIndex (never undefined),
// not parsed.x, because parsing:false leaves parsed shape dataset-dependent.
function tooltipTitle(items) {
  var idx = items && items.length ? items[0].dataIndex : -1;
  return (idx >= 0 && idx < currentItems.length) ? fmtTs(currentItems[idx].x) : '';
}

function pts(field) {
  return currentItems.map(function (it, i) {
    var v = it[field];
    return { x: i, y: (v === null || v === undefined) ? null : v };
  });
}

function lineDataset(field, label, color, dash) {
  return {
    type: 'line',
    label: label,
    data: pts(field),
    parsing: false,
    borderColor: color,
    borderWidth: 1.4,
    borderDash: dash || [],
    fill: false,
    spanGaps: false,
    tension: 0.2,
    pointRadius: 0,
    pointHoverRadius: 2,
    pointHitRadius: 4,
    order: 2,
    yAxisID: 'y'
  };
}

function candleDataset() {
  return {
    type: 'candlestick',
    label: currentSymbol,
    data: currentItems.map(function (it, i) {
      return { x: i, o: it.o, h: it.h, l: it.l, c: it.c };
    }),
    // NOTE: parsing must stay ENABLED for candlesticks — chartjs-chart-financial's
    // tooltip getLabelAndValue/getParsed path crashes on parsing:false.
    color: { up: UP, down: DOWN, unchanged: FLAT },
    borderColor: { up: UP, down: DOWN, unchanged: FLAT },
    // Explicit hover variants: the plugin's hover style resolution calls
    // .toString() on the value — missing hover colors crash the hover/tooltip
    // event pipeline (chart._active fills but tooltip never renders).
    hoverBackgroundColor: { up: 'rgba(63,185,80,0.40)', down: 'rgba(248,81,73,0.40)', unchanged: FLAT },
    hoverBorderColor: { up: UP, down: DOWN, unchanged: FLAT },
    order: 1
  };
}

function buildMainDatasets() {
  var ds = [candleDataset()];
  if (!hasIndicators) { return ds; }

  if (indicatorsState.sma20) {
    ds.push(lineDataset('sma20', 'SMA 20', C_SMA20, null));
  }
  if (indicatorsState.sma50) {
    ds.push(lineDataset('sma50', 'SMA 50', C_SMA50, null));
  }
  if (indicatorsState.ema) {
    ds.push(lineDataset('ema12', 'EMA 12', C_EMA12, null));
    ds.push(lineDataset('ema26', 'EMA 26', C_EMA26, [6, 4]));
  }
  if (indicatorsState.bb) {
    ds.push({
      type: 'line', label: 'BB Lower', data: pts('bb_lower'),
      parsing: false,
      borderColor: C_BB, borderWidth: 1.2, borderDash: [3, 3],
      fill: false, spanGaps: false, tension: 0.2,
      pointRadius: 0, pointHoverRadius: 2, pointHitRadius: 4,
      order: 2, yAxisID: 'y', id: 'bbLower'
    });
    ds.push({
      type: 'line', label: 'BB Upper', data: pts('bb_upper'),
      parsing: false,
      borderColor: C_BB, borderWidth: 1.2, borderDash: [3, 3],
      fill: { target: 'bbLower', above: 'rgba(57,197,207,0.07)' },
      spanGaps: false, tension: 0.2,
      pointRadius: 0, pointHoverRadius: 2, pointHitRadius: 4,
      order: 2, yAxisID: 'y', id: 'bbUpper'
    });
    ds.push({
      type: 'line', label: 'BB Mid', data: pts('bb_mid'),
      parsing: false,
      borderColor: C_BB, borderWidth: 1.2, borderDash: [8, 4],
      fill: false, spanGaps: false, tension: 0.2,
      pointRadius: 0, pointHoverRadius: 2, pointHitRadius: 4,
      order: 2, yAxisID: 'y'
    });
  }
  return ds;
}

function applyOverlays() {
  if (!chart) { return; }
  chart.data.datasets = buildMainDatasets();
  chart.update();
}

/* ---------- Chart rendering ---------- */
function destroyAllCharts() {
  if (chart) { chart.destroy(); chart = null; }
  if (volumeChart) { volumeChart.destroy(); volumeChart = null; }
  if (rsiChart) { rsiChart.destroy(); rsiChart = null; }
  if (macdChart) { macdChart.destroy(); macdChart = null; }
  lastHoverIndex = -1;
}

// Category (index-based) x scale shared by every chart. Position is the data
// ARRAY INDEX, so candles are ALWAYS evenly spaced — weekend/overnight gaps are
// structurally impossible. All charts share the same per-candle labels array;
// dataset x values are array indexes. hideTicks for the middle sub-panels
// (TradingView single-axis look: main + bottom panel show the x labels).
function xScaleOpts(labels, hideTicks) {
  var opts = {
    type: 'category',
    labels: labels,
    grid: { color: 'rgba(139,148,158,0.10)' },
    border: { color: '#30363d' }
  };
  if (hideTicks) {
    opts.ticks = { display: false };
  } else {
    opts.ticks = {
      color: '#8b949e',
      maxRotation: 0,
      autoSkip: true,
      maxTicksLimit: 10,
      font: { size: 9 }
    };
    if (currentTimeframe !== '1d') {
      // Intraday labels are 'HH:mm' and repeat across days — surface the date
      // on the first tick of each day (tick.value is the data index).
      opts.ticks.callback = intradayTickLabel;
    }
  }
  return opts;
}

// Ticks callback for intraday timeframes: show the full date when this tick's
// day differs from the previous DATA point, else the plain 'HH:mm'.
function intradayTickLabel(value, i, ticks) {
  var idx = Number(value);
  if (!isFinite(idx) || idx < 0 || idx >= currentItems.length) { return String(value); }
  var dt = toDateTime(currentItems[idx].x);
  if (!dt) { return String(value); }
  if (idx > 0) {
    var prev = toDateTime(currentItems[idx - 1].x);
    if (prev && dt.toFormat('yyyy-MM-dd') === prev.toFormat('yyyy-MM-dd')) {
      return dt.toFormat('HH:mm');
    }
  }
  return dt.toFormat('dd MMM HH:mm');
}

function renderMainChart() {
  if (chart) { chart.destroy(); chart = null; }

  chart = new Chart(el.chartCanvas, {
    type: 'candlestick',
    data: { labels: currentLabels, datasets: buildMainDatasets() },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 250 },
      interaction: { mode: 'index', intersect: false },
      // chartjs-chart-financial 0.2.1 overrides hover.mode to 'label' — a
      // Chart.js v3 mode that does NOT exist in v4.4's Interaction.modes, so
      // chart._active stays empty and onHover/active never fire. Force index.
      hover: { mode: 'index', intersect: false },
      onHover: function (evt, active) { syncHover(active); },
      plugins: {
        legend: { display: false },
        tooltip: Object.assign({}, TOOLTIP_STYLE, {
          callbacks: {
            title: tooltipTitle,
            label: function (ctx) {
              if (ctx.dataset.type === 'line') {
                var v = ctx.parsed && ctx.parsed.y;
                return ctx.dataset.label + ': ' + (v === null || v === undefined ? '—' : Number(v).toFixed(2));
              }
              var it = currentItems[ctx.dataIndex];
              if (!it) { return ''; }
              var chg = it.c - it.o;
              return [
                'Open:  ' + it.o.toFixed(2),
                'High:  ' + it.h.toFixed(2),
                'Low:   ' + it.l.toFixed(2),
                'Close: ' + it.c.toFixed(2) + '  ' + (chg >= 0 ? '+' : '') + chg.toFixed(2)
              ];
            }
          }
        })
      },
      scales: {
        x: xScaleOpts(currentLabels, false),
        y: {
          position: 'right',
          grid: { color: 'rgba(139,148,158,0.12)' },
          border: { color: '#30363d' },
          ticks: { color: '#8b949e' }
        }
      }
    }
  });
}

function renderVolumePanel() {
  if (volumeChart) { volumeChart.destroy(); volumeChart = null; }
  var volColors = currentItems.map(function (it) {
    return it.c >= it.o ? 'rgba(63,185,80,0.45)' : 'rgba(248,81,73,0.45)';
  });

  volumeChart = new Chart(el.volumeCanvas, {
    type: 'bar',
    data: {
      labels: currentLabels,
      datasets: [{
        label: 'Volume',
        data: currentItems.map(function (it, i) { return { x: i, y: it.v }; }),
        parsing: false,
        backgroundColor: volColors,
        borderWidth: 0,
        barPercentage: 0.9,
        categoryPercentage: 0.95,
        order: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 200 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: Object.assign({}, TOOLTIP_STYLE, {
          callbacks: {
            title: tooltipTitle,
            label: function (ctx) { return 'Volume: ' + fmtInt(ctx.parsed.y); }
          }
        })
      },
      scales: {
        x: xScaleOpts(currentLabels, true),
        y: {
          position: 'right',
          beginAtZero: true,
          grid: { color: 'rgba(139,148,158,0.10)' },
          border: { color: '#30363d' },
          ticks: { color: '#8b949e', font: { size: 9 }, maxTicksLimit: 4 }
        }
      }
    }
  });
}

function guideDataset(label, y, color) {
  return {
    type: 'line',
    label: label,
    data: [{ x: 0, y: y }, { x: currentItems.length - 1, y: y }],
    parsing: false,
    borderColor: color,
    borderWidth: 1,
    borderDash: [4, 4],
    pointRadius: 0,
    fill: false,
    spanGaps: true,
    order: 0
  };
}

function renderRsiPanel() {
  if (rsiChart) { rsiChart.destroy(); rsiChart = null; }

  rsiChart = new Chart(el.rsiCanvas, {
    type: 'line',
    data: {
      labels: currentLabels,
      datasets: [
        {
          type: 'line',
          label: 'RSI 14',
          data: pts('rsi14'),
          parsing: false,
          borderColor: C_SMA50,
          borderWidth: 1.5,
          fill: false,
          spanGaps: false,
          tension: 0.2,
          pointRadius: 0,
          pointHoverRadius: 2,
          order: 1
        },
        guideDataset('RSI 30', 30, 'rgba(139,148,158,0.55)'),
        guideDataset('RSI 70', 70, 'rgba(139,148,158,0.55)')
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 200 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: Object.assign({}, TOOLTIP_STYLE, {
          filter: function (item) { return item.datasetIndex < 1; },
          callbacks: {
            title: tooltipTitle,
            label: function (ctx) {
              var v = ctx.parsed.y;
              return 'RSI: ' + (v === null || v === undefined ? '—' : Number(v).toFixed(2));
            }
          }
        })
      },
      scales: {
        x: xScaleOpts(currentLabels, true),
        y: {
          position: 'right',
          min: 0,
          max: 100,
          grid: { color: 'rgba(139,148,158,0.10)' },
          border: { color: '#30363d' },
          ticks: { color: '#8b949e', font: { size: 9 }, maxTicksLimit: 4 }
        }
      }
    }
  });
}

function renderMacdPanel() {
  if (macdChart) { macdChart.destroy(); macdChart = null; }
  var histColors = currentItems.map(function (it) {
    var v = it.macd_hist;
    if (v === null || v === undefined) { return 'transparent'; }
    return v >= 0 ? 'rgba(63,185,80,0.5)' : 'rgba(248,81,73,0.5)';
  });

  macdChart = new Chart(el.macdCanvas, {
    type: 'bar',
    data: {
      labels: currentLabels,
      datasets: [
        {
          type: 'bar',
          label: 'Hist',
          data: pts('macd_hist'),
          parsing: false,
          backgroundColor: histColors,
          borderWidth: 0,
          barPercentage: 0.8,
          categoryPercentage: 0.9,
          order: 1
        },
        {
          type: 'line',
          label: 'MACD',
          data: pts('macd'),
          parsing: false,
          borderColor: C_EMA12,
          borderWidth: 1.4,
          fill: false,
          spanGaps: false,
          tension: 0.2,
          pointRadius: 0,
          pointHoverRadius: 2,
          order: 2
        },
        {
          type: 'line',
          label: 'Signal',
          data: pts('macd_signal'),
          parsing: false,
          borderColor: C_EMA26,
          borderWidth: 1.4,
          borderDash: [5, 3],
          fill: false,
          spanGaps: false,
          tension: 0.2,
          pointRadius: 0,
          pointHoverRadius: 2,
          order: 2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 200 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        // Legend hidden so every sub-panel is a pixel-identical column
        // (same fixed canvas height); dataset names still show in tooltips.
        legend: { display: false },
        tooltip: Object.assign({}, TOOLTIP_STYLE, {
          callbacks: {
            title: tooltipTitle,
            label: function (ctx) {
              var v = ctx.parsed.y;
              return ctx.dataset.label + ': ' + (v === null || v === undefined ? '—' : Number(v).toFixed(3));
            }
          }
        })
      },
      scales: {
        x: xScaleOpts(currentLabels, false),
        y: {
          position: 'right',
          grid: { color: 'rgba(139,148,158,0.10)' },
          border: { color: '#30363d' },
          ticks: { color: '#8b949e', font: { size: 9 }, maxTicksLimit: 4 }
        }
      }
    }
  });
}

/* ---------- Cross-chart hover sync (index-aligned category axes) ---------- */
function syncHover(active) {
  var i = -1;
  if (active && active.length) { i = active[0].index; }
  if (i === lastHoverIndex) { return; }
  lastHoverIndex = i;

  [chart, volumeChart, rsiChart, macdChart].forEach(function (ch) {
    if (!ch) { return; }
    try {
      if (i < 0) {
        ch.clearActiveElements();
        if (ch.tooltip) { ch.tooltip.setActiveElements([], {}); }
      } else {
        var el = ch.getDatasetMeta(0) && ch.getDatasetMeta(0).data[i];
        var pos = (el && typeof el.getCenterPoint === 'function') ? el.getCenterPoint() : {};
        ch.setActiveElements([{ datasetIndex: 0, index: i }]);
        // The financial plugin's hover override breaks the tooltip's own event
        // path (tooltip._active never fills) — force it from the synced index.
        if (ch.tooltip) { ch.tooltip.setActiveElements([{ datasetIndex: 0, index: i }], pos); }
      }
      ch.update('none');
    } catch (e) { /* cosmetic only */ }
  });
}

/* ---------- Chart meta ---------- */
function updateChartMeta() {
  el.chartTitle.textContent = currentSymbol + ' · ' + TF_LABEL[currentTimeframe];
  if (!currentItems.length) {
    el.chartRange.textContent = '';
    return;
  }
  var first = currentItems[0].x;
  var last = currentItems[currentItems.length - 1].x;
  var from = currentRange.from || first;
  var to = currentRange.to || last;
  el.chartRange.textContent = fmtRange(from, to) + ' · ' + currentItems.length + ' bars';
}

function setChipsEnabled(enabled) {
  el.indChips.forEach(function (btn) {
    btn.disabled = !enabled;
    btn.classList.toggle('disabled', !enabled);
  });
}

/* ---------- Movers ---------- */
function loadMovers() {
  fetchJson(API.movers + '?n=10').then(function (data) {
    renderMovers(el.gainers, data.gainers, true);
    renderMovers(el.losers, data.losers, false);
  }).catch(function (err) {
    showToast('Failed to load movers: ' + err.message, true);
  });
}

function renderMovers(ul, rows, isGainers) {
  ul.textContent = '';
  var list = Array.isArray(rows) ? rows : [];
  if (!list.length) {
    ul.appendChild(mkEl('li', 'mover-empty', 'No data yet'));
    return;
  }
  list.forEach(function (r) {
    var li = mkEl('li', 'mover-row');
    li.appendChild(mkEl('span', 'mover-symbol', r.symbol || '—'));
    var pct = (r.change_pct === null || r.change_pct === undefined) ? NaN : Number(r.change_pct);
    var cls = isFinite(pct) ? (pct >= 0 ? 'pos' : 'neg') : '';
    li.appendChild(mkEl('span', 'mover-pct ' + cls, (isGainers ? '▲ ' : '▼ ') + fmtPct(pct)));
    ul.appendChild(li);
  });
}

/* ---------- Portfolio ---------- */
function loadPortfolio() {
  fetchJson(API.portfolio).then(function (d) {
    renderPortfolio(d);
  }).catch(function () {
    renderPortfolio(null);
  });
}

function renderPortfolio(d) {
  var body = el.portfolioBody;
  body.textContent = '';

  var summary = (d && d.summary) ? d.summary : null;
  var positions = (d && Array.isArray(d.positions)) ? d.positions : [];
  var tradeCount = Number(summary && summary.trade_count);

  if (!summary || (!positions.length && !(isFinite(tradeCount) && tradeCount > 0))) {
    body.appendChild(emptyState(
      '💼',
      'Bot trading lands in Phase 5',
      'Paper-trade stats will appear here once the trading bot goes live.'
    ));
    return;
  }

  var grid = mkEl('div', 'pf-stats');
  grid.appendChild(pfStat('Trades', fmtInt(tradeCount), null));
  var pnl = summary.unrealized_pnl !== undefined ? summary.unrealized_pnl : summary.realized_pnl;
  var pnlNum = Number(pnl);
  var pnlCls = isFinite(pnlNum) ? (pnlNum >= 0 ? 'pos' : 'neg') : null;
  grid.appendChild(pfStat('Unrealized P&L', fmtMoney(pnl), pnlCls));
  grid.appendChild(pfStat('Win rate', fmtWinRate(summary.win_rate), null));
  body.appendChild(grid);

  if (positions.length) {
    var pl = mkEl('div', 'pf-positions');
    positions.slice(0, 8).forEach(function (p) {
      var row = mkEl('div', 'pf-row');
      row.appendChild(mkEl('span', 'pf-row-sym', p.symbol || '—'));
      row.appendChild(mkEl('span', null, fmtInt(p.qty) + ' @ ₹' + fmtMoney(p.invested).replace('+', '')));
      pl.appendChild(row);
    });
    body.appendChild(pl);
  }
}

function pfStat(k, v, cls) {
  var d = mkEl('div', 'pf-stat');
  d.appendChild(mkEl('div', 'pf-stat-k', k));
  d.appendChild(mkEl('div', 'pf-stat-v' + (cls ? ' ' + cls : ''), v));
  return d;
}

/* ---------- Sentiment ---------- */
function loadSentiment() {
  var seq = ++sentimentSeq;
  var sym = currentSymbol;
  fetchJson(API.sentiment + '?symbol=' + encodeURIComponent(sym)).then(function (d) {
    if (seq !== sentimentSeq) { return; }
    renderSentiment(d, sym);
  }).catch(function () {
    if (seq !== sentimentSeq) { return; }
    renderSentiment(null, sym);
  });
}

function renderSentiment(d, sym) {
  el.sentimentSymbol.textContent = sym;
  var body = el.sentimentBody;
  body.textContent = '';

  var rows = (d && Array.isArray(d.rows)) ? d.rows : [];
  if (!rows.length) {
    body.appendChild(emptyState(
      '📰',
      'News sentiment lands in Phase 3',
      'Scored headlines for ' + sym + ' will show here once the pipeline goes live.'
    ));
    return;
  }
  rows.slice(0, 6).forEach(function (r) {
    body.appendChild(sentiRow(r));
  });
}

function sentiRow(r) {
  var row = mkEl('div', 'senti-row');

  var title = mkEl('a', 'senti-title', r.title || '(untitled)');
  if (r.url) {
    title.href = r.url;
    title.target = '_blank';
    title.rel = 'noopener noreferrer';
  }

  var meta = mkEl('div', 'senti-meta');
  var score = Number(r.sentiment_compound);
  var cls = 'neu';
  var label = '—';
  if (isFinite(score)) {
    if (score >= 0.05) { cls = 'pos'; label = 'Positive'; }
    else if (score <= -0.05) { cls = 'neg'; label = 'Negative'; }
    else { label = 'Neutral'; }
  }
  meta.appendChild(mkEl('span', 'badge ' + cls, label));
  if (r.source) { meta.appendChild(mkEl('span', null, r.source)); }
  if (r.published_at) { meta.appendChild(mkEl('span', 'senti-time', fmtTs(r.published_at))); }

  row.appendChild(title);
  row.appendChild(meta);
  return row;
}

/* ---------- Empty state builder ---------- */
function emptyState(icon, title, hint) {
  var box = mkEl('div', 'empty-state');
  box.appendChild(mkEl('div', 'icon', icon));
  box.appendChild(mkEl('div', 'title', title));
  box.appendChild(mkEl('div', 'hint', hint));
  return box;
}

/* ---------- Events ---------- */
el.symbolSelect.addEventListener('change', function () {
  var next = el.symbolSelect.value;
  if (!next || next === currentSymbol) { return; }
  currentSymbol = next;
  loadChart();
  loadSentiment();
});

el.tfButtons.forEach(function (btn) {
  btn.addEventListener('click', function () {
    var tf = btn.getAttribute('data-tf');
    if (!tf || tf === currentTimeframe) { return; }
    currentTimeframe = tf;
    el.tfButtons.forEach(function (b) { b.classList.toggle('active', b === btn); });
    // Re-apply the current preset (or the new timeframe's default when the
    // user never picked one); keep a custom range untouched.
    if (!presetTouched) {
      setPreset(DEFAULT_PRESET_BY_TF[tf] || '6M');
    } else if (activePreset) {
      setPreset(activePreset);
    }
    loadChart();
  });
});

el.presetBtns.forEach(function (btn) {
  btn.addEventListener('click', function () {
    var key = btn.getAttribute('data-preset');
    if (key) { applyPreset(key); }
  });
});

el.applyRangeBtn.addEventListener('click', applyCustomRange);

[el.fromDate, el.toDate].forEach(function (inp) {
  inp.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') { applyCustomRange(); }
  });
});

el.indChips.forEach(function (btn) {
  btn.addEventListener('click', function () {
    var key = btn.getAttribute('data-ind');
    if (!key || !hasIndicators || !indicatorsState.hasOwnProperty(key)) { return; }
    indicatorsState[key] = !indicatorsState[key];
    btn.classList.toggle('active', indicatorsState[key]);
    applyOverlays();
  });
});

/* ---------- Init ---------- */
function init() {
  if (!window.Chart) {
    showToast('Chart library failed to load — check the CDN', true);
    return;
  }
  setupChartDefaults();

  // Initial range: the timeframe's default preset (1d → 6M, 15m → 3M, 5m → 1M).
  setPreset(DEFAULT_PRESET_BY_TF[DEFAULT_TIMEFRAME] || '6M');

  loadHealth();
  loadMovers();
  loadPortfolio();

  loadSymbols().then(function () {
    loadChart();
    loadSentiment();
  }).catch(function () {
    // symbols failed; still try the chart for the default symbol
    loadChart();
    loadSentiment();
  });

  // Silent auto-refresh: movers + status chip every 60s (never the chart).
  setInterval(function () {
    loadHealth();
    loadMovers();
  }, REFRESH_MS);
}

init();

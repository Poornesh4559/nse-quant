/* ==========================================================================
   NSE Quant — Dashboard frontend (vanilla JS, no build step)
   4-page hash-routed app: #/portfolio · #/charts · #/architecture · #/data

   API contract:
     GET /api/health        -> {status, db_ok, market_open, candles_total}
     GET /api/symbols       -> {symbols: [{symbol, name, instrument_type}]}
     GET /api/candles?symbol=&timeframe=&days=&limit=
                             -> {symbol, timeframe, candles:[{ts, open, high,
                                low, close, volume}]} (ts ASC)
     GET /api/indicators?symbol=&timeframe=&days=&limit=
                             -> same candles + sma20, sma50, ema12, ema26,
                                rsi14, macd, macd_signal, macd_hist,
                                bb_upper, bb_mid, bb_lower, vol_sma20
     GET /api/movers?n=     -> {gainers:[...], losers:[...]}
     GET /api/portfolio     -> {summary:{...}, positions:[{symbol,qty,invested}]}
     GET /api/recent-trades?limit=N -> {trades:[{trade_id, symbol, side, qty,
                                price, ts, fees, pnl, pnl_pct, exit_reason,
                                position_id, composite_score, mom_rank,
                                ml_p_up, sent_3d, market_sentiment,
                                global_cues, regime_score, regime_risk_on,
                                rsi14, macd, bb_pos, atr14, ret_1, ret_5,
                                ret_21, llm_rating, llm_reason, llm_model}]}
     GET /api/market       -> {market_sentiment:{direction, avg_compound,
                                n_articles,...}, global_cues:{..., themes:{}},
                                equity:{date, equity, cash, benchmark}}
     GET /api/equity       -> {points:[{date, equity, cash, benchmark}]}
     GET /api/sentiment?symbol= -> {rows:[...]}
     GET /api/raw?table=&limit=  -> {table, columns:[{column_name,data_type}],
                                rows:[{}]}
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
  recentTrades: '/api/recent-trades',
  market: '/api/market',
  equity: '/api/equity',
  sentiment: '/api/sentiment',
  raw: '/api/raw'
};

var DEFAULT_SYMBOL = 'TCS';
var DEFAULT_TIMEFRAME = '1d';
var REFRESH_MS = 60 * 1000;      // silent auto-refresh of side panels + status chip
var MAX_POINTS = 1_000_000;    // effectively disabled — full series kept for panning
var WIN_LEN = 150;             // FIXED candle count in the visible window
var winStart = 0;              // index into allItems (pan position)
var allItems = [];             // full series (unwindowed) for horizontal panning

// Default lookback windows per timeframe (no user date-range controls).
var DAYS_BY_TF = { '1d': 180, '15m': 90, '5m': 30 };   // 6M / 3M / 1M
var TF_LABEL = { '1d': '1D', '15m': '15M', '5m': '5M' };

// Whitelisted tables for the raw data viewer (/api/raw).
var RAW_TABLES = [
  'trades', 'trade_decisions', 'news_sentiment', 'market_sentiment',
  'global_cues', 'equity_curve', 'symbols', 'candles_1d'
];

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
var equityChart = null;

var symbols = [];
var currentSymbol = DEFAULT_SYMBOL;
var currentTimeframe = DEFAULT_TIMEFRAME;
var currentItems = [];       // last rendered (possibly downsampled) items
var currentLabels = [];      // per-candle x-axis labels (category scale, index-aligned)
var hasIndicators = false;   // true when the indicators feed is available
var indicatorsState = { sma20: true, sma50: true, ema: false, bb: false };

var loadSeq = 0;             // guards against out-of-order candle responses
var sentimentSeq = 0;        // guards out-of-order sentiment responses
var statusTimer = null;
var lastHoverIndex = -1;
var lastCloseCache = {};     // symbol -> last daily close (for position P&L)

var chartsInit = false;      // charts page lazy-init flag
var portfolioInit = false;   // portfolio page lazy-init flag
var dataInit = false;        // data page lazy-init flag

/* ---------- DOM ---------- */
function $(sel) { return document.querySelector(sel); }

var el = {
  marketChip: $('#market-chip'),
  candlesTotal: $('#candles-total'),
  lastUpdated: $('#last-updated'),
  statusMsg: $('#status-msg')
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

function mkEl(tag, className, text) {
  var e = document.createElement(tag);
  if (className) { e.className = className; }
  if (text !== undefined && text !== null) { e.textContent = text; }
  return e;
}

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

// Numeric with fixed decimals; null/undefined/NaN -> '—'
function fmtNum(v, d) {
  var n = Number(v);
  if (v === null || v === undefined || !isFinite(n)) { return '—'; }
  return n.toFixed(d === undefined ? 2 : d);
}

// Signed numeric (P&L-style), '—' on null
function fmtSigned(v, d) {
  var n = Number(v);
  if (v === null || v === undefined || !isFinite(n)) { return '—'; }
  return (n >= 0 ? '+' : '') + n.toFixed(d === undefined ? 2 : d);
}

function fmtBool(v) {
  if (v === null || v === undefined) { return '—'; }
  return v ? 'Yes' : 'No';
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

/* ==========================================================================
   HASH ROUTER
   ========================================================================== */
var ROUTES = ['portfolio', 'charts', 'architecture', 'data'];

function currentRoute() {
  var h = (location.hash || '').replace(/^#\/?/, '');
  return ROUTES.indexOf(h) >= 0 ? h : 'portfolio';
}

function showPage(route) {
  ROUTES.forEach(function (r) {
    var page = document.getElementById('page-' + r);
    if (page) { page.classList.toggle('active', r === route); }
  });
  document.querySelectorAll('.nav-link').forEach(function (a) {
    a.classList.toggle('active', a.getAttribute('data-route') === route);
  });
}

function route() {
  var r = currentRoute();
  showPage(r);

  // Lazy page init — charts must only build while visible (hidden canvases
  // render at 0x0); portfolio/data initialize on first visit.
  if (r === 'portfolio') {
    if (!portfolioInit) { portfolioInit = true; initPortfolio(); }
    else { refreshPortfolio(); }
  }
  if (r === 'charts') {
    if (!chartsInit) { chartsInit = true; initCharts(); }
    else { resizeAllCharts(); }
  }
  if (r === 'data') {
    if (!dataInit) { dataInit = true; initDataPage(); }
    else { loadRaw(); }
  }
}

/* ==========================================================================
   PAGE 1 · PORTFOLIO
   ========================================================================== */
function initPortfolio() {
  loadEquityChart();
  loadMarket();
  loadPositions();
  loadRecentTrades();
}

function refreshPortfolio() {
  loadEquityChart();
  loadMarket();
  loadPositions();
}

/* ---------- Equity vs benchmark chart ---------- */
function loadEquityChart() {
  fetchJson(API.equity).then(function (d) {
    var pts = Array.isArray(d.points) ? d.points : [];
    if (pts.length <= 1) {
      // empty state — needs at least 2 points to draw a curve
      var wrap = $('#equity-chart-wrap'), empty = $('#equity-empty');
      if (wrap) { wrap.style.display = 'none'; }
      if (empty) { empty.style.display = ''; }
      if (equityChart) { equityChart.destroy(); equityChart = null; }
      return;
    }
    var empty = $('#equity-empty');
    if (empty) { empty.style.display = 'none'; }
    var wrap = $('#equity-chart-wrap');
    if (wrap) { wrap.style.display = ''; }
    renderEquityChart(pts);
  }).catch(function (err) {
    showToast('Failed to load equity curve: ' + err.message, true);
  });
}

function renderEquityChart(pts) {
  var canvas = $('#equity-chart');
  if (!canvas || !window.Chart) { return; }
  if (equityChart) { equityChart.destroy(); equityChart = null; }

  var labels = pts.map(function (p) { return fmtDay(p.date); });
  var eq = pts.map(function (p, i) { return { x: i, y: Number(p.equity) }; });
  var bm = pts.map(function (p, i) { return { x: i, y: Number(p.benchmark) }; });

  equityChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Paper equity',
          data: eq,
          parsing: false,
          borderColor: UP,
          backgroundColor: 'rgba(63,185,80,0.08)',
          fill: true,
          tension: 0.25,
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 3,
          pointHitRadius: 6,
          order: 1
        },
        {
          label: 'Benchmark',
          data: bm,
          parsing: false,
          borderColor: C_EMA12,
          borderDash: [6, 4],
          fill: false,
          tension: 0.25,
          borderWidth: 1.6,
          pointRadius: 0,
          pointHoverRadius: 3,
          pointHitRadius: 6,
          order: 2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 250 },
      interaction: { mode: 'index', intersect: false },
      hover: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          align: 'end',
          labels: { color: '#8b949e', boxWidth: 14, font: { size: 11 } }
        },
        tooltip: Object.assign({}, TOOLTIP_STYLE, {
          callbacks: {
            title: function (items) {
              var idx = items && items.length ? items[0].dataIndex : -1;
              return (idx >= 0 && idx < pts.length) ? fmtTs(pts[idx].date) : '';
            },
            label: function (ctx) {
              var v = ctx.parsed && ctx.parsed.y;
              return ctx.dataset.label + ': ₹' + (v === null || v === undefined ? '—' : Number(v).toLocaleString('en-IN', { maximumFractionDigits: 2 }));
            }
          }
        })
      },
      scales: {
        x: {
          type: 'category',
          grid: { color: 'rgba(139,148,158,0.10)' },
          border: { color: '#30363d' },
          ticks: { color: '#8b949e', maxRotation: 0, autoSkip: true, maxTicksLimit: 8, font: { size: 9 } }
        },
        y: {
          position: 'right',
          grid: { color: 'rgba(139,148,158,0.12)' },
          border: { color: '#30363d' },
          ticks: { color: '#8b949e', callback: function (v) { return '₹' + Number(v).toLocaleString('en-IN'); } }
        }
      }
    }
  });
}

/* ---------- Market pulse (sentiment + global cues + equity snap) ---------- */
function loadMarket() {
  fetchJson(API.market).then(function (d) {
    renderMarket(d);
  }).catch(function (err) {
    var body = $('#market-body');
    if (body) {
      body.textContent = '';
      body.appendChild(emptyState('📊', 'Market data unavailable', err.message));
    }
  });
}

function dirClass(direction) {
  var d = String(direction || '').toUpperCase();
  if (d === 'BULLISH') { return 'bullish'; }
  if (d === 'BEARISH') { return 'bearish'; }
  return 'neutral';
}

function marketCard(title, m) {
  var card = mkEl('div', 'market-card');
  var head = mkEl('div', 'market-card-head');
  head.appendChild(mkEl('span', 'market-card-title', title));
  head.appendChild(mkEl('span', 'direction-badge ' + dirClass(m.direction), m.direction || '—'));
  card.appendChild(head);

  var stats = mkEl('div', 'market-stats');
  stats.appendChild(marketStat('Avg compound', fmtSigned(m.avg_compound, 3)));
  stats.appendChild(marketStat('Articles', fmtInt(m.n_articles)));
  stats.appendChild(marketStat('Pos / Neg', '<span class="pos">' + fmtInt(m.n_positive) + '</span> / <span class="neg">' + fmtInt(m.n_negative) + '</span>', true));
  card.appendChild(stats);
  return card;
}

function marketStat(k, v, isHtml) {
  var s = mkEl('div', 'market-stat');
  s.appendChild(mkEl('div', 'market-stat-k', k));
  var vEl = mkEl('div', 'market-stat-v');
  if (isHtml) { vEl.innerHTML = v; } else { vEl.textContent = v; }
  s.appendChild(vEl);
  return s;
}

function themeChip(key, t) {
  var chip = mkEl('span', 'theme-chip');
  chip.appendChild(mkEl('span', 'theme-name', key.replace('_', ' ')));
  var avg = Number(t.avg);
  chip.appendChild(mkEl('span', 'theme-avg ' + (isFinite(avg) ? (avg >= 0 ? 'pos' : 'neg') : ''), fmtSigned(avg, 3)));
  chip.appendChild(mkEl('span', 'theme-n', 'n=' + fmtInt(t.n)));
  return chip;
}

function renderMarket(d) {
  var body = $('#market-body');
  if (!body) { return; }
  body.textContent = '';

  var grid = mkEl('div', 'market-grid');
  var ms = (d && d.market_sentiment) || {};
  var gc = (d && d.global_cues) || {};

  grid.appendChild(marketCard('Market sentiment', ms));

  var gcCard = marketCard('Global cues', gc);
  var themes = (gc && gc.themes && typeof gc.themes === 'object') ? gc.themes : {};
  var themeKeys = Object.keys(themes).filter(function (k) { return themes[k] && typeof themes[k] === 'object'; });
  if (themeKeys.length) {
    var chips = mkEl('div', 'theme-chips');
    themeKeys.forEach(function (k) { chips.appendChild(themeChip(k, themes[k])); });
    gcCard.appendChild(chips);
  }
  grid.appendChild(gcCard);

  var eq = (d && d.equity) || {};
  if (eq && (eq.equity !== undefined || eq.benchmark !== undefined)) {
    var snap = mkEl('div', 'equity-snap');
    snap.appendChild(mkEl('span', null, 'Paper ₹' + fmtNum(eq.equity)));
    snap.appendChild(mkEl('span', null, 'Benchmark ₹' + fmtNum(eq.benchmark)));
    snap.appendChild(mkEl('span', null, String(eq.strategy || '')));
    grid.appendChild(snap);
  }

  body.appendChild(grid);
}

/* ---------- Open positions (with live last close) ---------- */
function fetchLastClose(sym) {
  if (lastCloseCache[sym] !== undefined) {
    return Promise.resolve(lastCloseCache[sym]);
  }
  return fetchJson(API.candles + '?symbol=' + encodeURIComponent(sym) + '&timeframe=1d&limit=1')
    .then(function (d) {
      var arr = (d && Array.isArray(d.candles)) ? d.candles : [];
      var c = arr.length ? Number(arr[arr.length - 1].close) : null;
      lastCloseCache[sym] = (isFinite(c) ? c : null);
      return lastCloseCache[sym];
    }).catch(function () {
      lastCloseCache[sym] = null;
      return null;
    });
}

function loadPositions() {
  fetchJson(API.portfolio).then(function (d) {
    var positions = (d && Array.isArray(d.positions)) ? d.positions : [];
    if (!positions.length) {
      var body = $('#positions-body');
      if (body) {
        body.textContent = '';
        body.appendChild(emptyState('💼', 'No open positions', 'Paper-trade positions will appear here once the bot trades.'));
      }
      return;
    }
    Promise.all(positions.map(function (p) {
      return fetchLastClose(p.symbol).then(function (close) {
        return { p: p, close: close };
      });
    })).then(function (rows) { renderPositions(rows); });
  }).catch(function (err) {
    var body = $('#positions-body');
    if (body) {
      body.textContent = '';
      body.appendChild(emptyState('💼', 'Positions unavailable', err.message));
    }
  });
}

function renderPositions(rows) {
  var body = $('#positions-body');
  if (!body) { return; }
  body.textContent = '';

  var tbl = mkEl('table', 'positions-table');
  var thead = mkEl('thead');
  var htr = mkEl('tr');
  ['Symbol', 'Qty', 'Invested', 'Last close', 'Unrealized P&L', 'P&L %'].forEach(function (h, i) {
    var th = mkEl('th', i >= 1 ? 'num' : '', h);
    htr.appendChild(th);
  });
  thead.appendChild(htr);
  tbl.appendChild(thead);

  var tbody = mkEl('tbody');
  rows.forEach(function (row) {
    var p = row.p;
    var invested = Number(p.invested);
    var qty = Number(p.qty);
    var close = row.close;
    var tr = mkEl('tr');

    tr.appendChild(mkEl('td', 'sym', p.symbol || '—'));
    tr.appendChild(mkEl('td', 'num', fmtInt(qty)));

    var invTd = mkEl('td', 'num', '₹' + fmtNum(invested));
    tr.appendChild(invTd);

    var closeTd = mkEl('td', 'num', close === null ? '—' : '₹' + fmtNum(close));
    tr.appendChild(closeTd);

    var pnlTd = mkEl('td', 'num');
    var pnlCls = '';
    if (close !== null && isFinite(invested) && isFinite(qty) && qty > 0) {
      var pnl = close * qty - invested;
      pnlCls = pnl >= 0 ? 'pos' : 'neg';
      pnlTd.textContent = fmtMoney(pnl);
    } else {
      pnlTd.textContent = '—';
    }
    pnlTd.classList.add(pnlCls);
    tr.appendChild(pnlTd);

    var pctTd = mkEl('td', 'num');
    if (close !== null && isFinite(invested) && invested > 0 && isFinite(qty) && qty > 0) {
      var pct = (close * qty - invested) / invested * 100;
      pctTd.textContent = fmtPct(pct);
      pctTd.classList.add(pct >= 0 ? 'pos' : 'neg');
    } else {
      pctTd.textContent = '—';
    }
    tr.appendChild(pctTd);

    tbody.appendChild(tr);
  });
  tbl.appendChild(tbody);
  body.appendChild(tbl);
}

/* ---------- Last 10 trades with full decision log ---------- */
function loadRecentTrades() {
  fetchJson(API.recentTrades + '?limit=10').then(function (d) {
    renderTrades((d && Array.isArray(d.trades)) ? d.trades : []);
  }).catch(function (err) {
    var body = $('#trades-body');
    if (body) {
      body.textContent = '';
      body.appendChild(emptyState('🧾', 'Trades unavailable', err.message));
    }
  });
}

function fmtPrice(v) {
  var n = Number(v);
  if (!isFinite(n)) { return '—'; }
  return '₹' + n.toLocaleString('en-IN', { maximumFractionDigits: 2, minimumFractionDigits: 2 });
}

function tradeHeader(t) {
  var btn = mkEl('button', 'trade-header');
  btn.type = 'button';
  btn.setAttribute('aria-expanded', 'false');

  btn.appendChild(mkEl('span', 'trade-sym', t.symbol || '—'));
  var side = String(t.side || '').toUpperCase();
  btn.appendChild(mkEl('span', 'side-badge ' + (side === 'SELL' ? 'sell' : 'buy'), side || '—'));
  btn.appendChild(mkEl('span', 'trade-qty', fmtInt(t.qty) + ' @ ' + fmtPrice(t.price)));
  btn.appendChild(mkEl('span', 'trade-time', fmtTs(t.ts)));

  var pnl = Number(t.pnl);
  if (t.pnl !== null && t.pnl !== undefined && isFinite(pnl)) {
    var pnlEl = mkEl('span', 'trade-pnl ' + (pnl >= 0 ? 'pos' : 'neg'), fmtMoney(pnl) + ' (' + fmtPct(t.pnl_pct) + ')');
    btn.appendChild(pnlEl);
  }

  btn.appendChild(mkEl('span', 'trade-chev', '▼'));
  return btn;
}

// label, value renderer
var DECISION_FIELDS = [
  ['Composite', function (t) { return fmtNum(t.composite_score, 3); }],
  ['ML P(up)', function (t) { return fmtNum(t.ml_p_up, 3); }],
  ['Mom rank', function (t) { return fmtNum(t.mom_rank, 2); }],
  ['Sent 3d', function (t) { return fmtSigned(t.sent_3d, 3); }],
  ['Mkt sentiment', function (t) { return fmtSigned(t.market_sentiment, 3); }],
  ['Global cues', function (t) { return fmtSigned(t.global_cues, 3); }],
  ['Regime score', function (t) { return fmtNum(t.regime_score, 3); }],
  ['Regime risk-on', function (t) { return fmtBool(t.regime_risk_on); }],
  ['RSI 14', function (t) { return fmtNum(t.rsi14, 1); }],
  ['MACD', function (t) { return fmtNum(t.macd, 2); }],
  ['BB pos', function (t) { return fmtNum(t.bb_pos, 3); }],
  ['ATR 14', function (t) { return fmtNum(t.atr14, 1); }],
  ['Ret 1d', function (t) { return fmtPct(t.ret_1); }],
  ['Ret 5d', function (t) { return fmtPct(t.ret_5); }],
  ['Ret 21d', function (t) { return fmtPct(t.ret_21); }],
  ['Fees', function (t) { return '₹' + fmtNum(t.fees); }],
  ['Exit reason', function (t) { return t.exit_reason || '—'; }]
];

function tradeDetail(t) {
  var detail = mkEl('div', 'trade-detail');

  var grid = mkEl('div', 'decision-grid');
  DECISION_FIELDS.forEach(function (f) {
    var pair = mkEl('div', 'dg-pair');
    pair.appendChild(mkEl('span', 'dg-label', f[0]));
    pair.appendChild(mkEl('span', 'dg-value', f[1](t)));
    grid.appendChild(pair);
  });
  detail.appendChild(grid);

  var llmBox = mkEl('div', 'llm-box');
  var rating = Number(t.llm_rating);
  var row = mkEl('div', 'llm-row');
  row.appendChild(mkEl('span', 'llm-label', 'LLM rating'));
  var barWrap = mkEl('div', 'llm-bar-wrap');
  var fill = mkEl('div', 'llm-bar-fill');
  var pct = isFinite(rating) ? Math.max(0, Math.min(1, rating)) * 100 : 0;
  fill.style.width = pct + '%';
  if (isFinite(rating)) {
    fill.className = 'llm-bar-fill ' + (rating >= 0.6 ? 'hi' : (rating >= 0.4 ? 'mid' : 'lo'));
  }
  barWrap.appendChild(fill);
  row.appendChild(barWrap);
  row.appendChild(mkEl('span', 'llm-val', isFinite(rating) ? rating.toFixed(2) : '—'));
  llmBox.appendChild(row);

  if (t.llm_reason) { llmBox.appendChild(mkEl('p', 'llm-reason', t.llm_reason)); }
  if (t.llm_model) { llmBox.appendChild(mkEl('div', 'llm-model', t.llm_model)); }
  detail.appendChild(llmBox);

  return detail;
}

function renderTrades(trades) {
  var body = $('#trades-body');
  if (!body) { return; }
  body.textContent = '';

  var sub = $('#trades-sub');
  if (sub) { sub.textContent = trades.length ? trades.length + ' of last 10 · full decision log' : '—'; }

  if (!trades.length) {
    body.appendChild(emptyState('🧾', 'No trades yet', 'Executed paper trades with their full decision log will show here.'));
    return;
  }

  trades.forEach(function (t) {
    var card = mkEl('div', 'trade-card');
    var header = tradeHeader(t);
    var detail = tradeDetail(t);
    header.addEventListener('click', function () {
      var open = card.classList.toggle('open');
      header.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    card.appendChild(header);
    card.appendChild(detail);
    body.appendChild(card);
  });
}

/* ==========================================================================
   PAGE 2 · CHARTS (existing chart page, no date-range controls)
   ========================================================================== */
var chartEl = null;

function initCharts() {
  chartEl = {
    symbolSelect: $('#symbol-select'),
    chartTitle: $('#chart-title'),
    chartRange: $('#chart-range'),
    chartCanvas: $('#candle-chart'),
    volumeCanvas: $('#volume-chart'),
    rsiCanvas: $('#rsi-chart'),
    macdCanvas: $('#macd-chart'),
    gainers: $('#gainers-list'),
    losers: $('#losers-list'),
    portfolioBody: $('#portfolio-body'),
    sentimentBody: $('#sentiment-body'),
    sentimentSymbol: $('#sentiment-symbol'),
    tfButtons: Array.prototype.slice.call(document.querySelectorAll('.tf-btn')),
    indChips: Array.prototype.slice.call(document.querySelectorAll('.ind-chip')),
    panPrev: $('#pan-prev'),
    panNext: $('#pan-next'),
    panLatest: $('#pan-latest')
  };

  if (chartEl.panPrev) { chartEl.panPrev.addEventListener('click', function () { panBy(-Math.floor(WIN_LEN / 3)); }); }
  if (chartEl.panNext) { chartEl.panNext.addEventListener('click', function () { panBy(Math.floor(WIN_LEN / 3)); }); }
  if (chartEl.panLatest) { chartEl.panLatest.addEventListener('click', panToLatest); }
  bindPanGestures();

  if (!window.Chart) {
    showToast('Chart library failed to load — check the CDN', true);
    return;
  }
  setupChartDefaults();

  bindChartsEvents();
  loadMovers();
  loadPortfolio();
  loadSymbols().then(function () {
    loadChart();
    loadSentiment();
  }).catch(function () {
    loadChart();
    loadSentiment();
  });
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
    chartEl.symbolSelect.innerHTML = html || '<option value="">No symbols</option>';

    var hasDefault = eq.some(function (s) { return s.symbol === DEFAULT_SYMBOL; });
    if (hasDefault) {
      currentSymbol = DEFAULT_SYMBOL;
    } else if (eq.length) {
      currentSymbol = eq[0].symbol;
    } else if (idx.length) {
      currentSymbol = idx[0].symbol;
    }
    chartEl.symbolSelect.value = currentSymbol;
  }).catch(function (err) {
    chartEl.symbolSelect.innerHTML = '<option value="' + esc(currentSymbol) + '">' + esc(currentSymbol) + '</option>';
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
// Default lookback windows per timeframe (no user date-range controls):
// 1d → 6M, 15m → 3M, 5m → 1M via the backend `days` param.
function fetchSeries(path, sym, tf) {
  var url = path + '?symbol=' + encodeURIComponent(sym) +
            '&timeframe=' + encodeURIComponent(tf) +
            '&days=' + (DAYS_BY_TF[tf] || 180);
  return fetchJson(url);
}

function loadChart() {
  var seq = ++loadSeq;
  var sym = currentSymbol, tf = currentTimeframe;

  showToast('Loading ' + sym + ' ' + TF_LABEL[tf] + '…');

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
      chartEl.chartTitle.textContent = sym + ' · ' + TF_LABEL[tf] + ' · unavailable';
      chartEl.chartRange.textContent = '';
      showToast('Failed to load chart data: ' + err.message, true);
    });
  });
}

function renderAll(data, withIndicators) {
  var candles = Array.isArray(data.candles) ? data.candles : [];
  if (!candles.length) {
    destroyAllCharts();
    setChipsEnabled(false);
    chartEl.chartTitle.textContent = currentSymbol + ' · ' + TF_LABEL[currentTimeframe] + ' · no data';
    chartEl.chartRange.textContent = '';
    showToast('No candle data for ' + currentSymbol + ' ' + TF_LABEL[currentTimeframe], true);
    return;
  }

  allItems = candles.map(function (c) {
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
  winStart = Math.max(0, allItems.length - WIN_LEN);  // default: latest window
  applyWindow();
}

/* ---------- Fixed candle window + horizontal pan ---------- */
function applyWindow() {
  if (allItems.length > WIN_LEN) {
    currentItems = allItems.slice(winStart, winStart + WIN_LEN);
  } else {
    winStart = 0;
    currentItems = allItems.slice();
  }
  currentLabels = buildLabels();
  renderMainChart();
  renderVolumePanel();
  renderRsiPanel();
  renderMacdPanel();
  updateChartMeta();
}

function panBy(delta) {
  if (!allItems.length) { return; }
  var maxStart = Math.max(0, allItems.length - WIN_LEN);
  winStart = Math.min(maxStart, Math.max(0, winStart + delta));
  applyWindow();
}

function panToLatest() {
  winStart = Math.max(0, allItems.length - WIN_LEN);
  applyWindow();
}

/* Wheel + drag-to-pan on the main chart canvas (horizontal panning). */
function bindPanGestures() {
  if (!chartEl || !chartEl.chartCanvas) { return; }
  var canvas = chartEl.chartCanvas;
  canvas.addEventListener('wheel', function (e) {
    if (!allItems.length) { return; }
    e.preventDefault();
    panBy(e.deltaY > 0 ? Math.floor(WIN_LEN / 5) : -Math.floor(WIN_LEN / 5));
  }, { passive: false });
  var downX = null;
  var dragging = false;
  canvas.addEventListener('pointerdown', function (e) {
    downX = e.clientX;
    dragging = false;
  });
  canvas.addEventListener('pointermove', function (e) {
    if (downX === null) { return; }
    var dx = e.clientX - downX;
    if (!dragging && Math.abs(dx) < 4) { return; }  // threshold: hover still works
    dragging = true;
    var w = canvas.clientWidth || 1;
    panBy(-Math.round((dx / w) * WIN_LEN));
    downX = e.clientX;
  });
  window.addEventListener('pointerup', function () { downX = null; });
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
    color: { up: UP, down: DOWN, unchanged: FLAT },
    borderColor: { up: UP, down: DOWN, unchanged: FLAT },
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
      opts.ticks.callback = intradayTickLabel;
    }
  }
  return opts;
}

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

  chart = new Chart(chartEl.chartCanvas, {
    type: 'candlestick',
    data: { labels: currentLabels, datasets: buildMainDatasets() },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 250 },
      interaction: { mode: 'index', intersect: false },
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

  volumeChart = new Chart(chartEl.volumeCanvas, {
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

  rsiChart = new Chart(chartEl.rsiCanvas, {
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

  macdChart = new Chart(chartEl.macdCanvas, {
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

/* ---------- Cross-chart hover sync ---------- */
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
        if (ch.tooltip) { ch.tooltip.setActiveElements([{ datasetIndex: 0, index: i }], pos); }
      }
      ch.update('none');
    } catch (e) { /* cosmetic only */ }
  });
}

/* ---------- Chart meta ---------- */
function updateChartMeta() {
  chartEl.chartTitle.textContent = currentSymbol + ' · ' + TF_LABEL[currentTimeframe];
  if (!currentItems.length) {
    chartEl.chartRange.textContent = '';
    return;
  }
  var first = currentItems[0].x;
  var last = currentItems[currentItems.length - 1].x;
  var total = allItems.length;
  var suffix = (total > WIN_LEN) ? (' · ' + currentItems.length + '/' + total + ' bars · ◀▶ drag/wheel') : (' · ' + currentItems.length + ' bars');
  chartEl.chartRange.textContent = fmtRange(first, last) + suffix;
}

function setChipsEnabled(enabled) {
  chartEl.indChips.forEach(function (btn) {
    btn.disabled = !enabled;
    btn.classList.toggle('disabled', !enabled);
  });
}

function resizeAllCharts() {
  [chart, volumeChart, rsiChart, macdChart, equityChart].forEach(function (ch) {
    if (ch && typeof ch.resize === 'function') {
      try { ch.resize(); } catch (e) { /* ignore */ }
    }
  });
}

/* ---------- Movers ---------- */
function loadMovers() {
  fetchJson(API.movers + '?n=10').then(function (data) {
    renderMovers(chartEl.gainers, data.gainers, true);
    renderMovers(chartEl.losers, data.losers, false);
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

/* ---------- Sidebar portfolio summary ---------- */
function loadPortfolio() {
  fetchJson(API.portfolio).then(function (d) {
    renderPortfolio(d);
  }).catch(function () {
    renderPortfolio(null);
  });
}

function renderPortfolio(d) {
  var body = chartEl.portfolioBody;
  body.textContent = '';

  var summary = (d && d.summary) ? d.summary : null;
  var positions = (d && Array.isArray(d.positions)) ? d.positions : [];
  var tradeCount = Number(summary && summary.trade_count);

  if (!summary || (!positions.length && !(isFinite(tradeCount) && tradeCount > 0))) {
    body.appendChild(emptyState(
      '💼',
      'No paper trades yet',
      'Paper-trade stats will appear here once the bot trades.'
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

/* ---------- Sentiment (sidebar) ---------- */
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
  chartEl.sentimentSymbol.textContent = sym;
  var body = chartEl.sentimentBody;
  body.textContent = '';

  var rows = (d && Array.isArray(d.rows)) ? d.rows : [];
  if (!rows.length) {
    body.appendChild(emptyState(
      '📰',
      'News sentiment',
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

/* ==========================================================================
   PAGE 4 · RAW DATA VIEWER
   ========================================================================== */
var rawEl = null;

function initDataPage() {
  rawEl = {
    tableSelect: $('#raw-table-select'),
    limitSelect: $('#raw-limit-select'),
    meta: $('#raw-meta'),
    wrap: $('#raw-table-wrap'),
    refreshBtn: $('#raw-refresh')
  };

  RAW_TABLES.forEach(function (t) {
    var opt = document.createElement('option');
    opt.value = t;
    opt.textContent = t;
    rawEl.tableSelect.appendChild(opt);
  });
  rawEl.tableSelect.value = RAW_TABLES[0];

  rawEl.tableSelect.addEventListener('change', loadRaw);
  rawEl.limitSelect.addEventListener('change', loadRaw);
  rawEl.refreshBtn.addEventListener('click', loadRaw);

  loadRaw();
}

function loadRaw() {
  if (!rawEl) { return; }
  var table = rawEl.tableSelect.value;
  var limit = rawEl.limitSelect.value;
  var wrap = rawEl.wrap;

  wrap.textContent = '';
  var loading = mkEl('div', 'raw-loading', 'Loading ' + table + '…');
  wrap.appendChild(loading);
  rawEl.meta.textContent = 'table: ' + table + ' · limit: ' + limit;

  fetchJson(API.raw + '?table=' + encodeURIComponent(table) + '&limit=' + encodeURIComponent(limit))
    .then(function (d) {
      renderRawTable(d);
    })
    .catch(function (err) {
      wrap.textContent = '';
      var e = mkEl('div', 'raw-error', 'Failed to load ' + table + ': ' + err.message);
      wrap.appendChild(e);
    });
}

function renderRawTable(d) {
  var wrap = rawEl.wrap;
  wrap.textContent = '';

  var columns = (d && Array.isArray(d.columns)) ? d.columns : [];
  var rows = (d && Array.isArray(d.rows)) ? d.rows : [];

  rawEl.meta.textContent = 'table: ' + (d.table || '?') + ' · ' + rows.length + ' rows · ' + columns.length + ' columns';

  if (!rows.length) {
    wrap.appendChild(emptyState('🗃️', 'No rows', 'Table ' + (d.table || '?') + ' is empty.'));
    return;
  }
  if (!columns.length) {
    wrap.appendChild(emptyState('🗃️', 'No columns', 'The backend returned no column metadata.'));
    return;
  }

  var tbl = mkEl('table', 'raw-table');
  var thead = mkEl('thead');
  var htr = mkEl('tr');
  columns.forEach(function (c) {
    var th = mkEl('th');
    th.appendChild(mkEl('div', 'raw-col-name', c.column_name));
    th.appendChild(mkEl('div', 'raw-col-type', c.data_type));
    htr.appendChild(th);
  });
  thead.appendChild(htr);
  tbl.appendChild(thead);

  var tbody = mkEl('tbody');
  rows.forEach(function (r) {
    var tr = mkEl('tr');
    columns.forEach(function (c) {
      var td = mkEl('td');
      var v = r[c.column_name];
      if (v === null || v === undefined) {
        td.textContent = '∅';
        td.style.color = '#6e7681';
      } else if (typeof v === 'object') {
        td.textContent = JSON.stringify(v);
      } else {
        td.textContent = String(v);
      }
      td.title = td.textContent;   // full value on hover
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  tbl.appendChild(tbody);
  wrap.appendChild(tbl);
}

/* ---------- Empty state builder ---------- */
function emptyState(icon, title, hint) {
  var box = mkEl('div', 'empty-state');
  box.appendChild(mkEl('div', 'icon', icon));
  box.appendChild(mkEl('div', 'title', title));
  box.appendChild(mkEl('div', 'hint', hint));
  return box;
}

/* ==========================================================================
   EVENTS / INIT
   ========================================================================== */
// Charts page events (bound once at lazy init)
function bindChartsEvents() {
  chartEl.symbolSelect.addEventListener('change', function () {
    var next = chartEl.symbolSelect.value;
    if (!next || next === currentSymbol) { return; }
    currentSymbol = next;
    loadChart();
    loadSentiment();
  });

  chartEl.tfButtons.forEach(function (btn) {
    btn.addEventListener('click', function () {
      var tf = btn.getAttribute('data-tf');
      if (!tf || tf === currentTimeframe) { return; }
      currentTimeframe = tf;
      chartEl.tfButtons.forEach(function (b) { b.classList.toggle('active', b === btn); });
      loadChart();
    });
  });

  chartEl.indChips.forEach(function (btn) {
    btn.addEventListener('click', function () {
      var key = btn.getAttribute('data-ind');
      if (!key || !hasIndicators || !indicatorsState.hasOwnProperty(key)) { return; }
      indicatorsState[key] = !indicatorsState[key];
      btn.classList.toggle('active', indicatorsState[key]);
      applyOverlays();
    });
  });
}

function boot() {
  setupChartDefaults();

  // Default route: #/portfolio (also when no hash at all).
  if (!location.hash) {
    try { history.replaceState(null, '', '#/portfolio'); } catch (e) { location.hash = '#/portfolio'; }
  }
  window.addEventListener('hashchange', route);
  route();

  loadHealth();

  // Silent auto-refresh: status chip every 60s; portfolio panel refresh while visible.
  setInterval(function () {
    loadHealth();
    if (currentRoute() === 'portfolio') { refreshPortfolio(); }
  }, REFRESH_MS);
}

boot();

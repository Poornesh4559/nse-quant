/* ==========================================================================
   NSE Quant — Dashboard frontend (vanilla JS, no build step)
   Consumes the FastAPI JSON contract:
     GET /api/health   -> {status, db_ok, market_open, candles_total}
     GET /api/symbols  -> {symbols: [{symbol, name, instrument_type}]}
     GET /api/candles?symbol=&timeframe=&limit= -> {candles: [{ts, open, high, low, close, volume}]}
     GET /api/movers?n= -> {gainers: [...], losers: [...]}  (row: {symbol, close, prev_close, change_pct, ts})
   ========================================================================== */
'use strict';

/* ---------- Constants ---------- */
var UP = '#3fb950';      // green — Indian convention: close >= open
var DOWN = '#f85149';    // red
var FLAT = '#8b949e';    // gray

var API = {
  health: '/api/health',
  symbols: '/api/symbols',
  candles: '/api/candles',
  movers: '/api/movers'
};

var DEFAULT_SYMBOL = 'TCS';
var DEFAULT_TIMEFRAME = '1d';
var CANDLE_LIMIT = 250;
var REFRESH_MS = 60 * 1000; // silent auto-refresh of movers + status chip

/* ---------- State ---------- */
var chart = null;
var symbols = [];
var currentSymbol = DEFAULT_SYMBOL;
var currentTimeframe = DEFAULT_TIMEFRAME;
var loadSeq = 0;          // guards against out-of-order candle responses
var statusTimer = null;

/* ---------- DOM ---------- */
function $(sel) { return document.querySelector(sel); }

var el = {
  marketChip: $('#market-chip'),
  candlesTotal: $('#candles-total'),
  lastUpdated: $('#last-updated'),
  symbolSelect: $('#symbol-select'),
  chartTitle: $('#chart-title'),
  chartCanvas: $('#candle-chart'),
  statusMsg: $('#status-msg'),
  gainers: $('#gainers-list'),
  losers: $('#losers-list'),
  tfButtons: Array.prototype.slice.call(document.querySelectorAll('.tf-btn'))
};

/* ---------- Chart.js setup ---------- */
if (window.Chart) {
  if (typeof ChartFinancial !== 'undefined') {
    Chart.register(ChartFinancial);
  }
  // chartjs-adapter-luxon auto-registers on load; explicit register is a no-op
  // if already registered but guards against CDN variants that don't.
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

function fmtTs(ts) {
  if (window.luxon && luxon.DateTime) {
    var dt = luxon.DateTime.fromISO(ts);
    if (dt.isValid) { return dt.toFormat('dd MMM yyyy HH:mm'); }
  }
  var d = new Date(ts);
  return isNaN(d.getTime()) ? String(ts) : d.toLocaleString('en-IN');
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

/* ---------- Candles ---------- */
function loadCandles(symbol, timeframe) {
  var seq = ++loadSeq;
  var url = API.candles + '?symbol=' + encodeURIComponent(symbol) +
            '&timeframe=' + encodeURIComponent(timeframe) +
            '&limit=' + CANDLE_LIMIT;

  showToast('Loading ' + symbol + ' ' + timeframe + '…');

  fetchJson(url).then(function (data) {
    if (seq !== loadSeq) { return; } // stale response
    var candles = Array.isArray(data.candles) ? data.candles : [];
    if (!candles.length) {
      destroyChart();
      el.chartTitle.textContent = symbol + ' · ' + timeframe + ' · no data';
      showToast('No candle data for ' + symbol + ' ' + timeframe, true);
      return;
    }
    renderChart(candles, symbol, timeframe);
  }).catch(function (err) {
    if (seq !== loadSeq) { return; }
    showToast('Failed to load candles: ' + err.message, true);
  });
}

/* ---------- Chart rendering ---------- */
function destroyChart() {
  if (chart) { chart.destroy(); chart = null; }
}

function renderChart(candles, symbol, timeframe) {
  destroyChart();
  el.chartTitle.textContent = symbol + ' · ' + timeframe;

  var items = candles.map(function (c) {
    return { x: c.ts, o: c.open, h: c.high, l: c.low, c: c.close };
  });
  var volume = candles.map(function (c) {
    return { x: c.ts, y: Number(c.volume) || 0 };
  });
  var volColors = candles.map(function (c) {
    return c.close >= c.open ? 'rgba(63,185,80,0.45)' : 'rgba(248,81,73,0.45)';
  });

  var unit = timeframe === '1d' ? 'day' : (timeframe === '15m' ? 'hour' : 'minute');

  chart = new Chart(el.chartCanvas, {
    type: 'candlestick',
    data: {
      datasets: [
        {
          label: symbol,
          data: items,
          color: { up: UP, down: DOWN, unchanged: FLAT },
          borderColor: { up: UP, down: DOWN, unchanged: FLAT }
        },
        {
          type: 'bar',
          label: 'Volume',
          data: volume,
          yAxisID: 'y2',
          backgroundColor: volColors,
          borderWidth: 0,
          barPercentage: 1,
          categoryPercentage: 1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 250 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1c2128',
          borderColor: '#30363d',
          borderWidth: 1,
          titleColor: '#e6edf3',
          bodyColor: '#c9d1d9',
          padding: 10,
          callbacks: {
            title: function (items) { return items && items.length ? fmtTs(items[0].parsed.x) : ''; },
            label: function (ctx) {
              if (ctx.dataset.type === 'bar') {
                return 'Volume: ' + fmtInt(ctx.parsed.y);
              }
              var d = ctx.parsed;
              var chg = d.c - d.o;
              return [
                'Open:  ' + d.o.toFixed(2),
                'High:  ' + d.h.toFixed(2),
                'Low:   ' + d.l.toFixed(2),
                'Close: ' + d.c.toFixed(2) + '  ' + (chg >= 0 ? '+' : '') + chg.toFixed(2)
              ];
            }
          }
        }
      },
      scales: {
        x: {
          type: 'time',
          time: {
            unit: unit,
            displayFormats: { day: 'dd MMM', hour: 'HH:mm', minute: 'HH:mm' }
          },
          grid: { color: 'rgba(139,148,158,0.12)' },
          border: { color: '#30363d' },
          ticks: { color: '#8b949e', maxRotation: 0, autoSkip: true, maxTicksLimit: 8 }
        },
        y: {
          position: 'right',
          grid: { color: 'rgba(139,148,158,0.12)' },
          border: { color: '#30363d' },
          ticks: { color: '#8b949e' }
        },
        y2: {
          position: 'left',
          display: false,
          grid: { drawOnChartArea: false },
          border: { display: false }
        }
      }
    }
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
  var list = Array.isArray(rows) ? rows : [];
  if (!list.length) {
    ul.innerHTML = '<li class="mover-empty">No data</li>';
    return;
  }
  ul.innerHTML = list.map(function (r) {
    var pct = Number(r.change_pct);
    var arrow = isGainers ? '▲' : '▼';
    var cls = pct >= 0 ? 'pos' : 'neg';
    return '<li class="mover-row">' +
             '<span class="mover-symbol">' + esc(r.symbol) + '</span>' +
             '<span class="mover-pct ' + cls + '">' + arrow + ' ' + fmtPct(pct) + '</span>' +
           '</li>';
  }).join('');
}

/* ---------- Events ---------- */
el.symbolSelect.addEventListener('change', function () {
  var next = el.symbolSelect.value;
  if (!next || next === currentSymbol) { return; }
  currentSymbol = next;
  loadCandles(currentSymbol, currentTimeframe);
});

el.tfButtons.forEach(function (btn) {
  btn.addEventListener('click', function () {
    var tf = btn.getAttribute('data-tf');
    if (!tf || tf === currentTimeframe) { return; }
    currentTimeframe = tf;
    el.tfButtons.forEach(function (b) { b.classList.toggle('active', b === btn); });
    loadCandles(currentSymbol, currentTimeframe);
  });
});

/* ---------- Init ---------- */
function init() {
  loadHealth();
  loadMovers();
  loadSymbols().then(function () {
    loadCandles(currentSymbol, currentTimeframe);
  }).catch(function () {
    // symbols failed; still try candles for the default symbol
    loadCandles(currentSymbol, currentTimeframe);
  });

  // Silent auto-refresh: movers + status chip every 60s.
  setInterval(function () {
    loadHealth();
    loadMovers();
  }, REFRESH_MS);
}

init();

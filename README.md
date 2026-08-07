# NSE Quant — Full-Stack NSE Market Data Pipeline + Trading Bot

> A personal quant stack for the Indian stock market: collect → store → analyze → visualize → (eventually) trade.
> Built for learning first, profits second. Runs on an Oracle Cloud free-tier VPS + Cloudflare free tier + GitHub.

---

## 🎯 Goal

Build a production-grade personal pipeline for NSE data:

1. **Collect** — live + historical NSE data (candles, ticks, fundamentals, news)
2. **Store** — time-series database, queryable, clean, versioned
3. **Analyze** — Python (pandas / backtesting / ML) + LLM-assisted news sentiment
4. **Show** — real-time dashboard on a subdomain
5. **Trade** — paper trading bot → small live positions (Phase 5)

---

## 🏗️ Architecture

```
┌─ COLLECT ────────────────┐   ┌─ STORE ──────────────┐   ┌─ ANALYZE ───────────────┐   ┌─ SHOW ───────────────────┐
│ Fyers API (primary)      │   │ TimescaleDB           │   │ Python (pandas)         │   │ Cloudflare Pages UI      │
│  ├─ REST: EOD / history  │──▶│  ├─ candles (1m/15m/…│   │  ├─ indicators           │   │  ├─ candles / volume     │
│  └─ WebSocket: live ticks│   │  │   daily, EOD)      │   │  ├─ backtests (vectorbt) │   │  ├─ movers / screener    │
│ Yahoo Finance (fallback) │   │  ├─ news + sentiment │   │  ├─ ML signals (lightgbm)│   │  └─ sentiment feed       │
│ Alpha Vantage (EOD alt)  │   │  │   scores           │   │  └─ Gemini summaries     │   └─────────────────────────┘
│ News RSS → VADER (done ✅)│   │  └─ trades (paper)   │   └──────────────────────────┘            ▲
└──────────────────────────┘   └──────────────────────┘                                            │
        │                              ▲                                                           │
        │                              │                                                           │
        └── cron jobs on VPS ─────────┘                                                           │
                              (Docker Compose)                                                    │
        Cloudflare Tunnel (poornesh.dev) ─────────────────────────────────────────────────────────┘
        Cloudflare Worker = thin JSON API over TimescaleDB (via tunnel) for the dashboard
```

## 🧩 Components

| Layer | Tool | Notes |
|---|---|---|
| **Collector** | Python + Fyers API (`fyers-apiv3`) | Primary source; websocket for live, REST for history |
| **Fallback data** | Yahoo Finance API | Free, no auth — used already, works from this VPS |
| **Storage** | TimescaleDB (Postgres ext) | Docker container on VPS; hypertables for candles |
| **Orchestration** | Cron jobs (Hermes) / Docker Compose | Every 10 min intraday snapshots + EOD batch |
| **Analysis** | pandas, vectorbt, scikit-learn, LightGBM | Signal research + backtesting |
| **Sentiment** | RSS fetch + VADER + LLM (Gemini) | News headline polarity; VADER prototype already working |
| **Dashboard** | Cloudflare Pages + Worker + Charts.js/Plotly | Static frontend, thin API worker, served via tunnel on a subdomain |
| **Bot** | Fyers order API | Paper trading first (Phase 5) |

## 📁 Repo Layout (planned)

```
nse-quant/
├── collector/          # data collection scripts (Fyers + Yahoo + news)
│   ├── fyers_client.py
│   ├── yahoo_eod.py
│   ├── news_sentiment.py     # VADER pipeline (prototype exists)
│   └── scheduler.py          # cron entrypoints
├── storage/            # DB schema, migrations, TimescaleDB setup
│   ├── schema.sql
│   └── docker-compose.yml    # timescaledb service
├── analysis/           # indicators, backtests, signals, ML
│   ├── indicators.py
│   ├── backtest.py
│   └── signals/
├── dashboard/          # Cloudflare Pages app + Worker API
│   ├── pages/          # static frontend
│   └── worker/         # JSON API (routes to tunnel → TimescaleDB)
├── bot/                # paper trading engine (Fyers)
├── scripts/            # one-off utilities
├── tests/
├── .env.example        # API keys template (NEVER commit real keys)
└── README.md
```

## 🗺️ Roadmap

| Phase | What | Status |
|---|---|---|
| **1** | Data collector: EOD OHLC → TimescaleDB, daily + intraday cron | 🟡 infra done (compose + schema) — collector in progress |
| **2** | Dashboard: candles, movers, sentiment on subdomain | ⏳ |
| **3** | News + sentiment pipeline (RSS → VADER → LLM summary) | 🟢 prototype done |
| **4** | Backtesting + signal research (pandas/vectorbt/ML) | ⏳ |
| **5** | Paper trading bot on Fyers (simulated first) | ⏳ |

## 🔐 Security Notes

- **Never commit** `.env` or API keys — `.gitignore` enforced; use `.env.example` with placeholders
- Fyers API keys: store in `~/.hermes/.env` or Docker secrets, **not** in the repo
- Dashboard API: Cloudflare Access or a shared secret on the Worker; don't expose trade endpoints publicly
- Repo can be flipped to private anytime (`gh repo edit --visibility private`)

## ⚠️ Disclaimer

This is a **learning project**. Nothing here is financial advice. Trading involves risk; start with paper trading, never trade money you can't afford to lose. Most signals do not beat the market — build it for the engineering, treat any profit as a bonus.

## 🚀 Getting Started

*(Phase 1 scaffolding — filled in as we build)*

```bash
# requirements
# - Docker + Docker Compose on the VPS (already present)
# - Fyers API creds (user's account)
# - Python 3.11 venv

git clone https://github.com/poornesh4559/nse-quant.git
cd nse-quant
cp .env.example .env   # fill in Fyers keys
docker compose -f storage/docker-compose.yml up -d
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# ... (to be documented per phase)
```

---

*Built with ☕ by Poornesh — Hermes-assisted. Phase 1 starting soon.*

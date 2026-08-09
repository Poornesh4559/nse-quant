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
│   ├── fyers_client.py      # auth + token cache + history/quotes wrapper
│   ├── auth.py              # headless login via vagator v2 HTTP flow (TOTP+PIN → token)
│   ├── symbols.py           # NIFTY 50 + indices seed (NSE API + fallback)
│   ├── backfill.py          # EOD + intraday historical backfill (chunked)
│   ├── intraday.py          # 5-min polling of 5m/15m candles
│   ├── eod.py               # post-close daily candle update
│   ├── scheduler.py         # CLI entrypoints for cron
│   ├── config.py            # .env loading (single source of truth)
│   └── db.py                # psycopg2 upserts for symbols/candles
├── storage/            # DB schema, migrations, TimescaleDB setup
│   ├── init/01_schema.sql
│   └── docker-compose.yml    # timescaledb service
├── analysis/           # indicators, backtests, signals, ML
├── dashboard/          # Cloudflare Pages app + Worker API
├── bot/                # paper trading engine (Fyers)
├── scripts/            # one-off utilities
├── tests/
├── .env.example        # API keys template (NEVER commit real keys)
└── README.md
```

## 🗺️ Roadmap

| Phase | What | Status |
|---|---|---|
| **1** | Data collector: EOD OHLC → TimescaleDB, daily + intraday cron | ✅ LIVE — cron runs Mon-Fri (intraday 5-min, EOD 18:00 IST, token refresh 8:00 IST) |
| **2** | Dashboard: candles, movers, market status | ✅ LIVE — **https://stock.poornesh.dev** (FastAPI + Chart.js via tunnel) |
| **3** | News + sentiment pipeline (Google News RSS + market feeds + reddit + GDELT → VADER+FinBERT dual-model avg → per-symbol scores) | ✅ LIVE — daily cron 19:00 IST, 60-day backfill, dashboard sentiment card |
| **4** | Backtesting + signal research (pandas/vectorbt/ML) | 🚧 FOUNDATION DONE — `analysis/` package: no-lookahead engine (next-open fills, ₹20-min/0.1% fees), 7 classic strategies, metrics, `python -m analysis.research run|bench` on 5y NIFTY 50 data; 13/13 pytest green |
| **5** | Paper trading bot on Fyers (simulated first) | ⏳ |

## ⏰ Scheduled Jobs (for Hermes / cron)

All collector commands run from the repo root with the project venv. Server TZ is
**Asia/Kolkata (IST)**; Hermes cron schedules evaluate in **UTC** (IST - 5:30).

| Job | Schedule (IST) | Command |
|---|---|---|
| Intraday 5m/15m poll | every 5 min, Mon–Fri 09:15–15:35 | `.venv/bin/python -m collector.scheduler intraday` |
| EOD daily candles | daily 18:00 | `.venv/bin/python -m collector.scheduler eod` |
| Token safety refresh | daily 09:00 (before market) | `.venv/bin/python -m collector.scheduler login` |

Notes:
- `intraday`/`eod` auto-refresh the token if expired (headless vagator login);
  the 09:00 job pre-warms it so market-hours runs never stall.
- Only run `intraday` when the market is open — skip weekends/holidays.
- Logs: the SDK writes `data/fyersApi.log` + `data/fyersRequests.log` (gitignored).
- Token cache: `data/fyers_token.json` (gitignored), ~23h validity before refresh.

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
# - Python 3.11 venv (use ~/.local/bin/python3.11; system pip is PEP-668 locked)

git clone https://github.com/poornesh4559/nse-quant.git
cd nse-quant
cp .env.example .env   # fill in Fyers keys
docker compose -f storage/docker-compose.yml up -d
/home/ubuntu/.local/bin/python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt

# seed the symbol universe (NIFTY 50 + indices)
.venv/bin/python -m collector.scheduler seed
# force a fresh token (headless vagator login; needs FYERS_ID/TOTP/PIN/APP_* in .env)
.venv/bin/python -m collector.scheduler login
# backfill history (EOD + 5m/15m), test run on 3 symbols first
.venv/bin/python -m collector.scheduler backfill --days 90 --timeframes 1d 5m 15m --limit 3
.venv/bin/python -m collector.scheduler backfill --days 90 --timeframes 1d 5m 15m
# daily: EOD after ~18:00 IST, intraday every 5 min during market hours
```

**Fyers auth notes:** the 1-day access token has no refresh token. The collector
re-authenticates automatically via the vagator v2 HTTP flow (TOTP + PIN) when the
token expires, and caches it in `data/fyers_token.json`. All creds live in `.env`
(`FYERS_APP_ID`/`FYERS_APP_TYPE`/`FYERS_APP_SECRET`, `FYERS_ID`, `FYERS_TOTP_KEY`, `FYERS_PIN`).

---

*Built with ☕ by Poornesh — Hermes-assisted. Phase 1 starting soon.*

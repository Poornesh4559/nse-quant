-- NSE Quant schema — applied automatically on first TimescaleDB boot.
-- Re-run manually for upgrades: docker compose exec db psql -U nse -d nse_quant -f /docker-entrypoint-initdb.d/01_schema.sql

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Reference list of instruments
CREATE TABLE IF NOT EXISTS symbols (
    symbol          TEXT PRIMARY KEY,          -- e.g. 'TCS', 'NIFTY 50'
    name            TEXT,
    isin            TEXT,
    sector          TEXT,
    instrument_type TEXT DEFAULT 'EQ',         -- EQ, INDEX, FUT, OPT...
    updated_at      TIMESTAMPTZ DEFAULT now()
);

-- OHLC candles (hypertable, partitioned by time)
CREATE TABLE IF NOT EXISTS candles (
    symbol    TEXT NOT NULL REFERENCES symbols(symbol) ON DELETE CASCADE,
    timeframe TEXT NOT NULL,                   -- '1m','5m','15m','1d',...
    ts        TIMESTAMPTZ NOT NULL,
    open      DOUBLE PRECISION,
    high      DOUBLE PRECISION,
    low       DOUBLE PRECISION,
    close     DOUBLE PRECISION,
    volume    BIGINT,
    PRIMARY KEY (symbol, timeframe, ts)
);
SELECT create_hypertable('candles', 'ts', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_candles_lookup ON candles (symbol, timeframe, ts DESC);

-- News headlines + VADER sentiment scores (Phase 3)
CREATE TABLE IF NOT EXISTS news_sentiment (
    id                 BIGSERIAL PRIMARY KEY,
    source             TEXT,
    symbol             TEXT,                      -- ticker this news maps to (NULL = market-wide)
    title              TEXT NOT NULL,
    url                TEXT,
    published_at       TIMESTAMPTZ,
    fetched_at         TIMESTAMPTZ DEFAULT now(),
    sentiment_compound DOUBLE PRECISION,       -- VADER compound (-1..1)
    sentiment_label    TEXT                    -- POSITIVE / NEUTRAL / NEGATIVE
);
CREATE INDEX IF NOT EXISTS idx_news_pub ON news_sentiment (published_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_news_url ON news_sentiment (url);

-- Paper trades (Phase 5)
CREATE TABLE IF NOT EXISTS trades (
    id            BIGSERIAL PRIMARY KEY,
    symbol        TEXT NOT NULL,
    side          TEXT NOT NULL,                    -- BUY / SELL
    qty           INTEGER,
    price         DOUBLE PRECISION,
    ts            TIMESTAMPTZ DEFAULT now(),
    strategy      TEXT,
    status        TEXT DEFAULT 'paper',
    position_id   TEXT,                             -- links entry/exit fills of one position
    fees          DOUBLE PRECISION DEFAULT 0,
    pnl           DOUBLE PRECISION,                 -- realized pnl (exit fills)
    pnl_pct       DOUBLE PRECISION,
    exit_reason   TEXT                              -- signal / stop_loss / end_of_test
);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol, ts);
-- Phase 4/5 migration: paper-trading columns (idempotent)
ALTER TABLE trades ADD COLUMN IF NOT EXISTS position_id TEXT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS fees DOUBLE PRECISION DEFAULT 0;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS pnl DOUBLE PRECISION;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS pnl_pct DOUBLE PRECISION;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS exit_reason TEXT;

-- Pre-market market-wide sentiment (Phase 4): the 8:30 IST daily direction call
CREATE TABLE IF NOT EXISTS market_sentiment (
    date             DATE PRIMARY KEY,
    avg_compound     DOUBLE PRECISION,              -- mean compound across articles
    n_articles       INTEGER,
    n_positive       INTEGER,
    n_negative       INTEGER,
    direction        TEXT,                          -- BULLISH / NEUTRAL / BEARISH
    created_at       TIMESTAMPTZ DEFAULT now()
);

-- Global cues / geopolitics (Phase 4): SEPARATE attribute — US markets, Fed,
-- crude, USD/INR, geo-risk. Feeds the paper bot's market-regime risk gate
-- alongside market_sentiment (60/40 weighted, tweakable).
CREATE TABLE IF NOT EXISTS global_cues (
    date             DATE PRIMARY KEY,
    avg_compound     DOUBLE PRECISION,              -- mean compound across all cue articles
    n_articles       INTEGER,
    n_positive       INTEGER,
    n_negative       INTEGER,
    direction        TEXT,                          -- BULLISH / NEUTRAL / BEARISH
    themes           JSONB,                         -- {"crude": {avg, n}, "fed": {...}, "us_markets": {...}, "geo": {...}, "usd_inr": {...}}
    created_at       TIMESTAMPTZ DEFAULT now()
);

-- Paper-bot daily equity snapshots (Phase 5): mark-to-market vs benchmark
CREATE TABLE IF NOT EXISTS equity_curve (
    date             DATE PRIMARY KEY,
    equity           DOUBLE PRECISION,              -- total paper equity (cash + positions)
    cash             DOUBLE PRECISION,
    benchmark        DOUBLE PRECISION,              -- buy-hold NIFTY 50 value, same start capital
    strategy         TEXT DEFAULT 'paper-v1',
    created_at       TIMESTAMPTZ DEFAULT now()
);

-- Decision log (Phase 5): FULL context of every trade the bot considered —
-- the training set for the NEXT model generation. Every row = one decision
-- with the exact features the bot saw at that moment, the LLM's final 0-1
-- rating, whether it executed, and the trade link (join trades.pnl later).
CREATE TABLE IF NOT EXISTS trade_decisions (
    id               BIGSERIAL PRIMARY KEY,
    decision_ts      TIMESTAMPTZ DEFAULT now(),
    symbol           TEXT NOT NULL,
    action           TEXT,                          -- BUY / SELL / HOLD / SKIP
    price            DOUBLE PRECISION,              -- reference/execution price
    qty              INTEGER,
    -- ranking inputs (the bot's own math)
    composite_score  DOUBLE PRECISION,              -- 0.4*mom + 0.35*ml + 0.25*sent
    mom_rank         DOUBLE PRECISION,
    ml_p_up          DOUBLE PRECISION,
    sent_3d          DOUBLE PRECISION,
    sent_7d          DOUBLE PRECISION,
    -- market context (the regime gate)
    market_sentiment DOUBLE PRECISION,
    global_cues      DOUBLE PRECISION,
    regime_score     DOUBLE PRECISION,
    regime_risk_on   BOOLEAN,
    -- technical snapshot at decision time
    rsi14            DOUBLE PRECISION,
    macd             DOUBLE PRECISION,
    bb_pos           DOUBLE PRECISION,
    atr14            DOUBLE PRECISION,
    ret_1            DOUBLE PRECISION,
    ret_5            DOUBLE PRECISION,
    ret_21           DOUBLE PRECISION,
    vol_z            DOUBLE PRECISION,
    -- LLM rating gate
    llm_rating       DOUBLE PRECISION,              -- final 0..1 from the LLM
    llm_reason       TEXT,                          -- one-line reasoning
    llm_model        TEXT,
    llm_gate_pass    BOOLEAN,                       -- rating >= threshold
    -- outcome
    executed         BOOLEAN DEFAULT FALSE,         -- did the bot actually fill it
    trade_id         BIGINT,                        -- -> trades.id (join for pnl)
    UNIQUE (symbol, decision_ts)
);
CREATE INDEX IF NOT EXISTS idx_trade_decisions_symbol ON trade_decisions (symbol, decision_ts DESC);

"""Sector classification for the NIFTY 50 universe (standard NSE sectoral
index mapping) + Asian-market early cues (Nikkei/Hang Seng/Shanghai/STI).

Commands:
    python -m collector.sectors seed        # UPDATE symbols.sector from the map
    python -m collector.sectors asia        # print Asian market % moves now
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import requests

SECTORS: dict[str, str] = {
    # IT
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "TECHM": "IT", "WIPRO": "IT",
    # Automobile
    "MARUTI": "AUTO", "M&M": "AUTO", "HEROMOTOCO": "AUTO", "BAJAJ-AUTO": "AUTO",
    "TMPV": "AUTO", "EICHERMOT": "AUTO",
    # Banking
    "HDFCBANK": "BANK", "ICICIBANK": "BANK", "SBIN": "BANK", "KOTAKBANK": "BANK",
    "AXISBANK": "BANK", "INDUSINDBK": "BANK",
    # Financial services / insurance
    "BAJFINANCE": "FINANCE", "BAJAJFINSV": "FINANCE",
    "LICI": "INSURANCE", "HDFCLIFE": "INSURANCE", "SBILIFE": "INSURANCE",
    # Metals & mining
    "TATASTEEL": "METAL", "JSWSTEEL": "METAL", "HINDALCO": "METAL",
    # Energy (oil/gas/coal)
    "RELIANCE": "ENERGY", "ONGC": "ENERGY", "BPCL": "ENERGY", "COALINDIA": "ENERGY",
    # Power & utilities
    "NTPC": "POWER", "POWERGRID": "POWER",
    # Pharma & healthcare
    "SUNPHARMA": "PHARMA", "CIPLA": "PHARMA", "DRREDDY": "PHARMA", "DIVISLAB": "PHARMA",
    "APOLLOHOSP": "HEALTHCARE",
    # FMCG / retail / consumer
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "TATACONSUM": "FMCG", "DMART": "RETAIL", "TITAN": "CONSUMER",
    "ASIANPAINT": "CONSUMER",
    # Telecom / infra / cement
    "BHARTIARTL": "TELECOM",
    "LT": "INFRA", "ADANIPORTS": "INFRA", "ADANIENT": "INFRA",
    "ULTRACEMCO": "CEMENT", "GRASIM": "CEMENT",
    # Indices
    "NIFTY 50": "INDEX", "BANKNIFTY": "INDEX",
}

# Asian indices that open BEFORE India (times IST): Nikkei 5:30, Shanghai/HK/SG 7:00.
# Yahoo chart API symbols + friendly names + sign convention (up = risk-on for us).
ASIA_INDICES: dict[str, dict] = {
    "^N225":   {"name": "Nikkei 225", "market": "Japan"},
    "^HSI":    {"name": "Hang Seng", "market": "Hong Kong"},
    "000001.SS": {"name": "Shanghai Comp", "market": "China"},
    "^STI":    {"name": "Straits Times", "market": "Singapore"},
}
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=5d&interval=1d"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}


def seed_sectors() -> int:
    """UPDATE symbols.sector from the static map (idempotent)."""
    from collector.config import settings
    import psycopg2
    conn = psycopg2.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        dbname=settings.postgres_db, user=settings.postgres_user,
        password=settings.postgres_password,
    )
    n = 0
    with conn.cursor() as cur:
        for sym, sector in SECTORS.items():
            cur.execute("UPDATE symbols SET sector=%s WHERE symbol=%s", (sector, sym))
            n += cur.rowcount
        conn.commit()
    conn.close()
    return n


def fetch_asia() -> dict[str, dict]:
    """Live % moves of Asian indices (Yahoo chart API). Never raises per-index."""
    out: dict[str, dict] = {}
    for sym, meta in ASIA_INDICES.items():
        try:
            r = requests.get(YAHOO_CHART.format(sym=sym), headers=UA, timeout=12)
            d = r.json()["chart"]["result"][0]
            closes = d["indicators"]["quote"][0]["close"]
            closes = [c for c in closes if c is not None]
            if len(closes) < 2:
                continue
            last, prev = closes[-1], closes[-2]
            out[sym] = {
                "name": meta["name"], "market": meta["market"],
                "last": round(float(last), 2), "prev": round(float(prev), 2),
                "chg_pct": round((last / prev - 1) * 100, 2),
            }
        except Exception as e:  # noqa: BLE001
            print(f"[sectors] {meta['name']} fetch failed: {e}", file=sys.stderr)
    return out


def cmd_asia() -> int:
    moves = fetch_asia()
    print(f"\n===== ASIAN MARKET CUES ({datetime.now(timezone.utc).astimezone().strftime('%H:%M %Z')}) =====")
    if not moves:
        print("no Asian index data available")
        return 1
    avg = sum(m["chg_pct"] for m in moves.values()) / len(moves)
    for sym, m in moves.items():
        arrow = "🟢" if m["chg_pct"] >= 0 else "🔴"
        print(f"  {arrow} {m['name']:<16} {m['chg_pct']:+.2f}%  (last {m['last']})")
    print(f"  ASIA AVG: {avg:+.2f}%  ->  {'risk-on tilt' if avg >= 0 else 'risk-off tilt'}")
    print("=" * 44)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "seed"
    if cmd == "seed":
        n = seed_sectors()
        print(f"seeded sectors for {n} symbols")
    elif cmd == "asia":
        raise SystemExit(cmd_asia())
    else:
        print("usage: python -m collector.sectors seed|asia")
        raise SystemExit(1)

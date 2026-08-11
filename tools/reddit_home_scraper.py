#!/usr/bin/env python3
"""Home-server Reddit scraper — runs on the home LXC (home IP, NOT blocked by
Reddit like the Oracle VPS). Fetches subreddit JSON, filters recent posts and
POSTs them to the nse-quant ingest endpoint where they get symbol-mapped and
scored (VADER+FinBERT) server-side.

STDLIB ONLY (urllib) — no pip installs needed on the tiny home box.
Cron every 30 min:  */30 * * * *  python3 /opt/reddit_scraper/reddit_home_scraper.py >> /var/log/reddit_scraper.log 2>&1

Config via env vars (or edit the defaults):
  REDDIT_INGEST_URL  default https://stock.poornesh.dev/api/ingest/reddit
  REDDIT_INGEST_TOKEN  (required — from the VPS .env)
  REDDIT_HOURS       default 12  (only posts newer than this)
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

SUBREDDITS = [
    "IndianStockMarket", "StockMarketIndia", "IndiaInvestments",
    "IndianStreetBets", "NSEbets",
]
INGEST_URL = os.getenv("REDDIT_INGEST_URL", "https://stock.poornesh.dev/api/ingest/reddit")
TOKEN = os.getenv("REDDIT_INGEST_TOKEN", "")
HOURS = float(os.getenv("REDDIT_HOURS", "12"))
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
DELAY_S = 2.0  # polite gap between subreddits


def fetch_json(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"  fetch failed {url}: {e}", file=sys.stderr)
        return None


def main() -> int:
    if not TOKEN:
        print("REDDIT_INGEST_TOKEN not set — aborting", file=sys.stderr)
        return 1
    cutoff = time.time() - HOURS * 3600
    posts: list[dict] = []
    for sub in SUBREDDITS:
        url = f"https://www.reddit.com/r/{sub}/new.json?limit=25"
        data = fetch_json(url)
        if not data:
            continue
        for child in (data.get("data", {}).get("children") or []):
            p = child.get("data", {})
            created = p.get("created_utc", 0)
            if created < cutoff:
                continue
            posts.append({
                "title": p.get("title", "")[:500],
                "permalink": p.get("permalink", ""),
                "created_utc": created,
                "subreddit": sub,
                "selftext": (p.get("selftext") or "")[:400],
            })
        time.sleep(DELAY_S)

    if not posts:
        print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] no new posts")
        return 0

    body = json.dumps({"posts": posts}).encode("utf-8")
    req = urllib.request.Request(INGEST_URL, data=body, method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOKEN}",
    })
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] "
              f"{len(posts)} posts -> {result.get('stored', 0)} stored")
    except Exception as e:  # noqa: BLE001
        print(f"POST to ingest failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

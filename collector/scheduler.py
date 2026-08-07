"""CLI entrypoints for the collector — these are what cron / Hermes invoke.

Examples:
    .venv/bin/python -m collector.scheduler seed
    .venv/bin/python -m collector.scheduler login
    .venv/bin/python -m collector.scheduler backfill --days 90 --timeframes 1d 5m 15m
    .venv/bin/python -m collector.scheduler eod
    .venv/bin/python -m collector.scheduler intraday
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta

from collector.backfill import backfill
from collector.config import settings
from collector.eod import collect_eod
from collector.fyers_client import FyersClient
from collector.intraday import collect_intraday
from collector.symbols import seed_symbols

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("scheduler")


def cmd_seed(_args) -> int:
    return seed_symbols()


def cmd_login(_args) -> int:
    client = FyersClient()
    token = client.get_access_token(force=True)
    print(f"access_token obtained and cached to {settings.token_cache_path}")
    print(token)
    return 0


def cmd_backfill(args) -> int:
    client = FyersClient()
    to_dt = datetime.now().astimezone()
    from_dt = to_dt - timedelta(days=args.days)
    rows = backfill(client, args.timeframes, from_dt, to_dt, limit=args.limit)
    logger.info("backfill complete: %d rows", rows)
    return 0


def cmd_eod(_args) -> int:
    client = FyersClient()
    rows = collect_eod(client)
    logger.info("eod complete: %d rows", rows)
    return 0


def cmd_intraday(args) -> int:
    client = FyersClient()
    rows = collect_intraday(client, args.timeframes)
    logger.info("intraday complete: %d rows", rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scheduler", description="NSE Quant collector")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("seed", help="seed `symbols` table with NIFTY 50 + indices")
    sub.add_parser("login", help="force a headless login and cache a fresh token")

    bp = sub.add_parser("backfill", help="backfill historical candles")
    bp.add_argument("--days", type=int, default=90, help="history depth in days (default 90)")
    bp.add_argument("--timeframes", nargs="+", default=["1d", "5m", "15m"])
    bp.add_argument("--limit", type=int, default=None, help="max symbols to process (test runs)")

    ep = sub.add_parser("eod", help="fetch today's daily candles")
    ip = sub.add_parser("intraday", help="poll recent intraday candles")
    ip.add_argument("--timeframes", nargs="+", default=["5m", "15m"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    handlers = {
        "seed": cmd_seed,
        "login": cmd_login,
        "backfill": cmd_backfill,
        "eod": cmd_eod,
        "intraday": cmd_intraday,
    }
    code = handlers[args.cmd](args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

"""Fyers v3 API client with automatic token lifecycle management.

The Fyers access_token is only valid for 24h and there is no refresh token.
This module caches the token to `data/fyers_token.json` and transparently
re-runs the headless login (collector.auth) when it expires.

Usage:
    from collector.fyers_client import FyersClient
    client = FyersClient()
    candles = client.history("NSE:TCS-EQ", "D", from_dt, to_dt)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

from fyers_apiv3 import fyersModel

from collector.config import settings

logger = logging.getLogger(__name__)

TOKEN_TTL_SECONDS = 23 * 60 * 60  # 23h buffer on the 24h token lifetime

# Fyers history API is aggressively rate-limited. Keep a minimum spacing
# between calls and back off on 429 responses.
MIN_CALL_INTERVAL_SECONDS = 1.5
RATE_LIMIT_MAX_RETRIES = 5
RATE_LIMIT_BACKOFF = 5.0

# Fyers history `resolution` values per DB timeframe convention.
RESOLUTION_MAP = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "60m": "60",
    "1d": "D",
    "1M": "M",
}


class TokenExpired(RuntimeError):
    """Raised when the cached/configured access token is no longer accepted."""


class FyersClient:
    """Thin, typed wrapper around the fyers-apiv3 SDK."""

    def __init__(self) -> None:
        self._client_id = settings.fyers_client_id
        self._model: Optional[fyersModel.FyersModel] = None
        self._last_call_ts = 0.0

    def _throttle(self) -> None:
        """Enforce a minimum interval between API calls."""
        elapsed = time.time() - self._last_call_ts
        if elapsed < MIN_CALL_INTERVAL_SECONDS:
            time.sleep(MIN_CALL_INTERVAL_SECONDS - elapsed)
        self._last_call_ts = time.time()

    def _call(self, fn, **kwargs):
        """Run an SDK call with throttling, 429 backoff, and re-auth on token expiry.

        The SDK signals errors either by raising or by returning
        {'s': 'error', 'code': ...}; both paths are handled. On auth errors the
        token is cleared and the call is retried once with a fresh token.
        """
        for attempt in range(2):
            result = None
            reauth = False
            for retry in range(RATE_LIMIT_MAX_RETRIES):
                self._throttle()
                result = fn(**kwargs)
                if isinstance(result, dict) and result.get("s") == "error":
                    if str(result.get("code")) == "429":
                        wait = RATE_LIMIT_BACKOFF * (retry + 1)
                        logger.warning("rate limited (429); backing off %.1fs", wait)
                        time.sleep(wait)
                        continue
                    if self._is_auth_error_resp(result):
                        logger.warning("auth error in response (%s); refreshing token", result)
                        self._clear_token()
                        self.reset_model()
                        reauth = True
                        break
                return result
            if not reauth:
                return result
        return result

    # ---- token lifecycle -------------------------------------------------

    def _load_cached_token(self) -> Optional[dict]:
        """Read the cached token dict, or None if missing/expired."""
        path = settings.token_cache_path
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if time.time() >= data.get("expires_at", 0):
            return None
        return data

    def _save_token(self, token: str) -> None:
        """Persist the access token with an expiry timestamp."""
        path = settings.token_cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "access_token": token,
            "obtained_at": time.time(),
            "expires_at": time.time() + TOKEN_TTL_SECONDS,
        }
        path.write_text(json.dumps(payload, indent=2))

    def _clear_token(self) -> None:
        path = settings.token_cache_path
        if path.exists():
            path.unlink()

    def _normalize_token(self, token: str) -> str:
        """Strip an optional '<appid>-<apptype>:' prefix to the raw JWT.

        The SDK builds the auth header itself as '<client_id>:<token>', so a
        pre-prefixed token (common when pasting from Fyers tools) must be cleaned.
        """
        if ":" in token:
            return token.split(":", 1)[1]
        return token

    def get_access_token(self, force: bool = False) -> str:
        """Return a valid access token: cached, configured, or via headless login."""
        if not force:
            cached = self._load_cached_token()
            if cached:
                return self._normalize_token(cached["access_token"])
            if settings.fyers_access_token:
                logger.info("using FYERS_ACCESS_TOKEN from env")
                return self._normalize_token(settings.fyers_access_token)
        from collector.auth import headless_login

        token = headless_login()
        if not token:
            raise RuntimeError("headless login produced no access_token")
        self._save_token(token)
        logger.info("obtained fresh access token (cached to %s)", settings.token_cache_path)
        return token

    @property
    def model(self) -> fyersModel.FyersModel:
        """Return a FyersModel bound to the current token, caching it."""
        if self._model is None:
            token = self.get_access_token()
            self._model = fyersModel.FyersModel(
                token=token,
                is_async=False,
                client_id=self._client_id,
                log_path=str(settings.token_cache_path.parent),
            )
        return self._model

    def reset_model(self) -> None:
        """Drop the cached model so the next call re-auths."""
        self._model = None

    @staticmethod
    def _is_auth_error_resp(resp: dict) -> bool:
        """Detect auth-related error responses from the Fyers API."""
        code = str(resp.get("code", ""))
        message = str(resp.get("message", "")).lower()
        if code in {"401", "-5", "-16", "1100", "1101"}:
            return True
        return any(m in message for m in ("unauthorised", "unauthorized", "invalid token", "token expired", "not a valid access token"))

    # ---- data APIs -------------------------------------------------------

    def history(
        self,
        symbol: str,
        timeframe: str,
        from_dt: datetime,
        to_dt: datetime,
    ) -> list[dict]:
        """Fetch OHLCV candles for a Fyers symbol over a datetime range.

        Args:
            symbol: Fyers symbol, e.g. 'NSE:TCS-EQ' or 'NSE:NIFTY50-INDEX'.
            timeframe: one of '1m','5m','15m','30m','60m','1d','1M'.
            from_dt/to_dt: inclusive window (naive datetimes assumed IST/local).

        Returns:
            List of dicts: {ts: datetime, open, high, low, close, volume}.
        """
        resolution = RESOLUTION_MAP.get(timeframe)
        if resolution is None:
            raise ValueError(f"unsupported timeframe {timeframe!r}")

        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "0",  # epoch seconds
            "range_from": int(from_dt.timestamp()),
            "range_to": int(to_dt.timestamp()),
            "cont_flag": "0",
        }

        def fetch():
            return self.model.history(data)

        resp = self._call(fetch)
        # Fyers returns 'no_data' for weekends/holidays — treat as empty, not error.
        if resp.get("s") not in ("ok", "no_data"):
            raise RuntimeError(f"fyers history failed for {symbol}: {resp}")
        candles = {}
        for row in resp.get("candles", []):
            # Fyers returns [epoch, open, high, low, close, volume]
            candles[row[0]] = {
                "ts": datetime.fromtimestamp(row[0], tz=timezone.utc).astimezone(
                    datetime.now().astimezone().tzinfo
                ),
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5],
            }
        return list(candles.values())

    def quotes(self, symbols: Sequence[str]) -> dict:
        """Fetch live quotes for Fyers symbols (comma-joined into one call)."""
        data = {"symbols": ",".join(symbols)}

        def fetch():
            return self.model.quotes(data)

        resp = self._call(fetch)
        if resp.get("s") != "ok":
            raise RuntimeError(f"fyers quotes failed: {resp}")
        return {d.get("n", d.get("symbol")): d.get("v", d) for d in resp.get("d", [])}

    def is_market_open(self) -> bool:
        """Rough India market-hours check (NSE: Mon–Fri 09:15–15:30 IST)."""
        now = datetime.now().astimezone()  # VPS is Asia/Kolkata
        if now.weekday() >= 5:
            return False
        t = now.time()
        return t >= datetime.strptime("09:15", "%H:%M").time() and t <= datetime.strptime(
            "15:30", "%H:%M"
        ).time()

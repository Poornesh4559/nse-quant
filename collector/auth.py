"""Headless Fyers login to obtain a fresh 1-day access_token.

Fyers access tokens expire every 24h with no refresh token. Instead of driving
a browser, this module replicates Fyers' own vagator v2 web-login API:

    send_login_otp -> verify_totp -> verify_pin -> token(auth_code)
    -> validate_authcode (appIdHash) -> access_token

All credentials come from `.env`:
    FYERS_ID          Fyers account id (alphanumeric, from your account)
    FYERS_TOTP_KEY    base32 TOTP secret from the Fyers app
    FYERS_PIN         account login PIN
    FYERS_APP_ID      app id part (from myapi.fyers.in dashboard)
    FYERS_APP_TYPE    app type suffix, e.g. 100/200 (default 100)
    FYERS_APP_SECRET  app secret from myapi.fyers.in
    FYERS_REDIRECT_URI redirect uri registered for the app
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any
from urllib import parse

import pyotp
import requests

from collector.config import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api-t2.fyers.in/vagator/v2"
BASE_URL_2 = "https://api-t1.fyers.in/api/v3"
URL_SEND_LOGIN_OTP = BASE_URL + "/send_login_otp"
URL_VERIFY_TOTP = BASE_URL + "/verify_otp"
URL_VERIFY_PIN = BASE_URL + "/verify_pin"
URL_TOKEN = BASE_URL_2 + "/token"
URL_VALIDATE_AUTH_CODE = BASE_URL_2 + "/validate-authcode"

SUCCESS = 1
ERROR = -1


def _post(session: requests.Session, url: str, payload: dict, headers: dict | None = None) -> tuple[int, Any]:
    """POST json over the shared session, returning (SUCCESS, parsed) or (ERROR, err).

    The session is essential: Fyers sets Cloudflare cookies on send_login_otp
    that must be echoed back or later steps reject the request_key.
    """
    try:
        resp = session.post(url=url, json=payload, headers=headers, timeout=30)
        if resp.status_code != 200:
            return ERROR, f"HTTP {resp.status_code}: {resp.text[:300]}"
        return SUCCESS, resp.json()
    except Exception as exc:  # network / json errors
        return ERROR, exc


def send_login_otp(session: requests.Session, fy_id: str, app_id: str) -> tuple[int, Any]:
    return _post(session, URL_SEND_LOGIN_OTP, {"fy_id": fy_id, "app_id": app_id})


def generate_totp(secret: str) -> tuple[int, Any]:
    try:
        return SUCCESS, pyotp.TOTP(secret).now()
    except Exception as exc:
        return ERROR, exc


def verify_totp(session: requests.Session, request_key: str, totp: str) -> tuple[int, Any]:
    return _post(session, URL_VERIFY_TOTP, {"request_key": request_key, "otp": totp})


def verify_pin(session: requests.Session, request_key: str, pin: str) -> tuple[int, Any]:
    return _post(
        session,
        URL_VERIFY_PIN,
        {"request_key": request_key, "identity_type": "pin", "identifier": pin},
    )


def get_auth_code(session, fy_id, app_id, redirect_uri, app_type, access_token) -> tuple[int, Any]:
    """Exchange the trade access_token for an auth_code (308 redirect carries it)."""
    payload = {
        "fyers_id": fy_id,
        "app_id": app_id,
        "redirect_uri": redirect_uri,
        "appType": app_type,
        "code_challenge": "",
        "state": "sample_state",
        "scope": "",
        "nonce": "",
        "response_type": "code",
        "create_cookie": True,
    }
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        resp = session.post(url=URL_TOKEN, json=payload, headers=headers, timeout=30)
        if resp.status_code != 308:
            return ERROR, f"expected 308, got HTTP {resp.status_code}: {resp.text[:300]}"
        result = resp.json()
        url = result["Url"]
        auth_code = parse.parse_qs(parse.urlparse(url).query)["auth_code"][0]
        return SUCCESS, auth_code
    except Exception as exc:
        return ERROR, exc


def generate_app_id_hash(app_id: str, app_type: str, app_secret: str) -> str:
    """Fyers v3: appIdHash = sha256('<app_id>-<app_type>:<app_secret>')."""
    return hashlib.sha256(f"{app_id}-{app_type}:{app_secret}".encode()).hexdigest()


def validate_authcode(session: requests.Session, app_id_hash: str, auth_code: str) -> tuple[int, Any]:
    return _post(
        session,
        URL_VALIDATE_AUTH_CODE,
        {"grant_type": "authorization_code", "appIdHash": app_id_hash, "code": auth_code},
    )


def _login_once() -> str:
    """Run the login flow once; returns the raw access_token."""
    fy_id = settings.fyers_id
    app_id = settings.fyers_app_id
    app_type = settings.fyers_app_type
    app_secret = settings.fyers_app_secret

    if not all([fy_id, settings.fyers_totp_key, settings.fyers_pin, app_id, app_secret]):
        missing = [
            name
            for name, val in [
                ("FYERS_ID", fy_id),
                ("FYERS_TOTP_KEY", settings.fyers_totp_key),
                ("FYERS_PIN", settings.fyers_pin),
                ("FYERS_APP_ID", app_id),
                ("FYERS_APP_SECRET", app_secret),
            ]
            if not val
        ]
        raise RuntimeError(f"missing Fyers creds in .env: {', '.join(missing)}")

    # Step 1-2: send OTP, generate TOTP (shared session keeps Cloudflare cookies)
    session = requests.Session()
    ok, resp = send_login_otp(session, fy_id=fy_id, app_id="2")  # app_id 2 = web login
    if ok != SUCCESS:
        raise RuntimeError(f"send_login_otp failed: {resp}")
    req_key = resp["request_key"]
    ok, totp = generate_totp(settings.fyers_totp_key)
    if ok != SUCCESS:
        raise RuntimeError(f"generate_totp failed: {totp}")

    # Step 3: verify TOTP -> new request_key
    ok, resp = verify_totp(session, request_key=req_key, totp=totp)
    if ok != SUCCESS:
        raise RuntimeError(f"verify_totp failed: {resp}")
    req_key = resp["request_key"]

    # Step 4: verify PIN -> trade access_token
    ok, trade_token = verify_pin(session, request_key=req_key, pin=settings.fyers_pin)
    if ok != SUCCESS:
        raise RuntimeError(f"verify_pin failed: {trade_token}")
    trade_token = trade_token["data"]["access_token"]

    # Step 5-7: exchange trade token for auth_code, then API token.
    # Use a FRESH session here — the vagator Cloudflare cookies from steps 1-4
    # cause /validate-authcode to reject the auth_code (observed 400 -1).
    api_session = requests.Session()
    ok, auth_code = get_auth_code(
        api_session,
        fy_id=fy_id,
        app_id=app_id,
        redirect_uri=settings.fyers_redirect_uri,
        app_type=app_type,
        access_token=trade_token,
    )
    if ok != SUCCESS:
        raise RuntimeError(f"get_auth_code failed: {auth_code}")

    # Step 6-7: appIdHash -> validate authcode -> API access_token
    app_id_hash = generate_app_id_hash(app_id, app_type, app_secret)
    ok, token = validate_authcode(api_session, app_id_hash=app_id_hash, auth_code=auth_code)
    if ok != SUCCESS:
        raise RuntimeError(f"validate_authcode failed: {token}")
    return token["access_token"]


def headless_login(retries: int = 6, pause: float = 2.5) -> str:
    """Run the full login flow with retries for transient failures.

    Fyers' /validate-authcode intermittently rejects the first 1-2 attempts
    (HTTP 400 \"invalid auth code\") even when everything is correct — observed
    at least once a day. Retry with a short pause; 6 attempts covers it.
    """
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            token = _login_once()
            logger.info("headless login succeeded for %s (attempt %d)", settings.fyers_id, attempt + 1)
            return token
        except (KeyError, RuntimeError) as exc:
            last_err = exc
            logger.warning("login attempt %d failed: %s", attempt + 1, exc)
            if attempt < retries - 1:
                time.sleep(pause)
    raise last_err  # type: ignore[misc]

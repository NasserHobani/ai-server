"""HTTP client for Odoo JSON routes (session auth + JSON-RPC body)."""

from __future__ import annotations

import os
from typing import Any

import httpx

from . import config


def load_odoo_credentials() -> dict[str, str] | None:
    email = os.getenv("CS_AI_BRIDGE_ODOO_EMAIL", "").strip()
    db_name = os.getenv("CS_AI_BRIDGE_ODOO_DB_NAME", "").strip()
    password = os.getenv("CS_AI_BRIDGE_ODOO_PASSWORD", "").strip()
    if not email or not db_name or not password:
        return None
    return {"email": email, "db_name": db_name, "password": password}


def authenticate_odoo(client: httpx.Client, base_url: str) -> None:
    creds = load_odoo_credentials()
    if creds is None:
        return

    response = client.post(
        f"{base_url}/web/session/authenticate",
        json={
            "jsonrpc": "2.0",
            "method": "call",
            "params": {
                "db": creds["db_name"],
                "login": creds["email"],
                "password": creds["password"],
            },
        },
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("Odoo authentication response must be a JSON object.")
    result = body.get("result", {})
    if not isinstance(result, dict) or not result.get("uid"):
        raise ValueError("Odoo authentication failed. Check email/db/password.")


def post_json(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    base_url = config.get_base_url()
    timeout_seconds = config.get_timeout_seconds()

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    headers.update(config.load_cookie_header())
    headers.update(config.load_extra_headers())

    with httpx.Client(timeout=timeout_seconds) as client:
        if "Cookie" not in headers:
            authenticate_odoo(client, base_url)
        response = client.post(
            f"{base_url}{endpoint}", json={"params": payload}, headers=headers
        )
        response.raise_for_status()

    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("API response must be a JSON object.")
    return data

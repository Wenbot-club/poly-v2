"""
Credential loading for live Polymarket trading.

Reads from environment variables.  No file I/O — secrets stay in env.

Required env vars:
  POLY_PRIVATE_KEY      — hex private key (with or without 0x prefix)
  POLY_API_KEY          — CLOB L2 API key UUID
  POLY_API_SECRET       — CLOB L2 API secret (base64)
  POLY_API_PASSPHRASE   — CLOB L2 passphrase
  POLY_FUNDER_ADDRESS   — EVM address that holds USDC / has CTF approvals

Optional:
  POLY_RPC_URL          — Polygon JSON-RPC URL (default: public Polygon RPC)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

_DEFAULT_RPC = "https://polygon-rpc.com"

_REQUIRED = [
    "POLY_PRIVATE_KEY",
    "POLY_API_KEY",
    "POLY_API_SECRET",
    "POLY_API_PASSPHRASE",
    "POLY_FUNDER_ADDRESS",
]


class CredentialError(ValueError):
    pass


@dataclass
class Credentials:
    private_key: str          # hex, without 0x
    api_key: str
    api_secret: str           # base64-encoded
    api_passphrase: str
    funder_address: str       # checksummed or lowercase EVM address
    rpc_url: str

    @property
    def private_key_hex(self) -> str:
        """Return private key without 0x prefix."""
        return self.private_key.removeprefix("0x")


def load_credentials(env: Optional[dict[str, str]] = None) -> Credentials:
    """
    Load credentials from environment (or from supplied dict for tests).

    Raises CredentialError listing all missing variables.
    """
    source = env if env is not None else os.environ

    missing = [k for k in _REQUIRED if not source.get(k, "").strip()]
    if missing:
        raise CredentialError(f"Missing required env vars: {', '.join(missing)}")

    return Credentials(
        private_key=source["POLY_PRIVATE_KEY"].strip(),
        api_key=source["POLY_API_KEY"].strip(),
        api_secret=source["POLY_API_SECRET"].strip(),
        api_passphrase=source["POLY_API_PASSPHRASE"].strip(),
        funder_address=source["POLY_FUNDER_ADDRESS"].strip().lower(),
        rpc_url=source.get("POLY_RPC_URL", _DEFAULT_RPC).strip() or _DEFAULT_RPC,
    )

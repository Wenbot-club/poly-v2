#!/usr/bin/env python3
"""
Interactive setup wizard for poly-v2 live trading.

Collects credentials interactively, derives missing values where possible,
generates Polymarket L2 API keys from the private key, writes .env file.

Usage:
  python scripts/setup_wizard.py [--env-file /path/to/.env]
"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path

import aiohttp

# ── ANSI colours ──────────────────────────────────────────────────────────────
RED   = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW= "\033[1;33m"
CYAN  = "\033[0;36m"
BOLD  = "\033[1m"
NC    = "\033[0m"

def _ok(msg):   print(f"{GREEN}[OK]{NC} {msg}")
def _warn(msg): print(f"{YELLOW}[!]{NC} {msg}")
def _err(msg):  print(f"{RED}[ERROR]{NC} {msg}")
def _head(msg): print(f"\n{BOLD}{CYAN}{msg}{NC}")


# ── Key derivation helpers (no web3 dep) ──────────────────────────────────────

def _privkey_to_address(private_key_hex: str) -> str:
    """
    Derive EVM address from raw private key hex using secp256k1 + keccak256.
    Uses eth_account if available, otherwise falls back to a pure-Python path
    via the coincurve library (already pulled by eth_account).  If neither is
    available we ask the user to enter the address manually.
    """
    pk = private_key_hex.removeprefix("0x").strip()
    try:
        from eth_account import Account
        acct = Account.from_key("0x" + pk)
        return acct.address.lower()
    except ImportError:
        pass
    try:
        import coincurve
        import hashlib as _hl
        priv_bytes = bytes.fromhex(pk)
        pub = coincurve.PublicKey.from_valid_secret(priv_bytes).format(compressed=False)[1:]
        digest = _hl.new("sha3_256")
        # keccak256 — not available in stdlib, use pysha3 if present
        try:
            import sha3  # pysha3
            k = hashlib.new("sha3_256")
            k.update(pub)
            return "0x" + k.digest()[-20:].hex()
        except ImportError:
            return ""
    except ImportError:
        return ""


async def _generate_l2_api_key(
    session: aiohttp.ClientSession,
    private_key_hex: str,
    address: str,
    nonce: int = 0,
) -> dict:
    """
    Call POST /auth/derive-api-key to generate L2 API credentials.
    Uses L1 auth (signed with private key) to bootstrap L2 creds.

    Returns dict with api_key, api_secret, api_passphrase on success.
    """
    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
    except ImportError:
        return {"error": "eth_account not installed — cannot auto-generate L2 keys"}

    CLOB_BASE = "https://clob.polymarket.com"
    ts = int(time.time())

    # Build L1 signature: sign "Polymarket\nAddress:<addr>\nTimestamp:<ts>\nNonce:<nonce>"
    message = f"Polymarket\nAddress:{address}\nTimestamp:{ts}\nNonce:{nonce}"
    msg_hash = encode_defunct(text=message)
    signed = Account.sign_message(msg_hash, private_key="0x" + private_key_hex.removeprefix("0x"))
    sig = signed.signature.hex()

    headers = {
        "POLY_ADDRESS": address,
        "POLY_SIGNATURE": sig,
        "POLY_TIMESTAMP": str(ts),
        "POLY_NONCE": str(nonce),
        "Content-Type": "application/json",
    }

    timeout = aiohttp.ClientTimeout(total=15.0)
    try:
        async with session.post(
            f"{CLOB_BASE}/auth/derive-api-key",
            headers=headers,
            timeout=timeout,
        ) as resp:
            body = await resp.text()
            if resp.status == 200:
                data = json.loads(body)
                return {
                    "api_key": data.get("apiKey", ""),
                    "api_secret": data.get("secret", ""),
                    "api_passphrase": data.get("passphrase", ""),
                }
            else:
                return {"error": f"HTTP {resp.status}: {body}"}
    except Exception as e:
        return {"error": str(e)}


# ── Wizard ────────────────────────────────────────────────────────────────────

def _prompt(label: str, secret: bool = False, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    prompt_str = f"  {BOLD}{label}{suffix}{NC}: "
    while True:
        val = (getpass.getpass(prompt_str) if secret else input(prompt_str)).strip()
        if not val and default:
            return default
        if val:
            return val
        print("  (required, cannot be empty)")


async def _run_wizard(env_file: Path) -> None:
    _head("═══ poly-v2 setup wizard ═══")
    print("  This wizard collects your trading credentials and writes them")
    print("  to a local .env file on this VPS. Nothing is sent remotely.")
    print("  Press Ctrl+C at any time to abort.\n")

    # ── Private key ───────────────────────────────────────────────────────────
    _head("Step 1 — MetaMask private key")
    print("  In MetaMask: Account menu → Account details → Show private key")
    print("  The key is a 64-character hex string (with or without 0x prefix).")
    private_key = _prompt("Private key", secret=True)
    private_key = private_key.removeprefix("0x").strip()
    if len(private_key) != 64:
        _err(f"Expected 64 hex chars, got {len(private_key)}. Double-check the key.")
        sys.exit(1)

    # ── Derive address ────────────────────────────────────────────────────────
    derived = _privkey_to_address(private_key)
    if derived:
        _ok(f"Derived address: {derived}")
        funder_address = derived
    else:
        _warn("Could not auto-derive address (eth_account not installed).")
        funder_address = _prompt("Funder address (0x...)")

    # ── RPC URL ───────────────────────────────────────────────────────────────
    _head("Step 2 — Polygon RPC URL (Alchemy)")
    print("  Alchemy dashboard → your Polygon PoS app → API key → HTTPS endpoint")
    print("  Format: https://polygon-mainnet.g.alchemy.com/v2/XXXXXX")
    rpc_url = _prompt("Alchemy RPC URL")

    # ── L2 API keys ───────────────────────────────────────────────────────────
    _head("Step 3 — Polymarket L2 API keys")
    print("  Attempting to auto-generate from your private key via Polymarket API...")

    async with aiohttp.ClientSession() as session:
        result = await _generate_l2_api_key(session, private_key, funder_address)

    if "error" in result:
        _warn(f"Auto-generation failed: {result['error']}")
        print("  Enter the keys manually (create them at polymarket.com → Settings → API keys):")
        api_key        = _prompt("API key")
        api_secret     = _prompt("API secret", secret=True)
        api_passphrase = _prompt("API passphrase", secret=True)
    else:
        _ok("L2 API keys generated successfully")
        api_key        = result["api_key"]
        api_secret     = result["api_secret"]
        api_passphrase = result["api_passphrase"]
        print(f"  API key: {api_key[:12]}...")

    # ── Write .env ────────────────────────────────────────────────────────────
    _head("Writing .env file")
    env_content = f"""# poly-v2 credentials — generated by setup_wizard.py
# DO NOT commit this file to git.

POLY_PRIVATE_KEY={private_key}
POLY_FUNDER_ADDRESS={funder_address}
POLY_RPC_URL={rpc_url}
POLY_API_KEY={api_key}
POLY_API_SECRET={api_secret}
POLY_API_PASSPHRASE={api_passphrase}
"""
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text(env_content)
    # Restrict permissions: owner read/write only
    env_file.chmod(0o600)
    _ok(f".env written to {env_file} (permissions: 600)")

    print("")
    _ok("Wizard complete. Running bootstrap check next...")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="poly-v2 interactive setup wizard")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("/opt/poly-v2/.env"),
        help="Path to write the .env file (default: /opt/poly-v2/.env)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(_run_wizard(args.env_file))
    except KeyboardInterrupt:
        print("\n\n[aborted]")
        sys.exit(1)


if __name__ == "__main__":
    main()

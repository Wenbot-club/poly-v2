"""
On-chain approval checks for Polymarket trading — no web3 dependency.

Uses raw JSON-RPC eth_call via aiohttp to check:
  1. USDC allowance(funder, ctf_exchange)  — must be > 0 (or max) to trade
  2. CTF Exchange isApprovedForAll(funder, neg_risk_exchange) — for neg-risk markets

Polygon contract addresses (mainnet):
  USDC (PoS):            0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174
  CTF Exchange:          0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E
  Neg Risk Exchange:     0xC5d563A36AE78145C45a50134d48A1215220f80a
  Neg Risk Adapter:      0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import aiohttp

# --- Contract addresses (Polygon mainnet) ------------------------------------

USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEG_RISK_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"

# --- ABI selectors (keccak256 first 4 bytes, pre-computed) -------------------
# allowance(address,address) = 0xdd62ed3e
# isApprovedForAll(address,address) = 0xe985e9c5

_SEL_ALLOWANCE = "dd62ed3e"
_SEL_IS_APPROVED_FOR_ALL = "e985e9c5"

_TIMEOUT = aiohttp.ClientTimeout(total=10.0)
_MAX_UINT256 = 2**256 - 1
_USDC_MIN_ALLOWANCE = 10_000_000  # 10 USDC (6 decimals) — sanity threshold


@dataclass
class ApprovalStatus:
    usdc_allowance_ctf: int           # raw uint256
    usdc_approved_ctf: bool           # True if allowance > threshold
    ctf_approved_neg_risk: bool       # isApprovedForAll result
    neg_risk_approved_adapter: bool   # funder → adapter approved on neg_risk_exchange
    error: Optional[str] = None

    @property
    def ready_to_trade(self) -> bool:
        return (
            self.error is None
            and self.usdc_approved_ctf
            and self.ctf_approved_neg_risk
            and self.neg_risk_approved_adapter
        )


def _pad_address(addr: str) -> str:
    """ABI-encode an address as a 32-byte hex string (no 0x prefix)."""
    return addr.removeprefix("0x").lower().zfill(64)


def _encode_call(selector: str, *addresses: str) -> str:
    """Build eth_call data: 0x + selector + ABI-encoded address args."""
    return "0x" + selector + "".join(_pad_address(a) for a in addresses)


async def _eth_call(
    session: aiohttp.ClientSession,
    rpc_url: str,
    to: str,
    data: str,
) -> str:
    """Make a single eth_call, return the hex result string."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_call",
        "params": [{"to": to, "data": data}, "latest"],
    }
    async with session.post(
        rpc_url,
        json=payload,
        timeout=_TIMEOUT,
        headers={"Content-Type": "application/json"},
    ) as resp:
        body = await resp.json(content_type=None)
        if "error" in body:
            raise RuntimeError(f"eth_call error: {body['error']}")
        return body["result"]


def _decode_uint256(hex_result: str) -> int:
    stripped = hex_result.removeprefix("0x").strip()
    if not stripped:
        return 0
    return int(stripped, 16)


def _decode_bool(hex_result: str) -> bool:
    return _decode_uint256(hex_result) != 0


async def check_approvals(
    session: aiohttp.ClientSession,
    funder_address: str,
    rpc_url: str,
) -> ApprovalStatus:
    """
    Check on-chain approvals needed for Polymarket trading.

    All calls are read-only (eth_call).  No transaction is sent.
    """
    funder = funder_address.lower()
    try:
        # 1. USDC allowance: funder → CTF Exchange
        allowance_data = _encode_call(_SEL_ALLOWANCE, funder, CTF_EXCHANGE)
        raw_allowance = await _eth_call(session, rpc_url, USDC_ADDRESS, allowance_data)
        usdc_allowance = _decode_uint256(raw_allowance)

        # 2. CTF isApprovedForAll: funder → Neg Risk Exchange
        ctf_approval_data = _encode_call(_SEL_IS_APPROVED_FOR_ALL, funder, NEG_RISK_EXCHANGE)
        raw_ctf = await _eth_call(session, rpc_url, CTF_EXCHANGE, ctf_approval_data)
        ctf_approved = _decode_bool(raw_ctf)

        # 3. Neg Risk isApprovedForAll: funder → Neg Risk Adapter
        neg_risk_data = _encode_call(_SEL_IS_APPROVED_FOR_ALL, funder, NEG_RISK_ADAPTER)
        raw_neg = await _eth_call(session, rpc_url, NEG_RISK_EXCHANGE, neg_risk_data)
        neg_approved = _decode_bool(raw_neg)

        return ApprovalStatus(
            usdc_allowance_ctf=usdc_allowance,
            usdc_approved_ctf=usdc_allowance >= _USDC_MIN_ALLOWANCE,
            ctf_approved_neg_risk=ctf_approved,
            neg_risk_approved_adapter=neg_approved,
        )

    except Exception as e:
        return ApprovalStatus(
            usdc_allowance_ctf=0,
            usdc_approved_ctf=False,
            ctf_approved_neg_risk=False,
            neg_risk_approved_adapter=False,
            error=str(e),
        )

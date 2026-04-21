#!/usr/bin/env python3
"""
Bootstrap check CLI — validates this machine can trade on Polymarket.

Checks (in order):
  1. Geoblock: Polymarket endpoints reachable from this IP
  2. Credentials: required env vars present
  3. CLOB dry-run: GET /profile succeeds with given API key
  4. On-chain approvals: USDC allowance + CTF/Neg Risk isApprovedForAll

Exit codes:
  0 — all checks passed (GO)
  1 — one or more checks failed (NO-GO)

Usage:
  export POLY_PRIVATE_KEY=0x...
  export POLY_API_KEY=...
  export POLY_API_SECRET=...
  export POLY_API_PASSPHRASE=...
  export POLY_FUNDER_ADDRESS=0x...
  export POLY_RPC_URL=https://polygon-rpc.com  # optional

  python demos/bootstrap_check.py
"""
from __future__ import annotations

import asyncio
import sys

from bot.trading.bootstrap import print_bootstrap_report, run_bootstrap


def main() -> int:
    report = asyncio.run(run_bootstrap())
    print_bootstrap_report(report)
    return 0 if report.ready else 1


if __name__ == "__main__":
    sys.exit(main())

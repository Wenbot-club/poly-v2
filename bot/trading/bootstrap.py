"""
Bootstrap orchestrator — full go/no-go check before live trading.

Runs in order:
  1. Geoblock check (Polymarket endpoints reachable from this machine/VPS)
  2. Credential loading (env vars present and non-empty)
  3. CLOB dry-run (GET /profile — verifies API key works)
  4. On-chain approvals (USDC allowance + CTF/Neg Risk isApprovedForAll)

Returns a BootstrapReport with a top-level `ready` flag and per-step details.
No orders are placed; all network calls are read-only.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

import aiohttp

from bot.trading.approvals import ApprovalStatus, check_approvals
from bot.trading.clob_client import dry_run_clob
from bot.trading.credentials import CredentialError, Credentials, load_credentials
from bot.trading.geoblock import GeoBlockResult, check_geoblock


@dataclass
class BootstrapReport:
    # Step results
    geoblock: Optional[GeoBlockResult] = None
    credentials: Optional[Credentials] = None
    credentials_error: Optional[str] = None
    clob_result: Optional[dict[str, Any]] = None
    approvals: Optional[ApprovalStatus] = None

    # Collected issues
    issues: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """True only if all checks passed with no issues."""
        return len(self.issues) == 0


async def run_bootstrap(
    session: Optional[aiohttp.ClientSession] = None,
    env: Optional[dict[str, str]] = None,
) -> BootstrapReport:
    """
    Execute all bootstrap checks and return a BootstrapReport.

    Pass env dict to override os.environ (useful in tests).
    Pass an existing aiohttp.ClientSession to reuse connections.
    """
    report = BootstrapReport()

    async def _run(http: aiohttp.ClientSession) -> None:
        # --- Step 1: Geoblock ---
        report.geoblock = await check_geoblock(session=http)
        if not report.geoblock.all_reachable:
            for name in report.geoblock.blocked_endpoints:
                s = report.geoblock.endpoints[name]
                report.issues.append(
                    f"geoblock: {name} ({s.url}) unreachable"
                    + (f" — {s.error}" if s.error else f" — HTTP {s.status_code}")
                )

        # --- Step 2: Credentials ---
        try:
            report.credentials = load_credentials(env)
        except CredentialError as e:
            report.credentials_error = str(e)
            report.issues.append(f"credentials: {e}")
            return  # cannot proceed without creds

        # --- Step 3: CLOB dry-run ---
        report.clob_result = await dry_run_clob(http, report.credentials)
        if not report.clob_result.get("ok"):
            report.issues.append(f"clob: {report.clob_result.get('error', 'unknown error')}")

        # --- Step 4: On-chain approvals ---
        report.approvals = await check_approvals(
            http,
            report.credentials.funder_address,
            report.credentials.rpc_url,
        )
        if report.approvals.error:
            report.issues.append(f"approvals: rpc error — {report.approvals.error}")
        else:
            if not report.approvals.usdc_approved_ctf:
                usdc = report.approvals.usdc_allowance_ctf
                report.issues.append(
                    f"approvals: USDC allowance to CTF Exchange is {usdc} "
                    f"(need ≥ 10 USDC = 10_000_000 units)"
                )
            if not report.approvals.ctf_approved_neg_risk:
                report.issues.append(
                    "approvals: CTF Exchange not approved for Neg Risk Exchange "
                    "(isApprovedForAll = false)"
                )
            if not report.approvals.neg_risk_approved_adapter:
                report.issues.append(
                    "approvals: Neg Risk Exchange not approved for Neg Risk Adapter "
                    "(isApprovedForAll = false)"
                )

    if session is not None:
        await _run(session)
    else:
        async with aiohttp.ClientSession() as http:
            await _run(http)

    return report


def print_bootstrap_report(report: BootstrapReport) -> None:
    """Print a human-readable go/no-go summary to stdout."""
    line = "=" * 60
    print(f"\n{line}")
    print("  BOOTSTRAP REPORT")
    print(line)

    # Geoblock
    if report.geoblock:
        print("\n  [geoblock]")
        for name, status in report.geoblock.endpoints.items():
            icon = "OK" if status.reachable else "BLOCKED"
            detail = f"HTTP {status.status_code}" if status.status_code else status.error or ""
            print(f"    {icon:8s}  {name}  ({detail})")

    # Credentials
    print("\n  [credentials]")
    if report.credentials_error:
        print(f"    FAIL  {report.credentials_error}")
    elif report.credentials:
        print(f"    OK    funder={report.credentials.funder_address}")
        print(f"          api_key={report.credentials.api_key[:8]}...")

    # CLOB
    print("\n  [clob dry-run]")
    if report.clob_result:
        if report.clob_result.get("ok"):
            print(f"    OK    address={report.clob_result.get('address')}")
        else:
            print(f"    FAIL  {report.clob_result.get('error')}")
    else:
        print("    SKIP  (credentials not loaded)")

    # Approvals
    print("\n  [on-chain approvals]")
    if report.approvals:
        if report.approvals.error:
            print(f"    ERROR {report.approvals.error}")
        else:
            usdc = report.approvals.usdc_allowance_ctf
            print(f"    {'OK' if report.approvals.usdc_approved_ctf else 'FAIL':6s}  "
                  f"USDC allowance to CTF Exchange: {usdc}")
            print(f"    {'OK' if report.approvals.ctf_approved_neg_risk else 'FAIL':6s}  "
                  f"CTF → Neg Risk Exchange approved")
            print(f"    {'OK' if report.approvals.neg_risk_approved_adapter else 'FAIL':6s}  "
                  f"Neg Risk → Adapter approved")
    else:
        print("    SKIP  (credentials not loaded)")

    # Final verdict
    print(f"\n{line}")
    if report.ready:
        print("  RESULT: GO — all checks passed")
    else:
        print(f"  RESULT: NO-GO — {len(report.issues)} issue(s):")
        for issue in report.issues:
            print(f"    - {issue}")
    print(f"{line}\n")

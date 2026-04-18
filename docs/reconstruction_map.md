# Reconstruction Map

The original upload to this repo had all file contents present but all filenames
scrambled. This document is the ground-truth mapping used to reconstruct the
correct package layout.

## File mapping: root filename → actual content → correct destination

| Root file (uploaded) | Actual content found | Correct destination |
|---|---|---|
| `demo_async_runner_local.py` | domain types (LocalState, TokenBook, …) | `bot/domain.py` |
| `settings.py` | settings (ConfigValue, RuntimeConfig) ✓ | `bot/settings.py` |
| `ws_user.py` | StateFactory, register_chainlink_tick | `bot/state.py` |
| `risk.py` | PTBLocker | `bot/ptb.py` |
| `ws_market.py` | RiskManager, RiskDecision | `bot/risk.py` |
| `replay.py` | HeartbeatMonitor, HeartbeatStatus | `bot/heartbeat.py` |
| `recorder.py` | FairValueEngine | `bot/fair_value.py` |
| `types.py` | replay_jsonl, ReplaySummary | `bot/replay.py` |
| `conftest.py` | UserMessageRouter, MockUserStream | `bot/routers/ws_user.py` |
| `demo_paper_local.py` | MarketMessageRouter, MockMarketStream | `bot/routers/ws_market.py` |
| `pytest.ini` | RTDSMessageRouter, MockRTDSStream | `bot/routers/ws_rtds.py` |
| `ptb.py` | MockGammaClient, DiscoveryService | `bot/providers/discovery.py` |
| `quote_policy.py` | MockExecutionEngine | `bot/execution/paper.py` |
| `state.py` | QuotePolicy | `bot/strategy/baseline.py` |
| `heartbeat.py` | AsyncLocalRunner, QueueingUserRouter | `bot/async_runner.py` |
| `discovery.py` | `.github/workflows/local_stack.yml` (YAML) | `.github/workflows/local_stack.yml` |
| `execution.py` | README.md content | `README.md` |
| `test_local_stack.py` | demo_paper_local.py demo script | `demos/demo_paper_local.py` |
| `ws_rtds.py` | settings.py duplicate | deleted (duplicate) |

## Content missing from the upload (reconstructed)

| Module | Status |
|---|---|
| `bot/recorder.py` (JSONLRecorder) | **Reconstructed** from usage in `async_runner.py` and `replay.py` |
| `demos/demo_async_runner_local.py` | **Reconstituted** — calls same demo pipeline as `demo_paper_local.py` |
| `tests/test_local_stack.py` | **Reconstructed smoke suite** — original not present in upload |
| `tests/conftest.py` | **Minimal clean conftest** — original not present in upload |
| `pytest.ini` (proper) | **Recreated** — original file contained RTDSMessageRouter code |

## What was NOT changed

- Zero logic changes in any reconstructed module
- All signatures preserved as-found
- All thresholds, constants, and parameters unchanged
- PTBLocker kept separate from QuotePolicy (not merged into strategy)

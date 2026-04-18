# poly

Runnable **local/mock core stack** for the Polymarket BTC 15m bot project.

What is present on `nucleus-local-stack` now:
- `.github/workflows/local_stack.yml`
- `pytest.ini`
- `tests/conftest.py`
- `tests/test_local_stack.py`
- `bot/settings.py`
- `bot/discovery.py`
- `bot/state.py`
- `bot/ptb.py`
- `bot/ws_market.py`
- `bot/ws_rtds.py`
- `bot/ws_user.py`
- `bot/execution.py`
- `bot/fair_value.py`
- `bot/quote_policy.py`
- `bot/risk.py`
- `bot/heartbeat.py`
- `bot/recorder.py`
- `bot/replay.py`
- `bot/async_runner.py`
- `demo_paper_local.py`
- `demo_async_runner_local.py`

What this branch is:
- a local paper/mock stack
- runnable without external network integrations
- suitable for local CI verification and deterministic replay

What this branch is not:
- **not** live Polymarket trading
- **not** real posting
- **not** boundary / external integration work

## Run locally

```bash
python -m pip install -U pytest
pytest tests
python demo_paper_local.py
python demo_async_runner_local.py
```

The demo scripts show:
- initial quote
- partial fill
- reprice/cancel
- final state
- JSONL replay verification

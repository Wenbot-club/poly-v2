# poly

Runnable **local/mock core stack** for the Polymarket BTC 15m bot project.

## What this is

- A local paper/mock stack
- Runnable without external network integrations
- Suitable for local CI verification and deterministic replay

## What this is not

- **Not** live Polymarket trading
- **Not** real order posting
- **Not** boundary / external integration work

## Package layout

```
bot/
├── domain.py          # core domain types
├── settings.py        # runtime config
├── state.py           # state factory and tick registration
├── ptb.py             # price-time-priority locker
├── risk.py            # risk manager
├── heartbeat.py       # heartbeat monitor
├── fair_value.py      # fair value computation
├── replay.py          # JSONL replay verification
├── recorder.py        # JSONL session recorder
├── async_runner.py    # async orchestrator (paper run)
├── providers/
│   ├── base.py        # DiscoveryProvider, MarketDataProvider, SignalProvider (Protocols)
│   └── discovery.py   # MockGammaClient, DiscoveryService
├── routers/
│   ├── ws_market.py   # market book message router
│   ├── ws_rtds.py     # RTDS price feed router
│   └── ws_user.py     # user order/trade router
├── strategy/
│   ├── base.py        # Strategy Protocol
│   └── baseline.py    # QuotePolicy (baseline strategy)
└── execution/
    ├── base.py        # ExecutionGateway Protocol
    └── paper.py       # MockExecutionEngine (paper-only, no real orders)
```

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

## Reconstruction note

This repo was originally uploaded with scrambled filenames (all content present,
all filenames wrong). This branch is the faithful reconstruction into the correct
`bot/` package layout. See `docs/reconstruction_map.md` for the full mapping.

The test suite (`tests/test_local_stack.py`) is a reconstructed smoke suite —
the original test file was not present in the upload. It covers the end-to-end
paper run and replay assertion.

`bot/recorder.py` (JSONLRecorder) was absent from the upload and has been
reconstructed from its usage in `async_runner.py` and `replay.py`.

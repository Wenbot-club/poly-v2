"""Tests for bot/strategy/directional_v2.py — deterministic, no network."""
from __future__ import annotations

from typing import Optional

import pytest

from bot.domain import (
    BestBidAsk,
    ClobMarketInfo,
    ClobToken,
    FairValueSnapshot,
    LocalState,
    MarketContext,
    PriceTick,
    TokenBook,
)
from bot.settings import DEFAULT_CONFIG, DirectionalThresholds, RuntimeConfig, Thresholds
from bot.strategy.directional_v2 import DirectionalPolicyV2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clob(min_order_size: float = 1.0, min_tick_size: float = 0.01) -> ClobMarketInfo:
    return ClobMarketInfo(
        tokens=[
            ClobToken(token_id="YES", outcome="Yes"),
            ClobToken(token_id="NO", outcome="No"),
        ],
        min_order_size=min_order_size,
        min_tick_size=min_tick_size,
        maker_base_fee_bps=0,
        taker_base_fee_bps=0,
        taker_delay_enabled=False,
        min_order_age_s=0.0,
        fee_rate=0.0,
        fee_exponent=1.0,
    )


def _make_market(end_ts_ms: int = 9_999_999_999_000) -> MarketContext:
    return MarketContext(
        market_id="test",
        condition_id="test",
        title="Test",
        slug="test",
        start_ts_ms=0,
        end_ts_ms=end_ts_ms,
        yes_token_id="YES",
        no_token_id="NO",
        clob=_make_clob(),
    )


def _make_state(
    yes_bid: float = 0.45,
    yes_ask: float = 0.55,
    yes_bid_size: float = 100.0,
    yes_ask_size: float = 100.0,
    pusd_free: float = 100.0,
    up_free: float = 0.0,
    position_qty: float = 0.0,
    position_cost: float = 0.0,
    binance_recv_ts: Optional[int] = None,
    chainlink_recv_ts: Optional[int] = None,
    end_ts_ms: int = 9_999_999_999_000,
) -> LocalState:
    market = _make_market(end_ts_ms=end_ts_ms)
    yes_book = TokenBook(asset_id="YES")
    yes_book.bids = {yes_bid: yes_bid_size}
    yes_book.asks = {yes_ask: yes_ask_size}
    yes_book.best = BestBidAsk(bid=yes_bid, ask=yes_ask, spread=yes_ask - yes_bid)
    yes_book.timestamp_ms = 1000

    state = LocalState(
        market=market,
        yes_book=yes_book,
        no_book=TokenBook(asset_id="NO"),
    )
    state.inventory.pusd_free = pusd_free
    state.inventory.up_free = up_free

    if position_qty > 0:
        state.position.qty = position_qty
        state.position.cost_basis = position_cost

    if binance_recv_ts is not None:
        state.last_binance = PriceTick(
            symbol="btc/usd", timestamp_ms=binance_recv_ts - 50,
            value=84000.0, recv_timestamp_ms=binance_recv_ts, sequence_no=1,
        )
    if chainlink_recv_ts is not None:
        state.last_chainlink = PriceTick(
            symbol="btc/usd", timestamp_ms=chainlink_recv_ts - 50,
            value=84000.0, recv_timestamp_ms=chainlink_recv_ts, sequence_no=1,
        )
    return state


def _make_fair(p_up: float = 0.55, tau_s: float = 900.0) -> FairValueSnapshot:
    return FairValueSnapshot(
        p_up=p_up, p_down=1.0 - p_up, z_score=0.0, sigma_60=0.001,
        denom=1.0, lead_adj=0.0, micro_adj=0.0, imbalance=0.0, tape=0.0,
        chainlink_last=84000.0, binance_last=84000.0, ptb=84000.0,
        tau_s=tau_s, timestamp_ms=1000,
    )


def _make_config(
    min_entry_edge_prob: float = 0.02,
    aggressive_entry_edge_prob: float = 0.05,
    take_profit_prob: float = 0.04,
    stop_loss_prob: float = -0.03,
    edge_lost_exit_prob: float = 0.01,
    min_tau_to_enter_s: float = 120.0,
    force_exit_tau_s: float = 45.0,
    entry_notional_usd: float = 5.0,
    chainlink_max_age_ms: int = 3_000,
    binance_max_age_ms: int = 10_000,
) -> RuntimeConfig:
    from bot.settings import (
        ConfigValue, DirectionalThresholds, FairThresholds, FreshnessThresholds,
        InventoryThresholds, PTBThresholds, QuoteThresholds, Thresholds,
    )
    return RuntimeConfig(
        thresholds=Thresholds(
            freshness=FreshnessThresholds(
                chainlink_max_age_ms=ConfigValue(chainlink_max_age_ms),
                binance_max_age_ms=ConfigValue(binance_max_age_ms),
            ),
            directional=DirectionalThresholds(
                min_entry_edge_prob=min_entry_edge_prob,
                aggressive_entry_edge_prob=aggressive_entry_edge_prob,
                take_profit_prob=take_profit_prob,
                stop_loss_prob=stop_loss_prob,
                edge_lost_exit_prob=edge_lost_exit_prob,
                min_tau_to_enter_s=min_tau_to_enter_s,
                force_exit_tau_s=force_exit_tau_s,
                entry_notional_usd=entry_notional_usd,
                chainlink_max_age_ms=chainlink_max_age_ms,
            ),
        )
    )


NOW_MS = 1_000_000


# ---------------------------------------------------------------------------
# Test 1: book gate — no book → disabled
# ---------------------------------------------------------------------------

def test_book_gate_no_bids_returns_disabled():
    state = _make_state()
    state.yes_book.bids = {}
    state.yes_book.best = BestBidAsk(bid=0.0, ask=0.55, spread=0.55)
    policy = DirectionalPolicyV2(config=DEFAULT_CONFIG)
    result = policy.build(state, _make_fair(), NOW_MS)
    assert not result.bid.enabled
    assert not result.ask.enabled
    assert result.bid.reason == "book_gate"


def test_book_gate_no_asks_returns_disabled():
    state = _make_state()
    state.yes_book.asks = {}
    state.yes_book.best = BestBidAsk(bid=0.45, ask=1.0, spread=0.55)
    policy = DirectionalPolicyV2(config=DEFAULT_CONFIG)
    result = policy.build(state, _make_fair(), NOW_MS)
    assert not result.bid.enabled
    assert result.bid.reason == "book_gate"


# ---------------------------------------------------------------------------
# Test 2: FLAT — no edge → disabled with "no_edge"
# ---------------------------------------------------------------------------

def test_flat_no_edge_returns_disabled():
    # fair.p_up=0.55, yes_ask=0.55 → entry_edge=0.0 < min_entry_edge_prob=0.02
    state = _make_state(yes_ask=0.55)
    fair = _make_fair(p_up=0.55, tau_s=900.0)
    policy = DirectionalPolicyV2(config=_make_config(min_entry_edge_prob=0.02))
    result = policy.build(state, fair, NOW_MS)
    assert not result.bid.enabled
    assert result.bid.reason == "no_edge"
    assert result.strategy_state == "flat"


# ---------------------------------------------------------------------------
# Test 3: FLAT — passive entry (edge >= min but < aggressive)
# ---------------------------------------------------------------------------

def test_flat_passive_entry_bids_just_below_ask():
    # fair.p_up=0.58, yes_ask=0.55 → entry_edge=0.03 (≥0.02, <0.05)
    state = _make_state(yes_ask=0.55, yes_bid=0.45)
    fair = _make_fair(p_up=0.58, tau_s=900.0)
    policy = DirectionalPolicyV2(config=_make_config(
        min_entry_edge_prob=0.02, aggressive_entry_edge_prob=0.05
    ))
    result = policy.build(state, fair, NOW_MS)
    assert result.bid.enabled
    assert result.bid.reason == "enter_long_passive"
    assert result.bid.price is not None
    assert result.bid.price < 0.55  # bid below ask
    assert result.strategy_state == "flat"
    assert result.entry_edge is not None
    assert abs(result.entry_edge - 0.03) < 1e-5


# ---------------------------------------------------------------------------
# Test 4: FLAT — aggressive entry (edge >= aggressive threshold)
# ---------------------------------------------------------------------------

def test_flat_aggressive_entry_bids_at_ask():
    # fair.p_up=0.61, yes_ask=0.55 → entry_edge=0.06 ≥ aggressive threshold 0.05
    state = _make_state(yes_ask=0.55, yes_bid=0.45)
    fair = _make_fair(p_up=0.61, tau_s=900.0)
    policy = DirectionalPolicyV2(config=_make_config(
        min_entry_edge_prob=0.02, aggressive_entry_edge_prob=0.05
    ))
    result = policy.build(state, fair, NOW_MS)
    assert result.bid.enabled
    assert result.bid.reason == "enter_long_aggressive"
    assert result.bid.price == 0.55  # bid at ask (lift the ask)
    assert result.strategy_state == "flat"


# ---------------------------------------------------------------------------
# Test 5: tau gate blocks entry
# ---------------------------------------------------------------------------

def test_tau_gate_blocks_entry_when_tau_too_small():
    state = _make_state(yes_ask=0.55)
    fair = _make_fair(p_up=0.62, tau_s=100.0)  # tau < min_tau_to_enter_s=120
    policy = DirectionalPolicyV2(config=_make_config(min_tau_to_enter_s=120.0))
    result = policy.build(state, fair, NOW_MS)
    assert not result.bid.enabled
    assert result.bid.reason == "tau_gate"
    assert result.strategy_state == "flat"


# ---------------------------------------------------------------------------
# Test 6: LONG — hold (no exit trigger)
# ---------------------------------------------------------------------------

def test_long_hold_when_no_exit_trigger():
    # pnl = best_bid - avg_cost = 0.50 - 0.50 = 0.0 (< take_profit=0.04, > stop_loss=-0.03)
    # unrealized_edge = fair.p_up - best_bid = 0.52 - 0.50 = 0.02 >= edge_lost=0.01 → no edge_lost
    state = _make_state(
        yes_bid=0.50, yes_ask=0.56,
        position_qty=10.0, position_cost=5.0,  # avg_cost=0.50
    )
    fair = _make_fair(p_up=0.52, tau_s=900.0)
    policy = DirectionalPolicyV2(config=_make_config(
        take_profit_prob=0.04, stop_loss_prob=-0.03, edge_lost_exit_prob=0.01
    ))
    result = policy.build(state, fair, NOW_MS)
    assert not result.bid.enabled
    assert result.bid.reason == "no_pyramid"
    assert not result.ask.enabled
    assert result.ask.reason == "hold"
    assert result.strategy_state == "long"
    assert result.pnl_per_share is not None
    assert abs(result.pnl_per_share - 0.0) < 1e-6


# ---------------------------------------------------------------------------
# Test 7: LONG — stop loss triggers aggressive exit
# ---------------------------------------------------------------------------

def test_long_stop_loss_triggers_aggressive_exit():
    # pnl = best_bid - avg_cost = 0.54 - 0.60 = -0.06 < stop_loss=-0.03
    state = _make_state(
        yes_bid=0.54, yes_ask=0.62,
        up_free=10.0,
        position_qty=10.0, position_cost=6.0,  # avg_cost=0.60
    )
    fair = _make_fair(p_up=0.55, tau_s=900.0)
    policy = DirectionalPolicyV2(config=_make_config(stop_loss_prob=-0.03))
    result = policy.build(state, fair, NOW_MS)
    assert result.ask.enabled
    assert result.ask.reason == "exit_stop_loss"
    assert result.ask.price == 0.54  # aggressive: at top_bid price
    assert result.strategy_state == "long"
    assert result.exit_candidate_reason == "exit_stop_loss"


# ---------------------------------------------------------------------------
# Test 8: LONG — take profit triggers passive exit
# ---------------------------------------------------------------------------

def test_long_take_profit_triggers_passive_exit():
    # pnl = best_bid - avg_cost = 0.54 - 0.50 = 0.04 >= take_profit=0.04
    # tau=900 >> aggressive_exit_tau_s=60 → passive exit
    state = _make_state(
        yes_bid=0.54, yes_ask=0.62,
        up_free=10.0,
        position_qty=10.0, position_cost=5.0,  # avg_cost=0.50
    )
    fair = _make_fair(p_up=0.56, tau_s=900.0)
    policy = DirectionalPolicyV2(config=_make_config(take_profit_prob=0.04))
    result = policy.build(state, fair, NOW_MS)
    assert result.ask.enabled
    assert result.ask.reason == "exit_take_profit_passive"
    assert result.ask.price == 0.55  # passive: top_bid + tick = 0.54 + 0.01
    assert result.strategy_state == "long"


# ---------------------------------------------------------------------------
# Test 9: LONG — edge lost triggers aggressive exit
# ---------------------------------------------------------------------------

def test_long_edge_lost_triggers_passive_exit():
    # unrealized_edge = fair.p_up - best_bid = 0.55 - 0.54 = 0.01 < edge_lost_exit_prob=0.02
    # pnl = best_bid - avg_cost = 0.54 - 0.55 = -0.01 > stop_loss=-0.03 → no stop_loss
    # tau=900 >> aggressive_exit_tau_s=60 → passive exit
    state = _make_state(
        yes_bid=0.54, yes_ask=0.62,
        up_free=10.0,
        position_qty=10.0, position_cost=5.5,  # avg_cost=0.55
    )
    fair = _make_fair(p_up=0.55, tau_s=900.0)
    policy = DirectionalPolicyV2(config=_make_config(
        stop_loss_prob=-0.03,
        edge_lost_exit_prob=0.02,
        take_profit_prob=0.04,
    ))
    result = policy.build(state, fair, NOW_MS)
    assert result.ask.enabled
    assert result.ask.reason == "exit_edge_lost_passive"
    assert result.ask.price == 0.55  # passive: top_bid + tick = 0.54 + 0.01


# ---------------------------------------------------------------------------
# Test 10: force exit when tau < force_exit_tau_s
# ---------------------------------------------------------------------------

def test_force_exit_when_tau_below_threshold():
    state = _make_state(
        yes_bid=0.50, yes_ask=0.60,
        up_free=10.0,
        position_qty=10.0, position_cost=5.0,
    )
    fair = _make_fair(p_up=0.52, tau_s=30.0)  # tau < force_exit_tau_s=45
    policy = DirectionalPolicyV2(config=_make_config(force_exit_tau_s=45.0))
    result = policy.build(state, fair, NOW_MS)
    assert result.ask.enabled
    assert result.ask.reason == "exit_force"
    assert result.ask.price == 0.50  # aggressive: at top_bid


# ---------------------------------------------------------------------------
# Test 11: force exit bypasses freshness gates
# ---------------------------------------------------------------------------

def test_force_exit_bypasses_freshness_gates():
    # Set stale binance tick (50s ago > 10s threshold) but force exit still triggers
    state = _make_state(
        yes_bid=0.50, yes_ask=0.60,
        up_free=10.0,
        position_qty=10.0, position_cost=5.0,
        binance_recv_ts=NOW_MS - 50_000,  # 50s stale
    )
    fair = _make_fair(p_up=0.52, tau_s=30.0)  # force exit zone
    policy = DirectionalPolicyV2(config=_make_config(
        force_exit_tau_s=45.0,
        binance_max_age_ms=10_000,
    ))
    result = policy.build(state, fair, NOW_MS)
    assert result.ask.enabled
    assert result.ask.reason == "exit_force"  # not "binance_stale"


# ---------------------------------------------------------------------------
# Test 12: chainlink stale blocks entry
# ---------------------------------------------------------------------------

def test_chainlink_stale_blocks_entry():
    state = _make_state(
        yes_ask=0.50,
        chainlink_recv_ts=NOW_MS - 10_000,  # 10s ago > chainlink_max_age_ms=3s
    )
    fair = _make_fair(p_up=0.60, tau_s=900.0)
    policy = DirectionalPolicyV2(config=_make_config(chainlink_max_age_ms=3_000))
    result = policy.build(state, fair, NOW_MS)
    assert not result.bid.enabled
    assert result.bid.reason == "chainlink_stale"


# ---------------------------------------------------------------------------
# Test 13: binance stale blocks entry
# ---------------------------------------------------------------------------

def test_binance_stale_blocks_entry():
    state = _make_state(
        yes_ask=0.50,
        binance_recv_ts=NOW_MS - 15_000,  # 15s ago > binance_max_age_ms=10s
    )
    fair = _make_fair(p_up=0.60, tau_s=900.0)
    policy = DirectionalPolicyV2(config=_make_config(binance_max_age_ms=10_000))
    result = policy.build(state, fair, NOW_MS)
    assert not result.bid.enabled
    assert result.bid.reason == "binance_stale"


# ---------------------------------------------------------------------------
# Test 14: entry_edge field in returned DesiredQuotes
# ---------------------------------------------------------------------------

def test_entry_edge_populated_on_flat_entry():
    state = _make_state(yes_ask=0.55, yes_bid=0.45)
    fair = _make_fair(p_up=0.58, tau_s=900.0)
    policy = DirectionalPolicyV2(config=_make_config(min_entry_edge_prob=0.02))
    result = policy.build(state, fair, NOW_MS)
    assert result.entry_edge is not None
    assert abs(result.entry_edge - (0.58 - 0.55)) < 1e-5


def test_entry_edge_none_when_in_long_state():
    state = _make_state(
        yes_bid=0.50, yes_ask=0.56,
        up_free=10.0, position_qty=10.0, position_cost=5.0,
    )
    fair = _make_fair(p_up=0.52, tau_s=900.0)
    policy = DirectionalPolicyV2(config=DEFAULT_CONFIG)
    result = policy.build(state, fair, NOW_MS)
    assert result.entry_edge is None
    assert result.strategy_state == "long"


# ---------------------------------------------------------------------------
# Test 15: ask disabled in flat state, bid disabled in long state
# ---------------------------------------------------------------------------

def test_ask_always_disabled_in_flat_state():
    state = _make_state(yes_ask=0.55)
    fair = _make_fair(p_up=0.62, tau_s=900.0)
    policy = DirectionalPolicyV2(config=_make_config(min_entry_edge_prob=0.02))
    result = policy.build(state, fair, NOW_MS)
    assert result.strategy_state == "flat"
    assert not result.ask.enabled
    assert result.ask.reason == "flat"


def test_bid_always_disabled_in_long_state():
    # In LONG state, bid should always be disabled (no pyramiding)
    state = _make_state(
        yes_bid=0.50, yes_ask=0.56,
        up_free=10.0, position_qty=10.0, position_cost=5.0,
    )
    fair = _make_fair(p_up=0.56, tau_s=900.0)  # take profit zone
    policy = DirectionalPolicyV2(config=_make_config(take_profit_prob=0.04))
    result = policy.build(state, fair, NOW_MS)
    assert result.strategy_state == "long"
    assert not result.bid.enabled
    assert result.bid.reason == "no_pyramid"


# ---------------------------------------------------------------------------
# Test 16: sizing — max_position_notional_usd cap
# ---------------------------------------------------------------------------

def test_entry_size_capped_by_max_position_notional():
    # entry_notional=50 at price 0.55 → raw size=90.9; max_notional=25 → max_size=45.4
    # bid_size = min(90.9, 45.4) = 45.4 — enabled
    state = _make_state(yes_ask=0.55, yes_bid=0.45, pusd_free=1000.0)
    fair = _make_fair(p_up=0.62, tau_s=900.0)
    from bot.settings import (
        ConfigValue, DirectionalThresholds, FreshnessThresholds,
        InventoryThresholds, PTBThresholds, QuoteThresholds, Thresholds,
    )
    cfg = RuntimeConfig(
        thresholds=Thresholds(
            directional=DirectionalThresholds(
                min_entry_edge_prob=0.02,
                aggressive_entry_edge_prob=0.05,
                entry_notional_usd=50.0,
                max_position_notional_usd=25.0,
            ),
        )
    )
    policy = DirectionalPolicyV2(config=cfg)
    result = policy.build(state, fair, NOW_MS)
    assert result.bid.enabled
    # max_size = floor(25 / 0.55 * 1000) / 1000 = floor(45454.5) / 1000 = 45.454
    assert result.bid.size <= 25.0 / 0.55 + 0.001


def test_entry_refused_when_size_below_min_order():
    # entry_notional=1 at price 0.55 → size≈1.818; min_order_size=5 → refused
    state = _make_state(yes_ask=0.55, yes_bid=0.45, pusd_free=1000.0)
    fair = _make_fair(p_up=0.62, tau_s=900.0)
    cfg = _make_config(entry_notional_usd=1.0)
    policy = DirectionalPolicyV2(config=cfg)
    result = policy.build(state, fair, NOW_MS)
    # min_order_size=1.0 from _make_clob default; 1/0.55≈1.818 >= 1.0 → enabled
    # Use a real min_order_size=5.0 via custom clob
    from bot.domain import BestBidAsk, ClobMarketInfo, ClobToken, TokenBook
    state2 = _make_state(yes_ask=0.55, yes_bid=0.45, pusd_free=1000.0)
    state2.market = _make_market()
    state2.market = MarketContext(
        market_id="test",
        condition_id="test",
        title="Test",
        slug="test",
        start_ts_ms=0,
        end_ts_ms=9_999_999_999_000,
        yes_token_id="YES",
        no_token_id="NO",
        clob=ClobMarketInfo(
            tokens=[ClobToken(token_id="YES", outcome="Yes")],
            min_order_size=10.0,  # large min_order_size
            min_tick_size=0.01,
            maker_base_fee_bps=0,
            taker_base_fee_bps=0,
            taker_delay_enabled=False,
            min_order_age_s=0.0,
            fee_rate=0.0,
            fee_exponent=1.0,
        ),
    )
    # entry_notional=1 at 0.55 → size≈1.818 < min_order=10 → refused
    result2 = policy.build(state2, fair, NOW_MS)
    assert not result2.bid.enabled
    assert result2.bid.reason == "size_too_small"


# ---------------------------------------------------------------------------
# Test 17: pnl_per_share uses best_bid not fair.p_up
# ---------------------------------------------------------------------------

def test_pnl_per_share_uses_best_bid_not_fair():
    # avg_cost=0.50, best_bid=0.52, fair.p_up=0.60
    # pnl = 0.52 - 0.50 = 0.02 (not 0.10)
    state = _make_state(
        yes_bid=0.52, yes_ask=0.62,
        up_free=10.0,
        position_qty=10.0, position_cost=5.0,  # avg_cost=0.50
    )
    fair = _make_fair(p_up=0.60, tau_s=900.0)
    policy = DirectionalPolicyV2(config=_make_config(
        take_profit_prob=0.10,  # high enough to not trigger at 0.02
        stop_loss_prob=-0.03,
        edge_lost_exit_prob=0.01,
    ))
    result = policy.build(state, fair, NOW_MS)
    assert result.pnl_per_share is not None
    assert abs(result.pnl_per_share - 0.02) < 1e-6  # best_bid - avg_cost


# ---------------------------------------------------------------------------
# Test 18: take_profit aggressive when tau <= aggressive_exit_tau_s
# ---------------------------------------------------------------------------

def test_take_profit_aggressive_when_tau_low():
    # pnl = 0.54 - 0.50 = 0.04 >= take_profit=0.04
    # tau=50 <= aggressive_exit_tau_s=60 → aggressive exit
    state = _make_state(
        yes_bid=0.54, yes_ask=0.62,
        up_free=10.0,
        position_qty=10.0, position_cost=5.0,  # avg_cost=0.50
    )
    fair = _make_fair(p_up=0.60, tau_s=50.0)  # 50 <= 60 → aggressive
    policy = DirectionalPolicyV2(config=_make_config(
        take_profit_prob=0.04, force_exit_tau_s=45.0
    ))
    result = policy.build(state, fair, NOW_MS)
    assert result.ask.enabled
    assert result.ask.reason == "exit_take_profit_aggressive"
    assert result.ask.price == 0.54  # aggressive: at top_bid


# ---------------------------------------------------------------------------
# Test 19: edge_lost aggressive when tau <= aggressive_exit_tau_s
# ---------------------------------------------------------------------------

def test_edge_lost_aggressive_when_tau_low():
    # unrealized_edge = 0.55 - 0.54 = 0.01 < edge_lost_exit_prob=0.02
    # tau=50 <= aggressive_exit_tau_s=60 → aggressive
    state = _make_state(
        yes_bid=0.54, yes_ask=0.62,
        up_free=10.0,
        position_qty=10.0, position_cost=5.5,  # avg_cost=0.55
    )
    fair = _make_fair(p_up=0.55, tau_s=50.0)
    policy = DirectionalPolicyV2(config=_make_config(
        stop_loss_prob=-0.03,
        edge_lost_exit_prob=0.02,
        take_profit_prob=0.04,
        force_exit_tau_s=45.0,
    ))
    result = policy.build(state, fair, NOW_MS)
    assert result.ask.enabled
    assert result.ask.reason == "exit_edge_lost_aggressive"
    assert result.ask.price == 0.54  # aggressive: at top_bid


# ---------------------------------------------------------------------------
# Test 20: desired_quotes set on state after build()
# ---------------------------------------------------------------------------

def test_desired_quotes_set_on_state_after_build():
    state = _make_state(yes_ask=0.55, yes_bid=0.45)
    fair = _make_fair(p_up=0.62, tau_s=900.0)
    policy = DirectionalPolicyV2(config=_make_config(min_entry_edge_prob=0.02))
    result = policy.build(state, fair, NOW_MS)
    assert state.desired_quotes is result

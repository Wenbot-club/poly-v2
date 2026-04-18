from __future__ import annotations

from dataclasses import dataclass
from statistics import pstdev
from typing import Optional

from .settings import RuntimeConfig
from .domain import LocalState, MarketContext, PriceTick, TokenBook


@dataclass
class StateFactory:
    config: RuntimeConfig

    def create(self, market: MarketContext) -> LocalState:
        yes_book = TokenBook(asset_id=market.yes_token_id, tick_size=market.clob.min_tick_size)
        no_book = TokenBook(asset_id=market.no_token_id, tick_size=market.clob.min_tick_size)
        state = LocalState(market=market, yes_book=yes_book, no_book=no_book)
        state.inventory.pusd_free = self.config.default_working_capital_usd
        state.inventory.up_target = self.config.thresholds.inventory.target_split_notional_usd.value
        state.set_clock(market.start_ts_ms)
        state.log("INFO", "state_created", ts_ms=market.start_ts_ms, market_id=market.market_id, condition_id=market.condition_id)
        return state


def register_chainlink_tick(state: LocalState, tick: PriceTick) -> None:
    state.set_clock(tick.recv_timestamp_ms)
    state.last_chainlink = tick
    state.chainlink_ticks.append(tick)
    state.log("INFO", "chainlink_tick", ts_ms=tick.recv_timestamp_ms, timestamp_ms=tick.timestamp_ms, recv_timestamp_ms=tick.recv_timestamp_ms, value=tick.value)


def register_binance_tick(state: LocalState, tick: PriceTick, config: RuntimeConfig) -> None:
    state.set_clock(tick.recv_timestamp_ms)
    state.last_binance = tick
    if state.binance_ticks:
        prev = state.binance_ticks[-1]
        signed_move = tick.value - prev.value
        if signed_move != 0:
            direction = 1.0 if signed_move > 0 else -1.0
            alpha = config.thresholds.fair.tape_ewma_alpha.value
            state.tape_ewma = alpha * direction + (1.0 - alpha) * state.tape_ewma
    state.binance_ticks.append(tick)
    state.log("INFO", "binance_tick", ts_ms=tick.recv_timestamp_ms, timestamp_ms=tick.timestamp_ms, recv_timestamp_ms=tick.recv_timestamp_ms, value=tick.value, tape_ewma=round(state.tape_ewma, 6))


def chainlink_age_ms(state: LocalState, now_ms: int) -> Optional[int]:
    return None if state.last_chainlink is None else now_ms - state.last_chainlink.recv_timestamp_ms


def binance_age_ms(state: LocalState, now_ms: int) -> Optional[int]:
    return None if state.last_binance is None else now_ms - state.last_binance.recv_timestamp_ms


def market_book_age_ms(state: LocalState, now_ms: int) -> int:
    return now_ms - max(state.yes_book.timestamp_ms, state.no_book.timestamp_ms)


def sigma_60_from_binance(state: LocalState, now_ms: Optional[int] = None, lookback_ms: int = 60_000) -> float:
    if now_ms is None:
        if state.last_binance is None:
            return 0.0
        now_ms = state.last_binance.timestamp_ms
    ticks = [t for t in state.binance_ticks if now_ms - lookback_ms <= t.timestamp_ms <= now_ms]
    if len(ticks) < 3:
        return 0.0
    diffs = [ticks[i].value - ticks[i - 1].value for i in range(1, len(ticks)) if ticks[i].timestamp_ms > ticks[i - 1].timestamp_ms]
    return 0.0 if not diffs else float(pstdev(diffs))

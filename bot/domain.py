from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Literal, Optional


Side = Literal["BUY", "SELL"]


def normalize_side(value: object, *, field_name: str = "side") -> Side:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string, got {type(value).__name__}")
    upper = value.upper()
    if upper not in {"BUY", "SELL"}:
        raise ValueError(f"{field_name} must be BUY or SELL, got {value!r}")
    return upper  # type: ignore[return-value]


@dataclass(slots=True)
class PriceLevel:
    price: float
    size: float


@dataclass(slots=True)
class BestBidAsk:
    bid: float = 0.0
    ask: float = 1.0
    spread: float = 1.0
    timestamp_ms: int = 0


@dataclass(slots=True)
class TokenBook:
    asset_id: str
    bids: Dict[float, float] = field(default_factory=dict)
    asks: Dict[float, float] = field(default_factory=dict)
    best: BestBidAsk = field(default_factory=BestBidAsk)
    last_trade_price: Optional[float] = None
    last_trade_side: Optional[Side] = None
    tick_size: float = 0.01
    timestamp_ms: int = 0

    def sorted_bids(self) -> List[PriceLevel]:
        return [
            PriceLevel(price=p, size=s)
            for p, s in sorted(self.bids.items(), key=lambda x: x[0], reverse=True)
            if s > 0
        ]

    def sorted_asks(self) -> List[PriceLevel]:
        return [
            PriceLevel(price=p, size=s)
            for p, s in sorted(self.asks.items(), key=lambda x: x[0])
            if s > 0
        ]

    def top_bid(self) -> Optional[PriceLevel]:
        bids = self.sorted_bids()
        return bids[0] if bids else None

    def top_ask(self) -> Optional[PriceLevel]:
        asks = self.sorted_asks()
        return asks[0] if asks else None


@dataclass(slots=True)
class PriceTick:
    symbol: str
    timestamp_ms: int
    value: float
    recv_timestamp_ms: int
    sequence_no: int


@dataclass(slots=True)
class ClobToken:
    token_id: str
    outcome: str


@dataclass(slots=True)
class ClobMarketInfo:
    tokens: List[ClobToken]
    min_order_size: float
    min_tick_size: float
    maker_base_fee_bps: int
    taker_base_fee_bps: int
    taker_delay_enabled: bool
    min_order_age_s: float
    fee_rate: float
    fee_exponent: float


@dataclass(slots=True)
class MarketContext:
    market_id: str
    condition_id: str
    title: str
    slug: str
    start_ts_ms: int
    end_ts_ms: int
    yes_token_id: str
    no_token_id: str
    clob: ClobMarketInfo

    @property
    def duration_ms(self) -> int:
        return self.end_ts_ms - self.start_ts_ms


@dataclass(slots=True)
class PTBDecision:
    locked: bool
    ptb_value: Optional[float]
    selected_tick: Optional[PriceTick]
    used_prestart_grace: bool
    collision_detected: bool
    collision_ticks: List[PriceTick]
    reason: str
    decision_ts_ms: int


@dataclass(slots=True)
class FairValueSnapshot:
    p_up: float
    p_down: float
    z_score: float      # legacy field; new callers populate via gap_z
    sigma_60: float
    denom: float
    lead_adj: float     # legacy field; new callers populate via signal_adj
    micro_adj: float    # legacy field; set to 0.0 by new callers
    imbalance: float
    tape: float
    chainlink_last: float
    binance_last: float
    ptb: float
    tau_s: float
    timestamp_ms: int
    # Additive fields (default 0.0 keeps existing construction sites unchanged)
    gap_z: float = 0.0          # gap_usd / sigma_usd — dimensionless, used by strategy
    signal_adj: float = 0.0     # clamp(K_SIGNAL * gap_z, ±SIGNAL_CAP) in prob space
    sigma_usd: float = 0.0      # max(sigma_60, SIGMA_FLOOR_USD) — effective sigma


@dataclass(slots=True)
class DesiredOrder:
    enabled: bool
    side: Side
    price: Optional[float]
    size: float
    reason: str


@dataclass(slots=True)
class DesiredQuotes:
    bid: DesiredOrder
    ask: DesiredOrder
    mode: str
    inventory_skew: float
    timestamp_ms: int


@dataclass(slots=True)
class InventoryState:
    up_free: float = 0.0
    down_free: float = 0.0
    pusd_free: float = 0.0
    pusd_reserved_for_bids: float = 0.0
    up_reserved_for_asks: float = 0.0
    up_live_bids: float = 0.0
    up_live_asks: float = 0.0
    up_target: float = 0.0

    def up_effective(self) -> float:
        return self.up_free + self.up_live_asks - self.up_live_bids

    def deviation(self) -> float:
        return self.up_effective() - self.up_target

    def available_pusd(self) -> float:
        return self.pusd_free - self.pusd_reserved_for_bids

    def available_up(self) -> float:
        return self.up_free - self.up_reserved_for_asks


@dataclass(slots=True)
class LiveOrder:
    order_id: str
    asset_id: str
    side: Side
    price: float
    size: float
    remaining: float
    status: str
    created_ts_ms: int
    updated_ts_ms: int
    client_order_id: Optional[str] = None


@dataclass(slots=True)
class UserOrderEvent:
    order_id: str
    asset_id: str
    side: Side
    price: float
    original_size: float
    size_matched: float
    remaining: float
    status: str
    timestamp_ms: int


@dataclass(slots=True)
class UserTradeEvent:
    trade_id: str
    asset_id: str
    side: Side
    price: float
    size: float
    timestamp_ms: int
    order_id: Optional[str] = None


@dataclass(slots=True)
class EventLogEntry:
    ts_ms: int
    level: Literal["INFO", "WARN", "ERROR"]
    message: str
    payload: dict = field(default_factory=dict)


@dataclass(slots=True)
class LocalState:
    market: MarketContext
    yes_book: TokenBook
    no_book: TokenBook
    chainlink_ticks: Deque[PriceTick] = field(default_factory=lambda: deque(maxlen=256))
    binance_ticks: Deque[PriceTick] = field(default_factory=lambda: deque(maxlen=512))
    last_chainlink: Optional[PriceTick] = None
    last_binance: Optional[PriceTick] = None
    ptb: Optional[PTBDecision] = None
    fair_value: Optional[FairValueSnapshot] = None
    desired_quotes: Optional[DesiredQuotes] = None
    inventory: InventoryState = field(default_factory=InventoryState)
    tape_ewma: float = 0.0
    simulated_now_ms: Optional[int] = None
    open_orders: Dict[str, LiveOrder] = field(default_factory=dict)
    live_bid_order_id: Optional[str] = None
    live_ask_order_id: Optional[str] = None
    user_order_events: List[UserOrderEvent] = field(default_factory=list)
    user_trade_events: List[UserTradeEvent] = field(default_factory=list)
    logs: List[EventLogEntry] = field(default_factory=list)

    def set_clock(self, now_ms: int) -> None:
        self.simulated_now_ms = now_ms

    def log(
        self,
        level: Literal["INFO", "WARN", "ERROR"],
        message: str,
        *,
        ts_ms: Optional[int] = None,
        **payload: object,
    ) -> None:
        resolved_ts_ms = ts_ms
        if resolved_ts_ms is None:
            for key in ("timestamp_ms", "recv_timestamp_ms", "decision_ts_ms", "payload_timestamp_ms"):
                candidate = payload.get(key)
                if isinstance(candidate, int):
                    resolved_ts_ms = candidate
                    break
        if resolved_ts_ms is None:
            resolved_ts_ms = self.simulated_now_ms if self.simulated_now_ms is not None else utc_now_ms()
        self.logs.append(EventLogEntry(ts_ms=resolved_ts_ms, level=level, message=message, payload=dict(payload)))


@dataclass(slots=True)
class DiscoveryCandidate:
    market_id: str
    condition_id: str
    title: str
    slug: str
    start_ts_ms: int
    end_ts_ms: int
    active: bool
    closed: bool
    clob_token_ids: List[str]


def parse_iso_to_ms(value: str) -> int:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def utc_now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)

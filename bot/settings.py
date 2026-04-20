from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ConfigValue:
    value: float


@dataclass(frozen=True, slots=True)
class FairThresholds:
    tape_ewma_alpha: ConfigValue = ConfigValue(0.35)


@dataclass(frozen=True, slots=True)
class PTBThresholds:
    prestart_grace_ms: ConfigValue = ConfigValue(250)
    max_delay_ms: ConfigValue = ConfigValue(2_500)
    collision_window_ms: ConfigValue = ConfigValue(25)


@dataclass(frozen=True, slots=True)
class FreshnessThresholds:
    chainlink_max_age_ms: ConfigValue = ConfigValue(3_000)
    market_book_max_age_ms: ConfigValue = ConfigValue(2_000)
    binance_max_age_ms: ConfigValue = ConfigValue(10_000)


@dataclass(frozen=True, slots=True)
class QuoteThresholds:
    size_change_reprice_ratio: ConfigValue = ConfigValue(0.10)


@dataclass(frozen=True, slots=True)
class InventoryThresholds:
    target_split_notional_usd: ConfigValue = ConfigValue(50.0)


@dataclass(frozen=True, slots=True)
class DirectionalThresholds:
    min_entry_edge_prob: float = 0.02
    aggressive_entry_edge_prob: float = 0.05
    take_profit_prob: float = 0.04
    stop_loss_prob: float = -0.03
    edge_lost_exit_prob: float = 0.01
    min_tau_to_enter_s: float = 120.0
    aggressive_exit_tau_s: float = 60.0
    force_exit_tau_s: float = 45.0
    entry_notional_usd: float = 5.0
    max_position_notional_usd: float = 25.0
    chainlink_max_age_ms: float = 15_000.0


@dataclass(frozen=True, slots=True)
class Thresholds:
    fair: FairThresholds = field(default_factory=FairThresholds)
    ptb: PTBThresholds = field(default_factory=PTBThresholds)
    freshness: FreshnessThresholds = field(default_factory=FreshnessThresholds)
    quote: QuoteThresholds = field(default_factory=QuoteThresholds)
    inventory: InventoryThresholds = field(default_factory=InventoryThresholds)
    directional: DirectionalThresholds = field(default_factory=DirectionalThresholds)


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    default_working_capital_usd: float = 125.0
    thresholds: Thresholds = field(default_factory=Thresholds)


DEFAULT_CONFIG = RuntimeConfig()

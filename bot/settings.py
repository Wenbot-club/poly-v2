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


@dataclass(frozen=True, slots=True)
class M5Config:
    # Window
    window_seconds: int = 300
    # PTB: don't attempt before window_ts + this delay
    ptb_fetch_delay_s: float = 15.0
    ptb_max_attempts: int = 10
    ptb_retry_delay_s: float = 3.0
    # Reject SSR when |ssr - api| > this (USD); use API instead
    ptb_max_ssr_api_delta_usd: float = 10.0
    # Entry scan window [start, end)
    entry_scan_start_s: float = 140.0
    entry_scan_end_s: float = 170.0
    early_poll_interval_s: float = 0.5
    # Probabilistic entry model
    sigma_lookback_s: float = 60.0
    sigma_floor_usd: float = 5.0
    z_gap_min: float = 0.35
    p_enter_up_min: float = 0.60
    p_enter_down_max: float = 0.40
    min_entry_edge: float = 0.06
    # Baseline entry
    baseline_elapsed_s: float = 170.0
    # Sizing
    leg1_bet_usd: float = 1.00
    hedge_bet_usd: float = 2.00
    # Hedge trigger (Binance only, no confirmation)
    hedge_threshold: float = 1.0
    hedge_cutoff_s: float = 250.0
    # Paper execution — FOK then FAK loop
    fok_price_offset: float = 0.05
    fok_max_price_leg1: float = 0.95   # cap for LEG1 FOK attempt
    fak_price_offset: float = 0.15
    fak_max_price: float = 0.99
    fak_duration_s: float = 10.0
    fak_retry_interval_s: float = 1.0
    price_insane_threshold: float = 0.995   # refuse if best_ask >= this
    # Token price background polling
    token_price_refresh_s: float = 2.0
    # Settlement
    settlement_initial_delay_s: float = 15.0
    settlement_poll_s: float = 4.0
    settlement_max_attempts: int = 20


DEFAULT_M5_CONFIG = M5Config()

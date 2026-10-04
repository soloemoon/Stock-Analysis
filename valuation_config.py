"""
Config loader for valuation_template.py.

Loads and validates valuation_config.yaml into a typed Config dataclass so
the template can consume it without editing Python source per run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from stock_valuation_portfolio import DCFAssumptions

DEFAULT_CONFIG_PATH = Path(__file__).parent / "valuation_config.yaml"

REQUIRED_TOP_LEVEL_KEYS = (
    "target_ticker",
    "peer_tickers",
    "portfolio_tickers",
    "price_history_period",
    "risk_free_rate",
    "allow_short_selling",
    "dcf_assumptions",
    "sensitivity",
)
REQUIRED_DCF_KEYS = ("projection_years", "fcf_growth_rate", "terminal_growth_rate", "discount_rate")
REQUIRED_SENSITIVITY_KEYS = ("discount_rates", "terminal_growth_rates", "flag_threshold")


@dataclass
class Config:
    target_ticker: str
    peer_tickers: list[str]
    sector: str | None
    portfolio_tickers: list[str]
    price_history_period: str
    risk_free_rate: float
    allow_short_selling: bool
    dcf_assumptions: DCFAssumptions
    sensitivity_discount_rates: list[float]
    sensitivity_terminal_growth_rates: list[float]
    sensitivity_flag_threshold: float


def _require_keys(d: dict, keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in d]
    if missing:
        raise ValueError(f"Missing required key(s) in {context}: {', '.join(missing)}")


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate the YAML config file into a Config object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} did not parse to a mapping (got {type(raw).__name__}).")

    _require_keys(raw, REQUIRED_TOP_LEVEL_KEYS, str(path))
    _require_keys(raw["dcf_assumptions"], REQUIRED_DCF_KEYS, "dcf_assumptions")
    _require_keys(raw["sensitivity"], REQUIRED_SENSITIVITY_KEYS, "sensitivity")

    dcf_raw = raw["dcf_assumptions"]
    dcf_assumptions = DCFAssumptions(
        projection_years=dcf_raw["projection_years"],
        fcf_growth_rate=dcf_raw["fcf_growth_rate"],
        terminal_growth_rate=dcf_raw["terminal_growth_rate"],
        discount_rate=dcf_raw["discount_rate"],
        net_debt_override=dcf_raw.get("net_debt_override"),
    )

    sensitivity_raw = raw["sensitivity"]

    return Config(
        target_ticker=raw["target_ticker"],
        peer_tickers=list(raw["peer_tickers"]),
        sector=raw.get("sector"),
        portfolio_tickers=list(raw["portfolio_tickers"]),
        price_history_period=raw["price_history_period"],
        risk_free_rate=raw["risk_free_rate"],
        allow_short_selling=raw["allow_short_selling"],
        dcf_assumptions=dcf_assumptions,
        sensitivity_discount_rates=list(sensitivity_raw["discount_rates"]),
        sensitivity_terminal_growth_rates=list(sensitivity_raw["terminal_growth_rates"]),
        sensitivity_flag_threshold=sensitivity_raw["flag_threshold"],
    )

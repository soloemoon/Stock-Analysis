"""
Stock valuation (DCF + relative valuation comps) and efficient portfolio
construction (mean-variance optimization).

Requires: yfinance, numpy, pandas, scipy
    pip install yfinance numpy pandas scipy

Usage:
    python stock_valuation_portfolio.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Discounted Cash Flow (DCF) valuation
# ---------------------------------------------------------------------------


@dataclass
class DCFAssumptions:
    """Assumptions driving a DCF valuation."""

    projection_years: int = 5
    fcf_growth_rate: float = 0.06
    terminal_growth_rate: float = 0.025
    discount_rate: float = 0.09  # required return / WACC
    net_debt_override: float | None = None


@dataclass
class DCFResult:
    ticker: str
    current_price: float
    shares_outstanding: float
    base_fcf: float
    projected_fcf: list[float]
    discounted_fcf: list[float]
    terminal_value: float
    discounted_terminal_value: float
    enterprise_value: float
    equity_value: float
    intrinsic_value_per_share: float
    upside_pct: float


def _latest_free_cash_flow(ticker: yf.Ticker) -> float:
    """Free cash flow = Operating Cash Flow - CapEx, from the most recent annual cash flow statement."""
    cashflow = ticker.cashflow
    if cashflow is None or cashflow.empty:
        raise ValueError("No cash flow statement available for this ticker.")

    latest = cashflow.iloc[:, 0]

    operating_cf = next(
        (latest[label] for label in ("Operating Cash Flow", "Total Cash From Operating Activities") if label in latest.index),
        None,
    )
    capex = next(
        (latest[label] for label in ("Capital Expenditure", "Capital Expenditures") if label in latest.index),
        None,
    )

    if operating_cf is None or capex is None:
        raise ValueError("Could not locate operating cash flow / capex line items.")

    # CapEx is typically reported as a negative number already.
    return float(operating_cf) + float(capex)


@dataclass
class WACCEstimate:
    """CAPM-based cost of equity, blended with a simple cost of debt, into a WACC."""

    ticker: str
    risk_free_rate: float
    beta: float
    equity_risk_premium: float
    cost_of_equity: float
    cost_of_debt: float
    tax_rate: float
    market_cap: float
    total_debt: float
    weight_equity: float
    weight_debt: float
    wacc: float


def estimate_discount_rate(
    ticker_symbol: str,
    risk_free_rate: float | None = None,
    treasury_maturity: str = "10y",
    equity_risk_premium: float = 0.05,
    default_tax_rate: float = 0.21,
) -> WACCEstimate:
    """
    Estimate a discount rate (WACC) for `ticker_symbol` from yfinance data.

    `risk_free_rate` defaults to None, which fetches the current US Treasury
    yield for `treasury_maturity` (default "10y", the conventional proxy for
    the risk-free rate in a WACC build-up) via `get_treasury_yield`. Pass an
    explicit `risk_free_rate` to override this and skip the network lookup.

    Cost of equity uses CAPM: risk_free_rate + beta * equity_risk_premium,
    with beta pulled from yfinance (defaults to 1.0 if unavailable, i.e. the
    market's average risk). Cost of debt is approximated as
    interestExpense / totalDebt from the latest income statement (falling
    back to risk_free_rate + 1.5% if unavailable), after-tax using the
    company's effective tax rate when available (falling back to
    `default_tax_rate`). Equity and debt are weighted by market cap and
    total debt respectively.

    This is a simplified estimate meant as a reasonable starting point, not
    a substitute for a full WACC build-up (e.g. it ignores preferred equity,
    operating leases, and target vs. current capital structure).
    """
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info

    if risk_free_rate is None:
        risk_free_rate = get_treasury_yield(treasury_maturity)

    beta = info.get("beta")
    if beta is None:
        beta = 1.0
    cost_of_equity = risk_free_rate + beta * equity_risk_premium

    market_cap = info.get("marketCap")
    total_debt = info.get("totalDebt") or 0.0
    if market_cap is None:
        raise ValueError(f"Missing market cap for {ticker_symbol}; cannot weight cost of capital.")

    pretax_cost_of_debt = None
    tax_rate = default_tax_rate
    financials = ticker.financials
    if financials is not None and not financials.empty:
        latest = financials.iloc[:, 0]
        interest_expense = next(
            (abs(float(latest[label])) for label in ("Interest Expense",) if label in latest.index),
            None,
        )
        if interest_expense is not None and not pd.isna(interest_expense) and total_debt:
            pretax_cost_of_debt = interest_expense / total_debt

        pretax_income = next(
            (float(latest[label]) for label in ("Pretax Income", "Income Before Tax") if label in latest.index),
            None,
        )
        tax_provision = next(
            (float(latest[label]) for label in ("Tax Provision", "Income Tax Expense") if label in latest.index),
            None,
        )
        if (
            pretax_income
            and not pd.isna(pretax_income)
            and tax_provision is not None
            and not pd.isna(tax_provision)
            and pretax_income != 0
        ):
            implied_rate = tax_provision / pretax_income
            if 0 <= implied_rate <= 0.6:  # sanity bound against distorted/one-off effective rates
                tax_rate = implied_rate

    if pretax_cost_of_debt is None or pd.isna(pretax_cost_of_debt):
        pretax_cost_of_debt = risk_free_rate + 0.015  # fallback: modest credit spread over risk-free

    after_tax_cost_of_debt = pretax_cost_of_debt * (1 - tax_rate)

    total_capital = market_cap + total_debt
    weight_equity = market_cap / total_capital
    weight_debt = total_debt / total_capital

    wacc = weight_equity * cost_of_equity + weight_debt * after_tax_cost_of_debt

    return WACCEstimate(
        ticker=ticker_symbol,
        risk_free_rate=risk_free_rate,
        beta=beta,
        equity_risk_premium=equity_risk_premium,
        cost_of_equity=cost_of_equity,
        cost_of_debt=after_tax_cost_of_debt,
        tax_rate=tax_rate,
        market_cap=market_cap,
        total_debt=total_debt,
        weight_equity=weight_equity,
        weight_debt=weight_debt,
        wacc=wacc,
    )


@dataclass
class FCFGrowthEstimate:
    """Historical free-cash-flow growth rate, estimated from annual filings."""

    ticker: str
    fcf_history: list[float]  # oldest to newest
    years: int
    method: str
    cagr: float
    yoy_growth_rates: list[float]  # one per year-over-year step, oldest to newest
    average_yoy_growth: float
    growth_rate: float  # the estimate selected by `method`


def estimate_fcf_growth_rate(
    ticker_symbol: str,
    max_years: int = 4,
    method: str = "cagr",
) -> FCFGrowthEstimate:
    """
    Estimate a forward FCF growth rate from the historical growth of Free
    Cash Flow (Operating Cash Flow - CapEx) over up to `max_years` of annual
    filings available from yfinance (yfinance typically exposes ~4 years of
    annual cash flow statements).

    `method` selects which estimate is returned as `growth_rate` (both are
    always computed and available on the result):
      - "cagr" (default): endpoint-to-endpoint compound annual growth rate.
        Simple and standard, but sensitive to whichever single year happens
        to land on the start/end points -- a one-off spike or dip in either
        endpoint year distorts the whole estimate.
      - "average_yoy": arithmetic mean of the year-over-year growth rates
        across all available periods. Smooths out a single outlier year
        since it only influences the two YoY steps touching it, rather than
        anchoring the entire multi-year growth rate.

    Year-over-year steps where the starting year's FCF is not positive are
    skipped (growth % is undefined/misleading from a non-positive base) when
    computing `average_yoy_growth`; if all steps are skipped, it is NaN.

    Raises ValueError if fewer than 2 years of history are available (need
    at least a start and end point), if `method` is not recognized, or if
    `method="cagr"` and the earliest FCF value is not positive (endpoint
    CAGR is undefined from a non-positive base -- use `method="average_yoy"`
    instead, or set the growth rate manually).
    """
    if method not in ("cagr", "average_yoy"):
        raise ValueError(f"method must be 'cagr' or 'average_yoy', got {method!r}.")

    ticker = yf.Ticker(ticker_symbol)
    cashflow = ticker.cashflow
    if cashflow is None or cashflow.empty:
        raise ValueError(f"No cash flow statement available for {ticker_symbol}.")

    n_years = min(max_years, cashflow.shape[1])
    if n_years < 2:
        raise ValueError(f"Need at least 2 years of cash flow history for {ticker_symbol}; found {n_years}.")

    # yfinance orders columns newest to oldest; reverse to oldest-to-newest.
    fcf_history = []
    for col_idx in reversed(range(n_years)):
        col = cashflow.iloc[:, col_idx]
        operating_cf = next(
            (col[label] for label in ("Operating Cash Flow", "Total Cash From Operating Activities") if label in col.index),
            None,
        )
        capex = next(
            (col[label] for label in ("Capital Expenditure", "Capital Expenditures") if label in col.index),
            None,
        )
        if operating_cf is None or capex is None:
            continue
        fcf_history.append(float(operating_cf) + float(capex))

    if len(fcf_history) < 2:
        raise ValueError(f"Could not locate operating cash flow / capex line items across enough years for {ticker_symbol}.")

    periods = len(fcf_history) - 1

    yoy_growth_rates = [
        (fcf_history[i + 1] / fcf_history[i]) - 1
        for i in range(periods)
        if fcf_history[i] > 0
    ]
    average_yoy_growth = float(np.mean(yoy_growth_rates)) if yoy_growth_rates else float("nan")

    start, end = fcf_history[0], fcf_history[-1]
    if start > 0:
        cagr = (end / start) ** (1 / periods) - 1
    else:
        cagr = float("nan")

    if method == "cagr":
        if start <= 0:
            raise ValueError(
                f"Earliest FCF for {ticker_symbol} is not positive (${start:,.0f}); CAGR is undefined. "
                "Use method='average_yoy' instead, or set the growth rate manually."
            )
        growth_rate = cagr
    else:
        if not yoy_growth_rates:
            raise ValueError(
                f"No valid year-over-year FCF growth steps for {ticker_symbol} "
                "(every starting year had non-positive FCF)."
            )
        growth_rate = average_yoy_growth

    return FCFGrowthEstimate(
        ticker=ticker_symbol,
        fcf_history=fcf_history,
        years=periods,
        method=method,
        cagr=cagr,
        yoy_growth_rates=yoy_growth_rates,
        average_yoy_growth=average_yoy_growth,
        growth_rate=growth_rate,
    )


@dataclass
class StockRiskProfile:
    """Beta and monthly-return volatility for a single ticker."""

    ticker: str
    beta: float | None
    monthly_return_std: float
    n_months: int


def monthly_return_volatility(prices: pd.DataFrame) -> pd.Series:
    """
    Standard deviation of monthly returns for each ticker column in `prices`.

    Prices are resampled to month-end (last observation per month) before
    taking percent change, so the result reflects realized monthly
    volatility rather than a daily or annualized figure.
    """
    monthly_prices = prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna(how="all")
    return monthly_returns.std()


def get_betas(tickers: list[str]) -> pd.Series:
    """Look up beta (as reported by yfinance) for each ticker; NaN if unavailable."""
    betas = {ticker: yf.Ticker(ticker).info.get("beta") for ticker in tickers}
    return pd.Series(betas, dtype=float)


def portfolio_beta(weights: pd.Series, betas: pd.Series) -> float:
    """
    Weighted-average portfolio beta (a standard approximation, since beta is
    linear in portfolio weights under CAPM).

    Tickers with a missing (NaN) beta are dropped and the remaining weights
    renormalized, so a missing beta does not silently zero out its weight
    share. Raises ValueError if no ticker has both a weight and a valid beta.
    """
    betas = betas.reindex(weights.index)
    available = betas.notna()
    if not available.any():
        raise ValueError("No tickers with a valid beta are available.")

    w = weights[available]
    w = w / w.sum()  # renormalize over available betas
    return float((w * betas[available]).sum())


def portfolio_monthly_volatility(prices: pd.DataFrame, weights: pd.Series) -> float:
    """
    Standard deviation of the portfolio's monthly returns, using `weights`
    (aligned to `prices` columns) applied to monthly returns directly -- this
    correctly reflects diversification/correlation across tickers, unlike a
    naive weighted average of each ticker's individual monthly volatility.
    """
    monthly_prices = prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna(how="all")
    w = weights.reindex(monthly_returns.columns)
    portfolio_returns = (monthly_returns * w).sum(axis=1)
    return float(portfolio_returns.std())


def compute_stock_risk_profile(ticker_symbol: str, period: str = "5y") -> StockRiskProfile:
    """
    Fetch beta (as reported by yfinance) and compute the standard deviation
    of monthly returns for a single ticker over `period` of price history.

    Monthly returns are derived by resampling daily adjusted close prices to
    month-end and taking percent change, so the result reflects realized
    monthly volatility rather than an annualized figure.

    Raises ValueError if fewer than 2 monthly observations are available
    (a standard deviation needs at least 2 points).
    """
    ticker = yf.Ticker(ticker_symbol)
    beta = ticker.info.get("beta")

    prices = download_price_history([ticker_symbol], period=period)
    monthly_prices = prices.iloc[:, 0].resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna()

    if len(monthly_returns) < 2:
        raise ValueError(
            f"Need at least 2 monthly return observations for {ticker_symbol}; found {len(monthly_returns)}."
        )

    return StockRiskProfile(
        ticker=ticker_symbol,
        beta=beta,
        monthly_return_std=float(monthly_returns.std()),
        n_months=len(monthly_returns),
    )


@dataclass
class OwnerEarningsResult:
    """
    Approximate "owner earnings" (a la Buffett): Net Income + Depreciation &
    Amortization - maintenance CapEx, from the most recent annual filings.
    """

    ticker: str
    net_income: float
    depreciation_amortization: float
    maintenance_capex: float
    owner_earnings: float


def compute_owner_earnings(ticker_symbol: str, maintenance_capex_pct_of_capex: float = 1.0) -> OwnerEarningsResult:
    """
    Compute approximate owner earnings for `ticker_symbol`:
        Net Income + Depreciation & Amortization - maintenance CapEx

    yfinance does not separately report maintenance vs. growth CapEx, so
    `maintenance_capex_pct_of_capex` (default 1.0, i.e. all of reported CapEx)
    lets the caller approximate maintenance-only CapEx as a fraction of total
    CapEx when a more refined estimate is available (e.g. from a 10-K
    disclosure). Using the default of 100% is a conservative simplification
    that will understate owner earnings for companies investing heavily in
    growth capex.

    Raises ValueError if the required cash flow line items are unavailable.
    """
    ticker = yf.Ticker(ticker_symbol)
    cashflow = ticker.cashflow
    if cashflow is None or cashflow.empty:
        raise ValueError(f"No cash flow statement available for {ticker_symbol}.")

    latest = cashflow.iloc[:, 0]

    net_income = next(
        (latest[label] for label in ("Net Income From Continuing Operations", "Net Income") if label in latest.index),
        None,
    )
    depreciation_amortization = next(
        (
            latest[label]
            for label in ("Depreciation And Amortization", "Depreciation Amortization Depletion")
            if label in latest.index
        ),
        None,
    )
    capex = next(
        (latest[label] for label in ("Capital Expenditure", "Capital Expenditures") if label in latest.index),
        None,
    )

    if net_income is None or depreciation_amortization is None or capex is None:
        raise ValueError(f"Could not locate net income / D&A / capex line items for {ticker_symbol}.")

    net_income = float(net_income)
    depreciation_amortization = float(depreciation_amortization)
    # CapEx is typically reported as a negative number already; take the magnitude as spend.
    maintenance_capex = abs(float(capex)) * maintenance_capex_pct_of_capex

    owner_earnings = net_income + depreciation_amortization - maintenance_capex

    return OwnerEarningsResult(
        ticker=ticker_symbol,
        net_income=net_income,
        depreciation_amortization=depreciation_amortization,
        maintenance_capex=maintenance_capex,
        owner_earnings=owner_earnings,
    )


@dataclass
class FundamentalsResult:
    """Basic profitability/leverage ratios as reported by yfinance."""

    ticker: str
    return_on_equity: float | None
    debt_to_equity: float | None  # yfinance reports this as a percentage (e.g. 150 = 150%)
    profit_margin: float | None


def compute_fundamentals(ticker_symbol: str) -> FundamentalsResult:
    """
    Fetch return on equity, debt-to-equity, and net profit margin for
    `ticker_symbol` from yfinance's `info` (trailing-twelve-month figures as
    reported by the data source). Any ratio not reported is returned as None
    rather than raising, since these are supplementary context rather than
    required valuation inputs.
    """
    info = yf.Ticker(ticker_symbol).info
    return FundamentalsResult(
        ticker=ticker_symbol,
        return_on_equity=info.get("returnOnEquity"),
        debt_to_equity=info.get("debtToEquity"),
        profit_margin=info.get("profitMargins"),
    )


# ---------------------------------------------------------------------------
# US government (Treasury) yields, for a data-driven risk-free rate
# ---------------------------------------------------------------------------

# yfinance ticker symbols for US Treasury yield indices (CBOE), keyed by a
# short maturity label. Values are index levels already expressed as a
# percentage (e.g. 5.25 means 5.25%), not decimals.
TREASURY_YIELD_TICKERS = {
    "13w": "^IRX",  # 13-week T-bill
    "5y": "^FVX",  # 5-year note
    "10y": "^TNX",  # 10-year note
    "30y": "^TYX",  # 30-year bond
}


def get_treasury_yield(maturity: str = "10y") -> float:
    """
    Fetch the current US Treasury yield for `maturity` ("13w", "5y", "10y",
    or "30y") from yfinance's CBOE Treasury yield index tickers, as a decimal
    (e.g. 0.0525 for 5.25%).

    Uses the most recent daily close over a short trailing window, since
    these indices don't always have data for the current day (e.g. weekends,
    holidays). Raises ValueError if no recent data is available or if
    `maturity` is not recognized.
    """
    if maturity not in TREASURY_YIELD_TICKERS:
        raise ValueError(f"maturity must be one of {sorted(TREASURY_YIELD_TICKERS)}, got {maturity!r}.")

    symbol = TREASURY_YIELD_TICKERS[maturity]
    history = yf.Ticker(symbol).history(period="5d")
    if history is None or history.empty:
        raise ValueError(f"No recent Treasury yield data available for {maturity} ({symbol}).")

    return float(history["Close"].iloc[-1]) / 100


def run_dcf(ticker_symbol: str, assumptions: DCFAssumptions = DCFAssumptions()) -> DCFResult:
    """Run a simple single-stage-growth DCF valuation for one ticker."""
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info

    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    shares_outstanding = info.get("sharesOutstanding")
    if current_price is None or shares_outstanding is None:
        raise ValueError(f"Missing price or shares outstanding for {ticker_symbol}.")

    base_fcf = _latest_free_cash_flow(ticker)

    g = assumptions.fcf_growth_rate
    r = assumptions.discount_rate
    tg = assumptions.terminal_growth_rate
    n = assumptions.projection_years

    projected_fcf = [base_fcf * (1 + g) ** year for year in range(1, n + 1)]
    discounted_fcf = [cf / (1 + r) ** year for year, cf in enumerate(projected_fcf, start=1)]

    terminal_value = projected_fcf[-1] * (1 + tg) / (r - tg)
    discounted_terminal_value = terminal_value / (1 + r) ** n

    enterprise_value = sum(discounted_fcf) + discounted_terminal_value

    net_debt = assumptions.net_debt_override
    if net_debt is None:
        total_debt = info.get("totalDebt") or 0.0
        cash = info.get("totalCash") or 0.0
        net_debt = total_debt - cash

    equity_value = enterprise_value - net_debt
    intrinsic_value_per_share = equity_value / shares_outstanding
    upside_pct = intrinsic_value_per_share / current_price - 1

    return DCFResult(
        ticker=ticker_symbol,
        current_price=current_price,
        shares_outstanding=shares_outstanding,
        base_fcf=base_fcf,
        projected_fcf=projected_fcf,
        discounted_fcf=discounted_fcf,
        terminal_value=terminal_value,
        discounted_terminal_value=discounted_terminal_value,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        intrinsic_value_per_share=intrinsic_value_per_share,
        upside_pct=upside_pct,
    )


def dcf_sensitivity(
    ticker_symbol: str,
    discount_rates: list[float],
    terminal_growth_rates: list[float],
    base_assumptions: DCFAssumptions = DCFAssumptions(),
) -> pd.DataFrame:
    """
    Compute intrinsic value per share across a grid of discount rates and
    terminal growth rates, holding all other DCF assumptions fixed.

    Returns a DataFrame indexed by discount_rate, with one column per
    terminal_growth_rate, containing intrinsic value per share. Combinations
    where terminal_growth_rate >= discount_rate are skipped (undefined
    Gordon-growth terminal value) and left as NaN.
    """
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info

    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    shares_outstanding = info.get("sharesOutstanding")
    if current_price is None or shares_outstanding is None:
        raise ValueError(f"Missing price or shares outstanding for {ticker_symbol}.")

    base_fcf = _latest_free_cash_flow(ticker)

    net_debt = base_assumptions.net_debt_override
    if net_debt is None:
        total_debt = info.get("totalDebt") or 0.0
        cash = info.get("totalCash") or 0.0
        net_debt = total_debt - cash

    g = base_assumptions.fcf_growth_rate
    n = base_assumptions.projection_years

    results = pd.DataFrame(index=discount_rates, columns=terminal_growth_rates, dtype=float)
    results.index.name = "discount_rate"
    results.columns.name = "terminal_growth_rate"

    for r in discount_rates:
        projected_fcf = [base_fcf * (1 + g) ** year for year in range(1, n + 1)]
        discounted_fcf_sum = sum(cf / (1 + r) ** year for year, cf in enumerate(projected_fcf, start=1))

        for tg in terminal_growth_rates:
            if tg >= r:
                continue  # Gordon growth terminal value is undefined/negative when tg >= r

            terminal_value = projected_fcf[-1] * (1 + tg) / (r - tg)
            discounted_terminal_value = terminal_value / (1 + r) ** n

            enterprise_value = discounted_fcf_sum + discounted_terminal_value
            equity_value = enterprise_value - net_debt
            intrinsic_value_per_share = equity_value / shares_outstanding

            results.loc[r, tg] = intrinsic_value_per_share

    results.attrs["current_price"] = current_price
    return results


def flag_sensitivity_divergence(
    sensitivity: pd.DataFrame,
    current_price: float | None = None,
    threshold: float = 0.20,
) -> pd.DataFrame:
    """
    Convert a DCF sensitivity grid of intrinsic values into % divergence from
    the current market price, and flag cells where the magnitude of that
    divergence exceeds `threshold` (default 20%).

    Returns a DataFrame of strings like "+34.2%*" where the trailing "*"
    marks cells flagged as diverging sharply. NaN cells (undefined Gordon
    growth combinations) are left blank.
    """
    if current_price is None:
        current_price = sensitivity.attrs.get("current_price")
    if current_price is None:
        raise ValueError("current_price not provided and not found in sensitivity.attrs.")

    pct_diff = sensitivity / current_price - 1

    def _format_cell(value: float) -> str:
        if pd.isna(value):
            return ""
        flag = "*" if abs(value) > threshold else ""
        return f"{value:+.1%}{flag}"

    formatted = pct_diff.map(_format_cell)
    formatted.attrs["threshold"] = threshold
    return formatted


def summarize_sensitivity_flags(sensitivity: pd.DataFrame, current_price: float | None = None, threshold: float = 0.20) -> None:
    """Print the sensitivity grid alongside a flagged version, plus a summary of flagged combinations."""
    if current_price is None:
        current_price = sensitivity.attrs.get("current_price")
    if current_price is None:
        raise ValueError("current_price not provided and not found in sensitivity.attrs.")

    pct_diff = sensitivity / current_price - 1
    flagged = flag_sensitivity_divergence(sensitivity, current_price=current_price, threshold=threshold)

    print(f"Current price: ${current_price:.2f}")
    print("\nIntrinsic value per share:")
    print(sensitivity.round(2))
    print(f"\n% divergence from current price (flagged with '*' when |divergence| > {threshold:.0%}):")
    print(flagged)

    flagged_combos = (
        pct_diff.abs()
        .stack()
        .loc[lambda s: s > threshold]
        .sort_values(ascending=False)
    )
    if flagged_combos.empty:
        print("\nNo assumption combinations diverge from current price by more than the threshold.")
    else:
        print(f"\n{len(flagged_combos)} combination(s) diverge by more than {threshold:.0%}:")
        for (discount_rate, terminal_growth), diff in flagged_combos.items():
            direction = "overvalued" if diff < 0 else "undervalued"
            actual_diff = pct_diff.loc[discount_rate, terminal_growth]
            print(
                f"  discount_rate={discount_rate:.1%}, terminal_growth={terminal_growth:.1%} "
                f"-> {actual_diff:+.1%} ({direction})"
            )


# ---------------------------------------------------------------------------
# Relative valuation (comps vs. peers)
# ---------------------------------------------------------------------------


COMP_MULTIPLES = ("trailingPE", "forwardPE", "enterpriseToEbitda", "priceToBook", "priceToSalesTrailing12Months")

_MULTIPLE_LABELS = {
    "trailingPE": "P/E (trailing)",
    "forwardPE": "P/E (forward)",
    "enterpriseToEbitda": "EV/EBITDA",
    "priceToBook": "P/B",
    "priceToSalesTrailing12Months": "P/S",
}

# Default reliability weights for the "average premium/discount" summary.
# These are subjective judgment calls, not derived from data, and should be
# treated as a starting point to override for a given sector/company:
#   - EV/EBITDA is capital-structure neutral and excludes non-cash charges,
#     generally considered one of the more reliable comps multiples.
#   - Forward P/E reflects expected earnings, but depends on analyst
#     estimate quality/consensus risk.
#   - Trailing P/E is distorted by one-off/non-recurring items and by
#     leverage differences across peers.
#   - P/B is most meaningful for asset-heavy or financial businesses, less
#     so for asset-light/tech names.
#   - P/S ignores profitability and margin differences entirely, so it is
#     the least informative multiple on its own.
#   - ROE and Profit margin are included as secondary "quality" checks
#     alongside the price multiples, at modest weight, since they aren't
#     price ratios (a premium multiple backed by higher profitability is a
#     more defensible premium than one that isn't).
#   - Debt/Equity is included at low weight as a leverage check. NOTE: unlike
#     the other entries, a "premium" (higher than peers) on Debt/Equity
#     reflects *more* leverage, not a richer valuation -- the sign here does
#     not mean the same thing as for the price multiples. Treat the combined
#     weighted average with that caveat in mind, or set its weight to 0 to
#     exclude it from the blended premium/discount score entirely.
DEFAULT_MULTIPLE_RELIABILITY_WEIGHTS = {
    "P/E (trailing)": 0.15,
    "P/E (forward)": 0.30,
    "EV/EBITDA": 0.35,
    "P/B": 0.10,
    "P/S": 0.10,
    "ROE": 0.10,
    "Debt/Equity": 0.05,
    "Profit margin": 0.10,
}

# Sector/sub-industry reliability weight presets, keyed by a short label.
# Broad sector labels ("financials", "tech", "industrials") are kept as
# aliases pointing at a representative sub-industry preset for convenience,
# but the sub-industry presets are the more precise choice when applicable.
#
# Rationale for the sub-industry splits:
#   - Banks: earnings and book value (net asset value of loans/deposits)
#     drive valuation; EV/EBITDA and P/S are not meaningful (no clean
#     "EBITDA" concept, and revenue is not comparable across lenders of
#     different balance-sheet sizes/funding mixes).
#   - Insurers: similar to banks in leaning on P/E and P/B, but book value
#     is somewhat less central than for banks (reserve accounting and
#     underwriting-cycle timing distort book value more), so P/E is
#     weighted a bit higher and P/B a bit lower than for banks.
#   - Software: highly asset-light and often unprofitable/low-margin while
#     scaling, so P/E is unreliable, P/B is nearly meaningless, and
#     EV/EBITDA and P/S (usable pre-profitability) dominate.
#   - Semiconductors: profitable but capital-intensive (fabs) and cyclical,
#     with earnings distorted across the cycle; EV/EBITDA is weighted most
#     heavily as the capital-structure- and cycle-neutral multiple, with
#     more weight on P/B than software given the heavier fixed-asset base.
#   - Industrials: capital-intensive, leverage varies across peers, so
#     EV/EBITDA (capital-structure neutral) is weighted most heavily.
#     P/B is more informative than for tech given heavier fixed-asset bases.
SECTOR_MULTIPLE_RELIABILITY_WEIGHTS = {
    "banks": {
        "P/E (trailing)": 0.25,
        "P/E (forward)": 0.25,
        "EV/EBITDA": 0.00,
        "P/B": 0.30,
        "P/S": 0.00,
        "ROE": 0.15,  # ROE is a standard bank profitability/quality metric
        "Debt/Equity": 0.00,  # leverage is inherent to the banking business model, not a valuation signal
        "Profit margin": 0.05,
    },
    "insurers": {
        "P/E (trailing)": 0.30,
        "P/E (forward)": 0.30,
        "EV/EBITDA": 0.00,
        "P/B": 0.20,
        "P/S": 0.00,
        "ROE": 0.15,
        "Debt/Equity": 0.00,  # underwriting leverage isn't comparable to industrial/operating leverage
        "Profit margin": 0.05,
    },
    "software": {
        "P/E (trailing)": 0.05,
        "P/E (forward)": 0.15,
        "EV/EBITDA": 0.30,
        "P/B": 0.05,
        "P/S": 0.30,
        "ROE": 0.05,  # can be distorted by low/negative book equity from buybacks; low weight
        "Debt/Equity": 0.00,  # asset-light, typically low and uninformative leverage
        "Profit margin": 0.10,
    },
    "semiconductors": {
        "P/E (trailing)": 0.15,
        "P/E (forward)": 0.20,
        "EV/EBITDA": 0.35,
        "P/B": 0.10,
        "P/S": 0.05,
        "ROE": 0.05,
        "Debt/Equity": 0.05,  # capital-intensive fabs make leverage more informative than for software
        "Profit margin": 0.05,
    },
    "industrials": {
        "P/E (trailing)": 0.15,
        "P/E (forward)": 0.20,
        "EV/EBITDA": 0.35,
        "P/B": 0.10,
        "P/S": 0.05,
        "ROE": 0.05,
        "Debt/Equity": 0.05,  # capital-intensive, leverage varies meaningfully across peers
        "Profit margin": 0.05,
    },
}

# Broad sector aliases, for callers who don't need sub-industry precision.
# Each points at a representative sub-industry preset above.
SECTOR_ALIASES = {
    "financials": "banks",
    "tech": "software",
}


def get_sector_weights(sector: str) -> dict[str, float]:
    """
    Look up a reliability-weight preset by sector/sub-industry label
    (case-insensitive).

    Checks SECTOR_MULTIPLE_RELIABILITY_WEIGHTS first (sub-industry presets
    like "banks", "software", "semiconductors"), then SECTOR_ALIASES (broad
    labels like "financials", "tech" that map to a representative
    sub-industry preset). Falls back to DEFAULT_MULTIPLE_RELIABILITY_WEIGHTS
    with a warning if the label is not recognized, rather than raising,
    since sector labels are a convenience layer on top of the more general
    `reliability_weights` param.
    """
    key = sector.lower()
    preset = SECTOR_MULTIPLE_RELIABILITY_WEIGHTS.get(key)
    if preset is None and key in SECTOR_ALIASES:
        preset = SECTOR_MULTIPLE_RELIABILITY_WEIGHTS.get(SECTOR_ALIASES[key])

    if preset is None:
        known = ", ".join(sorted(SECTOR_MULTIPLE_RELIABILITY_WEIGHTS)) + ", " + ", ".join(sorted(SECTOR_ALIASES))
        print(f"Warning: unrecognized sector '{sector}' (known: {known}). Falling back to default weights.")
        return DEFAULT_MULTIPLE_RELIABILITY_WEIGHTS
    return preset


def find_top_peers(
    ticker_symbol: str,
    n: int = 5,
    min_relative_weight: float = 0.05,
) -> list[str]:
    """
    Find the top `n` peer tickers for `ticker_symbol` by market weight,
    preferring its yfinance-reported industry (e.g. "consumer-electronics",
    "software-infrastructure") and falling back to the broader sector (e.g.
    "technology") when the industry doesn't have enough comparably-sized
    peers, excluding the ticker itself.

    "Comparably-sized" means a peer's market weight within the same
    classification is at least `min_relative_weight` times the target's own
    market weight (default 5%). This guards against industries dominated by
    a single company (e.g. Apple's "consumer-electronics" industry, where
    Apple holds ~99.9% of market weight and the remaining "top" names are
    tiny, unrelated companies) -- in that case the industry-level peer list
    is not meaningful, so the function falls back to the sector level, which
    draws from a larger pool of similarly-sized companies.

    Uses yfinance's `Industry.top_companies` / `Sector.top_companies`, which
    are already ranked by market weight, so the first `n` tickers (after
    removing the target and any too-small peers) are the largest comparable
    peers.

    Raises ValueError if neither the ticker's industry nor sector
    classification/peer data is available.
    """

    def _comparable_peers(top_companies: pd.DataFrame) -> list[str]:
        if ticker_symbol not in top_companies.index:
            # Target itself isn't in the ranked list (e.g. too small to be a
            # "top company" of its own sector) -- treat every listed peer as
            # comparable rather than discarding the whole classification.
            return [t for t in top_companies.index.tolist() if t != ticker_symbol]

        target_weight = top_companies.loc[ticker_symbol, "market weight"]
        weight_threshold = target_weight * min_relative_weight
        comparable = top_companies[top_companies["market weight"] >= weight_threshold]
        return [t for t in comparable.index.tolist() if t != ticker_symbol]

    info = yf.Ticker(ticker_symbol).info
    industry_key = info.get("industryKey")
    sector_key = info.get("sectorKey")

    industry_peers: list[str] = []
    if industry_key:
        industry_top_companies = yf.Industry(industry_key).top_companies
        if industry_top_companies is not None and not industry_top_companies.empty:
            industry_peers = _comparable_peers(industry_top_companies)

    if len(industry_peers) >= n:
        return industry_peers[:n]

    # Industry didn't yield enough comparably-sized peers -- fall back to
    # the broader sector, which draws from a larger pool of companies.
    if not sector_key:
        if industry_peers:
            return industry_peers
        raise ValueError(
            f"No industry or sector classification available for {ticker_symbol}; cannot find peers."
        )

    sector_top_companies = yf.Sector(sector_key).top_companies
    if sector_top_companies is None or sector_top_companies.empty:
        if industry_peers:
            return industry_peers
        raise ValueError(f"No peer companies found for {ticker_symbol}'s sector ({sector_key}).")

    sector_peers = _comparable_peers(sector_top_companies)

    # Prefer industry peers first (more precise comparables), then fill the
    # remainder with sector peers not already included.
    combined = industry_peers + [t for t in sector_peers if t not in industry_peers]
    if not combined:
        raise ValueError(
            f"No peers (other than {ticker_symbol} itself) found in industry {industry_key} or sector {sector_key}."
        )

    return combined[:n]


def _get_multiples(ticker_symbol: str) -> dict[str, float]:
    info = yf.Ticker(ticker_symbol).info
    return {multiple: info.get(multiple) for multiple in COMP_MULTIPLES}


def build_comps_table(ticker_symbol: str, peer_tickers: list[str]) -> pd.DataFrame:
    """
    Build a comparable-companies (comps) table of valuation multiples for a
    target ticker and a list of peer tickers.

    Returns a DataFrame indexed by ticker (target first, then peers) with one
    column per multiple in COMP_MULTIPLES. Missing values (multiple not
    reported by the data source, e.g. negative earnings for P/E) are left as
    NaN rather than dropped, so peers are not silently excluded.
    """
    all_tickers = [ticker_symbol] + [t for t in peer_tickers if t != ticker_symbol]
    data = {ticker: _get_multiples(ticker) for ticker in all_tickers}

    table = pd.DataFrame.from_dict(data, orient="index")
    table.columns = [_MULTIPLE_LABELS[col] for col in table.columns]
    table.index.name = "ticker"
    return table


_FUNDAMENTALS_LABELS = {
    "return_on_equity": "ROE",
    "debt_to_equity": "Debt/Equity",
    "profit_margin": "Profit margin",
}


def build_fundamentals_table(ticker_symbol: str, peer_tickers: list[str]) -> pd.DataFrame:
    """
    Build a table of profitability/leverage fundamentals (return on equity,
    debt-to-equity, profit margin) for a target ticker and its peers, for
    cross-peer comparison alongside the valuation-multiples comps table.

    Returns a DataFrame indexed by ticker (target first, then peers) with
    columns "ROE", "Debt/Equity", and "Profit margin". Missing values are
    left as NaN rather than dropped, consistent with `build_comps_table`.
    """
    all_tickers = [ticker_symbol] + [t for t in peer_tickers if t != ticker_symbol]
    data = {ticker: vars(compute_fundamentals(ticker)) for ticker in all_tickers}

    table = pd.DataFrame.from_dict(data, orient="index").drop(columns="ticker")
    table = table.rename(columns=_FUNDAMENTALS_LABELS)
    table.index.name = "ticker"
    return table


def comps_valuation(
    ticker_symbol: str,
    peer_tickers: list[str],
    aggregation: str = "median",
    include_fundamentals: bool = True,
) -> pd.DataFrame:
    """
    Compare a target ticker's valuation multiples (and, by default,
    profitability/leverage fundamentals) to a peer aggregate (median or mean
    by default) and flag whether it looks cheap or expensive on each metric.

    When `include_fundamentals` is True (default), ROE, Debt/Equity, and
    Profit margin from `build_fundamentals_table` are appended as additional
    rows alongside the price multiples, so `weighted_average_premium` can
    incorporate them via DEFAULT_MULTIPLE_RELIABILITY_WEIGHTS /
    SECTOR_MULTIPLE_RELIABILITY_WEIGHTS. Note that for Debt/Equity, a
    "premium" (target higher than peers) means more leverage, not a richer
    valuation -- it does not carry the same interpretation as the price
    multiples and the ROE/margin quality checks. Set `include_fundamentals`
    to False to restrict the comparison to price multiples only.

    Returns a DataFrame with one row per metric: the target's value, the
    peer aggregate, and the % premium/discount of the target vs. peers.
    For the price multiples and ROE/Profit margin, a positive value means
    the target trades at a premium (or is more profitable) than peers;
    negative means a discount (or less profitable). For Debt/Equity,
    positive means more levered than peers (see note above).
    """
    if aggregation not in ("median", "mean"):
        raise ValueError("aggregation must be 'median' or 'mean'.")

    table = build_comps_table(ticker_symbol, peer_tickers)
    if include_fundamentals:
        fundamentals = build_fundamentals_table(ticker_symbol, peer_tickers)
        table = table.join(fundamentals)

    target_row = table.loc[ticker_symbol]
    peer_rows = table.drop(index=ticker_symbol)

    peer_aggregate = peer_rows.median() if aggregation == "median" else peer_rows.mean()
    premium_discount = target_row / peer_aggregate - 1

    result = pd.DataFrame(
        {
            f"{ticker_symbol}": target_row,
            f"Peer {aggregation}": peer_aggregate,
            "Premium/(Discount) to Peers": premium_discount,
        }
    )
    result.index.name = "multiple"
    return result


def weighted_average_premium(
    valuation: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Compute a reliability-weighted average premium/discount from a
    `comps_valuation` result, instead of a simple unweighted mean.

    `weights` maps multiple label -> relative reliability weight (need not
    sum to 1; they are renormalized). Defaults to
    DEFAULT_MULTIPLE_RELIABILITY_WEIGHTS. Multiples with missing (NaN)
    premium/discount values are dropped and the remaining weights
    renormalized, so a missing multiple does not silently zero out its
    weight share.

    Raises ValueError if no multiples with both a value and a positive
    weight are available.
    """
    if weights is None:
        weights = DEFAULT_MULTIPLE_RELIABILITY_WEIGHTS

    premiums = valuation["Premium/(Discount) to Peers"]
    weight_series = pd.Series(weights).reindex(premiums.index).fillna(0.0)

    available = premiums.notna() & (weight_series > 0)
    if not available.any():
        raise ValueError("No multiples with both a valid value and a positive weight are available.")

    premiums = premiums[available]
    weight_series = weight_series[available]
    weight_series = weight_series / weight_series.sum()  # renormalize over available multiples

    return float((premiums * weight_series).sum())


def summarize_comps(
    ticker_symbol: str,
    peer_tickers: list[str],
    aggregation: str = "median",
    reliability_weights: dict[str, float] | None = None,
    sector: str | None = None,
) -> None:
    """
    Print the full comps table and the target-vs-peer premium/discount summary.

    `sector` is a convenience shortcut that looks up a preset from
    SECTOR_MULTIPLE_RELIABILITY_WEIGHTS (sub-industry labels like "banks",
    "insurers", "software", "semiconductors", "industrials") or
    SECTOR_ALIASES (broad labels like "financials", "tech"). If both
    `sector` and `reliability_weights` are given, `reliability_weights`
    takes precedence and `sector` is ignored.
    """
    if reliability_weights is None and sector is not None:
        reliability_weights = get_sector_weights(sector)

    table = build_comps_table(ticker_symbol, peer_tickers)
    valuation = comps_valuation(ticker_symbol, peer_tickers, aggregation=aggregation)

    print(f"Comps table ({ticker_symbol} vs. {len(peer_tickers)} peers):")
    print(table.round(2))

    print(f"\n{ticker_symbol} vs. peer {aggregation}:")
    print(valuation.round(3))

    valid = valuation["Premium/(Discount) to Peers"].dropna()
    if valid.empty:
        print("\nNo multiples available for comparison (all NaN).")
        return

    weighted_premium = weighted_average_premium(valuation, weights=reliability_weights)
    direction = "premium" if weighted_premium > 0 else "discount"
    weight_label = sector if (sector and reliability_weights is not None) else "custom/default"
    print(
        f"\nReliability-weighted average ({weight_label} weights) across available multiples: "
        f"{weighted_premium:+.1%} {direction} to peer {aggregation}."
    )


# ---------------------------------------------------------------------------
# Efficient portfolio construction (mean-variance optimization)
# ---------------------------------------------------------------------------


@dataclass
class PortfolioResult:
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float


def download_price_history(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers."""
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    return data.dropna(how="all")


def annualized_return_cov(prices: pd.DataFrame, trading_days: int = 252) -> tuple[pd.Series, pd.DataFrame]:
    """Compute annualized mean returns and covariance matrix from a price history."""
    daily_returns = prices.pct_change().dropna(how="all")
    mean_returns = daily_returns.mean() * trading_days
    cov_matrix = daily_returns.cov() * trading_days
    return mean_returns, cov_matrix


def _portfolio_performance(
    weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame
) -> tuple[float, float]:
    port_return = float(np.dot(weights, mean_returns))
    port_vol = float(np.sqrt(weights @ cov_matrix.to_numpy() @ weights))
    return port_return, port_vol


def _minimize_volatility(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    target_return: float | None,
    allow_short: bool,
) -> np.ndarray:
    n = len(mean_returns)
    init_guess = np.repeat(1 / n, n)
    bounds = None if allow_short else tuple((0.0, 1.0) for _ in range(n))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if target_return is not None:
        constraints.append(
            {"type": "eq", "fun": lambda w: _portfolio_performance(w, mean_returns, cov_matrix)[0] - target_return}
        )

    result = minimize(
        fun=lambda w: _portfolio_performance(w, mean_returns, cov_matrix)[1],
        x0=init_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result.x


def max_sharpe_portfolio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.04,
    allow_short: bool = False,
) -> PortfolioResult:
    """Find the portfolio on the efficient frontier with the highest Sharpe ratio."""
    n = len(mean_returns)
    init_guess = np.repeat(1 / n, n)
    bounds = None if allow_short else tuple((0.0, 1.0) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    def neg_sharpe(w: np.ndarray) -> float:
        port_return, port_vol = _portfolio_performance(w, mean_returns, cov_matrix)
        return -(port_return - risk_free_rate) / port_vol

    result = minimize(
        fun=neg_sharpe,
        x0=init_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    weights = result.x
    port_return, port_vol = _portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe = (port_return - risk_free_rate) / port_vol

    return PortfolioResult(
        weights=pd.Series(weights, index=mean_returns.index),
        expected_return=port_return,
        volatility=port_vol,
        sharpe_ratio=sharpe,
    )


def efficient_frontier(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_points: int = 25,
    allow_short: bool = False,
) -> pd.DataFrame:
    """Compute the minimum-volatility portfolio for a range of target returns."""
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), n_points)
    records = []
    for target in target_returns:
        weights = _minimize_volatility(mean_returns, cov_matrix, target_return=target, allow_short=allow_short)
        port_return, port_vol = _portfolio_performance(weights, mean_returns, cov_matrix)
        records.append({"target_return": target, "return": port_return, "volatility": port_vol})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------


def main() -> None:
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JNJ"]

    print("=== DCF Valuation ===")
    for symbol in tickers:
        try:
            result = run_dcf(symbol)
        except (ValueError, KeyError) as exc:
            print(f"{symbol}: could not value ({exc})")
            continue
        print(
            f"{symbol}: price=${result.current_price:.2f}  "
            f"intrinsic=${result.intrinsic_value_per_share:.2f}  "
            f"upside={result.upside_pct:.1%}"
        )

    print("\n=== DCF Sensitivity Analysis (AAPL) ===")
    discount_rates = [0.07, 0.08, 0.09, 0.10, 0.11]
    terminal_growth_rates = [0.015, 0.02, 0.025, 0.03]
    sensitivity = dcf_sensitivity("AAPL", discount_rates, terminal_growth_rates)
    summarize_sensitivity_flags(sensitivity, threshold=0.20)

    print("\n=== Comps vs. Peers (AAPL) ===")
    peer_tickers = ["MSFT", "GOOGL", "AMZN", "META"]
    summarize_comps("AAPL", peer_tickers, aggregation="median", sector="tech")

    print("\n=== Efficient Portfolio ===")
    prices = download_price_history(tickers, period="5y")
    mean_returns, cov_matrix = annualized_return_cov(prices)

    max_sharpe = max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate=0.04, allow_short=False)
    print("Max-Sharpe portfolio weights:")
    print(max_sharpe.weights.round(4))
    print(
        f"Expected return: {max_sharpe.expected_return:.2%}  "
        f"Volatility: {max_sharpe.volatility:.2%}  "
        f"Sharpe ratio: {max_sharpe.sharpe_ratio:.2f}"
    )

    frontier = efficient_frontier(mean_returns, cov_matrix, n_points=25, allow_short=False)
    print("\nEfficient frontier (target return -> volatility):")
    print(frontier.round(4))


if __name__ == "__main__":
    main()

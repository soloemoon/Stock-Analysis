from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from shiny import reactive, render
from shiny import ui as core_ui
from shiny.express import input, ui

from stock_valuation_portfolio import (
    DCFAssumptions,
    annualized_return_cov,
    build_comps_table,
    build_fundamentals_table,
    comps_valuation,
    compute_fundamentals,
    compute_owner_earnings,
    compute_stock_risk_profile,
    dcf_sensitivity,
    download_price_history,
    efficient_frontier,
    estimate_discount_rate,
    estimate_fcf_growth_rate,
    find_top_peers,
    flag_sensitivity_divergence,
    get_betas,
    get_sector_weights,
    get_treasury_yield,
    max_sharpe_portfolio,
    monthly_return_volatility,
    portfolio_beta,
    portfolio_monthly_volatility,
    run_dcf,
    weighted_average_premium,
)
from valuation_config import DEFAULT_CONFIG_PATH, load_config

# Load valuation_config.yaml (if present) to seed input defaults, so the
# dashboard's starting values stay in sync with the YAML config used by
# valuation_template.py. Falls back to the hardcoded defaults below if the
# config file is missing or invalid, so the app still works standalone.
try:
    _config = load_config(DEFAULT_CONFIG_PATH)
except (FileNotFoundError, ValueError) as _config_error:
    _config = None
    _config_load_error = str(_config_error)
else:
    _config_load_error = None


def _config_default(getter, fallback):
    """Pull a default value from the loaded config, or fall back if unavailable."""
    if _config is None:
        return fallback
    return getter(_config)


def _join_tickers(tickers: list[str]) -> str:
    return ", ".join(tickers)


def _join_floats(values: list[float]) -> str:
    return ", ".join(str(v) for v in values)

ui.page_opts(title="Stock Valuation & Portfolio Dashboard", fillable=True)

ui.tags.style(
    """
    .value-box .value-box-value {
        font-size: 0.85rem !important;
    }
    .value-box .value-box-title {
        font-size: 0.85rem !important;
    }
    .value-box .value-box-showcase {
        font-size: 0.85rem !important;
    }
    .value-box .value-box-area {
        display: flex !important;
        flex-direction: column !important;
        align-items: flex-start !important;
        justify-content: flex-start !important;
        gap: 0.25rem !important;
        overflow: hidden !important;
        white-space: normal !important;
    }
    .value-box .value-box-title,
    .value-box .value-box-value,
    .value-box .value-box-area > *:not(.value-box-title):not(.value-box-value) {
        position: static !important;
        white-space: normal !important;
        overflow-wrap: break-word !important;
        line-height: 1.3 !important;
    }
    """
)

with ui.sidebar(open="desktop", width=320):
    ui.input_text(
        "ticker",
        "Target ticker",
        value=_config_default(lambda c: c.target_ticker, "AAPL"),
    ).add_style("text-transform:uppercase;")
    ui.input_text(
        "peer_tickers",
        "Peer tickers (comma-separated)",
        value=_config_default(lambda c: _join_tickers(c.peer_tickers), "MSFT, GOOGL, AMZN, META"),
    )
    ui.input_action_button(
        "auto_peers", "Auto-populate top 5 peers", class_="btn-outline-secondary btn-sm"
    )

    @render.text
    def auto_peers_status():
        return auto_peers_message.get()

    ui.input_select(
        "sector",
        "Sector (comps weighting preset)",
        choices=[
            "none",
            "tech",
            "financials",
            "banks",
            "insurers",
            "software",
            "semiconductors",
            "industrials",
        ],
        selected=_config_default(lambda c: c.sector or "none", "tech"),
    )

    with ui.accordion(open=["DCF assumptions"]):
        with ui.accordion_panel("DCF assumptions"):
            ui.input_numeric(
                "projection_years",
                "Projection years",
                value=_config_default(lambda c: c.dcf_assumptions.projection_years, 5),
                min=1,
                max=15,
            )
            ui.input_numeric(
                "fcf_growth_rate",
                "FCF growth rate",
                value=_config_default(lambda c: c.dcf_assumptions.fcf_growth_rate, 0.06),
                step=0.01,
            )
            ui.input_numeric(
                "terminal_growth_rate",
                "Terminal growth rate",
                value=_config_default(lambda c: c.dcf_assumptions.terminal_growth_rate, 0.025),
                step=0.005,
            )
            ui.input_numeric(
                "discount_rate",
                "Discount rate (WACC)",
                value=_config_default(lambda c: c.dcf_assumptions.discount_rate, 0.09),
                step=0.005,
            )
            ui.input_select(
                "fcf_growth_method",
                "FCF growth estimation method",
                choices={
                    "cagr": "Endpoint CAGR",
                    "average_yoy": "Average year-over-year (smoothed)",
                },
                selected="average_yoy",
            )
            ui.input_action_button("auto_fill", "Auto-compute from yfinance", class_="btn-outline-secondary btn-sm")

            @render.text
            def auto_fill_status():
                return auto_fill_message.get()

        with ui.accordion_panel("Sensitivity grid"):
            ui.input_text(
                "sens_discount_rates",
                "Discount rates",
                value=_config_default(
                    lambda c: _join_floats(c.sensitivity_discount_rates), "0.07, 0.08, 0.09, 0.10, 0.11"
                ),
            )
            ui.input_text(
                "sens_terminal_growth",
                "Terminal growth rates",
                value=_config_default(
                    lambda c: _join_floats(c.sensitivity_terminal_growth_rates), "0.015, 0.02, 0.025, 0.03"
                ),
            )
            ui.input_numeric(
                "sens_threshold",
                "Flag threshold",
                value=_config_default(lambda c: c.sensitivity_flag_threshold, 0.20),
                step=0.05,
            )

        with ui.accordion_panel("Portfolio"):
            ui.input_text(
                "portfolio_tickers",
                "Portfolio candidate tickers (comma-separated)",
                value=_config_default(lambda c: _join_tickers(c.portfolio_tickers), "AAPL, MSFT, GOOGL, AMZN, JNJ"),
            )
            ui.input_select(
                "price_history_period",
                "Price history period",
                choices=["1y", "3y", "5y", "10y"],
                selected=_config_default(lambda c: c.price_history_period, "5y"),
            )
            ui.input_switch(
                "use_treasury_yield",
                "Base risk-free rate on current US Treasury yield",
                value=True,
            )
            ui.input_select(
                "treasury_maturity",
                "Treasury maturity",
                choices={
                    "13w": "13-week T-bill",
                    "5y": "5-year note",
                    "10y": "10-year note",
                    "30y": "30-year bond",
                },
                selected="10y",
            )
            ui.input_numeric(
                "risk_free_rate",
                "Risk-free rate (used when Treasury toggle is off)",
                value=_config_default(lambda c: c.risk_free_rate, 0.04),
                step=0.005,
            )
            ui.input_switch(
                "allow_short_selling",
                "Allow short selling",
                value=_config_default(lambda c: c.allow_short_selling, False),
            )
            ui.input_switch("annualize_volatility", "Show annualized volatility (\u03c3 \u00d7 \u221a12)", value=False)

    ui.input_action_button("run", "Run analysis", class_="btn-primary")

    if _config_load_error is not None:

        @render.text
        def config_load_warning():
            return f"Note: valuation_config.yaml could not be loaded ({_config_load_error}); using built-in defaults."


def _effective_risk_free_rate() -> float:
    """Current US Treasury yield if the toggle is on, else the manual risk-free-rate input."""
    if input.use_treasury_yield():
        return get_treasury_yield(input.treasury_maturity())
    return float(input.risk_free_rate())


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_ticker_list(text: str) -> list[str]:
    return [t.strip().upper() for t in text.split(",") if t.strip()]


@reactive.calc
@reactive.event(input.run)
def target_ticker() -> str:
    return input.ticker().strip().upper()


auto_fill_message: reactive.Value[str] = reactive.Value("")
auto_peers_message: reactive.Value[str] = reactive.Value("")


@reactive.effect
@reactive.event(input.auto_peers)
def _auto_populate_peers() -> None:
    symbol = input.ticker().strip().upper()
    if not symbol:
        auto_peers_message.set("Enter a ticker before auto-populating peers.")
        return

    try:
        peers = find_top_peers(symbol, n=5)
        ui.update_text("peer_tickers", value=", ".join(peers))
        auto_peers_message.set(f"Top peers for {symbol}: {', '.join(peers)}")
    except ValueError as exc:
        auto_peers_message.set(f"Could not auto-populate peers: {exc}")


@reactive.effect
@reactive.event(input.auto_fill)
def _auto_fill_dcf_inputs() -> None:
    symbol = input.ticker().strip().upper()
    if not symbol:
        auto_fill_message.set("Enter a ticker before auto-computing.")
        return

    messages = []
    risk_free_rate = _effective_risk_free_rate()
    messages.append(f"Risk-free rate used: {risk_free_rate:.2%}")

    try:
        wacc_estimate = estimate_discount_rate(symbol, risk_free_rate=risk_free_rate)
        ui.update_numeric("discount_rate", value=round(wacc_estimate.wacc, 4))
        messages.append(
            f"WACC \u2248 {wacc_estimate.wacc:.2%} "
            f"(beta={wacc_estimate.beta:.2f}, cost of equity={wacc_estimate.cost_of_equity:.2%}, "
            f"after-tax cost of debt={wacc_estimate.cost_of_debt:.2%})"
        )
    except ValueError as exc:
        messages.append(f"Could not estimate WACC: {exc}")

    try:
        fcf_estimate = estimate_fcf_growth_rate(symbol, method=input.fcf_growth_method())
        ui.update_numeric("fcf_growth_rate", value=round(fcf_estimate.growth_rate, 4))
        method_label = "average YoY" if fcf_estimate.method == "average_yoy" else "CAGR"
        messages.append(
            f"FCF growth \u2248 {fcf_estimate.growth_rate:.2%} ({method_label} over {fcf_estimate.years} year(s); "
            f"per-year growth: {', '.join(f'{g:+.1%}' for g in fcf_estimate.yoy_growth_rates)})"
        )
    except ValueError as exc:
        messages.append(f"Could not estimate FCF growth: {exc}")

    auto_fill_message.set(" | ".join(messages))


@reactive.calc
@reactive.event(input.run)
def dcf_assumptions() -> DCFAssumptions:
    return DCFAssumptions(
        projection_years=int(input.projection_years()),
        fcf_growth_rate=float(input.fcf_growth_rate()),
        terminal_growth_rate=float(input.terminal_growth_rate()),
        discount_rate=float(input.discount_rate()),
    )


@reactive.calc
@reactive.event(input.run)
def dcf_result():
    return run_dcf(target_ticker(), dcf_assumptions())


@reactive.calc
@reactive.event(input.run)
def sensitivity_result() -> pd.DataFrame:
    discount_rates = _parse_float_list(input.sens_discount_rates())
    terminal_growth_rates = _parse_float_list(input.sens_terminal_growth())
    return dcf_sensitivity(
        target_ticker(),
        discount_rates,
        terminal_growth_rates,
        base_assumptions=dcf_assumptions(),
    )


@reactive.calc
@reactive.event(input.run)
def comps_result():
    peers = _parse_ticker_list(input.peer_tickers())
    raw_table = build_comps_table(target_ticker(), peers)
    fundamentals = build_fundamentals_table(target_ticker(), peers)
    combined_table = raw_table.join(fundamentals)
    valuation = comps_valuation(target_ticker(), peers, aggregation="median")
    sector = None if input.sector() == "none" else input.sector()
    weights = get_sector_weights(sector) if sector else None
    weighted_premium = None
    if weights is not None:
        try:
            weighted_premium = weighted_average_premium(valuation, weights=weights)
        except ValueError:
            weighted_premium = None
    return combined_table, valuation, weighted_premium, sector


@reactive.calc
@reactive.event(input.run)
def portfolio_result():
    tickers = _parse_ticker_list(input.portfolio_tickers())
    prices = download_price_history(tickers, period=input.price_history_period())
    mean_returns, cov_matrix = annualized_return_cov(prices)
    max_sharpe = max_sharpe_portfolio(
        mean_returns,
        cov_matrix,
        risk_free_rate=_effective_risk_free_rate(),
        allow_short=input.allow_short_selling(),
    )
    frontier = efficient_frontier(
        mean_returns,
        cov_matrix,
        n_points=25,
        allow_short=input.allow_short_selling(),
    )
    betas = get_betas(tickers)
    monthly_vol = monthly_return_volatility(prices)

    try:
        port_beta = portfolio_beta(max_sharpe.weights, betas)
    except ValueError:
        port_beta = None
    port_monthly_vol = portfolio_monthly_volatility(prices, max_sharpe.weights)

    return max_sharpe, frontier, betas, monthly_vol, port_beta, port_monthly_vol


@reactive.calc
@reactive.event(input.run)
def risk_profile():
    return compute_stock_risk_profile(target_ticker(), period=input.price_history_period())


@reactive.calc
@reactive.event(input.run)
def owner_earnings_result():
    return compute_owner_earnings(target_ticker())


@reactive.calc
@reactive.event(input.run)
def fundamentals_result():
    return compute_fundamentals(target_ticker())


with ui.layout_column_wrap(width=1 / 3, fill=False):

    @render.ui
    def price_box():
        result = dcf_result()
        return core_ui.value_box(
            "Current price",
            f"${result.current_price:.2f}",
            theme="primary",
            height="150px",
        )

    @render.ui
    def intrinsic_box():
        result = dcf_result()
        return core_ui.value_box(
            "Intrinsic value / share",
            f"${result.intrinsic_value_per_share:.2f}",
            theme="primary",
            height="150px",
        )

    @render.ui
    def upside_box():
        result = dcf_result()
        return core_ui.value_box(
            "Upside vs. current price",
            f"{result.upside_pct:+.1%}",
            theme="primary",
            height="150px",
        )

with ui.layout_column_wrap(width=1 / 3, fill=False):

    @render.ui
    def risk_box():
        profile = risk_profile()
        beta_text = f"{profile.beta:.2f}" if profile.beta is not None else "n/a"
        vol = profile.monthly_return_std
        vol_label = "monthly"
        if input.annualize_volatility():
            vol = vol * (12**0.5)
            vol_label = "annualized"
        return core_ui.value_box(
            "Beta",
            f"\u03b2 {beta_text}",
            f"\u03c3({vol_label}) {vol:.2%} (n={profile.n_months})",
            theme="primary",
            height="150px",
        )

    @render.ui
    def owner_earnings_box():
        result = owner_earnings_result()
        return core_ui.value_box(
            "Owner Earnings",
            f"${result.owner_earnings/1000000:,.0f}M",
            theme="primary",
            height="150px",
        )

    @render.ui
    def fundamentals_box():
        result = fundamentals_result()
        roe_text = f"{result.return_on_equity:.1%}" if result.return_on_equity is not None else "n/a"
        dte_text = f"{result.debt_to_equity:.1f}" if result.debt_to_equity is not None else "n/a"
        margin_text = f"{result.profit_margin:.1%}" if result.profit_margin is not None else "n/a"
        return core_ui.value_box(
            "Financial Metrics",
            f"ROE {roe_text}",
            f"D/E {dte_text}  |  Margin {margin_text}",
            theme="primary",
            height="150px",
        )


with ui.navset_card_underline(title="Analysis"):

    with ui.nav_panel("DCF Sensitivity"):

        @render.data_frame
        def sensitivity_table():
            sensitivity = sensitivity_result()
            return render.DataGrid(sensitivity.round(2).reset_index())

        @render.data_frame
        def sensitivity_flagged_table():
            sensitivity = sensitivity_result()
            flagged = flag_sensitivity_divergence(sensitivity, threshold=float(input.sens_threshold()))
            return render.DataGrid(flagged.reset_index())

    with ui.nav_panel("Comps vs. Peers"):

        @render.data_frame
        def comps_table():
            combined_table, _valuation, _weighted_premium, _sector = comps_result()
            return render.DataGrid(combined_table.round(3).reset_index())

        @render.text
        def comps_summary():
            _combined_table, _valuation, weighted_premium, sector = comps_result()
            if weighted_premium is None:
                return "No reliability-weighted summary available (missing weights or all multiples NaN)."
            direction = "premium" if weighted_premium > 0 else "discount"
            label = sector if sector else "custom/default"
            return (
                f"Reliability-weighted average ({label} weights) across available multiples: "
                f"{weighted_premium:+.1%} {direction} to peer median."
            )

    with ui.nav_panel("Efficient Portfolio"):

        @render.data_frame
        def portfolio_weights_table():
            max_sharpe, _frontier, betas, monthly_vol, _port_beta, _port_monthly_vol = portfolio_result()
            table = (
                max_sharpe.weights.round(4)
                .rename_axis("ticker")
                .rename("weight")
                .reset_index()
                .set_index("ticker")
            )
            table["beta"] = betas.round(2)
            if input.annualize_volatility():
                table["annualized_return_std"] = (monthly_vol * (12**0.5)).round(4)
            else:
                table["monthly_return_std"] = monthly_vol.round(4)
            return render.DataGrid(table.reset_index())

        @render.text
        def portfolio_risk_text():
            _max_sharpe, _frontier, _betas, _monthly_vol, port_beta, port_monthly_vol = portfolio_result()
            beta_text = f"{port_beta:.2f}" if port_beta is not None else "n/a"
            vol = port_monthly_vol
            vol_label = "monthly"
            if input.annualize_volatility():
                vol = vol * (12**0.5)
                vol_label = "annualized"
            return (
                f"Portfolio beta (max-Sharpe weights): \u03b2 {beta_text}  |  "
                f"{vol_label.capitalize()} volatility: \u03c3({vol_label}) {vol:.2%}"
            )

        @render.text
        def portfolio_summary():
            max_sharpe, _frontier, _betas, _monthly_vol, _port_beta, _port_monthly_vol = portfolio_result()
            return (
                f"Expected return: {max_sharpe.expected_return:.2%}  "
                f"Volatility: {max_sharpe.volatility:.2%}  "
                f"Sharpe ratio: {max_sharpe.sharpe_ratio:.2f}"
            )

        @render.plot
        def frontier_plot():
            _max_sharpe, frontier, _betas, _monthly_vol, _port_beta, _port_monthly_vol = portfolio_result()
            fig, ax = plt.subplots()
            ax.plot(frontier["volatility"], frontier["return"], marker="o", markersize=3)
            ax.set_xlabel("Volatility")
            ax.set_ylabel("Expected return")
            ax.set_title("Efficient Frontier")
            return fig

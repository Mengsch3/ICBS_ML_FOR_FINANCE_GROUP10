import pandas as pd
from pathlib import Path
from typing import TypeAlias


EXCLUDED_INSTRUMENT_IDS = {643}  # AKR: corrupted OHLC and market-cap history
PathLike: TypeAlias = str | Path


def load_prices(data_dir: PathLike, cutoff_date: str = "2024-12-31") -> pd.DataFrame:
    """Return the sorted price panel through cutoff_date, excluding corrupted instruments."""
    file_path = Path(data_dir) / "prices.parquet"

    columns = [
        "ticker",
        "instrument_id",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "market_cap",
        "status",
    ]

    df = pd.read_parquet(file_path, columns=columns)

    df["date"] = pd.to_datetime(df["date"])
    cutoff_date = pd.Timestamp(cutoff_date)
    df = df[df["date"] <= cutoff_date]

    # Exclude AKR because its historical OHLC and market-cap observations contain unrecoverable data corruption.
    df = df[~df["instrument_id"].isin(EXCLUDED_INSTRUMENT_IDS)].copy()

    price = df.drop_duplicates(["instrument_id", "date"], keep="last")

    price = price.sort_values(["instrument_id", "date"]).reset_index(drop=True)

    return price


def add_adjusted_prices(price: pd.DataFrame) -> pd.DataFrame:
    """Return price with split-adjusted OHLC fields and raw dollar volume."""
    price = price.copy()

    price["adjustment_factor"] = price["adjusted_close"] / price["close"]

    price["adj_open"] = price["adjustment_factor"] * price["open"]
    price["adj_close"] = price["adjusted_close"]
    price["adj_high"] = price["adjustment_factor"] * price["high"]
    price["adj_low"] = price["adjustment_factor"] * price["low"]

    price['raw_dollar_volume'] = price["close"] * price["volume"]

    price["valid_price_row"] = (
        price["adj_open"].notna()
        & price["adj_close"].notna()
        & (price["adj_open"] > 0)
        & (price["adj_close"] > 0)
        & price["volume"].notna()
        & (price["volume"] >= 0)
    )

    return price


def add_return_objects(price: pd.DataFrame) -> pd.DataFrame:
    """Return price with overnight, intraday, close-to-close, and next-overnight returns."""
    price = price.copy()

    grouped = price.groupby("instrument_id", sort=False)

    price["prev_adj_close"] = grouped["adj_close"].shift(1)
    # Buy at the previous close and sell at today's open.
    price["r_overnight"] = price["adj_open"] / price["prev_adj_close"] - 1
    # Buy at today's open and sell at today's close.
    price["r_intraday"] = price["adj_close"] / price["adj_open"] - 1
    # Buy at the previous close and sell at today's close.
    price["r_close_to_close"] = price["adj_close"] / price["prev_adj_close"] - 1

    price["target_r_overnight_next"] = grouped["r_overnight"].shift(-1)
    return price


def check_return_reconciliation(price: pd.DataFrame, tolerance: float = 1e-6) -> pd.DataFrame:
    """Return price with diagnostics verifying that overnight and intraday returns reconcile."""
    price = price.copy()
    reconstructed = (1 + price["r_overnight"]) * (1 + price["r_intraday"]) - 1
    price["return_reconciliation_residual"] = (
            reconstructed - price["r_close_to_close"]
    ).abs()

    price["return_reconciliation_ok"] = (
            price["return_reconciliation_residual"].isna()
            | (price["return_reconciliation_residual"] <= tolerance)
    )
    return price


def build_yearly_universe(
    price: pd.DataFrame,
    start_year: int = 2010,
    end_year: int = 2024,
    universe_size: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the daily panel plus a fixed annual market-cap universe and summaries."""
    price = price.copy()

    first_date_by_stock = price.groupby("instrument_id")["date"].min()

    universe_rows = []
    summary_rows = []

    # Rank each year from the previous year's final trading-day snapshot.
        # Use the previous year's final available trading date as the ranking date.
    # Read point-in-time short-interest history.
    # Match each price row to the latest strictly prior short-interest release for the same instrument.
    # The day t open is known by 15:50 ET and can enter the day t signal.
    for year in range(start_year, end_year + 1):
    # These returns depend on the day t close, so they must be lagged by one day.
        previous_year_dates = price[price["date"].dt.year == year - 1]["date"]
        rank_date = previous_year_dates.max()

        snapshot = price[price["date"] == rank_date].copy()

        snapshot = snapshot[
            snapshot["market_cap"].notna()
            & snapshot["valid_price_row"]
            ]
        snapshot = snapshot[snapshot["status"].astype(str) == "1"]

        history_start = rank_date - pd.DateOffset(years=1)

        snapshot = snapshot[
            snapshot["instrument_id"].map(first_date_by_stock) <= history_start
            ]

        selected = snapshot.sort_values("market_cap", ascending=False).head(universe_size).copy()

        selected["universe_rank"] = range(1, len(selected) + 1)
        selected["universe_year"] = year
        selected["rank_date"] = rank_date

        selected = selected[
            [
                "instrument_id",
                "ticker",
                "universe_year",
                "rank_date",
                "universe_rank",
                "market_cap",
                "status",
            ]
        ]

        universe_rows.append(selected)
        summary_rows.append({
            "year": year,
            "rank_date": rank_date,
            "eligible_after_history_filter": len(snapshot),
            "selected_names": len(selected)
        })

    universe_table = pd.concat(universe_rows, ignore_index=True)
    universe_summary = pd.DataFrame(summary_rows)

    price["universe_year"] = price["date"].dt.year

    price = price.merge(
        universe_table[["instrument_id", "universe_year", "universe_rank"]],
        on=["instrument_id", "universe_year"],
        how="left"
    )
    price["yearly_universe_member"] = price["universe_rank"].notna()

    return price, universe_summary, universe_table


def add_earnings_features(price: pd.DataFrame, data_dir: PathLike) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return price with BMO/AMC-adjusted earnings flags and the raw earnings table."""
    trading_dates = (price["date"].drop_duplicates().sort_values().reset_index(drop=True))
    next_trading_date = dict(zip(trading_dates[:-1], trading_dates[1:]))

    file_path = Path(data_dir) / "earnings_calendar.parquet"
    earnings = pd.read_parquet(file_path)
    earnings = earnings.rename(columns={"stock_id": "instrument_id"})
    earnings["reporting_date"] = pd.to_datetime(earnings["reporting_date"])
    earnings["before_after_market"] = earnings["before_after_market"].str.lower()

    earnings["earnings_effective_date"] = earnings["reporting_date"]
    after_mask = earnings["before_after_market"] == "after"

    earnings.loc[after_mask, "earnings_effective_date"] = (
        earnings.loc[after_mask, "reporting_date"].map(next_trading_date)
    )

    unknown_mask = earnings["before_after_market"].isna()

    earnings.loc[unknown_mask, "earnings_effective_date"] = (
        earnings.loc[unknown_mask, "reporting_date"].map(next_trading_date)
    )

    daily_earnings = (
        earnings
        .groupby(["instrument_id", "earnings_effective_date"])
        .size()
        .reset_index(name="earnings_event_count")
    )

    daily_earnings = daily_earnings.rename(
        columns={"earnings_effective_date": "date"}
    )

    price = price.merge(
        daily_earnings,
        on=["instrument_id", "date"],
        how="left"
    )

    price["earnings_event_count"] = (
        price["earnings_event_count"]
        .fillna(0)
        .astype(int))

    price["is_earnings_effective_date"] = price["earnings_event_count"] > 0

    return price, earnings

def add_short_interest_features(price: pd.DataFrame, data_dir: PathLike) -> pd.DataFrame:
    """Return price merged to short-interest data with a strict two-day availability lag."""
    # Volume and dollar volume are end-of-day data, so they are also lagged by one day.
    file_path = Path(data_dir) / "short_interest_transfo.parquet"
    short_interest = pd.read_parquet(file_path)

    short_interest = short_interest.rename(columns={
        "stock_id": "instrument_id",
        "date": "short_interest_available_date",
        "dsi": "short_dsi",
        "dtcn": "short_days_to_cover",
        "ddtcn": "short_days_to_cover_chg",
    })

    short_interest["short_interest_available_date"] = (
        pd.to_datetime(short_interest["short_interest_available_date"])
        + pd.Timedelta(days=2)
    )

    price_sorted = price.sort_values(["date", "instrument_id"]).reset_index(drop=True)

    short_interest = short_interest.sort_values([
        "short_interest_available_date",
        "instrument_id"
    ]).reset_index(drop=True)

    # Earnings effective dates already reflect the BMO/AMC convention.
    price_merged = pd.merge_asof(
        price_sorted,
        short_interest,
        by="instrument_id",
        left_on="date",
        right_on="short_interest_available_date",
        direction="backward",
        allow_exact_matches=False,
    )

    price_merged = price_merged.sort_values(
        ["instrument_id", "date"]
    ).reset_index(drop=True)

    return price_merged


def add_gics_and_regime(price: pd.DataFrame, data_dir: PathLike) -> pd.DataFrame:
    """Return price enriched with GICS and daily regime fields."""
    gics_path = Path(data_dir) / "gics_info.parquet"
    gics = pd.read_parquet(gics_path)

    price = price.merge(
        gics,
        on="instrument_id",
        how="left"
    )

    regime_path = Path(data_dir) / "regime.parquet"
    regime = pd.read_parquet(regime_path)

    regime = regime.drop(columns=["Unnamed: 0"], errors="ignore")

    regime["date"] = pd.to_datetime(regime["date"])

    price = price.merge(
        regime,
        on="date",
        how="left"
    )

    return price


def add_point_in_time_features(price: pd.DataFrame) -> pd.DataFrame:
    """Return features observable by 15:50 ET on day t; close-based fields are lagged."""

    price = price.copy()
    price = price.sort_values(["instrument_id", "date"]).reset_index(drop=True)

    grouped = price.groupby("instrument_id", sort=False)

    # Short-interest fields already use a strict lagged merge_asof join.
    price["feature_r_overnight_t"] = price["r_overnight"]

    # Non-ASCII comment replaced with English documentation.
    price["feature_r_intraday_lag1"] = grouped["r_intraday"].shift(1)
    price["feature_r_close_to_close_lag1"] = grouped["r_close_to_close"].shift(1)

    # Non-ASCII comment replaced with English documentation.
    price["feature_dollar_volume_lag1"] = grouped["raw_dollar_volume"].shift(1)

    shifted_dollar_volume = grouped["raw_dollar_volume"].shift(1)
    price["feature_adv20_lag1"] = (
        shifted_dollar_volume
        .groupby(price["instrument_id"])
        .rolling(20, min_periods=10)
        .mean()
        .reset_index(level=0, drop=True)
    )

    shifted_close_to_close = grouped["r_close_to_close"].shift(1)
    price["feature_vol20_lag1"] = (
        shifted_close_to_close
        .groupby(price["instrument_id"])
        .rolling(20, min_periods=10)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Non-ASCII comment replaced with English documentation.
    price["feature_earnings_today"] = price["is_earnings_effective_date"].astype(int)

    # Non-ASCII comment replaced with English documentation.
    price["feature_short_dsi"] = price["short_dsi"]
    price["feature_short_days_to_cover"] = price["short_days_to_cover"]
    price["feature_short_days_to_cover_chg"] = price["short_days_to_cover_chg"]

    return price


def build_stylized_fact_table(
    price: pd.DataFrame,
    start_date: str = "2010-01-01",
    cutoff_date: str = "2024-12-31",
) -> pd.DataFrame:
    """Return equal-weighted daily return streams and cumulative growth diagnostics."""
    df = price[
        (price["date"] >= pd.Timestamp(start_date))
        & (price["date"] <= pd.Timestamp(cutoff_date))
        & (price["yearly_universe_member"])
        & price[["r_overnight", "r_intraday", "r_close_to_close"]].notna().all(axis=1)
        ].copy()

    daily = (
        df.groupby("date")
        .agg(
            n_names=("instrument_id", "nunique"),
            r_overnight=("r_overnight", "mean"),
            r_intraday=("r_intraday", "mean"),
            r_close_to_close=("r_close_to_close", "mean"),
        )
        .reset_index()
        .sort_values("date")
    )

    daily["growth_r_overnight"] = (1 + daily["r_overnight"]).cumprod()
    daily["growth_r_intraday"] = (1 + daily["r_intraday"]).cumprod()
    daily["growth_r_close_to_close"] = (1 + daily["r_close_to_close"]).cumprod()

    return daily


def build_universe_evolution_table(
    price: pd.DataFrame,
    universe_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return annual universe evolution diagnostics and detail tables."""
    price = price.copy()
    universe_table = universe_table.copy()

    price["date"] = pd.to_datetime(price["date"])
    price["universe_year"] = price["date"].dt.year

    universe_table["rank_date"] = pd.to_datetime(universe_table["rank_date"])
    universe_table["universe_year"] = universe_table["universe_year"].astype(int)

    # -----------------------------------------------------
    # 1. Selected names at year-start
    # -----------------------------------------------------
    selected_summary = (
        universe_table.groupby("universe_year", as_index=False)
        .agg(
            rank_date=("rank_date", "first"),
            eligible_names_at_year_start=("instrument_id", "nunique")
        )
    )

    # -----------------------------------------------------
    # 2. Observed selected names in the daily panel
    # -----------------------------------------------------
    observed = (
        price[price["yearly_universe_member"]]
        .groupby(["universe_year", "instrument_id"], as_index=False)
        .agg(
            ticker=("ticker", "last"),
            first_observed_date=("date", "min"),
            last_observed_date=("date", "max"),
            n_daily_rows=("date", "size"),
        )
    )

    observed_summary = (
        observed.groupby("universe_year", as_index=False)
        .agg(
            observed_names_in_daily_panel=("instrument_id", "nunique")
        )
    )

    # -----------------------------------------------------
    # 3. Selected but not observed in that year's daily panel
    # -----------------------------------------------------
    selected_vs_observed = universe_table.merge(
        observed,
        on=["universe_year", "instrument_id"],
        how="left",
        indicator=True,
        suffixes=("_selected", "_observed")
    )

    selected_vs_observed["observed_in_daily_panel"] = (
        selected_vs_observed["_merge"] == "both"
    )

    missing_selected = selected_vs_observed[
        ~selected_vs_observed["observed_in_daily_panel"]
    ].copy()

    missing_summary = (
        selected_vs_observed.groupby("universe_year", as_index=False)
        .agg(
            selected_names=("instrument_id", "nunique"),
            observed_names=("observed_in_daily_panel", "sum")
        )
    )

    missing_summary["selected_but_unobserved"] = (
        missing_summary["selected_names"] - missing_summary["observed_names"]
    )

    # -----------------------------------------------------
    # 4. Mid-year exits among observed selected names
    # -----------------------------------------------------
    year_end_dates = (
        price.groupby("universe_year", as_index=False)
        .agg(year_end_date=("date", "max"))
    )

    observed_with_year_end = observed.merge(
        year_end_dates,
        on="universe_year",
        how="left"
    )

    observed_with_year_end["mid_year_exit"] = (
        observed_with_year_end["last_observed_date"]
        < observed_with_year_end["year_end_date"]
    )

    mid_year_exit_summary = (
        observed_with_year_end.groupby("universe_year", as_index=False)
        .agg(
            mid_year_exits=("mid_year_exit", "sum")
        )
    )

    mid_year_exit_details = observed_with_year_end[
        observed_with_year_end["mid_year_exit"]
    ].copy()

    # -----------------------------------------------------
    # 5. Final summary table
    # -----------------------------------------------------
    universe_evolution = (
        selected_summary
        .merge(observed_summary, on="universe_year", how="left")
        .merge(
            missing_summary[
                ["universe_year", "selected_but_unobserved"]
            ],
            on="universe_year",
            how="left"
        )
        .merge(mid_year_exit_summary, on="universe_year", how="left")
    )

    universe_evolution = universe_evolution.rename(
        columns={"universe_year": "year"}
    )

    universe_evolution = universe_evolution[
        [
            "year",
            "rank_date",
            "eligible_names_at_year_start",
            "observed_names_in_daily_panel",
            "selected_but_unobserved",
            "mid_year_exits",
        ]
    ]

    return (
        universe_evolution,
        selected_vs_observed,
        missing_selected,
        mid_year_exit_details,
    )


def save_stylized_fact_diagnostics(
    stylized: pd.DataFrame,
    output_dir: PathLike = "Output",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write stylized-fact figures and CSV diagnostics; returns annual and summary tables."""
    output_dir = Path(output_dir)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    stylized = stylized.copy()
    stylized["date"] = pd.to_datetime(stylized["date"])
    stylized["year"] = stylized["date"].dt.year

    # -----------------------------------------------------
    # 1. Save cumulative-growth figure
    # -----------------------------------------------------
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    plt.plot(
        stylized["date"],
        stylized["growth_r_overnight"],
        label="Overnight close-to-open"
    )

    plt.plot(
        stylized["date"],
        stylized["growth_r_intraday"],
        label="Intraday open-to-close"
    )

    plt.plot(
        stylized["date"],
        stylized["growth_r_close_to_close"],
        label="Total close-to-close"
    )

    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.title("Equal-weighted eligible universe: return decomposition")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        diagnostics_dir / "step1_stylized_fact_equal_weighted.png",
        dpi=300
    )
    plt.close()

    # -----------------------------------------------------
    # 2. Annual compounded returns
    # -----------------------------------------------------
    annual_returns = (
        stylized.groupby("year")
        .agg(
            n_days=("date", "size"),
            avg_n_names=("n_names", "mean"),
            annual_overnight=(
                "r_overnight",
                lambda x: (1 + x).prod() - 1
            ),
            annual_intraday=(
                "r_intraday",
                lambda x: (1 + x).prod() - 1
            ),
            annual_close_to_close=(
                "r_close_to_close",
                lambda x: (1 + x).prod() - 1
            ),
        )
        .reset_index()
    )

    annual_returns.to_csv(
        diagnostics_dir / "step1_stylized_fact_annual_returns.csv",
        index=False
    )

    # -----------------------------------------------------
    # 3. Dispersion summary across years
    # -----------------------------------------------------
    return_cols = [
        "annual_overnight",
        "annual_intraday",
        "annual_close_to_close",
    ]

    dispersion = (
        annual_returns[return_cols]
        .agg(["mean", "std", "min", "median", "max"])
        .T
        .reset_index()
        .rename(columns={"index": "return_stream"})
    )

    dispersion.to_csv(
        diagnostics_dir / "step1_stylized_fact_dispersion.csv",
        index=False
    )

    # -----------------------------------------------------
    # 4. Additional comparison diagnostics
    # -----------------------------------------------------
    comparison = pd.DataFrame([{
        "final_growth_overnight": stylized["growth_r_overnight"].iloc[-1],
        "final_growth_intraday": stylized["growth_r_intraday"].iloc[-1],
        "final_growth_close_to_close": stylized["growth_r_close_to_close"].iloc[-1],
        "years_overnight_gt_intraday": (
            annual_returns["annual_overnight"]
            > annual_returns["annual_intraday"]
        ).sum(),
        "total_years": annual_returns["year"].nunique(),
        "mean_annual_overnight": annual_returns["annual_overnight"].mean(),
        "mean_annual_intraday": annual_returns["annual_intraday"].mean(),
        "mean_annual_close_to_close": annual_returns["annual_close_to_close"].mean(),
        "sd_annual_overnight": annual_returns["annual_overnight"].std(),
        "sd_annual_intraday": annual_returns["annual_intraday"].std(),
        "sd_annual_close_to_close": annual_returns["annual_close_to_close"].std(),
    }])

    comparison.to_csv(
        diagnostics_dir / "step1_stylized_fact_comparison_summary.csv",
        index=False
    )

    return annual_returns, dispersion, comparison


def save_outputs(
    price: pd.DataFrame,
    stylized: pd.DataFrame,
    universe_summary: pd.DataFrame,
    universe_table: pd.DataFrame,
    output_dir: PathLike = "Output",
) -> None:
    """Write Step 1 parquet and diagnostics to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    price.to_parquet(output_dir / "step1_daily_panel.parquet", index=False)
    stylized.to_csv(diagnostics_dir / "step1_stylized_fact_daily.csv", index=False)
    universe_summary.to_csv(
        diagnostics_dir / "step1_yearly_universe_summary.csv",
        index=False
    )

    universe_table.to_csv(
        diagnostics_dir / "step1_yearly_universe_table.csv",
        index=False
    )

    recon_summary = pd.DataFrame([{
        "stock_days": len(price),
        "failed_days": (~price["return_reconciliation_ok"]).sum(),
        "failure_fraction": (~price["return_reconciliation_ok"]).mean(),
        "max_residual": price["return_reconciliation_residual"].max(),
    }])

    recon_summary.to_csv(
        diagnostics_dir / "step1_return_reconciliation_summary.csv",
        index=False
    )

    (
        universe_evolution,
        selected_vs_observed,
        missing_selected,
        mid_year_exit_details,
    ) = build_universe_evolution_table(price, universe_table)

    universe_evolution.to_csv(
        diagnostics_dir / "step1_yearly_universe_evolution.csv",
        index=False
    )

    selected_vs_observed.to_csv(
        diagnostics_dir / "step1_selected_vs_observed.csv",
        index=False
    )

    missing_selected.to_csv(
        diagnostics_dir / "step1_selected_but_unobserved_names.csv",
        index=False
    )

    mid_year_exit_details.to_csv(
        diagnostics_dir / "step1_mid_year_exit_details.csv",
        index=False
    )

    save_stylized_fact_diagnostics(stylized, output_dir)

if __name__ == "__main__":
    data_dir = "../Data"
    output_dir = "../Output"

    prices = load_prices(data_dir)
    prices = add_adjusted_prices(prices)
    prices = add_return_objects(prices)
    prices = check_return_reconciliation(prices)

    prices, universe_summary, universe_table = build_yearly_universe(prices)

    prices, earnings = add_earnings_features(prices, data_dir)
    prices = add_short_interest_features(prices, data_dir)
    prices = add_gics_and_regime(prices, data_dir)
    prices = add_point_in_time_features(prices)

    stylized = build_stylized_fact_table(prices)

    save_outputs(prices, stylized, universe_summary, universe_table, output_dir)

    print((Path(output_dir) / "step1_daily_panel.parquet").exists())
    print((Path(output_dir) / "diagnostics" / "step1_stylized_fact_daily.csv").exists())
    print((Path(output_dir) / "diagnostics" / "step1_yearly_universe_summary.csv").exists())

    print("Done")

import pandas as pd
from pathlib import Path
from collections.abc import Sequence
from typing import TypeAlias
import warnings
warnings.filterwarnings("ignore")

PathLike: TypeAlias = str | Path


def load_step1_panel(input_path: PathLike) -> pd.DataFrame:
    """Return the Step 1 daily panel sorted by instrument and date."""
    price = pd.read_parquet(input_path)
    price["date"] = pd.to_datetime(price["date"])
    price = price.drop_duplicates(["instrument_id", "date"], keep="last")
    price = price.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    return price


def add_capacity_base_columns(price: pd.DataFrame, universe_table_path: PathLike) -> pd.DataFrame:
    """Return price with lagged price, volatility, ADV, and year-start market-cap capacity fields."""
    price = price.copy()
    price = price.sort_values(["instrument_id", "date"]).reset_index(drop=True)

    grouped = price.groupby("instrument_id", sort=False)

    # Lagged price observable before the day-t close decision
    price["feature_price_lag1"] = grouped["adj_close"].shift(1)

    # Annualised realised volatility from lagged 20-day close-to-close volatility
    price["feature_vol20_annualized_lag1"] = (
        price["feature_vol20_lag1"] * (252 ** 0.5)
    )

    # Year-start / previous-year-end market cap from Step 1 universe table
    universe_table = pd.read_csv(universe_table_path)
    universe_table["rank_date"] = pd.to_datetime(universe_table["rank_date"])

    year_start_mcap = universe_table[
        [
            "instrument_id",
            "universe_year",
            "rank_date",
            "market_cap",
        ]
    ].copy()

    year_start_mcap = year_start_mcap.rename(
        columns={
            "market_cap": "feature_market_cap_year_start",
            "rank_date": "market_cap_rank_date",
        }
    )

    price = price.merge(
        year_start_mcap,
        on=["instrument_id", "universe_year"],
        how="left",
    )

    return price



def add_earnings_window_flag(price: pd.DataFrame, window_days: int = 0) -> pd.DataFrame:
    """Return price with an earnings exclusion flag widened by window_days on each side."""
    price = price.copy()
    price = price.sort_values(["instrument_id", "date"]).reset_index(drop=True)

    if window_days == 0:
        price["is_earnings_window"] = price["is_earnings_effective_date"]
        return price

    grouped = price.groupby("instrument_id", sort=False)

    price["is_earnings_window"] = False

    for lag in range(-window_days, window_days + 1):
        shifted_flag = grouped["is_earnings_effective_date"].shift(lag).fillna(False)
        price["is_earnings_window"] = price["is_earnings_window"] | shifted_flag

    return price


def assign_eligibility_reason(
    price: pd.DataFrame,
    min_price: float = 5,
    min_adv20: float = 20_000_000,
    min_market_cap: float = 1_000_000_000,
    min_annualized_vol: float = 0.05,
    max_annualized_vol: float = 1.50,
) -> pd.DataFrame:
    """Return price with point-in-time trade-eligibility reasons and flags."""
    price = price.copy()
    price = price.sort_values(["instrument_id", "date"]).reset_index(drop=True)

    # Default: all stock-days are tradable unless a filter binds
    price["eligibility_reason"] = "OK"

    # Must be in the yearly top-1000 universe
    price.loc[
        ~price["yearly_universe_member"],
        "eligibility_reason"
    ] = "NOT_YEARLY_UNIVERSE"

    # Missing required point-in-time fields
    missing_mask = (
        price["feature_price_lag1"].isna()
        | price["feature_adv20_lag1"].isna()
        | price["feature_market_cap_year_start"].isna()
        | price["feature_vol20_annualized_lag1"].isna()
        | price["target_r_overnight_next"].isna()
    )

    price.loc[
        (price["eligibility_reason"] == "OK") & missing_mask,
        "eligibility_reason"
    ] = "MISSING_DATA"

    # Price floor
    price.loc[
        (price["eligibility_reason"] == "OK")
        & (price["feature_price_lag1"] < min_price),
        "eligibility_reason"
    ] = "PRICE_FAIL"

    # Dollar-volume floor
    price.loc[
        (price["eligibility_reason"] == "OK")
        & (price["feature_adv20_lag1"] < min_adv20),
        "eligibility_reason"
    ] = "ADV_FAIL"

    # Year-start market-cap floor
    price.loc[
        (price["eligibility_reason"] == "OK")
        & (price["feature_market_cap_year_start"] < min_market_cap),
        "eligibility_reason"
    ] = "MCAP_FAIL"

    # Realised-volatility band
    vol_fail = (
        (price["feature_vol20_annualized_lag1"] < min_annualized_vol)
        | (price["feature_vol20_annualized_lag1"] > max_annualized_vol)
    )

    price.loc[
        (price["eligibility_reason"] == "OK") & vol_fail,
        "eligibility_reason"
    ] = "VOL_FAIL"

    # Earnings exclusion window
    price.loc[
        (price["eligibility_reason"] == "OK")
        & price["is_earnings_window"],
        "eligibility_reason"
    ] = "EARN_WINDOW"

    price["is_trade_eligible"] = price["eligibility_reason"] == "OK"

    return price


def add_participation_capacity(price: pd.DataFrame, participation_cap: float = 0.05) -> pd.DataFrame:
    """Return price with per-stock dollar capacity under the ADV participation cap."""
    price = price.copy()
    price["participation_cap"] = participation_cap

    price["max_position_by_adv"] = (
            price["feature_adv20_lag1"] * participation_cap
    )

    # Non-eligible stock-days receive zero tradable capacity.
    price.loc[
        ~price["is_trade_eligible"],
        "max_position_by_adv"
    ] = 0.0

    return price


def summarize_eligibility(price: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return aggregate, yearly, and daily-year eligibility diagnostics."""
    reason_summary = (
        price["eligibility_reason"]
        .value_counts()
        .rename_axis("eligibility_reason")
        .reset_index(name="stock_days")
    )

    reason_summary["fraction"] = reason_summary["stock_days"] / len(price)

    temp = price.copy()
    temp["year"] = temp["date"].dt.year

    reason_by_year = (
        temp
        .groupby(["year", "eligibility_reason"])
        .size()
        .reset_index(name="stock_days")
    )

    daily_eligible = (
        temp
        .groupby("date")
        .agg(
            eligible_names=("instrument_id", lambda x: x[temp.loc[x.index, "is_trade_eligible"]].nunique()),
            universe_names=("instrument_id", lambda x: x[temp.loc[x.index, "yearly_universe_member"]].nunique()),
        )
        .reset_index()
    )

    daily_eligible["year"] = daily_eligible["date"].dt.year

    eligible_by_year = (
        daily_eligible
        .groupby("year")
        .agg(
            avg_eligible_names=("eligible_names", "mean"),
            min_eligible_names=("eligible_names", "min"),
            max_eligible_names=("eligible_names", "max"),
            avg_universe_names=("universe_names", "mean"),
        )
        .reset_index()
    )
    return reason_summary, reason_by_year, eligible_by_year


def summarize_slippage_examples(
    price: pd.DataFrame,
    k: float = 0.7,
    participation_cap: float = 0.05,
    mid_cap_lower: float = 1_000_000_000,
    mid_cap_upper: float = 10_000_000_000,
    large_cap_lower: float = 10_000_000_000,
) -> pd.DataFrame:
    """Return square-root impact examples at the participation cap."""

    price = price.copy()

    df = price[
        price["is_trade_eligible"]
        & price["feature_adv20_lag1"].notna()
        & price["feature_market_cap_year_start"].notna()
        & price["feature_vol20_annualized_lag1"].notna()
    ].copy()

    df["feature_vol20_daily_lag1"] = (
        df["feature_vol20_annualized_lag1"] / (252 ** 0.5)
    )

    df["impact_bps_at_cap"] = (
        k
        * df["feature_vol20_daily_lag1"]
        * (participation_cap ** 0.5)
        * 10_000
    )

    df["max_position_at_cap"] = (
        df["feature_adv20_lag1"] * participation_cap
    )

    mid_cap = df[
        (df["feature_market_cap_year_start"] >= mid_cap_lower)
        & (df["feature_market_cap_year_start"] < mid_cap_upper)
    ].copy()

    large_cap = df[
        df["feature_market_cap_year_start"] >= large_cap_lower
    ].copy()

    rows = []

    for bucket_name, sub in [
        ("Typical mid-cap", mid_cap),
        ("Typical large-cap", large_cap),
    ]:
        rows.append({
            "bucket": bucket_name,
            "n_stock_days": len(sub),
            "market_cap_definition": (
                "$1bn-$10bn" if bucket_name == "Typical mid-cap"
                else ">= $10bn"
            ),
            "participation_cap": participation_cap,
            "k": k,
            "median_year_start_market_cap": sub["feature_market_cap_year_start"].median(),
            "median_adv20": sub["feature_adv20_lag1"].median(),
            "median_annualized_vol": sub["feature_vol20_annualized_lag1"].median(),
            "median_daily_vol": sub["feature_vol20_daily_lag1"].median(),
            "median_max_position_at_cap": sub["max_position_at_cap"].median(),
            "median_impact_bps_at_cap": sub["impact_bps_at_cap"].median(),
            "average_year_start_market_cap": sub["feature_market_cap_year_start"].mean(),
            "average_adv20": sub["feature_adv20_lag1"].mean(),
            "average_annualized_vol": sub["feature_vol20_annualized_lag1"].mean(),
            "average_daily_vol": sub["feature_vol20_daily_lag1"].mean(),
            "average_max_position_at_cap": sub["max_position_at_cap"].mean(),
            "average_impact_bps_at_cap": sub["impact_bps_at_cap"].mean(),
        })

    slippage_examples = pd.DataFrame(rows)

    return slippage_examples


def summarize_aum_eligible_set(
    price: pd.DataFrame,
    aum_levels: Sequence[float] = (50_000_000, 250_000_000, 1_000_000_000),
    n_long: int = 100,
    n_short: int = 100,
) -> pd.DataFrame:
    """Return yearly capacity-eligible counts for each portfolio AUM level."""
    price = price.copy()
    price["year"] = price["date"].dt.year

    basket_size = n_long + n_short
    all_rows = []

    for aum in aum_levels:
        per_stock_target = aum / basket_size

        temp = price.copy()
        temp["aum"] = aum
        temp["per_stock_target"] = per_stock_target

        temp["capacity_eligible"] = (
            temp["is_trade_eligible"]
            & (temp["max_position_by_adv"] >= per_stock_target)
        )

        daily = (
            temp.groupby("date")
            .agg(
                base_eligible_names=(
                    "instrument_id",
                    lambda x: x[temp.loc[x.index, "is_trade_eligible"]].nunique()
                ),
                capacity_eligible_names=(
                    "instrument_id",
                    lambda x: x[temp.loc[x.index, "capacity_eligible"]].nunique()
                ),
            )
            .reset_index()
        )

        daily["year"] = daily["date"].dt.year
        daily["aum"] = aum
        daily["per_stock_target"] = per_stock_target

        yearly = (
            daily.groupby(["aum", "year"], as_index=False)
            .agg(
                per_stock_target=("per_stock_target", "first"),
                avg_base_eligible_names=("base_eligible_names", "mean"),
                avg_capacity_eligible_names=("capacity_eligible_names", "mean"),
                min_capacity_eligible_names=("capacity_eligible_names", "min"),
                max_capacity_eligible_names=("capacity_eligible_names", "max"),
            )
        )

        all_rows.append(yearly)

    aum_eligible_by_year = pd.concat(all_rows, ignore_index=True)

    aum_eligible_by_year = aum_eligible_by_year[
        [
            "aum",
            "year",
            "per_stock_target",
            "avg_base_eligible_names",
            "avg_capacity_eligible_names",
            "min_capacity_eligible_names",
            "max_capacity_eligible_names",
        ]
    ]

    return aum_eligible_by_year


def make_binding_constraint_table(reason_by_year: pd.DataFrame) -> pd.DataFrame:
    """Return a wide yearly table of eligibility binding constraints."""
    binding_by_year = (
        reason_by_year
        .pivot_table(
            index="year",
            columns="eligibility_reason",
            values="stock_days",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    # Stable column order for reporting
    preferred_order = [
        "year",
        "OK",
        "MISSING_DATA",
        "PRICE_FAIL",
        "ADV_FAIL",
        "MCAP_FAIL",
        "VOL_FAIL",
        "EARN_WINDOW",
        "NOT_YEARLY_UNIVERSE",
    ]

    existing_cols = [c for c in preferred_order if c in binding_by_year.columns]
    other_cols = [c for c in binding_by_year.columns if c not in existing_cols]

    binding_by_year = binding_by_year[existing_cols + other_cols]

    return binding_by_year


def save_step2_outputs(
    price: pd.DataFrame,
    reason_summary: pd.DataFrame,
    reason_by_year: pd.DataFrame,
    eligible_by_year: pd.DataFrame,
    slippage_examples: pd.DataFrame,
    aum_eligible_by_year: pd.DataFrame,
    binding_by_year: pd.DataFrame,
    output_dir: PathLike = "../Output",
) -> None:
    """Write Step 2 capacity parquet and report diagnostics to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    price.to_parquet(output_dir / "step2_capacity_universe.parquet", index=False)

    reason_summary.to_csv(
        diagnostics_dir / "step2_reason_summary.csv",
        index=False
    )

    reason_by_year.to_csv(
        diagnostics_dir / "step2_reason_by_year.csv",
        index=False
    )

    eligible_by_year.to_csv(
        diagnostics_dir / "step2_eligible_by_year.csv",
        index=False
    )

    slippage_examples.to_csv(
        diagnostics_dir / "step2_slippage_examples.csv",
        index=False
    )

    aum_eligible_by_year.to_csv(
        diagnostics_dir / "step2_aum_eligible_by_year.csv",
        index=False
    )

    binding_by_year.to_csv(
        diagnostics_dir / "step2_binding_by_year.csv",
        index=False
    )


if __name__ == '__main__':
    input_path = Path("../Output/step1_daily_panel.parquet")
    output_dir = Path("../Output")

    prices = load_step1_panel(input_path)

    universe_table_path = output_dir / "diagnostics" / "step1_yearly_universe_table.csv"

    prices = add_capacity_base_columns(
        prices,
        universe_table_path=universe_table_path,
    )
    prices = add_earnings_window_flag(prices, window_days=1)


    prices = assign_eligibility_reason(prices)
    prices = add_participation_capacity(prices, participation_cap=0.05)

    reason_summary, reason_by_year, eligible_by_year = summarize_eligibility(prices)

    slippage_examples = summarize_slippage_examples(
        prices,
        k=0.7,
        participation_cap=0.05,
    )

    aum_eligible_by_year = summarize_aum_eligible_set(
        prices,
        aum_levels=(50_000_000, 250_000_000, 1_000_000_000),
        n_long=100,
        n_short=100,
    )

    binding_by_year = make_binding_constraint_table(reason_by_year)

    print(slippage_examples)
    print(aum_eligible_by_year)
    print(reason_summary)
    print(reason_by_year.head(20))
    print(eligible_by_year)

    save_step2_outputs(
        prices,
        reason_summary,
        reason_by_year,
        eligible_by_year,
        slippage_examples,
        aum_eligible_by_year,
        binding_by_year,
        output_dir=output_dir,
    )

    print((output_dir / "step2_capacity_universe.parquet").exists())
    print((output_dir / "diagnostics" / "step2_reason_summary.csv").exists())
    print((output_dir / "diagnostics" / "step2_reason_by_year.csv").exists())
    print((output_dir / "diagnostics" / "step2_eligible_by_year.csv").exists())
    print((output_dir / "diagnostics" / "step2_aum_eligible_by_year.csv").exists())
    print((output_dir / "diagnostics" / "step2_binding_by_year.csv").exists())

    print("Step 2 done")

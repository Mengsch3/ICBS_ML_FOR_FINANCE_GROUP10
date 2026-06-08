from pathlib import Path
import pandas as pd
from typing import TypeAlias


BORROW_COST_MAP = {
    "A": 0.004,   # 40 bps
    "B": 0.020,   # 200 bps
    "C": 0.080,   # 800 bps
}
PathLike: TypeAlias = str | Path


def load_step2_panel(input_path: PathLike) -> pd.DataFrame:
    """Return the Step 2 panel sorted by instrument and date."""
    price = pd.read_parquet(input_path)

    price["date"] = pd.to_datetime(price["date"])

    price = (
        price
        .sort_values(["instrument_id", "date"])
        .reset_index(drop=True)
    )

    return price


def add_short_interest_percentiles(price: pd.DataFrame) -> pd.DataFrame:
    """Return price with daily cross-sectional short-interest percentiles."""

    price = price.copy()

    # DSI percentile
    price["short_dsi_pct"] = (
        price
        .groupby("date")["feature_short_dsi"]
        .rank(pct=True)
    )

    # Days-to-cover percentile
    price["short_dtc_pct"] = (
        price
        .groupby("date")["feature_short_days_to_cover"]
        .rank(pct=True)
    )

    return price


def assign_borrow_tier(
    price: pd.DataFrame,
    moderate_threshold: float = 0.80,
    severe_threshold: float = 0.95,
) -> pd.DataFrame:
    """Return price with borrow tiers A, B, and C from short-interest percentiles."""

    price = price.copy()

    price["borrow_tier"] = "A"

    moderate_mask = (
        (price["short_dsi_pct"] >= moderate_threshold)
        | (price["short_dtc_pct"] >= moderate_threshold)
    )

    severe_mask = (
        (price["short_dsi_pct"] >= severe_threshold)
        | (price["short_dtc_pct"] >= severe_threshold)
    )

    price.loc[moderate_mask, "borrow_tier"] = "B"

    price.loc[severe_mask, "borrow_tier"] = "C"

    return price


def add_borrow_cost(price: pd.DataFrame) -> pd.DataFrame:
    """Return price with annual and daily borrow costs in decimal return units."""
    price = price.copy()

    price["borrow_cost_annual"] = (
        price["borrow_tier"]
        .map(BORROW_COST_MAP)
    )

    price["borrow_cost_daily"] = (
        price["borrow_cost_annual"] / 252
    )

    return price


def add_shortability_flag(price: pd.DataFrame) -> pd.DataFrame:
    """Return price with shortability equal to trade eligibility; HTB names carry higher costs."""

    price = price.copy()

    price["is_shortable"] = (
        price["is_trade_eligible"]
    )

    return price


def missing_si_data(price: pd.DataFrame) -> pd.DataFrame:
    """Return yearly missing-short-interest fractions among trade-eligible stock-days."""
    temp = price[price["is_trade_eligible"]].copy()

    temp["year"] = temp["date"].dt.year

    temp["missing_si_data"] = (
        temp["feature_short_dsi"].isna()
        | temp["feature_short_days_to_cover"].isna()
    )

    missing_si_summary = (
        temp
        .groupby("year")["missing_si_data"]
        .mean()
        .reset_index(name="missing_fraction")
    )

    return missing_si_summary


def summarize_borrow(price: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return borrow tier, yearly tier, and borrow-cost diagnostics."""

    temp = price[price["is_trade_eligible"]].copy()

    temp["year"] = temp["date"].dt.year

    # overall
    borrow_summary = (
        temp["borrow_tier"]
        .value_counts()
        .rename_axis("borrow_tier")
        .reset_index(name="stock_days")
    )

    borrow_summary["fraction"] = (
        borrow_summary["stock_days"] / len(temp)
    )

    # by year
    borrow_by_year = (
        temp
        .groupby(["year", "borrow_tier"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # make sure all three tiers exist as columns
    for tier in ["A", "B", "C"]:
        if tier not in borrow_by_year.columns:
            borrow_by_year[tier] = 0

    # rename count columns
    borrow_by_year = borrow_by_year.rename(
        columns={
            "year": "Year",
            "A": "Tier A",
            "B": "Tier B",
            "C": "Tier C",
        }
    )

    # add total stock-days
    borrow_by_year["Total stock-days"] = (
            borrow_by_year["Tier A"]
            + borrow_by_year["Tier B"]
            + borrow_by_year["Tier C"]
    )

    # add percentages
    borrow_by_year["Tier A pct"] = (
            borrow_by_year["Tier A"] / borrow_by_year["Total stock-days"]
    )

    borrow_by_year["Tier B pct"] = (
            borrow_by_year["Tier B"] / borrow_by_year["Total stock-days"]
    )

    borrow_by_year["Tier C pct"] = (
            borrow_by_year["Tier C"] / borrow_by_year["Total stock-days"]
    )

    # optional: convert to percentage format, e.g. 72.35 instead of 0.7235
    borrow_by_year["Tier A pct"] = borrow_by_year["Tier A pct"] * 100
    borrow_by_year["Tier B pct"] = borrow_by_year["Tier B pct"] * 100
    borrow_by_year["Tier C pct"] = borrow_by_year["Tier C pct"] * 100

    # keep columns in required order
    borrow_by_year = borrow_by_year[
        [
            "Year",
            "Tier A",
            "Tier B",
            "Tier C",
            "Total stock-days",
            "Tier A pct",
            "Tier B pct",
            "Tier C pct",
        ]
    ]

    # cost stats
    borrow_cost_summary = (
        temp["borrow_cost_annual"]
        .describe()
        .reset_index()
    )

    return (
        borrow_summary,
        borrow_by_year,
        borrow_cost_summary,
    )


def save_outputs(
    price: pd.DataFrame,
    borrow_summary: pd.DataFrame,
    borrow_by_year: pd.DataFrame,
    borrow_cost_summary: pd.DataFrame,
    missing_si_summary: pd.DataFrame,
    output_dir: PathLike = "../Output",
) -> None:
    """Write Step 3 borrow parquet and diagnostic CSV files to output_dir."""
    output_dir = Path(output_dir)

    diagnostics_dir = output_dir / "diagnostics"

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    price.to_parquet(
        output_dir / "step3_borrow_model.parquet",
        index=False,
    )

    borrow_summary.to_csv(
        diagnostics_dir / "step3_borrow_tier_summary.csv",
        index=False,
    )

    borrow_by_year.to_csv(
        diagnostics_dir / "step3_borrow_tier_by_year.csv",
        index=False,
    )

    borrow_cost_summary.to_csv(
        diagnostics_dir / "step3_borrow_cost_summary.csv",
        index=False,
    )
    
    missing_si_summary.to_csv(
    diagnostics_dir / "step3_missing_si_summary.csv",
    index=False,
    )


if __name__ == "__main__":

    input_path = "../Output/step2_capacity_universe.parquet"

    price = load_step2_panel(input_path)

    price = add_short_interest_percentiles(price)

    price = assign_borrow_tier(
        price,
        moderate_threshold=0.80,
        severe_threshold=0.95,
    )

    price = add_borrow_cost(price)

    price = add_shortability_flag(price)

    (
        borrow_summary,
        borrow_by_year,
        borrow_cost_summary,
    ) = summarize_borrow(price)

    missing_si_summary = missing_si_data(price)
    
    print(borrow_summary)
    print(borrow_by_year.head())
    print(borrow_cost_summary)
    print(missing_si_summary)
    
    save_outputs(
        price,
        borrow_summary,
        borrow_by_year,
        borrow_cost_summary,
        missing_si_summary
    )

    print("Step 3 done")

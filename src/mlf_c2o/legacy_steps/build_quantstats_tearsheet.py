"""
Generate the 250M AUM tear-sheet for the overnight long-short strategy.

Run from Src/:
    python build_quantstats_tearsheet.py

Inputs:
    Output/diagnostics/step5_net_returns_250m.csv
    Data/sp500_tr.parquet

Outputs:
    Output/tearsheet_250m.html
    Output/diagnostics/step8_quantstats_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent
DIAG = PROJECT / "Output" / "diagnostics"
DATA_DIR = PROJECT / "Data"
OUT_HTML = PROJECT / "Output" / "tearsheet_250m.html"
OUT_SUMMARY = DIAG / "step8_quantstats_summary.csv"
TRADING_DAYS = 252


def main() -> None:
    """Generate a QuantStats HTML tear-sheet."""
    strategy, benchmark = _load_returns()
    summary = _summary_table(strategy, benchmark)
    DIAG.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_SUMMARY, index=False)
    _write_tearsheet(strategy, benchmark)
    print(f"Tear-sheet saved to {OUT_HTML.resolve()}")
    print(f"Summary saved to {OUT_SUMMARY.resolve()}")


def _load_returns() -> tuple[pd.Series, pd.Series]:
    strategy = pd.read_csv(
        DIAG / "step5_net_returns_250m.csv",
        parse_dates=["date"],
        index_col="date",
    )["net_daily_return"].dropna()
    strategy.name = "Overnight L/S"

    sp500 = pd.read_parquet(DATA_DIR / "sp500_tr.parquet", columns=["date", "adjusted_close"])
    sp500["date"] = pd.to_datetime(sp500["date"])
    sp500 = sp500.set_index("date").sort_index()
    benchmark = sp500["adjusted_close"].pct_change().dropna()
    benchmark.name = "SP500 TR"

    common = strategy.index.intersection(benchmark.index)
    return strategy.reindex(common).dropna(), benchmark.reindex(common).dropna()


def _summary_table(strategy: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    rows = [
        _metrics_row("Overnight L/S", strategy),
        _metrics_row("SP500 TR", benchmark),
    ]
    return pd.DataFrame(rows)


def _metrics_row(name: str, returns: pd.Series) -> dict[str, float | str]:
    cumulative = (1.0 + returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1.0
    volatility = returns.std() * np.sqrt(TRADING_DAYS)
    annual_return = returns.mean() * TRADING_DAYS
    sharpe = annual_return / volatility if volatility > 0 else np.nan
    downside = returns[returns < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = annual_return / downside if downside > 0 else np.nan
    years = len(returns) / TRADING_DAYS
    cagr = cumulative.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else np.nan
    calmar = cagr / abs(drawdown.min()) if drawdown.min() < 0 else np.nan
    return {
        "series": name,
        "start_date": str(returns.index.min().date()),
        "end_date": str(returns.index.max().date()),
        "n_days": int(len(returns)),
        "annual_return": annual_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": drawdown.min(),
        "calmar": calmar,
    }


def _write_tearsheet(strategy: pd.Series, benchmark: pd.Series) -> None:
    try:
        import quantstats as qs
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "quantstats is required for Step 8. Install the project dependencies "
            "with `pip install -e .` from MLF_C2O_Final_Deliverable, or run "
            "`pip install quantstats` in the active Python environment."
        ) from None

    qs.extend_pandas()
    qs.reports.html(
        strategy,
        benchmark=benchmark,
        title="Overnight Long/Short Strategy - $250M AUM - 2013-2024",
        output=str(OUT_HTML),
        download_filename="tearsheet_250m.html",
    )


if __name__ == "__main__":
    main()

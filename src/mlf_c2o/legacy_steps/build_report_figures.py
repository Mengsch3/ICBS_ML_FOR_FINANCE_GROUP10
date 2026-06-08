"""
Generate the nine report figures as code deliverables.

Run from Src/:
    python build_report_figures.py

Inputs:
    Output/diagnostics/*.csv
    Output/step1_daily_panel.parquet
    Output/diagnostics/step5_pnl_*.csv

Outputs:
    Output/figures/fig01_short_interest_series_aapl.png
    ...
    Output/figures/fig09_drawdown_250m.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent
DIAG_DIR = PROJECT / "Output" / "diagnostics"
FIG_DIR = PROJECT / "Output" / "figures"

AUM_CONFIG = {
    "50M": ("50m", 50_000_000.0),
    "250M": ("250m", 250_000_000.0),
    "1B": ("1b", 1_000_000_000.0),
}


def _prepare_output_dir(output_dir: Path | None = None) -> Path:
    out = Path(output_dir) if output_dir is not None else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _load_pnl(label: str) -> tuple[pd.DataFrame, float]:
    suffix, aum = AUM_CONFIG[label]
    pnl = pd.read_csv(DIAG_DIR / f"step5_pnl_{suffix}.csv", parse_dates=["date"])
    pnl = pnl.set_index("date").sort_index()
    return pnl, aum


def figure_short_interest_series(output_dir: Path | None = None, ticker: str = "AAPL") -> None:
    """Generate Figure 1: point-in-time short-interest series."""
    out = _prepare_output_dir(output_dir)
    price = pd.read_parquet(PROJECT / "Output" / "step1_daily_panel.parquet", columns=["date", "ticker", "short_dsi"])
    price["date"] = pd.to_datetime(price["date"])
    one = price[
        (price["ticker"] == ticker)
        & (price["date"] >= "2010-01-01")
        & (price["date"] <= "2024-12-31")
    ].sort_values("date")
    if one.empty:
        raise ValueError(f"No rows found for ticker={ticker}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(one["date"], one["short_dsi"], where="post", label=f"{ticker} short-interest ratio", linewidth=1.8)
    ax.set_title(f"Point-in-time short-interest series: {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Short-interest ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, out, f"fig01_short_interest_series_{ticker.lower()}.png")


def figure_stylised_fact(output_dir: Path | None = None) -> None:
    """Generate Figure 2: equal-weighted overnight/intraday growth."""
    out = _prepare_output_dir(output_dir)
    df = pd.read_csv(DIAG_DIR / "step1_stylized_fact_daily.csv", parse_dates=["date"])
    df = df[(df["date"] >= "2010-01-01") & (df["date"] <= "2024-12-31")]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["date"], df["growth_r_overnight"], label="Overnight", linewidth=2.0)
    ax.plot(df["date"], df["growth_r_intraday"], label="Intraday", linewidth=2.0)
    ax.plot(df["date"], df["growth_r_close_to_close"], label="Close-to-close", linewidth=2.0)
    ax.set_title("Equal-weighted return decomposition, 2010-2024")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, out, "fig02_stylised_fact.png")


def figure_capacity_by_aum(output_dir: Path | None = None) -> None:
    """Generate Figure 3: capacity-eligible names by AUM."""
    out = _prepare_output_dir(output_dir)
    df = pd.read_csv(DIAG_DIR / "step2_aum_eligible_by_year.csv")
    df = df[df["year"].between(2010, 2024)]
    label_map = {50_000_000: "50M", 250_000_000: "250M", 1_000_000_000: "1B"}
    df["aum_label"] = df["aum"].map(label_map)

    fig, ax = plt.subplots(figsize=(9, 5))
    for label in ["50M", "250M", "1B"]:
        g = df[df["aum_label"] == label]
        ax.plot(g["year"], g["avg_capacity_eligible_names"], marker="o", label=label)
    ax.set_title("Capacity-eligible names under the 5% ADV cap")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average capacity-eligible names")
    ax.set_xticks(sorted(df["year"].unique()))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="AUM", frameon=False)
    _save(fig, out, "fig03_capacity_by_aum.png")


def figure_feature_importance(output_dir: Path | None = None) -> None:
    """Generate Figure 4: top 20 model feature importances."""
    out = _prepare_output_dir(output_dir)
    df = pd.read_csv(DIAG_DIR / "step4_feature_importance.csv").head(20).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df["feature"], df["importance"], color="#4C78A8")
    ax.set_title("Top 20 alpha-model feature importances")
    ax.set_xlabel("Average LightGBM importance")
    ax.grid(True, axis="x", alpha=0.25)
    _save(fig, out, "fig04_feature_importance.png")


def figure_feature_group_importance(output_dir: Path | None = None) -> None:
    """Generate Figure 5: model importance by feature group."""
    out = _prepare_output_dir(output_dir)
    df = pd.read_csv(DIAG_DIR / "step4_feature_group_importance.csv").sort_values("importance_share", ascending=True)
    labels = df["group_id"].astype(str) + ": " + df["group_name"].astype(str)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels, df["importance_share"] * 100, color="#72B7B2")
    ax.set_title("Feature-group contribution to LightGBM gain")
    ax.set_xlabel("Share of total feature importance (%)")
    ax.grid(True, axis="x", alpha=0.25)
    for i, value in enumerate(df["importance_share"] * 100):
        ax.text(value + 0.15, i, f"{value:.1f}%", va="center", fontsize=8)
    _save(fig, out, "fig05_feature_group_importance.png")


def figure_feature_importance_concentration(output_dir: Path | None = None) -> None:
    """Generate Figure 6: feature-importance concentration."""
    out = _prepare_output_dir(output_dir)
    df = pd.read_csv(DIAG_DIR / "step4_feature_importance_concentration.csv").copy()
    label_map = {"top_feature": "Top feature", "top_5_features": "Top 5 features", "top_group": "Top group"}
    df["label"] = df["metric"].map(label_map).fillna(df["metric"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["label"], df["importance_share"] * 100, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_title("Feature-importance concentration diagnostics")
    ax.set_ylabel("Share of total feature importance (%)")
    ax.grid(True, axis="y", alpha=0.25)
    for i, value in enumerate(df["importance_share"] * 100):
        ax.text(i, value + 0.5, f"{value:.1f}%", ha="center", fontsize=9)
    _save(fig, out, "fig06_feature_importance_concentration.png")


def figure_signal_gate(output_dir: Path | None = None) -> None:
    """Generate Figure 7: signal-spread and rolling-IC gate."""
    out = _prepare_output_dir(output_dir)
    df = pd.read_csv(DIAG_DIR / "step5_signal_gate.csv", parse_dates=["date"])
    allowed = df[df["trade_allowed"]]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(df["date"], df["alpha_spread"], label="Alpha spread", linewidth=1.0)
    axes[0].plot(df["date"], df["spread_threshold"], label="Rolling spread threshold", linewidth=1.4)
    axes[0].scatter(allowed["date"], allowed["alpha_spread"], s=8, color="#54A24B", alpha=0.6, label="Trade allowed")
    axes[0].set_ylabel("p90-p10 alpha spread")
    axes[0].set_title("Point-in-time signal gate")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(df["date"], df["rolling_ic"], color="#4C78A8", label="Rolling IC")
    axes[1].axhline(0.02, color="#E45756", linestyle="--", label="IC threshold = 0.02")
    axes[1].scatter(allowed["date"], allowed["rolling_ic"], s=8, color="#54A24B", alpha=0.6)
    axes[1].set_ylabel("Trailing mean IC")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    _save(fig, out, "fig07_signal_gate.png")


def figure_cumulative_returns(output_dir: Path | None = None) -> None:
    """Generate Figure 8: cumulative net returns by AUM."""
    out = _prepare_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(9, 5))
    for label in ["50M", "250M", "1B"]:
        pnl, aum = _load_pnl(label)
        net_ret = pnl["net_pnl"] / aum
        cum = (1 + net_ret.fillna(0)).cumprod() - 1
        ax.plot(cum.index, cum * 100, label=label, linewidth=2.0)
    ax.set_title("Cumulative net return by AUM")
    ax.set_ylabel("Cumulative net return (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(title="AUM", frameon=False)
    _save(fig, out, "fig08_cumulative_returns.png")


def figure_drawdown_250m(output_dir: Path | None = None) -> None:
    """Generate Figure 9: 250M drawdown curve."""
    out = _prepare_output_dir(output_dir)
    pnl, aum = _load_pnl("250M")
    net_ret = pnl["net_pnl"] / aum
    cum = (1 + net_ret.fillna(0)).cumprod()
    drawdown = cum / cum.cummax() - 1

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(drawdown.index, drawdown * 100, 0, color="#E45756", alpha=0.35)
    ax.plot(drawdown.index, drawdown * 100, color="#B22222", linewidth=1.2)
    ax.set_title("Drawdown: 250M AUM strategy")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.25)
    _save(fig, out, "fig09_drawdown_250m.png")


def generate_all_figures(output_dir: Path | None = None) -> None:
    """Generate exactly the nine figures used by the final report."""
    figure_short_interest_series(output_dir)
    figure_stylised_fact(output_dir)
    figure_capacity_by_aum(output_dir)
    figure_feature_importance(output_dir)
    figure_feature_group_importance(output_dir)
    figure_feature_importance_concentration(output_dir)
    figure_signal_gate(output_dir)
    figure_cumulative_returns(output_dir)
    figure_drawdown_250m(output_dir)


def main() -> None:
    """Generate the nine report figures into the legacy Output/figures directory."""
    generate_all_figures(FIG_DIR)
    print(f"Figures saved to: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()

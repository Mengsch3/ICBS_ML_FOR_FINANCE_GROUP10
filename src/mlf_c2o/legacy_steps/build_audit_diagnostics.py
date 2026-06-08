"""
Step 6 audit diagnostics for report Sections 7.3--7.5.

This script is intentionally diagnostic-only.  It does not change the headline
Step 5 back-test outputs; instead it reconstructs portfolio weights from the
current Step 4/Step 5 code and saves named CSV files that make the capacity,
execution, borrow and reporting-integrity numbers traceable.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Local parquet dependency used in the Codex workspace.  Normal user
# environments with pyarrow installed do not need this path.
for dep_dir in (PROJECT_ROOT / ".codex_pyarrow", PROJECT_ROOT / ".codex_deps"):
    if dep_dir.exists():
        sys.path.insert(0, str(dep_dir))

import numpy as np
import pandas as pd

# build_backtest imports lightgbm only for the optional meta-model functions.
# These Step 6 audits call deterministic portfolio-construction/PnL functions
# only, so a stub keeps the audit reproducible in environments without
# LightGBM installed.
try:
    import lightgbm  # noqa: F401
except ModuleNotFoundError:
    import types

    sys.modules["lightgbm"] = types.SimpleNamespace()

import build_backtest as bt


OUTPUT_DIR = PROJECT_ROOT / "Output"
DIAG_DIR = OUTPUT_DIR / "diagnostics"

AUM_LEVELS = {
    "50m": 50_000_000.0,
    "250m": 250_000_000.0,
    "1b": 1_000_000_000.0,
}

HIGH_SHORT_INTEREST_THRESHOLD = 0.10


def _sharpe(daily_returns: pd.Series) -> float:
    vol = daily_returns.std()
    if vol == 0 or np.isnan(vol):
        return np.nan
    return float(daily_returns.mean() / vol * np.sqrt(252))


def _max_drawdown(cum_return: pd.Series) -> float:
    running_max = cum_return.cummax()
    drawdown = cum_return / running_max - 1.0
    return float(drawdown.min())


def _performance_metrics(pnl: pd.DataFrame, aum: float) -> dict[str, float]:
    net_ret = pnl["net_pnl"] / aum
    gross_ret = pnl["gross_pnl"] / aum
    cum_net = (1.0 + net_ret).cumprod()
    ann = 252

    return {
        "net_ann_ret_pct": float(net_ret.mean() * ann * 100),
        "net_ann_vol_pct": float(net_ret.std() * np.sqrt(ann) * 100),
        "net_sharpe": _sharpe(net_ret),
        "gross_sharpe": _sharpe(gross_ret),
        "max_drawdown_pct": _max_drawdown(cum_net) * 100,
        "avg_daily_turnover_one_way_pct": float((pnl["turnover_$"] / aum / 2).mean() * 100),
        "avg_gross_exposure_pct": float((pnl["gross_notional"] / aum).mean() * 100),
        "peak_gross_exposure_pct": float((pnl["gross_notional"] / aum).max() * 100),
        "frac_days_below_95pct_gross_pct": float((pnl["gross_notional"] < aum * 0.95).mean() * 100),
        "trade_cost_ann_bps": float(pnl["trade_cost"].mean() * ann / aum * 1e4),
        "borrow_cost_ann_bps": float(pnl["borrow_cost"].mean() * ann / aum * 1e4),
        "active_days": int((pnl["gross_notional"] > 0).sum()),
        "total_days": int(len(pnl)),
    }


def _load_backtest_panel() -> pd.DataFrame:
    """Load Step 5 panel and attach point-in-time short-interest diagnostics."""
    df = bt.load_data(str(OUTPUT_DIR))
    df["date"] = pd.to_datetime(df["date"])

    si_cols = [
        "instrument_id",
        "date",
        "feature_short_dsi",
        "short_dsi",
        "short_dsi_pct",
        "feature_adv20_lag1",
    ]
    si = pd.read_parquet(OUTPUT_DIR / "step3_borrow_model.parquet", columns=si_cols)
    si["date"] = pd.to_datetime(si["date"])
    si = si[(si["date"] >= bt.BACKTEST_START) & (si["date"] <= bt.BACKTEST_END)]
    si = si.drop_duplicates(["date", "instrument_id"])

    merge_cols = ["instrument_id", "date"]
    extra_cols = [c for c in si.columns if c not in merge_cols and c not in df.columns]
    return df.merge(si[merge_cols + extra_cols], on=merge_cols, how="left")


def _build_gated_weights(df: pd.DataFrame, aum: float, signal_gate: pd.Series) -> dict:
    weights = bt.build_daily_weights(
        df,
        aum,
        basket_frac=bt.BASKET_FRAC,
        min_alpha_spread=bt.MIN_ALPHA_SPREAD,
        scale_by_vol=True,
    )
    weights = bt.apply_date_gate(weights, signal_gate)
    weights = bt.apply_gross_exposure_cap(weights, aum)
    return weights


def _weights_to_positions(df: pd.DataFrame, weights: dict, aum_label: str, aum: float) -> pd.DataFrame:
    rows = []
    for date, day_weights in weights.items():
        for instrument_id, signed_position in day_weights.items():
            if signed_position == 0.0:
                continue
            rows.append(
                {
                    "aum_label": aum_label,
                    "aum": aum,
                    "date": pd.Timestamp(date),
                    "instrument_id": instrument_id,
                    "signed_position": float(signed_position),
                    "abs_position": abs(float(signed_position)),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "aum_label",
                "aum",
                "date",
                "instrument_id",
                "signed_position",
                "abs_position",
                "max_position_by_adv",
                "adv_dollar",
                "adv_share_pct",
                "cap_utilization",
                "feature_short_dsi",
                "short_dsi",
                "short_dsi_pct",
                "borrow_tier",
                "target_r_overnight_next",
                "gross_pnl_contribution",
                "high_short_interest_gt_10pct",
            ]
        )

    pos = pd.DataFrame(rows)
    info_cols = [
        "date",
        "instrument_id",
        "max_position_by_adv",
        "feature_short_dsi",
        "short_dsi",
        "short_dsi_pct",
        "borrow_tier",
        "target_r_overnight_next",
    ]
    info = df[info_cols].drop_duplicates(["date", "instrument_id"])
    pos = pos.merge(info, on=["date", "instrument_id"], how="left")

    finite_cap = pos["max_position_by_adv"].replace([np.inf, -np.inf], np.nan)
    pos["adv_dollar"] = finite_cap / bt.PARTICIPATION_CAP
    pos["adv_share_pct"] = pos["abs_position"] / pos["adv_dollar"] * 100
    pos["cap_utilization"] = pos["abs_position"] / finite_cap
    pos["gross_pnl_contribution"] = (
        pos["signed_position"] * pos["target_r_overnight_next"].fillna(0.0)
    )
    pos["high_short_interest_gt_10pct"] = (
        pos["feature_short_dsi"] > HIGH_SHORT_INTEREST_THRESHOLD
    )
    return pos.sort_values(["date", "instrument_id"]).reset_index(drop=True)


def _summarise_positions(pos: pd.DataFrame, aum_label: str, aum: float) -> dict[str, float]:
    if pos.empty:
        return {
            "aum_label": aum_label,
            "aum": aum,
            "active_days": 0,
            "position_count": 0,
            "avg_per_stock_abs_position": 0.0,
            "avg_daily_peak_abs_position": 0.0,
            "peak_per_stock_abs_position": 0.0,
            "avg_position_adv_share_pct": np.nan,
            "peak_position_adv_share_pct": np.nan,
            "max_cap_utilization": np.nan,
            "cap_binding_positions": 0,
            "cap_binding_days": 0,
            "cap_violations_gt_5pct_adv": 0,
        }

    cap_util = pos["cap_utilization"].replace([np.inf, -np.inf], np.nan)
    adv_share = pos["adv_share_pct"].replace([np.inf, -np.inf], np.nan)
    daily_peak = pos.groupby("date")["abs_position"].max()
    daily_cap_bind = pos.groupby("date")["cap_utilization"].max()

    return {
        "aum_label": aum_label,
        "aum": aum,
        "active_days": int(pos["date"].nunique()),
        "position_count": int(len(pos)),
        "avg_per_stock_abs_position": float(pos["abs_position"].mean()),
        "avg_daily_peak_abs_position": float(daily_peak.mean()),
        "peak_per_stock_abs_position": float(pos["abs_position"].max()),
        "avg_position_adv_share_pct": float(adv_share.mean()),
        "peak_position_adv_share_pct": float(adv_share.max()),
        "max_cap_utilization": float(cap_util.max()),
        "cap_binding_positions": int((cap_util >= 0.999999).sum()),
        "cap_binding_days": int((daily_cap_bind >= 0.999999).sum()),
        "cap_violations_gt_5pct_adv": int((adv_share > 5.000001).sum()),
    }


def _borrow_contribution_summary(pos: pd.DataFrame, pnl: pd.DataFrame, aum_label: str, aum: float) -> dict[str, float]:
    total_days = len(pnl)
    total_gross_pnl = float(pnl["gross_pnl"].sum())
    high = pos[pos["high_short_interest_gt_10pct"]]
    high_pnl = float(high["gross_pnl_contribution"].sum())

    return {
        "aum_label": aum_label,
        "aum": aum,
        "threshold_feature_short_dsi": HIGH_SHORT_INTEREST_THRESHOLD,
        "high_si_position_count": int(len(high)),
        "all_position_count": int(len(pos)),
        "high_si_notional_share_pct": float(high["abs_position"].sum() / pos["abs_position"].sum() * 100)
        if len(pos) else np.nan,
        "high_si_gross_pnl": high_pnl,
        "total_gross_pnl": total_gross_pnl,
        "high_si_total_period_gross_bps": float(high_pnl / aum * 1e4),
        "high_si_ann_gross_bps": float(high_pnl / aum / total_days * 252 * 1e4),
        "high_si_share_of_total_gross_pnl_pct": float(high_pnl / total_gross_pnl * 100)
        if total_gross_pnl != 0 else np.nan,
    }


def _reporting_integrity_rows() -> list[dict[str, str]]:
    return [
        {
            "audit_item": "chart_reproducibility",
            "status": "NOT_APPLICABLE",
            "evidence": "This final delivery is code-only and does not include LaTeX report figures.",
            "reproduction_command": "python -m mlf_c2o.main --dry-run",
        },
        {
            "audit_item": "pipeline_reproducibility",
            "status": "PASS",
            "evidence": "mlf_c2o.run_pipeline runs Step 1 through Step 6 in order and checks expected outputs after each step.",
            "reproduction_command": "python -m mlf_c2o.main",
        },
        {
            "audit_item": "random_seed",
            "status": "PASS",
            "evidence": "Step 4 sets RANDOM_SEED=42 and passes seed=RANDOM_SEED to LightGBM; the unused Step 5 meta-model also fixes random_state=42.",
            "reproduction_command": "rg -n \"RANDOM_SEED|random_state|seed\" Src",
        },
        {
            "audit_item": "assumptions_on_binding_figures",
            "status": "NOT_APPLICABLE",
            "evidence": "No report figures are shipped in this code-only delivery; binding assumptions are declared in config/default.yaml.",
            "reproduction_command": "python -m mlf_c2o.main --dry-run",
        },
    ]


def run_audits() -> None:
    """Write Step 6 capacity, borrow, and reporting-integrity audit diagnostics."""
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading back-test panel with short-interest diagnostics...")
    df = _load_backtest_panel()

    print("Building baseline point-in-time signal gate...")
    signal_gate, _ = bt.build_rolling_signal_gate(df)

    cap_comparison_rows = []
    position_summary_rows = []
    borrow_contribution_rows = []

    baseline_weights_by_aum = {}
    baseline_pnl_by_aum = {}

    print("Running 5% ADV-cap baseline and saving per-position diagnostics...")
    for label, aum in AUM_LEVELS.items():
        weights = _build_gated_weights(df, aum, signal_gate)
        pnl = bt.run_backtest_with_weights(df, weights)
        pos = _weights_to_positions(df, weights, label, aum)

        baseline_weights_by_aum[label] = weights
        baseline_pnl_by_aum[label] = pnl

        pos.to_csv(DIAG_DIR / f"step6_capacity_positions_{label}.csv", index=False)

        row = {"aum_label": label, "aum": aum, "cap_setting": "5pct_adv_cap"}
        row.update(_performance_metrics(pnl, aum))
        cap_comparison_rows.append(row)

        position_summary_rows.append(_summarise_positions(pos, label, aum))
        borrow_contribution_rows.append(_borrow_contribution_summary(pos, pnl, label, aum))

    print("Running no-cap counterfactual...")
    original_caps = df["max_position_by_adv"].copy()
    df["max_position_by_adv"] = np.inf
    try:
        for label, aum in AUM_LEVELS.items():
            weights = _build_gated_weights(df, aum, signal_gate)
            pnl = bt.run_backtest_with_weights(df, weights)
            pnl.to_csv(DIAG_DIR / f"step6_pnl_no_cap_{label}.csv", index=False)

            row = {"aum_label": label, "aum": aum, "cap_setting": "no_cap"}
            row.update(_performance_metrics(pnl, aum))
            cap_comparison_rows.append(row)
    finally:
        df["max_position_by_adv"] = original_caps

    print("Running high-short-interest hard-exclusion counterfactual...")
    original_eligible = df["is_trade_eligible"].copy()
    high_si_mask = df["feature_short_dsi"] > HIGH_SHORT_INTEREST_THRESHOLD
    df.loc[high_si_mask, "is_trade_eligible"] = False
    try:
        exclusion_gate, _ = bt.build_rolling_signal_gate(df)
        exclusion_rows = []
        for label, aum in AUM_LEVELS.items():
            weights = _build_gated_weights(df, aum, exclusion_gate)
            pnl = bt.run_backtest_with_weights(df, weights)
            pnl.to_csv(DIAG_DIR / f"step6_pnl_exclude_high_si_{label}.csv", index=False)

            row = {
                "aum_label": label,
                "aum": aum,
                "threshold_feature_short_dsi": HIGH_SHORT_INTEREST_THRESHOLD,
                "excluded_stock_days": int(high_si_mask.sum()),
            }
            row.update(_performance_metrics(pnl, aum))
            exclusion_rows.append(row)
    finally:
        df["is_trade_eligible"] = original_eligible

    pd.DataFrame(cap_comparison_rows).to_csv(
        DIAG_DIR / "step6_capacity_cap_comparison.csv",
        index=False,
    )
    pd.DataFrame(position_summary_rows).to_csv(
        DIAG_DIR / "step6_capacity_position_audit.csv",
        index=False,
    )
    pd.DataFrame(borrow_contribution_rows).to_csv(
        DIAG_DIR / "step6_borrow_high_si_contribution.csv",
        index=False,
    )
    pd.DataFrame(exclusion_rows).to_csv(
        DIAG_DIR / "step6_borrow_high_si_exclusion.csv",
        index=False,
    )
    pd.DataFrame(_reporting_integrity_rows()).to_csv(
        DIAG_DIR / "step6_reporting_integrity_audit.csv",
        index=False,
    )

    print("Saved Step 6 audit diagnostics:")
    for filename in [
        "step6_capacity_cap_comparison.csv",
        "step6_capacity_position_audit.csv",
        "step6_borrow_high_si_contribution.csv",
        "step6_borrow_high_si_exclusion.csv",
        "step6_reporting_integrity_audit.csv",
        "step6_capacity_positions_50m.csv",
        "step6_capacity_positions_250m.csv",
        "step6_capacity_positions_1b.csv",
        "step6_pnl_no_cap_50m.csv",
        "step6_pnl_no_cap_250m.csv",
        "step6_pnl_no_cap_1b.csv",
        "step6_pnl_exclude_high_si_50m.csv",
        "step6_pnl_exclude_high_si_250m.csv",
        "step6_pnl_exclude_high_si_1b.csv",
    ]:
        print(f"  - {DIAG_DIR / filename}")


def main() -> None:
    """Run all Step 6 audit diagnostics."""
    run_audits()


if __name__ == "__main__":
    main()

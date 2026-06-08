"""
Step 5 Back-test: dollar-neutral overnight L/S equity strategy.

---------------------------------------------------------------------
  Commission  0.5 bps / leg  on notional traded
  Slippage    1.5 bps / leg  on notional traded
  Borrow Tier A  40 bps p.a. on short notional
  Borrow Tier B 200 bps p.a. on short notional
  Borrow Tier C 800 bps p.a. on short notional

Trading cost model:
  Each "leg" costs 2 bps (commission + slippage).
  Positions are sized daily; changes between consecutive days determine
  the number of legs required (enter new / exit old).
  Positions that PERSIST unchanged from t-1 to t are still cycled through
  the overnight hold: they exit at MOO_t (2 bps) and re-enter at MOC_t
  ---------------------------------------------------------------------
  This is conservative and consistent with a pure overnight strategy.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
import atexit
import sys
from collections.abc import Sequence
from typing import TextIO
from typing import TypeAlias

PathLike: TypeAlias = str | Path
DailyWeights: TypeAlias = dict[pd.Timestamp, dict[int, float]]

# Cost constants
COMMISSION_BPS  = 0.5
SLIPPAGE_BPS    = 1.5
COST_PER_LEG    = (COMMISSION_BPS + SLIPPAGE_BPS) * 1e-4   # 2 bps as fraction

# Strategy parameters
BASKET_FRAC       = 0.05    # high-conviction top/bottom 5%
PARTICIPATION_CAP = 0.05   # 5 % of ADV per position
GROSS_EXPOSURE_CAP = None   # no portfolio-level gross exposure cap
AUM_LEVELS        = [50e6, 250e6, 1_000e6]

# Point-in-time signal-conviction filter. Current-day alpha dispersion is known
# before the MOC trade; thresholds below use only prior days.
MIN_ALPHA_SPREAD     = 0.0   # legacy fixed threshold disabled for headline run
SPREAD_LOOKBACK_DAYS = 252
SPREAD_MIN_PERIODS   = 60
SPREAD_QUANTILE      = 0.80
IC_LOOKBACK_DAYS     = 126
IC_MIN_PERIODS       = 20
MIN_ROLLING_IC       = 0.02


BACKTEST_START = pd.Timestamp("2013-01-01")
BACKTEST_END   = pd.Timestamp("2024-12-31")


class TeeStdout:
    """Write console output to the terminal and a log file."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        """Write data to every stream and return the number of characters written."""
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        """Flush every wrapped stream."""
        for stream in self.streams:
            stream.flush()


def start_console_log(log_path: Path) -> Path:
    """Mirror all Step 5 print output to a diagnostics text file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = TeeStdout(original_stdout, log_file)

    def _close_log():
        try:
            sys.stdout.flush()
        finally:
            sys.stdout = original_stdout
            log_file.close()

    atexit.register(_close_log)
    return log_path


def _allocate_with_caps(raw_scores, target_notional, caps):
    """Allocate notional proportionally and iteratively redistribute cap overflows.

    raw_scores and caps must share the same index and contain non-negative values.
    The function returns a Series of allocated notional that sums to at most
    target_notional, with the unused amount redistributed pro-rata among names
    that are still below cap until convergence.
    """
    if len(raw_scores) == 0 or target_notional <= 0:
        return pd.Series(dtype=float)

    weights = raw_scores.clip(lower=0.0).astype(float).copy()
    caps = caps.reindex(weights.index).astype(float).copy()
    caps = caps.replace([np.inf, -np.inf], np.nan)

    # Names without a usable cap are treated as uncapped.
    caps = caps.fillna(np.inf)

    remaining = float(target_notional)
    active = weights.index[weights > 0]
    allocation = pd.Series(0.0, index=weights.index, dtype=float)

    # If all raw scores are zero, fall back to equal weights.
    if weights.sum() <= 0:
        weights.loc[:] = 1.0
        active = weights.index

    while len(active) > 0 and remaining > 0:
        active_weights = weights.loc[active]
        active_caps = caps.loc[active]
        weight_sum = float(active_weights.sum())

        if weight_sum <= 0:
            proposal = pd.Series(remaining / len(active), index=active, dtype=float)
        else:
            proposal = active_weights / weight_sum * remaining

        capped = proposal > active_caps + 1e-12
        if not capped.any():
            allocation.loc[active] = proposal
            remaining = 0.0
            break

        allocation.loc[capped.index[capped]] = active_caps.loc[capped.index[capped]]
        remaining -= float(active_caps.loc[capped.index[capped]].sum())
        active = active[~capped.values]

        # Numerical safety: if the remaining amount is effectively exhausted,
        # stop rather than looping on floating-point noise.
        if remaining <= 1e-12:
            break

    return allocation


# I/O

def load_data(output_dir: PathLike = "../Output") -> pd.DataFrame:
    """Return the Step 4 alpha panel merged with Step 3 capacity fields."""
    base = Path(output_dir)

    alpha = pd.read_parquet(base / "step4_alpha_scores.parquet")
    alpha["date"] = pd.to_datetime(alpha["date"])

    # Participation cap and risk-scaling inputs live in step3.
    step3_cols = [
        "instrument_id", "date", "max_position_by_adv",
        "feature_vol20_lag1", "feature_adv20_lag1", "regime",
    ]
    cap_df = pd.read_parquet(
        base / "step3_borrow_model.parquet",
        columns=step3_cols,
    )
    cap_df["date"] = pd.to_datetime(cap_df["date"])
    # Filter to backtest window before merging to minimise memory
    cap_df = cap_df[cap_df["date"] >= BACKTEST_START].reset_index(drop=True)

    df = alpha.merge(cap_df, on=["instrument_id", "date"], how="left")
    del cap_df  # free memory

    mask = (df["date"] >= BACKTEST_START) & (df["date"] <= BACKTEST_END)
    df = df[mask].sort_values(["date", "instrument_id"]).reset_index(drop=True)
    if "feature_r_overnight_std20" not in df.columns:
        df["feature_r_overnight_std20"] = df["feature_vol20_lag1"]
    df["feature_day_of_week"] = df["date"].dt.dayofweek.astype(np.int8)
    df["feature_month"] = df["date"].dt.month.astype(np.int8)
    return df


# Portfolio construction per date

def build_daily_weights(
    df: pd.DataFrame,
    portfolio_aum: float,
    basket_frac: float | None = None,
    min_alpha_spread: float = 0.0,
    scale_by_vol: bool = False,
) -> DailyWeights:
    """
    ---------------------------------------------------------------------
    {date: {instrument_id: signed_dollar_weight}}).

    Long  = positive dollar weight (top quintile by alpha_score).
    Short = negative dollar weight (bottom quintile), filtered for borrow cost:
      - Tier C (800 bps/year) is hard-excluded from the short basket: borrow
        ---------------------------------------------------------------------
      - Remaining short candidates are sorted so Tier A names are preferred
        over Tier B before the basket is filled to n_basket names.
    Sizes are score-proportional within each side, then capped by ADV participation.

    min_alpha_spread : float
        ---------------------------------------------------------------------
        inter-decile range of alpha_score (p90 - p10) is computed for the
        eligible universe.  If the spread is below this threshold the day is
        skipped (weights = {}) because the expected gross alpha is unlikely to
        cover the round-trip cost of 4 bps per dollar of gross exposure.
        Default 0.0 = no filtering (backward-compatible).
    """
    if basket_frac is None:
        basket_frac = BASKET_FRAC
    target_side_notional = portfolio_aum * 0.5   # 50 % long, 50 % short

    daily_weights = {}

    for date, g in df.groupby("date"):
        eligible = g[g["is_trade_eligible"] & g["alpha_score"].notna()].copy()
        if len(eligible) < 20:
            daily_weights[date] = {}
            continue

        # Signal-conviction filter
        # Skip days where the cross-sectional signal dispersion is so low that
        # expected alpha cannot recover the ~4 bps round-trip cost.
        if min_alpha_spread > 0.0:
            spread = (eligible["alpha_score"].quantile(0.9)
                      - eligible["alpha_score"].quantile(0.1))
            if spread < min_alpha_spread:
                daily_weights[date] = {}
                continue

        n_basket = max(5, int(len(eligible) * basket_frac))
        eligible = eligible.sort_values("alpha_score")

        long_idx  = eligible.tail(n_basket).index

        # Hard-exclude Tier C from short basket ( 4.3 hard exclusion).
        # Among remaining shortable candidates (sorted by alpha_score ascending),
        # prefer Tier A over Tier B so the cheapest-to-borrow names fill first.
        short_pool = eligible[
            eligible["is_shortable"]
            & (eligible.get("borrow_tier", "A") != "C")
        ].copy()
        # Tier A = 0, Tier B = 1 for sort priority; alpha_score already ascending.
        _tier_order = short_pool.get("borrow_tier", pd.Series("A", index=short_pool.index))
        short_pool["_tier_rank"] = _tier_order.map({"A": 0, "B": 1}).fillna(0).astype(int)
        short_pool = short_pool.sort_values(["_tier_rank", "alpha_score"])
        short_idx  = short_pool.head(n_basket).index

        weights = {}

        # Score-proportional weighting: weight_i |score_i - cross-sectional median|
        # Per 6.1: "score-weighted (proportional to s - median)".
        cs_median = eligible["alpha_score"].median()

        for idx_set, sign in [(long_idx, 1.0), (short_idx, -1.0)]:
            if len(idx_set) == 0:
                continue
            scores_sub = eligible.loc[idx_set, "alpha_score"]
            if sign == 1.0:
                raw_scores = (scores_sub - cs_median).clip(lower=0.0)
            else:
                raw_scores = (cs_median - scores_sub).clip(lower=0.0)
            # Optionally scale raw scores by lagged instrument return volatility.
            if scale_by_vol:
                if "feature_r_overnight_std20" in eligible.columns:
                    vol_col = eligible["feature_r_overnight_std20"]
                elif "feature_vol20_lag1" in eligible.columns:
                    vol_col = eligible["feature_vol20_lag1"]
                else:
                    vol_col = None
                if vol_col is None:
                    vol_series = pd.Series(1.0, index=scores_sub.index)
                else:
                    vol_series = vol_col.loc[scores_sub.index].replace(0.0, np.nan).fillna(1e-4)
                risk_scale = 1.0 / vol_series.clip(lower=1e-4)
                raw_scores = raw_scores * risk_scale
            cap_vals = g.loc[idx_set, "max_position_by_adv"].copy()
            cap_vals = cap_vals.where(cap_vals.notna() & (cap_vals > 0), np.inf)
            alloc = _allocate_with_caps(raw_scores, target_side_notional, cap_vals)
            for idx in idx_set:
                iid = g.loc[idx, "instrument_id"]
                weights[iid] = sign * float(alloc.get(idx, 0.0))

        daily_weights[date] = weights

    return daily_weights


def compute_daily_alpha_spread(df: pd.DataFrame) -> pd.Series:
    """Daily p90-p10 alpha dispersion, known before the MOC trade."""
    rows = []
    for date, g in df.groupby("date"):
        eligible = g[g["is_trade_eligible"] & g["alpha_score"].notna()]
        if len(eligible) < 20:
            rows.append((date, np.nan))
            continue
        rows.append((
            date,
            eligible["alpha_score"].quantile(0.9)
            - eligible["alpha_score"].quantile(0.1),
        ))
    return pd.Series(dict(rows)).sort_index()


def compute_daily_ic(df: pd.DataFrame) -> pd.Series:
    """
    Daily cross-sectional IC between alpha_score and realised next overnight
    return.  This is used only after shift(1), so the gate for date t uses ICs
    already realised before the MOC decision on date t.
    """
    rows = []
    for date, g in df.groupby("date"):
        valid = (
            g["is_trade_eligible"]
            & g["alpha_score"].notna()
            & g["target_r_overnight_next"].notna()
        )
        if valid.sum() < 20:
            rows.append((date, np.nan))
            continue
        rho = (
            g.loc[valid, "alpha_score"].rank()
            .corr(g.loc[valid, "target_r_overnight_next"].rank())
        )
        rows.append((date, rho))
    return pd.Series(dict(rows)).sort_index()


def build_rolling_signal_gate(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    Point-in-time day filter:
      1. trade only when today's alpha spread clears a rolling prior threshold;
      2. trade only when trailing realised IC is positive enough.

    Both thresholds use shifted histories, so no same-day realised return enters
    the decision.
    """
    spread = compute_daily_alpha_spread(df)
    ic = compute_daily_ic(df)

    spread_threshold = (
        spread.shift(1)
        .rolling(SPREAD_LOOKBACK_DAYS, min_periods=SPREAD_MIN_PERIODS)
        .quantile(SPREAD_QUANTILE)
    )
    rolling_ic = (
        ic.shift(1)
        .rolling(IC_LOOKBACK_DAYS, min_periods=IC_MIN_PERIODS)
        .mean()
    )
    trade_allowed = (spread >= spread_threshold) & (rolling_ic >= MIN_ROLLING_IC)

    diag = pd.DataFrame({
        "date": spread.index,
        "alpha_spread": spread.values,
        "spread_threshold": spread_threshold.reindex(spread.index).values,
        "rolling_ic": rolling_ic.reindex(spread.index).values,
        "trade_allowed": trade_allowed.reindex(spread.index).fillna(False).values,
    })
    return trade_allowed.fillna(False), diag


def apply_date_gate(daily_weights: DailyWeights, signal_gate: pd.Series | None) -> DailyWeights:
    """Set weights to empty on dates that fail the point-in-time signal gate."""
    if signal_gate is None:
        return daily_weights
    return {
        date: (weights.copy() if bool(signal_gate.get(date, False)) else {})
        for date, weights in daily_weights.items()
    }


def apply_gross_exposure_cap(
    daily_weights: DailyWeights,
    portfolio_aum: float,
    gross_exposure_cap: float | None = GROSS_EXPOSURE_CAP,
) -> DailyWeights:
    """Scale each day's positions so gross notional does not exceed cap * AUM."""
    if gross_exposure_cap is None:
        return daily_weights

    capped = {}
    max_gross = portfolio_aum * gross_exposure_cap
    for date, weights in daily_weights.items():
        gross = sum(abs(w) for w in weights.values())
        if gross > max_gross and gross > 0:
            scale = max_gross / gross
            capped[date] = {iid: w * scale for iid, w in weights.items()}
        else:
            capped[date] = weights.copy()
    return capped


# Daily P&L engine

def run_backtest(
    df: pd.DataFrame,
    portfolio_aum: float,
    basket_frac: float | None = None,
    min_alpha_spread: float = 0.0,
    signal_gate: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Iterate over dates and compute daily net P&L.

    Cost model (pure overnight, fully cycled each night):
      - Every dollar of notional held overnight incurs 4 bps total
        (2 bps MOC entry + 2 bps MOO exit next morning).
      - Short positions additionally incur daily borrow cost.
    """
    daily_weights = build_daily_weights(
        df, portfolio_aum,
        basket_frac=basket_frac,
        min_alpha_spread=min_alpha_spread,
        scale_by_vol=True,
    )
    daily_weights = apply_date_gate(daily_weights, signal_gate)
    daily_weights = apply_gross_exposure_cap(daily_weights, portfolio_aum)
    dates = sorted(daily_weights.keys())

    # Lookup tables indexed by (date, instrument_id)
    ret_lookup  = df.set_index(["date", "instrument_id"])["target_r_overnight_next"]
    borrow_lookup = df.set_index(["date", "instrument_id"])["borrow_cost_daily"]

    records = []
    prev_weights = {}

    for date in dates:
        curr_weights = daily_weights[date]

        # Gross P&L
        gross_pnl = 0.0
        for iid, dw in curr_weights.items():
            ret = ret_lookup.get((date, iid), np.nan)
            if not np.isnan(ret):
                gross_pnl += dw * ret

        # Trading cost: 4 bps per persisting dollar + 2 bps on new/exited
        all_ids = set(curr_weights) | set(prev_weights)
        trade_cost = 0.0
        for iid in all_ids:
            curr_dw = curr_weights.get(iid, 0.0)
            prev_dw = prev_weights.get(iid, 0.0)
            if prev_dw != 0.0 and curr_dw != 0.0:
                # Position persists: full round-trip (exit MOO + re-enter MOC)
                continuing = min(abs(prev_dw), abs(curr_dw))
                change     = abs(curr_dw - prev_dw)
                trade_cost += continuing * 2 * COST_PER_LEG   # 4 bps
                trade_cost += change     * COST_PER_LEG        # 2 bps (one leg)
            elif prev_dw != 0.0:
                # Position exited: 2 bps MOO exit
                trade_cost += abs(prev_dw) * COST_PER_LEG
            else:
                # New position: 2 bps MOC entry
                trade_cost += abs(curr_dw) * COST_PER_LEG

        # Borrow cost on short notional
        borrow_cost = 0.0
        for iid, dw in curr_weights.items():
            if dw < 0:
                b = borrow_lookup.get((date, iid), 0.0)
                borrow_cost += abs(dw) * b

        net_pnl = gross_pnl - trade_cost - borrow_cost

        gross_long  = sum(dw for dw in curr_weights.values() if dw > 0)
        gross_short = sum(abs(dw) for dw in curr_weights.values() if dw < 0)
        gross_notional = gross_long + gross_short

        # One-way turnover as fraction of gross notional
        turnover_dollars = sum(
            abs(curr_weights.get(iid, 0.0) - prev_weights.get(iid, 0.0))
            for iid in all_ids
        )

        records.append({
            "date"          : date,
            "gross_pnl"     : gross_pnl,
            "trade_cost"    : trade_cost,
            "borrow_cost"   : borrow_cost,
            "net_pnl"       : net_pnl,
            "gross_long"    : gross_long,
            "gross_short"   : gross_short,
            "gross_notional": gross_notional,
            "turnover_$"    : turnover_dollars,
            "n_long"        : sum(1 for dw in curr_weights.values() if dw > 0),
            "n_short"       : sum(1 for dw in curr_weights.values() if dw < 0),
        })

        prev_weights = {iid: dw for iid, dw in curr_weights.items() if dw != 0}

    return pd.DataFrame(records).set_index("date")


def _compute_rolling_hit_rate(df, window=20):
    """Per-instrument rolling hit rate: fraction of last `window` days where
    sign(alpha_score) == sign(target_r_overnight_next).  Returns a Series
    indexed by (date, instrument_id)."""
    tmp = df[["date", "instrument_id", "alpha_score", "target_r_overnight_next"]].copy()
    tmp = tmp.dropna(subset=["alpha_score", "target_r_overnight_next"])
    tmp["hit"] = np.sign(tmp["alpha_score"]) == np.sign(tmp["target_r_overnight_next"])
    tmp = tmp.sort_values(["instrument_id", "date"])
    # rolling mean within each instrument, shift(1) so we never use today's return
    tmp["hit_rate"] = (
        tmp.groupby("instrument_id")["hit"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=5).mean())
    )
    return tmp.set_index(["date", "instrument_id"])["hit_rate"]


def _compute_rolling_ic(df, window=20):
    """Daily cross-sectional IC (rank correlation of alpha_score vs next-day return),
    then rolling z-score over `window` days.  Returns a Series indexed by date."""
    def _cs_ic(g: pd.DataFrame) -> float:
        """Return the rank IC for one daily cross-section."""
        a = g["alpha_score"]
        r = g["target_r_overnight_next"]
        valid = a.notna() & r.notna()
        if valid.sum() < 5:
            return np.nan
        return a[valid].rank().corr(r[valid].rank())

    daily_ic = df.groupby("date").apply(_cs_ic)
    # rolling z-score (shift 1 so today's IC is not used as feature for today)
    ic_roll_mean = daily_ic.shift(1).rolling(window, min_periods=5).mean()
    ic_roll_std  = daily_ic.shift(1).rolling(window, min_periods=5).std().replace(0.0, np.nan)
    ic_zscore = ((daily_ic.shift(1) - ic_roll_mean) / ic_roll_std).fillna(0.0)
    return ic_zscore


def generate_meta_samples(df: pd.DataFrame, daily_weights: DailyWeights) -> pd.DataFrame:
    """Generate per-date meta-training samples with a day-level label.

    Label design (day-level, not single-trade):
      label=1 if the weighted portfolio signal hit rate on this date > 0.5,
      i.e. more than half of the dollar-weighted positions had
      sign(alpha_score) == sign(actual overnight return).

    Features include per-instrument rolling hit-rate (momentum of signal
    quality) and cross-sectional IC z-score (regime indicator).
    """
    # Pre-compute rolling features
    hit_rate_idx = _compute_rolling_hit_rate(df, window=20)
    ic_zscore    = _compute_rolling_ic(df, window=20)

    feat_cols = [
        "target_r_overnight_next", "borrow_cost_daily", "alpha_score",
        "target_rank_normalized", "feature_vol20_lag1", "feature_adv20_lag1",
        "feature_r_overnight_std20", "feature_ds_ma5", "feature_beta_on",
        "is_shortable", "feature_day_of_week", "feature_month",
    ]
    available = [c for c in feat_cols if c in df.columns]
    df_idx = df.set_index(["date", "instrument_id"])[available]

    rows = []
    for date, weights in daily_weights.items():
        if not weights:
            continue
        total_abs_w = sum(abs(w) for w in weights.values())
        if total_abs_w == 0:
            continue

        # Day-level label: weighted hit rate > 0.5
        weighted_hits = 0.0
        n_valid = 0
        for iid, w in weights.items():
            try:
                r_row = df_idx.loc[(date, iid)]
            except KeyError:
                continue
            ret = r_row.get("target_r_overnight_next", np.nan)
            alpha = r_row.get("alpha_score", np.nan)
            if pd.isna(ret) or pd.isna(alpha):
                continue
            hit = 1.0 if np.sign(alpha) == np.sign(ret) else 0.0
            weighted_hits += abs(w) * hit
            n_valid += 1
        if n_valid == 0:
            continue
        day_hit_rate = weighted_hits / sum(
            abs(w) for iid, w in weights.items()
            if not pd.isna(df_idx.loc[(date, iid)].get("target_r_overnight_next", np.nan))
               if (date, iid) in df_idx.index
        ) if n_valid > 0 else 0.5
        label = 1 if day_hit_rate > 0.5 else 0

        # Day-level features: cross-sectional stats of the basket on this date
        basket_keys = [(date, iid) for iid in weights if (date, iid) in df_idx.index]
        if not basket_keys:
            continue
        basket = df_idx.loc[basket_keys]

        # Average rolling hit rate across basket instruments
        hr_vals = [hit_rate_idx.get((date, iid), np.nan) for iid in weights]
        avg_hit_rate = np.nanmean(hr_vals) if any(not np.isnan(v) for v in hr_vals) else 0.5

        alpha_spread = basket["alpha_score"].quantile(0.9) - basket["alpha_score"].quantile(0.1) \
            if "alpha_score" in basket.columns else 0.0
        avg_vol = basket["feature_vol20_lag1"].mean() if "feature_vol20_lag1" in basket.columns else np.nan
        avg_r_on_std = basket["feature_r_overnight_std20"].mean() if "feature_r_overnight_std20" in basket.columns else np.nan
        avg_ds_ma5 = basket["feature_ds_ma5"].mean() if "feature_ds_ma5" in basket.columns else np.nan
        avg_beta = basket["feature_beta_on"].mean() if "feature_beta_on" in basket.columns else np.nan
        n_basket = len(weights)

        rows.append({
            "date": date,
            "label": label,
            "day_hit_rate": day_hit_rate,
            "avg_rolling_hit_rate": avg_hit_rate,   # primary new feature
            "ic_zscore": ic_zscore.get(date, 0.0),  # regime feature
            "alpha_spread": alpha_spread,
            "avg_vol": avg_vol,
            "avg_r_on_std": avg_r_on_std,
            "avg_ds_ma5": avg_ds_ma5,
            "avg_beta": avg_beta,
            "n_basket": n_basket,
            "day_of_week": int(pd.to_datetime(date).dayofweek),
            "month": int(pd.to_datetime(date).month),
        })

    return pd.DataFrame(rows)


def train_yearly_meta_models(samples: pd.DataFrame, years: Sequence[int]) -> dict[int, object | None]:
    """Train one meta-model per predict year using samples up to year-1.

    Day-level model: predict whether tomorrow will be a profitable trading day
    based on rolling signal quality metrics and regime indicators.
    """
    models = {}
    feature_cols = [
        "avg_rolling_hit_rate",  # recent signal quality per instrument
        "ic_zscore",             # cross-sectional IC regime
        "alpha_spread",          # signal conviction today
        "avg_vol",
        "avg_r_on_std",
        "avg_ds_ma5",
        "avg_beta",
        "n_basket",
        "day_of_week",
        "month",
    ]
    for yr in years:
        cutoff = pd.Timestamp(f"{yr-1}-12-31")
        train = samples[samples["date"] <= cutoff]
        if len(train) < 100:  # day-level samples are fewer than trade-level
            models[yr] = None
            continue
        X = train[feature_cols].fillna(0.0)
        y = train["label"].astype(int)
        clf = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            num_leaves=16, min_child_samples=10,
            random_state=42, verbosity=-1,
        )
        clf.fit(X, y)
        models[yr] = (clf, feature_cols)
    return models


def apply_meta_filter_to_weights(
    daily_weights: DailyWeights,
    models: dict[int, object | None],
    df: pd.DataFrame,
    prob_threshold: float = 0.5,
    min_days_per_year: int = 0,
) -> DailyWeights:
    """Return a filtered copy of daily_weights; day-level model gates entire days.

    The metamodel predicts whether a TRADING DAY should be traded at all.
    If prob < prob_threshold the whole day's weights are set to {}.

    min_days_per_year : int
        Hard floor: for each calendar year, if the number of trading days
        passing the threshold is below this value, the top-scoring rejected
        days (by metamodel prob) are reinstated until the floor is met.
        This prevents years like 2023/2024 from having zero activity even
        when the model sees no day above 0.5.
    """
    # Pre-compute rolling features needed for inference
    hit_rate_idx = _compute_rolling_hit_rate(df, window=20)
    ic_zscore    = _compute_rolling_ic(df, window=20)

    feat_lookup_cols = [
        "alpha_score", "feature_vol20_lag1",
        "feature_r_overnight_std20", "feature_ds_ma5", "feature_beta_on",
    ]
    available = [c for c in feat_lookup_cols if c in df.columns]
    df_idx = df.set_index(["date", "instrument_id"])[available]

    # First pass: compute prob for every date, store both result and prob
    date_probs = {}   # date -> prob (NaN if no model)
    for date, weights in daily_weights.items():
        yr = int(pd.to_datetime(date).year)
        model_tuple = models.get(yr)
        if model_tuple is None or not weights:
            date_probs[date] = np.nan
            continue
        clf, feature_cols = model_tuple

        basket_keys = [(date, iid) for iid in weights if (date, iid) in df_idx.index]
        if not basket_keys:
            date_probs[date] = np.nan
            continue
        basket = df_idx.loc[basket_keys]

        hr_vals = [hit_rate_idx.get((date, iid), np.nan) for iid in weights]
        avg_hit_rate = np.nanmean(hr_vals) if any(not np.isnan(v) for v in hr_vals) else 0.5

        alpha_spread = basket["alpha_score"].quantile(0.9) - basket["alpha_score"].quantile(0.1) \
            if "alpha_score" in basket.columns else 0.0

        X_day = pd.DataFrame([{
            "avg_rolling_hit_rate": avg_hit_rate,
            "ic_zscore": ic_zscore.get(date, 0.0),
            "alpha_spread": alpha_spread,
            "avg_vol": basket["feature_vol20_lag1"].mean() if "feature_vol20_lag1" in basket.columns else 0.0,
            "avg_r_on_std": basket["feature_r_overnight_std20"].mean() if "feature_r_overnight_std20" in basket.columns else 0.0,
            "avg_ds_ma5": basket["feature_ds_ma5"].mean() if "feature_ds_ma5" in basket.columns else 0.0,
            "avg_beta": basket["feature_beta_on"].mean() if "feature_beta_on" in basket.columns else 0.0,
            "n_basket": len(weights),
            "day_of_week": int(pd.to_datetime(date).dayofweek),
            "month": int(pd.to_datetime(date).month),
        }])
        for fc in feature_cols:
            if fc not in X_day.columns:
                X_day[fc] = 0.0
        date_probs[date] = float(clf.predict_proba(X_day[feature_cols].fillna(0.0))[0, 1])

    # Second pass: apply threshold, then enforce min_days_per_year floor
    # Group dates by year to identify shortfall years
    from collections import defaultdict
    dates_by_year = defaultdict(list)
    for date in daily_weights:
        dates_by_year[int(pd.to_datetime(date).year)].append(date)

    # For each year, determine which extra dates to reinstate
    extra_dates = set()
    for yr, yr_dates in dates_by_year.items():
        # dates that have a valid prob (i.e. model existed and basket non-empty)
        scored = [(d, date_probs[d]) for d in yr_dates
                  if not np.isnan(date_probs.get(d, np.nan)) and daily_weights[d]]
        passing = [d for d, p in scored if p >= prob_threshold]
        shortfall = min_days_per_year - len(passing)
        if shortfall > 0:
            # reinstate top-prob rejected days
            rejected = sorted(
                [(d, p) for d, p in scored if p < prob_threshold],
                key=lambda x: -x[1]
            )
            for d, _ in rejected[:shortfall]:
                extra_dates.add(d)

    # Build filtered dict
    filtered = {}
    for date, weights in daily_weights.items():
        prob = date_probs.get(date, np.nan)
        if np.isnan(prob):
            # no model for this year or empty basket pass through unchanged
            filtered[date] = weights.copy()
        elif prob >= prob_threshold or date in extra_dates:
            filtered[date] = weights.copy()
        else:
            filtered[date] = {}
    return filtered


def run_backtest_with_weights(df: pd.DataFrame, daily_weights: DailyWeights) -> pd.DataFrame:
    """Compute PnL table given a precomputed daily_weights dict."""
    # Lookup tables indexed by (date, instrument_id)
    ret_lookup  = df.set_index(["date", "instrument_id"])["target_r_overnight_next"].to_dict()
    borrow_lookup = df.set_index(["date", "instrument_id"]).get("borrow_cost_daily", pd.Series(0.0)).to_dict()

    records = []
    prev_weights = {}
    dates = sorted(daily_weights.keys())
    for date in dates:
        curr_weights = daily_weights[date]
        gross_pnl = 0.0
        for iid, dw in curr_weights.items():
            ret = ret_lookup.get((date, iid), np.nan)
            if not np.isnan(ret):
                gross_pnl += dw * ret
        all_ids = set(curr_weights) | set(prev_weights)
        trade_cost = 0.0
        for iid in all_ids:
            curr_dw = curr_weights.get(iid, 0.0)
            prev_dw = prev_weights.get(iid, 0.0)
            if prev_dw != 0.0 and curr_dw != 0.0:
                continuing = min(abs(prev_dw), abs(curr_dw))
                change     = abs(curr_dw - prev_dw)
                trade_cost += continuing * 2 * COST_PER_LEG
                trade_cost += change     * COST_PER_LEG
            elif prev_dw != 0.0:
                trade_cost += abs(prev_dw) * COST_PER_LEG
            else:
                trade_cost += abs(curr_dw) * COST_PER_LEG
        borrow_cost = 0.0
        for iid, dw in curr_weights.items():
            if dw < 0:
                b = borrow_lookup.get((date, iid), 0.0)
                borrow_cost += abs(dw) * b
        net_pnl = gross_pnl - trade_cost - borrow_cost
        gross_long  = sum(dw for dw in curr_weights.values() if dw > 0)
        gross_short = sum(abs(dw) for dw in curr_weights.values() if dw < 0)
        gross_notional = gross_long + gross_short
        turnover_dollars = sum(
            abs(curr_weights.get(iid, 0.0) - prev_weights.get(iid, 0.0))
            for iid in all_ids
        )
        records.append({
            "date": date,
            "gross_pnl": gross_pnl,
            "trade_cost": trade_cost,
            "borrow_cost": borrow_cost,
            "net_pnl": net_pnl,
            "gross_long": gross_long,
            "gross_short": gross_short,
            "gross_notional": gross_notional,
            "turnover_$": turnover_dollars,
            "n_long": sum(1 for dw in curr_weights.values() if dw > 0),
            "n_short": sum(1 for dw in curr_weights.values() if dw < 0),
        })
        prev_weights = {iid: dw for iid, dw in curr_weights.items() if dw != 0}
    return pd.DataFrame(records).set_index("date")


# Performance metrics

def sharpe(ret_series: pd.Series) -> float:
    """Annualised Sharpe (daily returns input)."""
    m, s = ret_series.mean(), ret_series.std()
    return m / s * np.sqrt(252) if s > 0 else np.nan


def max_drawdown(cum_ret: pd.Series) -> float:
    """Return the minimum percentage drawdown from a cumulative return curve."""
    roll_max = cum_ret.expanding().max()
    return ((cum_ret - roll_max) / roll_max).min()


def performance_table(pnl_df: pd.DataFrame, portfolio_aum: float) -> dict[str, float]:
    """Return headline annualised performance and exposure metrics."""
    net_ret   = pnl_df["net_pnl"]   / portfolio_aum
    gross_ret = pnl_df["gross_pnl"] / portfolio_aum

    cum_net = (1 + net_ret).cumprod()

    ann      = 252
    n        = len(net_ret)

    net_ann_ret  = net_ret.mean()  * ann
    net_ann_vol  = net_ret.std()   * np.sqrt(ann)
    net_sharpe   = sharpe(net_ret)
    gross_sharpe = sharpe(gross_ret)
    mdd          = max_drawdown(cum_net)

    # Cost contributions
    trade_cost_ann  = pnl_df["trade_cost"].mean()  * ann / portfolio_aum
    borrow_cost_ann = pnl_df["borrow_cost"].mean() * ann / portfolio_aum

    # Turnover: one-way daily as fraction of AUM
    avg_ow_turnover = (pnl_df["turnover_$"] / portfolio_aum / 2).mean()

    avg_gross_exp = pnl_df["gross_notional"].mean() / portfolio_aum

    # Fraction of days where ADV participation cap is binding: actual gross
    # exposure falls materially below the target AUM ( 6.5 required metric).
    frac_at_cap = (pnl_df["gross_notional"] < portfolio_aum * 0.95).mean()

    return {
        "net_ann_ret (%)":              round(net_ann_ret  * 100, 2),
        "net_ann_vol (%)":              round(net_ann_vol  * 100, 2),
        "net_sharpe":                   round(net_sharpe,  3),
        "gross_sharpe":                 round(gross_sharpe,3),
        "max_drawdown (%)":             round(mdd          * 100, 2),
        "avg_daily_turnover (1-way %)": round(avg_ow_turnover * 100, 1),
        "avg_gross_exposure (%)":       round(avg_gross_exp * 100, 1),
        "frac_days_at_cap (%)":         round(frac_at_cap  * 100, 1),
        "trade_cost_ann (bps)":         round(trade_cost_ann * 1e4, 1),
        "borrow_cost_ann (bps)":        round(borrow_cost_ann * 1e4, 1),
    }


def cost_decomposition(pnl_df: pd.DataFrame, portfolio_aum: float, aum_label: str) -> dict[str, float | str]:
    """Decompose gross-to-net Sharpe degradation by cost component.

    The back-test stores total trading cost as commission plus slippage.  Since
    the cost schedule is 0.5 bps commission and 1.5 bps slippage per leg, the
    saved trade cost is split in the same 1:3 ratio.
    """
    trade_cost = pnl_df["trade_cost"].astype(float)
    trade_cost_per_leg_bps = COMMISSION_BPS + SLIPPAGE_BPS
    commission_share = COMMISSION_BPS / trade_cost_per_leg_bps
    slippage_share = SLIPPAGE_BPS / trade_cost_per_leg_bps

    commission_cost = trade_cost * commission_share
    slippage_cost = trade_cost * slippage_share
    borrow_cost = pnl_df["borrow_cost"].astype(float)
    gross_pnl = pnl_df["gross_pnl"].astype(float)

    gross_ret = gross_pnl / portfolio_aum
    after_commission_ret = (gross_pnl - commission_cost) / portfolio_aum
    after_slippage_ret = (
        gross_pnl - commission_cost - slippage_cost
    ) / portfolio_aum
    net_ret = (
        gross_pnl - commission_cost - slippage_cost - borrow_cost
    ) / portfolio_aum

    gross_sh = sharpe(gross_ret)
    after_commission_sh = sharpe(after_commission_ret)
    after_slippage_sh = sharpe(after_slippage_ret)
    net_sh = sharpe(net_ret)

    ann = 252
    return {
        "aum_label": aum_label,
        "aum": portfolio_aum,
        "gross_sharpe": gross_sh,
        "after_commission_sharpe": after_commission_sh,
        "after_slippage_sharpe": after_slippage_sh,
        "net_sharpe": net_sh,
        "total_sharpe_degradation": gross_sh - net_sh,
        "commission_sharpe_drag": gross_sh - after_commission_sh,
        "slippage_sharpe_drag": after_commission_sh - after_slippage_sh,
        "trading_cost_sharpe_drag": gross_sh - after_slippage_sh,
        "borrow_sharpe_drag": after_slippage_sh - net_sh,
        "commission_cost_ann_bps": commission_cost.mean() * ann / portfolio_aum * 1e4,
        "slippage_cost_ann_bps": slippage_cost.mean() * ann / portfolio_aum * 1e4,
        "trading_cost_ann_bps": trade_cost.mean() * ann / portfolio_aum * 1e4,
        "borrow_cost_ann_bps": borrow_cost.mean() * ann / portfolio_aum * 1e4,
        "total_cost_ann_bps": (
            commission_cost.mean() + slippage_cost.mean() + borrow_cost.mean()
        ) * ann / portfolio_aum * 1e4,
    }


def yearly_sharpe(pnl_df: pd.DataFrame, portfolio_aum: float) -> pd.DataFrame:
    """Return yearly net Sharpe, annual return, trade count, and basket sizes."""
    net_ret = (pnl_df["net_pnl"] / portfolio_aum).copy()
    net_ret.index = pd.to_datetime(net_ret.index)
    pnl_df = pnl_df.copy()
    pnl_df.index = pd.to_datetime(pnl_df.index)
    results = []
    for yr, g in net_ret.groupby(net_ret.index.year):
        pnl_year = pnl_df.loc[pnl_df.index.year == yr]
        active = pnl_year[(pnl_year["n_long"] > 0) | (pnl_year["n_short"] > 0)]
        results.append({
            "year": yr,
            "net_sharpe": round(sharpe(g), 3),
            "net_ret (%)": round(g.mean() * 252 * 100, 2),
            "trade_days": int(len(active)),
            "n_long_avg_all": round(pnl_year["n_long"].mean(), 2),
            "n_short_avg_all": round(pnl_year["n_short"].mean(), 2),
            "n_long_avg_traded": round(active["n_long"].mean(), 1) if len(active) else 0.0,
            "n_short_avg_traded": round(active["n_short"].mean(), 1) if len(active) else 0.0,
        })
    return pd.DataFrame(results)


# Entry point

if __name__ == "__main__":
    out_dir = Path("../Output/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    console_log_path = start_console_log(out_dir / "step5_console_output.txt")
    print(f"Step 5 console output will be saved -> {console_log_path}")

    print("Loading alpha scores + capacity data...")
    df = load_data()
    aum_labels = {50e6: "50M", 250e6: "250M", 1_000e6: "1B"}

    print(f"Backtest window: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Participation cap: {PARTICIPATION_CAP:.0%} ADV\n")

    print("Building point-in-time rolling signal gate...")
    signal_gate, gate_diag = build_rolling_signal_gate(df)
    print(
        "Rolling gate: "
        f"spread >= prior {SPREAD_LOOKBACK_DAYS}d p{SPREAD_QUANTILE:.1%}, "
        f"rolling IC({IC_LOOKBACK_DAYS}d) >= {MIN_ROLLING_IC:.3f}"
    )
    print(
        f"Trade days allowed: {int(signal_gate.sum())} / {len(signal_gate)} "
        f"({signal_gate.mean():.1%})\n"
    )

    # Diagnostic only: the headline uses the fixed BASKET_FRAC constant above.
    SWEEP_FRACS = [0.05, 0.075, 0.10, 0.125, 0.15]
    print("-" * 72)
    print("BASKET_FRAC DIAGNOSTIC (@ $250M AUM, fixed rolling gate)")
    print("-" * 72)
    print(f"  {'frac':>6}  {'net_sharpe':>12}  {'gross_sharpe':>13}  "
          f"{'net_ret%':>9}  {'net_vol%':>9}  {'trade_cost_bps':>15}")
    basket_diag_rows = []
    for frac in SWEEP_FRACS:
        pnl_s = run_backtest(
            df, 250e6, basket_frac=frac,
            min_alpha_spread=MIN_ALPHA_SPREAD,
            signal_gate=signal_gate,
        )
        m = performance_table(pnl_s, 250e6)
        basket_diag_rows.append({
            "basket_frac": frac,
            "aum_label": "250M",
            "aum": 250e6,
            **m,
        })
        print(f"  {frac:>6.1%}  {m['net_sharpe']:>12.3f}  "
              f"{m['gross_sharpe']:>13.3f}  "
              f"{m['net_ann_ret (%)']:>9.2f}  "
              f"{m['net_ann_vol (%)']:>9.2f}  "
              f"{m['trade_cost_ann (bps)']:>15.1f}")
    print("-" * 72)
    print(f"\nHeadline uses fixed BASKET_FRAC = {BASKET_FRAC:.1%}; no test-period best-frac selection.\n")

    # Headline table: all three AUM levels
    all_metrics = {}
    pnl_by_label = {}

    for aum in AUM_LEVELS:
        label = aum_labels[aum]
        print(f"Running backtest @ AUM = ${label} ...", flush=True)
        pnl = run_backtest(
            df, aum, basket_frac=BASKET_FRAC,
            min_alpha_spread=MIN_ALPHA_SPREAD,
            signal_gate=signal_gate,
        )
        metrics = performance_table(pnl, aum)
        all_metrics[label] = metrics
        pnl_by_label[label] = pnl

    print("\n" + "=" * 80)
    print("HEADLINE PERFORMANCE TABLE  (2013-2024, net of all costs)")
    print("=" * 80)
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.index.name = "AUM"
    print(metrics_df.to_string())
    cost_rows = []

    # Cost decomposition @ 250M
    for aum in AUM_LEVELS:
        label = aum_labels[aum]
        pnl = pnl_by_label[label]
        cost_row = cost_decomposition(pnl, aum, label)
        cost_rows.append(cost_row)

        print("\n" + "-" * 60)
        print(f"COST DECOMPOSITION @ ${label} AUM")
        print("-" * 60)
        print(f"  Gross Sharpe:             {cost_row['gross_sharpe']:>7.3f}")
        print(f"  After commission Sharpe:  {cost_row['after_commission_sharpe']:>7.3f}")
        print(f"  After slippage Sharpe:    {cost_row['after_slippage_sharpe']:>7.3f}")
        print(f"  Net   Sharpe:             {cost_row['net_sharpe']:>7.3f}")
        print(f"  Commission drag:          {cost_row['commission_cost_ann_bps']:>7.1f} bps/yr "
              f"({cost_row['commission_sharpe_drag']:>6.3f} Sharpe)")
        print(f"  Slippage drag:            {cost_row['slippage_cost_ann_bps']:>7.1f} bps/yr "
              f"({cost_row['slippage_sharpe_drag']:>6.3f} Sharpe)")
        print(f"  Borrow drag:              {cost_row['borrow_cost_ann_bps']:>7.1f} bps/yr "
              f"({cost_row['borrow_sharpe_drag']:>6.3f} Sharpe)")
        print(f"  Total cost drag:          {cost_row['total_cost_ann_bps']:>7.1f} bps/yr "
              f"({cost_row['total_sharpe_degradation']:>6.3f} Sharpe)")

    # Year-by-year @ 250M
    yearly_rows = []
    for aum in AUM_LEVELS:
        label = aum_labels[aum]
        yr_df = yearly_sharpe(pnl_by_label[label], aum)
        yr_save = yr_df.copy()
        yr_save.insert(0, "aum_label", label)
        yr_save.insert(1, "aum", aum)
        yearly_rows.append(yr_save)
        print("\n" + "-" * 60)
        print(f"YEAR-BY-YEAR PERFORMANCE @ ${label} AUM")
        print("-" * 60)
        print(yr_df.to_string(index=False))

    # Save daily returns for 250M
    gate_diag.to_csv(out_dir / "step5_signal_gate.csv", index=False)
    pd.DataFrame(basket_diag_rows).to_csv(
        out_dir / "step5_basket_frac_diagnostic.csv", index=False
    )
    metrics_df.to_csv(out_dir / "step5_headline_performance.csv")
    pd.DataFrame(cost_rows).to_csv(
        out_dir / "step5_cost_decomposition.csv", index=False
    )
    pd.concat(yearly_rows, ignore_index=True).to_csv(
        out_dir / "step5_year_by_year_performance.csv", index=False
    )
    for aum in AUM_LEVELS:
        label = aum_labels[aum]
        suffix = label.lower()
        pnl = pnl_by_label[label]
        pnl.to_csv(out_dir / f"step5_pnl_{suffix}.csv")
        (pnl["net_pnl"] / aum).rename("net_daily_return").to_csv(
            out_dir / f"step5_net_returns_{suffix}.csv"
        )
    print("\nDaily P&L saved -> Output/diagnostics/step5_pnl_{50m,250m,1b}.csv")
    print(f"Signal gate saved -> Output/diagnostics/step5_signal_gate.csv")
    print("Basket diagnostic saved -> Output/diagnostics/step5_basket_frac_diagnostic.csv")
    print("Headline table saved -> Output/diagnostics/step5_headline_performance.csv")
    print("Cost decomposition saved -> Output/diagnostics/step5_cost_decomposition.csv")
    print("Year-by-year table saved -> Output/diagnostics/step5_year_by_year_performance.csv")
    print("Console output saved -> Output/diagnostics/step5_console_output.txt")
    print("Back-test done.")

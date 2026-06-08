"""
Step 4: Cross-Sectional Alpha Model - Literature-Driven Feature Set
=====================================================================
LightGBM walk-forward model to score eligible stocks by expected overnight
return for the next session.  All features are observable at 15:50 ET on
day t (decision time before the MOC closing auction).
---------------------------------------------------------------------
WALK-FORWARD SCHEME
  Initial training  : 2010-01-01 to 2012-12-31
  Quarterly retrains: 2013Q1 to 2024Q4  (48 folds)
  Rolling lookback  : 7 years from fold train_cutoff
  Strict causality  : no data from predict period enters training

TARGET
  target_rank_normalized: cross-sectional rank of r_ON_{t+1} mapped to
  [-0.5, +0.5].  Equivalent to rank of overnight excess return (rank is
  invariant to subtraction of cross-sectional mean).
---------------------------------------------------------------------
"""

from pathlib import Path
from collections.abc import Sequence
from typing import TypeAlias
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PathLike: TypeAlias = str | Path

# Reproducibility
RANDOM_SEED = 42

# Walk-forward window parameters
TRAIN_START_DATE        = pd.Timestamp("2010-01-01")
TRAIN_INITIAL_END_YEAR  = 2012
PREDICT_START_YEAR      = 2013
PREDICT_END_YEAR        = 2024
TRAINING_LOOKBACK_YEARS = 7

# Feature IC screening parameters
IC_SCREEN_END_DATE      = pd.Timestamp("2012-12-31")
IC_SCREEN_WINDOW_YEARS  = 1
IC_DROP_THRESHOLD       = 0.01
NAN_FRAC_THRESHOLD      = 0.70

# Target column names
TARGET_RAW  = "target_r_overnight_next"
TARGET_RANK = "target_rank_normalized"

# LightGBM categorical features
_CATEGORICAL_FEATURES = {
    "feature_day_of_week",
    "feature_month",
    "feature_earnings_today",
}

# Feature manifest
ALL_FEATURE_COLS = [
    # Group A: current-day open signals
    # [Lou, Polk & Skouras 2019 JFE 134; Hendershott et al. 2020 JFE 138]
    "feature_r_overnight_t",
    "feature_earnings_today",
    "feature_r_overnight_t_norm",
    "feature_r_overnight_t_rank",
    "feature_r_overnight_t_norm_rank",
    "feature_abs_r_overnight_t_norm",
    # Group B: prior-day close-based returns
    # [Akbas et al. 2022 JFE 145]
    "feature_r_intraday_lag1",
    "feature_r_close_to_close_lag1",
    # Group C: overnight momentum lags + rolling stats
    # [Lou et al. 2019; Western Securities 2024 TOI factor]
    "feature_r_overnight_lag1",
    "feature_r_overnight_lag2",
    "feature_r_overnight_lag3",
    "feature_r_overnight_lag4",
    "feature_r_overnight_lag5",
    "feature_r_overnight_ma5",
    "feature_r_overnight_ma10",
    "feature_r_overnight_ma20",
    "feature_r_overnight_std20",
    "feature_r_overnight_cum5",
    # Group D: intraday history
    # [Akbas et al. 2022]
    "feature_r_intraday_lag2",
    "feature_r_intraday_lag3",
    "feature_r_intraday_ma5",
    "feature_r_intraday_sum5_rank",
    "feature_r_intraday_sum10",
    "feature_r_intraday_sum10_rank",
    "feature_r_intraday_sum20_rank",
    # Group E: volatility / liquidity
    # [Parkinson 1980; Garman & Klass 1980]
    "feature_adv20_lag1",
    "feature_vol20_lag1",
    "feature_vol5_lag1",
    "feature_vol60_lag1",
    "feature_parkinson_vol20_lag1",
    "feature_gk_vol20_lag1",
    # Group F: short interest (Step 3, with publication lag)
    "feature_short_dsi",
    "feature_short_days_to_cover",
    "feature_short_days_to_cover_chg",
    "short_dsi_pct",
    "short_dtc_pct",
    # Group G: calendar
    "feature_day_of_week",
    "feature_month",
    # Group H: OHLC gap/range signals
    # [Kakushadze 2016, Alpha101]
    "feature_open_to_prev_high",
    "feature_open_to_prev_low",
    "feature_open_in_prev_range",
    "feature_close_position_lag1",
    "feature_range_ratio_lag1",
    # Group I: volume signals
    # [Kakushadze 2016, Alpha101 section 2/12]
    "feature_rel_volume_lag1",
    "feature_vol_sign_x_id_lag1",
    # Group J: overnight-intraday difference (DS, simplified TOI)
    # [Lou et al. 2019; Western Securities 2024 TOI factor]
    "feature_ds_lag1",
    "feature_ds_ma5",
    "feature_ds_ma10",
    "feature_ds_ma10_rank",
    "feature_ds_ma20",
    "feature_ds_ma20_rank",
    # Group K: scaled excess overnight return + session betas
    # [Kuru 2025; Hendershott, Livdan & Rosch 2020 JFE 138]
    "feature_seor_lag1",
    "feature_beta_on",
    "feature_beta_id",
    # Group L: overnight return jump / ORJ (PEAD event signal)
    # [Chan & Marsh 2025]
    "feature_orj_lag1",
    "feature_pead_lag1",
    # Group M: paper 4-factor set
    # [arXiv:1410.5513, "4-Factor Model for Overnight Returns"]
    "feature_pre",
    "feature_mom",
    "feature_hlv",
    "feature_vol",
    # Group N: enhanced momentum / reversal
    # [Akbas et al. 2022]
    "feature_mom5d_lag1",
    "feature_mom5d_rank",
    "feature_r_overnight_sum20_rank",
    "feature_on_pos_frac20",
    "feature_r_overnight_lag10",
    "feature_r_overnight_lag20",
]


FEATURE_GROUPS = {
    "A": {
        "name": "Current overnight gap",
        "features": [
            "feature_r_overnight_t",
            "feature_earnings_today",
            "feature_r_overnight_t_norm",
            "feature_r_overnight_t_rank",
            "feature_r_overnight_t_norm_rank",
            "feature_abs_r_overnight_t_norm",
        ],
    },
    "B": {
        "name": "Prior close returns",
        "features": [
            "feature_r_intraday_lag1",
            "feature_r_close_to_close_lag1",
        ],
    },
    "C": {
        "name": "Overnight history",
        "features": [
            "feature_r_overnight_lag1",
            "feature_r_overnight_lag2",
            "feature_r_overnight_lag3",
            "feature_r_overnight_lag4",
            "feature_r_overnight_lag5",
            "feature_r_overnight_ma5",
            "feature_r_overnight_ma10",
            "feature_r_overnight_ma20",
            "feature_r_overnight_std20",
            "feature_r_overnight_cum5",
        ],
    },
    "D": {
        "name": "Intraday history",
        "features": [
            "feature_r_intraday_lag2",
            "feature_r_intraday_lag3",
            "feature_r_intraday_ma5",
            "feature_r_intraday_sum5_rank",
            "feature_r_intraday_sum10",
            "feature_r_intraday_sum10_rank",
            "feature_r_intraday_sum20_rank",
        ],
    },
    "E": {
        "name": "Volatility and liquidity",
        "features": [
            "feature_adv20_lag1",
            "feature_vol20_lag1",
            "feature_vol5_lag1",
            "feature_vol60_lag1",
            "feature_parkinson_vol20_lag1",
            "feature_gk_vol20_lag1",
        ],
    },
    "F": {
        "name": "Short interest and borrow pressure",
        "features": [
            "feature_short_dsi",
            "feature_short_days_to_cover",
            "feature_short_days_to_cover_chg",
            "short_dsi_pct",
            "short_dtc_pct",
        ],
    },
    "G": {
        "name": "Calendar",
        "features": [
            "feature_day_of_week",
            "feature_month",
        ],
    },
    "H": {
        "name": "Gap and range",
        "features": [
            "feature_open_to_prev_high",
            "feature_open_to_prev_low",
            "feature_open_in_prev_range",
            "feature_close_position_lag1",
            "feature_range_ratio_lag1",
        ],
    },
    "I": {
        "name": "Volume interaction",
        "features": [
            "feature_rel_volume_lag1",
            "feature_vol_sign_x_id_lag1",
        ],
    },
    "J": {
        "name": "DS/TOI overnight-intraday spread",
        "features": [
            "feature_ds_lag1",
            "feature_ds_ma5",
            "feature_ds_ma10",
            "feature_ds_ma10_rank",
            "feature_ds_ma20",
            "feature_ds_ma20_rank",
        ],
    },
    "K": {
        "name": "Session beta and SEOR",
        "features": [
            "feature_seor_lag1",
            "feature_beta_on",
            "feature_beta_id",
        ],
    },
    "L": {
        "name": "Jump and PEAD",
        "features": [
            "feature_orj_lag1",
            "feature_pead_lag1",
        ],
    },
    "M": {
        "name": "Four-factor overnight style",
        "features": [
            "feature_pre",
            "feature_mom",
            "feature_hlv",
            "feature_vol",
        ],
    },
    "N": {
        "name": "Enhanced momentum/reversal",
        "features": [
            "feature_mom5d_lag1",
            "feature_mom5d_rank",
            "feature_r_overnight_sum20_rank",
            "feature_on_pos_frac20",
            "feature_r_overnight_lag10",
            "feature_r_overnight_lag20",
        ],
    },
}


def feature_to_group_map() -> pd.DataFrame:
    """Return a feature-to-group lookup table for importance diagnostics."""
    rows = []
    for group_id, meta in FEATURE_GROUPS.items():
        for feature in meta["features"]:
            rows.append(
                {
                    "feature": feature,
                    "group_id": group_id,
                    "group_name": meta["name"],
                }
            )
    return pd.DataFrame(rows)


# I/O

def load_step3_panel(input_path: PathLike) -> pd.DataFrame:
    """Load the Step 3 borrow-model panel from the training start date."""
    _needed_cols = [
        "instrument_id", "date", "ticker",
        "adj_open", "adj_high", "adj_low", "adj_close",
        "raw_dollar_volume",
        "r_overnight", "r_intraday", "r_close_to_close",
        "target_r_overnight_next",
        "feature_r_overnight_t",
        "feature_r_intraday_lag1",
        "feature_r_close_to_close_lag1",
        "feature_dollar_volume_lag1",
        "feature_adv20_lag1",
        "feature_vol20_lag1",
        "feature_earnings_today",
        "feature_short_dsi",
        "feature_short_days_to_cover",
        "feature_short_days_to_cover_chg",
        "short_dsi_pct", "short_dtc_pct",
        "borrow_tier", "borrow_cost_annual", "borrow_cost_daily", "is_shortable",
        "is_trade_eligible",
    ]
    price = pd.read_parquet(input_path, columns=_needed_cols)
    price["date"] = pd.to_datetime(price["date"])
    price = price[price["date"] >= TRAIN_START_DATE].copy()
    price = price.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    return price


def trim_memory(price: pd.DataFrame) -> pd.DataFrame:
    """Drop raw OHLC columns and downcast feature columns to float32."""
    _drop = [
        "adj_open", "adj_high", "adj_low", "adj_close",
        "raw_dollar_volume",
        "r_overnight", "r_intraday", "r_close_to_close",
    ]
    price = price.drop(columns=[c for c in _drop if c in price.columns])
    for col in price.columns:
        if col.startswith("feature_") and price[col].dtype == np.float64:
            price[col] = price[col].astype("float32")
    return price


# Feature engineering

def add_alpha_features(price: pd.DataFrame) -> pd.DataFrame:
    """
    Add all point-in-time alpha features observable at 15:50 ET on day t.

    INFORMATION SET RULE:
      Any quantity from Close_t, High_t, Low_t, Volume_t uses shift(1) or
      further lags (observable <= 16:00 ET on day t-1).
      Exception: adj_open on day t is observable at 09:30 ET day t.
      Exception: r_overnight = Open_t / Close_{t-1} - 1, published 09:30.
    """
    price = price.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    grouped = price.groupby("instrument_id", sort=False)

    # Precompute key lagged OHLC series
    _lag_close = grouped["adj_close"].shift(1).clip(lower=1e-6)
    _lag_open  = grouped["adj_open"].shift(1).clip(lower=1e-6)
    _lag_high  = grouped["adj_high"].shift(1).clip(lower=1e-6)
    _lag_low   = grouped["adj_low"].shift(1).clip(lower=1e-6)
    _lag_range = (_lag_high - _lag_low).clip(lower=1e-6)
    _lag_dvol  = grouped["raw_dollar_volume"].shift(1).clip(lower=1e-6)

    # Group A: volatility-normalised overnight return
    # [Lou, Polk & Skouras 2019, JFE 134]
    price["feature_r_overnight_t_norm"] = (
        price["feature_r_overnight_t"]
        / price["feature_vol20_lag1"].clip(lower=1e-6)
    )
    price["feature_r_overnight_t_rank"] = (
        price.groupby("date")["feature_r_overnight_t"].rank(pct=True) - 0.5
    )
    price["feature_r_overnight_t_norm_rank"] = (
        price.groupby("date")["feature_r_overnight_t_norm"].rank(pct=True) - 0.5
    )
    price["feature_abs_r_overnight_t_norm"] = price["feature_r_overnight_t_norm"].abs()

    # Group C: overnight return lags + rolling stats
    # [Lou et al. 2019] overnight momentum
    for lag in range(1, 6):
        price[f"feature_r_overnight_lag{lag}"] = grouped["r_overnight"].shift(lag)

    _lagged_on = grouped["r_overnight"].shift(1)

    price["feature_r_overnight_ma5"] = (
        _lagged_on.groupby(price["instrument_id"]).rolling(5,  min_periods=3).mean()
        .reset_index(level=0, drop=True)
    )
    price["feature_r_overnight_ma10"] = (
        _lagged_on.groupby(price["instrument_id"]).rolling(10, min_periods=5).mean()
        .reset_index(level=0, drop=True)
    )
    price["feature_r_overnight_ma20"] = (
        _lagged_on.groupby(price["instrument_id"]).rolling(20, min_periods=10).mean()
        .reset_index(level=0, drop=True)
    )
    price["feature_r_overnight_std20"] = (
        _lagged_on.groupby(price["instrument_id"]).rolling(20, min_periods=10).std()
        .reset_index(level=0, drop=True)
    )

    # 5-day compounded overnight return ending at t-1
    _cum5 = (
        (1.0 + price["feature_r_overnight_lag1"].fillna(0.0))
        * (1.0 + price["feature_r_overnight_lag2"].fillna(0.0))
        * (1.0 + price["feature_r_overnight_lag3"].fillna(0.0))
        * (1.0 + price["feature_r_overnight_lag4"].fillna(0.0))
        * (1.0 + price["feature_r_overnight_lag5"].fillna(0.0))
        - 1.0
    )
    _all_nan5 = (
        price["feature_r_overnight_lag1"].isna()
        & price["feature_r_overnight_lag2"].isna()
        & price["feature_r_overnight_lag3"].isna()
        & price["feature_r_overnight_lag4"].isna()
        & price["feature_r_overnight_lag5"].isna()
    )
    price["feature_r_overnight_cum5"] = _cum5.where(~_all_nan5, np.nan)

    # Group D: lagged intraday returns
    # [Akbas et al. 2022] intraday daytime reversal signal
    for lag in range(2, 4):
        price[f"feature_r_intraday_lag{lag}"] = grouped["r_intraday"].shift(lag)

    _lagged_id = grouped["r_intraday"].shift(1)
    price["feature_r_intraday_ma5"] = (
        _lagged_id.groupby(price["instrument_id"]).rolling(5, min_periods=3).mean()
        .reset_index(level=0, drop=True)
    )
    _intraday_sum5 = (
        _lagged_id.groupby(price["instrument_id"]).rolling(5, min_periods=3).sum()
        .reset_index(level=0, drop=True)
    )
    _intraday_sum10 = (
        _lagged_id.groupby(price["instrument_id"]).rolling(10, min_periods=5).sum()
        .reset_index(level=0, drop=True)
    )
    _intraday_sum20 = (
        _lagged_id.groupby(price["instrument_id"]).rolling(20, min_periods=10).sum()
        .reset_index(level=0, drop=True)
    )
    price["feature_r_intraday_sum5_rank"] = (
        _intraday_sum5.groupby(price["date"]).rank(pct=True) - 0.5
    )
    price["feature_r_intraday_sum10"] = _intraday_sum10
    price["feature_r_intraday_sum10_rank"] = (
        _intraday_sum10.groupby(price["date"]).rank(pct=True) - 0.5
    )
    price["feature_r_intraday_sum20_rank"] = (
        _intraday_sum20.groupby(price["date"]).rank(pct=True) - 0.5
    )

    # Group E: volatility
    _lagged_ctc = grouped["r_close_to_close"].shift(1)

    price["feature_vol5_lag1"] = (
        _lagged_ctc.groupby(price["instrument_id"]).rolling(5,  min_periods=3).std()
        .reset_index(level=0, drop=True)
    )
    price["feature_vol60_lag1"] = (
        _lagged_ctc.groupby(price["instrument_id"]).rolling(60, min_periods=30).std()
        .reset_index(level=0, drop=True)
    )

    # Parkinson (1980) high-low volatility estimator
    # sigma^2 = 1/(4*ln2) * (ln H_{t-1} - ln L_{t-1})^2
    # [Parkinson 1980, Journal of Business]
    _park_daily = (
        np.log(_lag_high / _lag_low).clip(lower=0).pow(2)
        / (4.0 * np.log(2.0))
    )
    price["feature_parkinson_vol20_lag1"] = np.sqrt(
        _park_daily.groupby(price["instrument_id"]).rolling(20, min_periods=10).mean()
        .reset_index(level=0, drop=True)
        .clip(lower=0)
    )

    # Garman-Klass (1980) OHLC volatility estimator
    # sigma^2 = 0.5*(ln H/L)^2 - (2*ln2-1)*(ln C/O)^2
    # [Garman & Klass 1980, Journal of Business]
    _gk_daily = (
        0.5 * np.log(_lag_high / _lag_low).clip(lower=0).pow(2)
        - (2.0 * np.log(2.0) - 1.0) * np.log(_lag_close / _lag_open).pow(2)
    ).clip(lower=0)
    price["feature_gk_vol20_lag1"] = np.sqrt(
        _gk_daily.groupby(price["instrument_id"]).rolling(20, min_periods=10).mean()
        .reset_index(level=0, drop=True)
        .clip(lower=0)
    )

    # Group G: calendar
    price["feature_day_of_week"] = price["date"].dt.dayofweek.astype(np.int8)
    price["feature_month"]       = price["date"].dt.month.astype(np.int8)

    # Group H: OHLC gap/range signals (Alpha101 section 20)
    # [Kakushadze 2016, "101 Formulaic Alphas"]
    # adj_open on day t: known 09:30 ET; all lag cols: known 16:00 t-1.
    price["feature_open_to_prev_high"] = (
        (price["adj_open"] - _lag_high) / _lag_close
    )
    price["feature_open_to_prev_low"] = (
        (price["adj_open"] - _lag_low) / _lag_close
    )
    price["feature_open_in_prev_range"] = (
        (price["adj_open"] - _lag_low) / _lag_range
    ).clip(-3.0, 4.0)
    price["feature_close_position_lag1"] = (
        (_lag_close - _lag_low) / _lag_range
    ).clip(0.0, 1.0)
    price["feature_range_ratio_lag1"] = _lag_range / _lag_close

    # Group I: volume signals (Alpha101 section 2/12)
    # [Kakushadze 2016]
    price["feature_rel_volume_lag1"] = (
        price["feature_dollar_volume_lag1"]
        / price["feature_adv20_lag1"].clip(lower=1e-6)
        - 1.0
    )
    _log_vol_lag1 = np.log1p(grouped["raw_dollar_volume"].shift(1).clip(lower=0))
    _log_vol_lag3 = np.log1p(grouped["raw_dollar_volume"].shift(3).clip(lower=0))
    price["feature_vol_sign_x_id_lag1"] = (
        np.sign(_log_vol_lag1 - _log_vol_lag3)
        * price["feature_r_intraday_lag1"]
    )

    # Group J: Overnight-Intraday Difference / DS (simplified TOI)
    # DS_i = r_ON_{t-1} - r_ID_{t-1}: daily "tug-of-war" direction
    # [Lou, Polk & Skouras 2019 JFE 134; Western Securities 2024 TOI factor]
    # Full TOI = monthly Pearson corr(DS, V_ID/V_total) on days r_ON>0 & DS>0;
    # here we use the daily DS and its 5-day mean as simplified predictive features.
    _ds_lag1 = grouped["r_overnight"].shift(1) - grouped["r_intraday"].shift(1)
    price["feature_ds_lag1"] = _ds_lag1
    price["feature_ds_ma5"] = (
        _ds_lag1.groupby(price["instrument_id"]).rolling(5, min_periods=3).mean()
        .reset_index(level=0, drop=True)
    )
    price["feature_ds_ma10"] = (
        _ds_lag1.groupby(price["instrument_id"]).rolling(10, min_periods=5).mean()
        .reset_index(level=0, drop=True)
    )
    price["feature_ds_ma10_rank"] = (
        price["feature_ds_ma10"].groupby(price["date"]).rank(pct=True) - 0.5
    )
    price["feature_ds_ma20"] = (
        _ds_lag1.groupby(price["instrument_id"]).rolling(20, min_periods=10).mean()
        .reset_index(level=0, drop=True)
    )
    price["feature_ds_ma20_rank"] = (
        price["feature_ds_ma20"].groupby(price["date"]).rank(pct=True) - 0.5
    )

    # Group K: Scaled Excess Overnight Return + session betas
    # [Kuru 2025] SEOR = r_ON / beta_ON
    # [Hendershott, Livdan & Rosch 2020 JFE 138] beta_ON and beta_ID
    # Rolling OLS beta = Cov(y, x) / Var(x) estimated over 60-day window.
    # Fully vectorized via rolling means: Cov = E[xy]-E[x]*E[y], Var = E[x^2]-E[x]^2.
    # Market returns are equal-weighted cross-sectional means per date.
    # All series shifted by 1 so beta on day t uses data up to t-1.
    _mkt_on = price.groupby("date")["r_overnight"].transform("mean")
    _mkt_id = price.groupby("date")["r_intraday"].transform("mean")

    price["_mkt_on"] = _mkt_on
    price["_mkt_id"] = _mkt_id

    BETA_W = 60
    BETA_M = 30
    _grp   = price["instrument_id"]

    def _rolling_beta(y_col, x_col):
        """Vectorized rolling OLS beta = Cov(y,x)/Var(x) per instrument group."""
        y = grouped[y_col].shift(1)
        x = price.groupby(_grp)[x_col].shift(1)
        xy = y * x
        x2 = x ** 2

        def _roll(s):
            return (
                s.groupby(_grp)
                .rolling(BETA_W, min_periods=BETA_M)
                .mean()
                .reset_index(level=0, drop=True)
            )

        mean_y  = _roll(y)
        mean_x  = _roll(x)
        mean_xy = _roll(xy)
        mean_x2 = _roll(x2)

        cov = mean_xy - mean_x * mean_y
        var = (mean_x2 - mean_x ** 2).clip(lower=1e-12)
        return cov / var

    price["feature_beta_on"] = _rolling_beta("r_overnight", "_mkt_on")
    price["feature_beta_id"] = _rolling_beta("r_intraday",  "_mkt_id")

    # SEOR: r_ON_{t-1} / beta_ON; clip beta to [0.1, 5.0] for safety
    price["feature_seor_lag1"] = (
        grouped["r_overnight"].shift(1)
        / price["feature_beta_on"].clip(lower=0.1, upper=5.0)
    )

    price.drop(columns=["_mkt_on", "_mkt_id"], inplace=True, errors="ignore")

    # Group L: Overnight Return Jump / ORJ
    # ORJ_{i,t-1} = I(r_ON_{t-1} > mu_ON + 2.5 * sigma_ON), rolling 60d
    # [Chan & Marsh 2025] Post-overnight-jump drift (PEAD for overnight signals)
    _ron_lag1 = grouped["r_overnight"].shift(1)
    _ron_mu = (
        _ron_lag1.groupby(price["instrument_id"]).rolling(60, min_periods=20).mean()
        .reset_index(level=0, drop=True)
    )
    _ron_sigma = (
        _ron_lag1.groupby(price["instrument_id"]).rolling(60, min_periods=20).std()
        .reset_index(level=0, drop=True)
    ).clip(lower=1e-6)
    price["feature_orj_lag1"] = (
        (_ron_lag1 > (_ron_mu + 2.5 * _ron_sigma)).astype(np.int8)
    )
    # PEAD: lagged earnings flag magnitude x lagged overnight return
    price["feature_pead_lag1"] = (
        grouped["feature_earnings_today"].shift(1).abs().astype(float)
        * _ron_lag1
    )

    # Group M: paper 4-factor set (arXiv:1410.5513)
    price["feature_pre"] = np.log(_lag_close)
    price["feature_mom"] = np.log(_lag_close / _lag_open)
    _hlv_daily = ((_lag_high - _lag_low) / _lag_close).pow(2)
    _hlv_u = (
        _hlv_daily.groupby(price["instrument_id"]).rolling(21, min_periods=10).mean()
        .reset_index(level=0, drop=True)
    )
    _hlv_raw = 0.5 * np.log(_hlv_u.clip(lower=1e-12))
    _vol_avg = (
        _lag_dvol.groupby(price["instrument_id"]).rolling(21, min_periods=10).mean()
        .reset_index(level=0, drop=True)
    )
    _vol_raw = np.log(_vol_avg.clip(lower=1e-12))
    price["feature_hlv"] = _hlv_raw - _hlv_raw.groupby(price["date"]).transform("mean")
    price["feature_vol"] = _vol_raw - _vol_raw.groupby(price["date"]).transform("mean")

    # Group N: enhanced momentum / reversal
    # [Akbas et al. 2022] medium-horizon overnight reversal
    price["feature_mom5d_lag1"] = (
        _lagged_ctc.groupby(price["instrument_id"]).rolling(5, min_periods=3).sum()
        .reset_index(level=0, drop=True)
    )
    price["feature_mom5d_rank"] = (
        price["feature_mom5d_lag1"].groupby(price["date"]).rank(pct=True) - 0.5
    )
    _overnight_sum20 = (
        _lagged_on.groupby(price["instrument_id"]).rolling(20, min_periods=10).sum()
        .reset_index(level=0, drop=True)
    )
    price["feature_r_overnight_sum20_rank"] = (
        _overnight_sum20.groupby(price["date"]).rank(pct=True) - 0.5
    )
    _on_pos = (grouped["r_overnight"].shift(1) > 0).astype(float)
    price["feature_on_pos_frac20"] = (
        _on_pos.groupby(price["instrument_id"]).rolling(20, min_periods=10).mean()
        .reset_index(level=0, drop=True)
    )
    price["feature_r_overnight_lag10"] = grouped["r_overnight"].shift(10)
    price["feature_r_overnight_lag20"] = grouped["r_overnight"].shift(20)

    return price


# Target construction

def build_cs_rank_target(price: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional rank-normalised target mapped to [-0.5, +0.5].

    For each date, rank target_r_overnight_next among eligible stocks:
        rank_norm = (rank - 1) / (n - 1) - 0.5
    Mathematically equivalent to ranking overnight excess return (rank is
    invariant to subtracting the cross-sectional mean).
    [Gu, Kelly & Xiu 2020 RFS] standardisation improves walk-forward stability.
    """
    price = price.copy()
    price[TARGET_RANK] = np.nan
    eligible_mask = price["is_trade_eligible"] & price[TARGET_RAW].notna()

    def _rank_norm(series):
        n = series.notna().sum()
        if n < 2:
            return pd.Series(np.nan, index=series.index)
        ranked = series.rank(method="average", na_option="keep")
        return (ranked - 1.0) / (n - 1.0) - 0.5

    ranks = (
        price.loc[eligible_mask]
        .groupby("date")[TARGET_RAW]
        .transform(_rank_norm)
    )
    price.loc[eligible_mask, TARGET_RANK] = ranks.values
    return price


# Per-fold IC screening helper

def _screen_fold(screen_df, feature_cols, drop_threshold, nan_frac_threshold):
    """
    Per-fold: compute daily Spearman IC, return (kept_cols, sign_dict, report_df).
    Decision rules:
      - categorical: always keep
      - NaN frac > nan_frac_threshold: drop
      - |mean_IC| < drop_threshold: drop
      - mean_IC < -drop_threshold: flip (negate feature)
      - mean_IC >= drop_threshold: keep
    """
    eligible = screen_df[screen_df[TARGET_RAW].notna()]
    numeric_feats = [
        f for f in feature_cols
        if f in eligible.columns and f not in _CATEGORICAL_FEATURES
    ]
    missing_feats = set(f for f in feature_cols if f not in eligible.columns)

    high_nan: dict = {}
    valid_numeric: list = []
    for f in numeric_feats:
        nf = float(eligible[f].isna().mean()) if len(eligible) > 0 else 1.0
        if nf > nan_frac_threshold:
            high_nan[f] = nf
        else:
            valid_numeric.append(f)

    daily_ic: list = []
    for _, grp in eligible.groupby("date"):
        sub = grp[valid_numeric + [TARGET_RAW]].dropna(how="all", subset=[TARGET_RAW])
        if len(sub) < 10:
            continue
        tgt = sub[TARGET_RAW]
        try:
            row = sub[valid_numeric].apply(
                lambda col: col.dropna().corr(
                    tgt.loc[col.dropna().index], method="spearman"
                )
            )
            daily_ic.append(row)
        except Exception:
            pass

    ic_df = (
        pd.DataFrame(daily_ic, columns=valid_numeric)
        if daily_ic
        else pd.DataFrame(columns=valid_numeric)
    )

    rows: list = []
    dropped: set = set()
    sign_dict: dict = {}

    for feat in feature_cols:
        if feat in _CATEGORICAL_FEATURES:
            sign_dict[feat] = 1
            rows.append({"feature": feat, "mean_ic": np.nan, "n_days": 0,
                         "decision": "keep (categorical)"})
            continue
        if feat in missing_feats:
            sign_dict[feat] = 1
            rows.append({"feature": feat, "mean_ic": np.nan, "n_days": 0,
                         "decision": "keep (missing col)"})
            continue
        if feat in high_nan:
            dropped.add(feat)
            rows.append({"feature": feat, "mean_ic": np.nan, "n_days": 0,
                         "decision": f"drop (nan {high_nan[feat]:.0%})"})
            continue
        if feat not in ic_df.columns:
            sign_dict[feat] = 1
            rows.append({"feature": feat, "mean_ic": np.nan, "n_days": 0,
                         "decision": "keep (no ic data)"})
            continue

        s = ic_df[feat].dropna()
        n = len(s)
        if n < 20:
            sign_dict[feat] = 1
            rows.append({"feature": feat, "mean_ic": np.nan, "n_days": n,
                         "decision": "keep (few days)"})
            continue

        mean_ic = float(s.mean())
        if abs(mean_ic) < drop_threshold:
            dropped.add(feat)
            rows.append({"feature": feat, "mean_ic": round(mean_ic, 5), "n_days": n,
                         "decision": "drop"})
        elif mean_ic < -drop_threshold:
            sign_dict[feat] = -1
            rows.append({"feature": feat, "mean_ic": round(mean_ic, 5), "n_days": n,
                         "decision": "flip"})
        else:
            sign_dict[feat] = 1
            rows.append({"feature": feat, "mean_ic": round(mean_ic, 5), "n_days": n,
                         "decision": "keep"})

    kept_cols = [f for f in feature_cols if f not in dropped]
    return kept_cols, sign_dict, pd.DataFrame(rows)


# Walk-forward model

def walk_forward_train_predict(
    price: pd.DataFrame,
    feature_cols: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward expanding-window LightGBM alpha model with quarterly retraining.

    - Training window: rolling 7-year lookback ending at fold_end
    - Prediction: next calendar quarter
    - Per-fold IC screening (last 1 year of training data)
    - LightGBM: gradient-boosted trees, regression objective on rank target

    Reference: [Gu, Kelly & Xiu 2020 RFS] walk-forward ML for asset pricing.
    [Lou et al. 2019] established overnight alpha predictability framework.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("pip install lightgbm") from exc

    price = price.copy()
    price["alpha_score"] = np.nan

    lgb_params = dict(
        objective="regression",
        metric="rmse",
        num_leaves=63,
        learning_rate=0.05,
        n_estimators=500,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=50,
        lambda_l1=0.1,
        lambda_l2=0.1,
        verbose=-1,
        n_jobs=-1,
        seed=RANDOM_SEED,
    )

    trainable   = price[price["is_trade_eligible"] & price[TARGET_RANK].notna()]
    predictable = price[price["is_trade_eligible"]]

    feature_importances: list = []
    fold_screen_reports: list = []

    # ---- Annual screening: compute one screening result per calendar year
    # The screening for predict_year Y uses data up to year Y-1 (inclusive).
    year_screen_kept: dict = {}
    year_screen_signs: dict = {}
    year_screen_report: dict = {}
    # Per-year factor stability weight (applied as multiplicative scale to features)
    year_screen_weights: dict = {}
    for yr in range(PREDICT_START_YEAR, PREDICT_END_YEAR + 1):
        screen_end = pd.Timestamp(f"{yr-1}-12-31")
        screen_df = trainable[trainable["date"] <= screen_end]
        # If insufficient screening history, fallback to keeping all features
        if len(screen_df) < 2000:
            kept = list(feature_cols)
            signs = {f: 1 for f in kept}
            report = pd.DataFrame([{"feature": f, "mean_ic": np.nan, "std_ic": np.nan, "n_days": 0, "decision": "keep (insufficient data)"} for f in feature_cols])
        else:
            kept, signs, report = _screen_fold(screen_df, feature_cols, IC_DROP_THRESHOLD, NAN_FRAC_THRESHOLD)
        # Compute per-feature stability weight: use t-statistic if present, else mean_ic/std_ic
        w = {}
        if "t_statistic" in report.columns:
            stat_col = "t_statistic"
            stat = report.set_index("feature")[stat_col].reindex(feature_cols)
            # Positive t-stat increases weight, negative -> zero
            stat = stat.fillna(0.0).clip(lower=0.0)
            # normalize
            if stat.sum() > 0:
                norm = stat / stat.sum()
            else:
                norm = pd.Series(1.0, index=feature_cols) / len(feature_cols)
            for f in feature_cols:
                w[f] = float(norm.get(f, 0.0))
        else:
            report_idx = report.set_index("feature").reindex(feature_cols)
            mean_ic = report_idx["mean_ic"].fillna(0.0)
            # std_ic may be missing if screening did not compute per-feature std;
            # handle safely by falling back to abs(mean_ic) (with small floor).
            if "std_ic" in report_idx.columns:
                std_ic = report_idx["std_ic"].fillna(mean_ic.abs().replace(0.0, 1e-6))
            else:
                std_ic = mean_ic.abs().replace(0.0, 1e-6)
            stability = (mean_ic.abs() / std_ic).fillna(0.0)
            if stability.sum() > 0:
                norm = stability / stability.sum()
            else:
                norm = pd.Series(1.0, index=feature_cols) / len(feature_cols)
            for f in feature_cols:
                w[f] = float(norm.get(f, 0.0))
        year_screen_weights[yr] = w
        year_screen_kept[yr] = kept
        year_screen_signs[yr] = signs
        report["screen_year"] = yr - 1
        year_screen_report[yr] = report

    fold_ends = pd.date_range(
        start=pd.Timestamp(f"{PREDICT_START_YEAR - 1}-12-31"),
        end=pd.Timestamp(f"{PREDICT_END_YEAR}-12-31"),
        freq="QE",
    )

    for i in range(len(fold_ends) - 1):
        train_cutoff = fold_ends[i]
        pred_start   = fold_ends[i] + pd.Timedelta(days=1)
        pred_end     = fold_ends[i + 1]
        predict_year = int(pred_end.year)

        rolling_start = max(
            TRAIN_START_DATE,
            train_cutoff - pd.DateOffset(years=TRAINING_LOOKBACK_YEARS),
        )
        train_df = trainable[
            (trainable["date"] >= rolling_start)
            & (trainable["date"] <= train_cutoff)
        ]
        pred_df = predictable[
            (predictable["date"] >= pred_start)
            & (predictable["date"] <= pred_end)
        ]

        if len(train_df) < 5_000:
            print(f"Skipping fold ending {train_cutoff.date()}: insufficient training rows.")
            continue
        if len(pred_df) == 0:
            print(f"Skipping fold ending {train_cutoff.date()}: no prediction rows.")
            continue

        # Use the annual screening result for this predict_year
        fold_kept = year_screen_kept.get(predict_year, list(feature_cols))
        fold_signs = year_screen_signs.get(predict_year, {f: 1 for f in fold_kept})
        fold_report = year_screen_report.get(predict_year, pd.DataFrame())
        fold_report = fold_report.copy()
        fold_report["fold_end"] = train_cutoff.date()
        fold_screen_reports.append(fold_report)

        X_train = train_df[fold_kept].copy()
        X_pred  = pred_df[fold_kept].copy()
        # Apply annual factor stability weights (scale features)
        weights_for_year = year_screen_weights.get(predict_year, {f: 1.0 for f in feature_cols})
        for feat in fold_kept:
            scale = weights_for_year.get(feat, 1.0)
            if scale != 1.0:
                X_train[feat] = X_train[feat] * scale
                X_pred[feat]  = X_pred[feat]  * scale
        for feat, sign in fold_signs.items():
            if sign == -1 and feat in fold_kept:
                X_train[feat] = -X_train[feat]
                X_pred[feat]  = -X_pred[feat]

        fold_cat_idx = [j for j, c in enumerate(fold_kept) if c in _CATEGORICAL_FEATURES]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_train,
            train_df[TARGET_RANK],
            categorical_feature=fold_cat_idx if fold_cat_idx else "auto",
        )

        scores = pd.Series(model.predict(X_pred), index=pred_df.index)
        price.loc[scores.index, "alpha_score"] = scores

        fi_df = pd.DataFrame({
            "feature":      fold_kept,
            "importance":   model.feature_importances_,
            "predict_year": predict_year,
        })
        feature_importances.append(fi_df)

        n_scored   = scores.notna().sum()
        n_dropped  = len(feature_cols) - len(fold_kept)
        n_flipped  = sum(1 for s in fold_signs.values() if s == -1)
        print(
            f"fold_end={train_cutoff.date()} | "
            f"trained on {len(train_df):>9,} rows | "
            f"scored {n_scored:>5,} rows | "
            f"kept {len(fold_kept):2d} / dropped {n_dropped:2d} / flipped {n_flipped:2d}"
        )

    fi_all = (
        pd.concat(feature_importances, ignore_index=True)
        if feature_importances
        else pd.DataFrame(columns=["feature", "importance", "predict_year"])
    )
    fold_reports_all = (
        pd.concat(fold_screen_reports, ignore_index=True)
        if fold_screen_reports
        else pd.DataFrame()
    )
    return price, fi_all, fold_reports_all


# IC statistics

def compute_ic_statistics(price: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-sectional Spearman IC between alpha_score and r_ON_{t+1}."""
    scored = price[
        price["is_trade_eligible"]
        & price["alpha_score"].notna()
        & price[TARGET_RAW].notna()
    ]
    rows = []
    for date, grp in scored.groupby("date"):
        if len(grp) < 10:
            continue
        rho, pval = stats.spearmanr(grp["alpha_score"], grp[TARGET_RAW], nan_policy="omit")
        rows.append({"date": date, "ic": rho, "pval": pval, "n_stocks": len(grp)})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def summarize_ic(ic_df: pd.DataFrame) -> pd.DataFrame:
    """Return mean IC, t-statistic, hit rate, information ratio, and sample size."""
    if ic_df.empty:
        return pd.DataFrame()
    ic = ic_df["ic"].dropna()
    t_stat, p_val = stats.ttest_1samp(ic, 0.0)
    return pd.DataFrame([{
        "mean_ic":              round(ic.mean(), 6),
        "std_ic":               round(ic.std(),  6),
        "t_statistic":          round(float(t_stat), 4),
        "p_value":              round(float(p_val), 6),
        "ic_positive_fraction": round((ic > 0).mean(), 4),
        "ir":                   round(ic.mean() / ic.std(), 4) if ic.std() > 0 else np.nan,
        "n_days":               len(ic),
    }])


def compute_ic_by_year(ic_df: pd.DataFrame) -> pd.DataFrame:
    """Return yearly cross-sectional IC summary statistics."""
    if ic_df.empty:
        return pd.DataFrame()
    ic_df = ic_df.copy()
    ic_df["year"] = ic_df["date"].dt.year
    rows = []
    for year, grp in ic_df.groupby("year"):
        ic = grp["ic"].dropna()
        if len(ic) < 5:
            continue
        t_stat, _ = stats.ttest_1samp(ic, 0.0)
        rows.append({
            "year":                 int(year),
            "mean_ic":              round(ic.mean(), 6),
            "std_ic":               round(ic.std(),  6),
            "t_statistic":          round(float(t_stat), 4),
            "ic_positive_fraction": round((ic > 0).mean(), 4),
            "n_days":               len(ic),
        })
    return pd.DataFrame(rows)


def summarize_feature_importance(fi_all: pd.DataFrame) -> pd.DataFrame:
    """Return average LightGBM feature importance sorted descending."""
    if fi_all.empty:
        return pd.DataFrame()
    return (
        fi_all.groupby("feature")["importance"]
        .mean()
        .reset_index()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# Output

def summarize_feature_group_importance(fi_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate LightGBM feature gain by pre-defined feature group."""
    group_map = feature_to_group_map()
    all_groups = pd.DataFrame(
        [
            {
                "group_id": group_id,
                "group_name": meta["name"],
                "n_group_features": len(meta["features"]),
            }
            for group_id, meta in FEATURE_GROUPS.items()
        ]
    )

    if fi_summary.empty:
        out = all_groups.copy()
        out["importance"] = 0.0
        out["importance_share"] = 0.0
        out["remaining_gain_share_if_removed"] = 1.0
        out["active_features"] = 0
        out["top_feature"] = ""
        out["top_feature_importance"] = 0.0
        out["rank"] = np.nan
        return out

    tmp = fi_summary.merge(group_map, on="feature", how="left")
    tmp["group_id"] = tmp["group_id"].fillna("UNMAPPED")
    tmp["group_name"] = tmp["group_name"].fillna("Unmapped")
    group_imp = (
        tmp.groupby(["group_id", "group_name"], as_index=False)
        .agg(
            importance=("importance", "sum"),
            active_features=("feature", "nunique"),
            top_feature=("feature", "first"),
            top_feature_importance=("importance", "first"),
        )
    )

    out = all_groups.merge(group_imp, on=["group_id", "group_name"], how="left")
    out["importance"] = out["importance"].fillna(0.0)
    out["active_features"] = out["active_features"].fillna(0).astype(int)
    out["top_feature"] = out["top_feature"].fillna("")
    out["top_feature_importance"] = out["top_feature_importance"].fillna(0.0)

    total = float(out["importance"].sum())
    if total > 0:
        out["importance_share"] = out["importance"] / total
        out["remaining_gain_share_if_removed"] = 1.0 - out["importance_share"]
    else:
        out["importance_share"] = 0.0
        out["remaining_gain_share_if_removed"] = 1.0

    out = out.sort_values("importance", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out[
        [
            "rank",
            "group_id",
            "group_name",
            "importance",
            "importance_share",
            "remaining_gain_share_if_removed",
            "active_features",
            "n_group_features",
            "top_feature",
            "top_feature_importance",
        ]
    ]


def summarize_feature_group_importance_by_year(fi_all: pd.DataFrame) -> pd.DataFrame:
    """Track group gain share through prediction years."""
    if fi_all.empty or "predict_year" not in fi_all.columns:
        return pd.DataFrame()

    group_map = feature_to_group_map()
    tmp = fi_all.merge(group_map, on="feature", how="left")
    tmp["group_id"] = tmp["group_id"].fillna("UNMAPPED")
    tmp["group_name"] = tmp["group_name"].fillna("Unmapped")
    out = (
        tmp.groupby(["predict_year", "group_id", "group_name"], as_index=False)
        .agg(importance=("importance", "sum"))
        .sort_values(["predict_year", "importance"], ascending=[True, False])
        .reset_index(drop=True)
    )
    totals = out.groupby("predict_year")["importance"].transform("sum")
    out["importance_share"] = np.where(totals > 0, out["importance"] / totals, 0.0)
    return out


def summarize_feature_importance_concentration(
    fi_summary: pd.DataFrame,
    fi_group_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize whether the model is dominated by one feature or group."""
    if fi_summary.empty:
        return pd.DataFrame()

    total_gain = float(fi_summary["importance"].sum())
    top_feature = fi_summary.iloc[0]
    top5_gain = float(fi_summary.head(5)["importance"].sum())

    rows = [
        {
            "metric": "top_feature",
            "name": top_feature["feature"],
            "importance": float(top_feature["importance"]),
            "importance_share": (
                float(top_feature["importance"] / total_gain)
                if total_gain > 0
                else np.nan
            ),
        },
        {
            "metric": "top_5_features",
            "name": ",".join(fi_summary.head(5)["feature"].astype(str)),
            "importance": top5_gain,
            "importance_share": top5_gain / total_gain if total_gain > 0 else np.nan,
        },
    ]

    if not fi_group_summary.empty:
        top_group = fi_group_summary.iloc[0]
        rows.append(
            {
                "metric": "top_group",
                "name": f"{top_group['group_id']}: {top_group['group_name']}",
                "importance": float(top_group["importance"]),
                "importance_share": float(top_group["importance_share"]),
            }
        )

    return pd.DataFrame(rows)


def summarize_feature_screening_by_group(screening_report: pd.DataFrame) -> pd.DataFrame:
    """Summarize historical IC-screen decisions by group and screening year."""
    if screening_report.empty:
        return pd.DataFrame()

    group_map = feature_to_group_map()
    tmp = screening_report.merge(group_map, on="feature", how="left")
    tmp["group_id"] = tmp["group_id"].fillna("UNMAPPED")
    tmp["group_name"] = tmp["group_name"].fillna("Unmapped")
    tmp["decision_base"] = (
        tmp["decision"]
        .astype(str)
        .str.extract(r"^(keep|drop|flip)", expand=False)
        .fillna(tmp["decision"].astype(str))
    )
    return (
        tmp.groupby(["screen_year", "group_id", "group_name", "decision_base"], as_index=False)
        .size()
        .rename(columns={"size": "feature_count"})
        .sort_values(["screen_year", "group_id", "decision_base"])
        .reset_index(drop=True)
    )


def save_outputs(
    price: pd.DataFrame,
    ic_df: pd.DataFrame,
    ic_summary: pd.DataFrame,
    ic_by_year: pd.DataFrame,
    fi_summary: pd.DataFrame,
    fi_group_summary: pd.DataFrame | None = None,
    fi_group_by_year: pd.DataFrame | None = None,
    fi_concentration: pd.DataFrame | None = None,
    screening_by_group: pd.DataFrame | None = None,
    output_dir: PathLike = "../Output",
) -> None:
    """Write Step 4 alpha scores and IC/importance diagnostics to output_dir."""
    output_dir = Path(output_dir)
    diag_dir   = output_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    _output_cols = [
        "instrument_id", "date", "ticker",
        "alpha_score", TARGET_RANK, TARGET_RAW,
        "is_trade_eligible",
        "borrow_tier", "borrow_cost_annual", "borrow_cost_daily", "is_shortable",
    ]
    price[_output_cols].to_parquet(output_dir / "step4_alpha_scores.parquet", index=False)
    ic_df.to_csv(diag_dir / "step4_ic_daily.csv",               index=False)
    ic_summary.to_csv(diag_dir / "step4_ic_summary.csv",        index=False)
    ic_by_year.to_csv(diag_dir / "step4_ic_by_year.csv",        index=False)
    fi_summary.to_csv(diag_dir / "step4_feature_importance.csv", index=False)
    if fi_group_summary is not None:
        fi_group_summary.to_csv(
            diag_dir / "step4_feature_group_importance.csv",
            index=False,
        )
    if fi_group_by_year is not None:
        fi_group_by_year.to_csv(
            diag_dir / "step4_feature_group_importance_by_year.csv",
            index=False,
        )
    if fi_concentration is not None:
        fi_concentration.to_csv(
            diag_dir / "step4_feature_importance_concentration.csv",
            index=False,
        )
    if screening_by_group is not None:
        screening_by_group.to_csv(
            diag_dir / "step4_feature_screening_by_group.csv",
            index=False,
        )


def save_screening_report(report_df: pd.DataFrame, output_dir: PathLike = "../Output") -> None:
    """Write per-fold feature IC screening decisions to output_dir."""
    diag_dir = Path(output_dir) / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(diag_dir / "step4_feature_ic_screening.csv", index=False)


# Entry point

if __name__ == "__main__":
    input_path = Path("../Output/step3_borrow_model.parquet")
    output_dir = Path("../Output")

    print("Loading Step 3 panel...")
    price = load_step3_panel(input_path)

    print("Adding alpha features (literature-driven feature set)...")
    price = add_alpha_features(price)

    print("Building cross-sectional rank target...")
    price = build_cs_rank_target(price)

    print("Trimming memory (drop OHLC, downcast to float32)...")
    price = trim_memory(price)

    print(
        "\nWalk-forward training and prediction "
        f"(quarterly retraining, per-fold IC screening, threshold={IC_DROP_THRESHOLD})..."
    )
    price, fi_all, fold_reports_all = walk_forward_train_predict(price, ALL_FEATURE_COLS)
    save_screening_report(fold_reports_all, output_dir)

    print("\nComputing IC statistics...")
    ic_df      = compute_ic_statistics(price)
    ic_summary = summarize_ic(ic_df)
    ic_by_year = compute_ic_by_year(ic_df)
    fi_summary = summarize_feature_importance(fi_all)
    fi_group_summary = summarize_feature_group_importance(fi_summary)
    fi_group_by_year = summarize_feature_group_importance_by_year(fi_all)
    fi_concentration = summarize_feature_importance_concentration(
        fi_summary,
        fi_group_summary,
    )
    screening_by_group = summarize_feature_screening_by_group(fold_reports_all)

    print("\n-- IC Summary ----------------------------------------------------------")
    print(ic_summary.to_string(index=False))
    print("\n-- IC by Year ----------------------------------------------------------")
    print(ic_by_year.to_string(index=False))
    print("\n-- Feature Importance (top 15) -----------------------------------------")
    print(fi_summary.head(15).to_string(index=False))
    print("\n-- Feature Group Importance ---------------------------------------------")
    print(fi_group_summary.head(10).to_string(index=False))
    print("\n-- Importance Concentration ---------------------------------------------")
    print(fi_concentration.to_string(index=False))

    print("\nSaving outputs...")
    save_outputs(
        price,
        ic_df,
        ic_summary,
        ic_by_year,
        fi_summary,
        fi_group_summary,
        fi_group_by_year,
        fi_concentration,
        screening_by_group,
        output_dir,
    )
    print("\nStep 4 done.")

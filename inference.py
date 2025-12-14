from fastapi import FastAPI, Query, HTTPException
import numpy as np
import pandas as pd
from xgboost import XGBRanker
from typing import Dict, Tuple

# ----------------------------
# Config (match your notebook)
# ----------------------------
DATA_DIR = "data/raw"

XGB_FEATURE_COLS = ["MKT", "SMB", "HML", "RMW", "CMA"]
XGB_N_BINS_DEFAULT = 5

xgb_model_params = dict(
    objective="rank:pairwise",
    n_estimators=120,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    verbosity=0,
)

# Globals cached in-memory after startup
panel_ml_base: pd.DataFrame | None = None
panel_xgb_default: pd.DataFrame | None = None
_panel_xgb_by_bins: Dict[int, pd.DataFrame] = {}
_model_by_bins_and_month: Dict[Tuple[int, str], XGBRanker] = {}


# ----------------------------
# Helpers (same logic as notebook)
# ----------------------------
def ols_numpy(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def rolling_factor_loadings(df_excess, ticker, window=36, min_obs=24):
    betas = []
    dates = []

    y_all = df_excess[ticker].astype(float).values
    X_all = df_excess[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].astype(float).values
    X_all = np.column_stack([np.ones(len(X_all)), X_all])

    for i in range(window, len(df_excess)):
        y = y_all[i - window : i]
        X = X_all[i - window : i]

        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if mask.sum() < min_obs:
            continue

        beta = ols_numpy(X[mask], y[mask])
        betas.append(beta)
        dates.append(df_excess["date"].iloc[i])

    if len(betas) == 0:
        return pd.DataFrame(columns=["alpha", "MKT", "SMB", "HML", "RMW", "CMA"])

    return pd.DataFrame(
        betas,
        index=dates,
        columns=["alpha", "MKT", "SMB", "HML", "RMW", "CMA"],
    )


def _add_rank_labels(panel_ml: pd.DataFrame, xgb_n_bins: int) -> pd.DataFrame:
    panel_xgb_local = panel_ml.copy()
    B = int(xgb_n_bins)
    is_pos = panel_xgb_local["y_excess"] > 0
    panel_xgb_local["rel"] = 0

    pct = panel_xgb_local.loc[is_pos].groupby("date")["y_excess"].rank(pct=True, method="first")
    panel_xgb_local.loc[is_pos, "rel"] = (1 + np.floor(pct * (B - 1)).clip(0, B - 2)).astype(int)
    return panel_xgb_local


def build_panel_ml_base(data_dir: str = DATA_DIR) -> pd.DataFrame:
    # ---- Load + aggregate FF5 daily to monthly (same as notebook) ----
    ff = pd.read_csv(f"{data_dir}/ff5_data.csv")
    ff = ff[ff["Date"].astype(str).str.isdigit()].copy()
    ff["date"] = pd.to_datetime(ff["Date"].astype(str), format="%Y%m%d", errors="coerce")
    ff = ff.dropna(subset=["date"]).drop(columns=["Date"]).sort_values("date").reset_index(drop=True)

    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    ff[factor_cols] = ff[factor_cols].ffill()
    ff[factor_cols] = ff[factor_cols] / 100.0
    ff = ff.set_index("date")[factor_cols].sort_index()

    rf_monthly = (1 + ff["RF"]).resample("M").prod() - 1
    mkt_total_daily = ff["Mkt-RF"] + ff["RF"]
    mkt_total_monthly = (1 + mkt_total_daily).resample("M").prod() - 1
    mkt_rf_monthly = mkt_total_monthly - rf_monthly

    other_factors = ["SMB", "HML", "RMW", "CMA"]
    other_monthly = (1 + ff[other_factors]).resample("M").prod() - 1

    ff_monthly = pd.concat(
        [mkt_rf_monthly.rename("Mkt-RF"), other_monthly, rf_monthly.rename("RF")],
        axis=1,
    )

    # align to month-start timestamps
    ff_monthly.index = ff_monthly.index.to_period("M").to_timestamp()
    ff_monthly = ff_monthly.reset_index().rename(columns={"date": "date"})

    # ---- Load market monthly returns ----
    market = pd.read_csv(f"{data_dir}/market_data.csv", parse_dates=["Date"])
    market = market.rename(columns={"Date": "date"}).sort_values("date").reset_index(drop=True)

    # ---- Merge ----
    df_total = market.merge(ff_monthly, on="date", how="inner")
    if "RF_y" in df_total.columns:
        df_total = df_total.rename(columns={"RF_y": "RF"})
    if "RF_x" in df_total.columns:
        df_total = df_total.drop(columns=["RF_x"])

    tickers = [c for c in df_total.columns if c not in ["date"] + factor_cols]

    # ---- Excess returns for regressions ----
    df_excess = df_total.copy()
    df_excess[tickers] = df_excess[tickers].sub(df_excess["RF"], axis=0)

    # ---- Rolling betas to long ----
    beta_long_parts = []
    for tkr in tickers:
        bdf = rolling_factor_loadings(df_excess, tkr)
        if bdf is None or bdf.empty:
            continue
        tmp = bdf[XGB_FEATURE_COLS].copy()
        tmp["ticker"] = tkr
        tmp["date"] = pd.to_datetime(tmp.index)
        beta_long_parts.append(tmp.reset_index(drop=True))

    beta_long = pd.concat(beta_long_parts, ignore_index=True)

    # ---- y_excess label to long ----
    realized = df_total.set_index("date")[tickers]
    rf = df_total.set_index("date")["RF"]
    y_excess_wide = realized.sub(rf, axis=0)

    y_excess_long = (
        y_excess_wide.stack(dropna=False)
        .rename("y_excess")
        .rename_axis(["date", "ticker"])
        .reset_index()
    )

    panel_ml = beta_long.merge(y_excess_long, on=["date", "ticker"], how="left")
    panel_ml = panel_ml.replace([np.inf, -np.inf], np.nan)
    panel_ml = panel_ml.dropna(subset=XGB_FEATURE_COLS + ["y_excess"]).sort_values(["date", "ticker"]).reset_index(drop=True)

    return panel_ml


def build_panel_xgb(data_dir: str = DATA_DIR, xgb_n_bins: int = XGB_N_BINS_DEFAULT) -> pd.DataFrame:
    base = build_panel_ml_base(data_dir=data_dir)
    return _add_rank_labels(base, xgb_n_bins=int(xgb_n_bins))


def _month_key(dt) -> str:
    ts = pd.to_datetime(dt).to_period("M").to_timestamp()
    return ts.strftime("%Y-%m-%d")


def _parse_as_of_month(as_of_month: str | None, available_dates) -> pd.Timestamp:
    if as_of_month is None or str(as_of_month).strip() == "":
        return pd.to_datetime(sorted(available_dates)[-1]).to_period("M").to_timestamp()
    ts = pd.to_datetime(as_of_month, errors="coerce")
    if pd.isna(ts):
        raise HTTPException(status_code=400, detail="Invalid as_of_month. Use YYYY-MM or YYYY-MM-DD.")
    return ts.to_period("M").to_timestamp()


def _get_panel_for_bins(xgb_n_bins: int) -> pd.DataFrame:
    global panel_ml_base, panel_xgb_default
    B = int(xgb_n_bins)

    if B == int(XGB_N_BINS_DEFAULT) and panel_xgb_default is not None:
        return panel_xgb_default

    cached = _panel_xgb_by_bins.get(B)
    if cached is not None:
        return cached

    if panel_ml_base is None:
        panel_ml_base = build_panel_ml_base()

    panel_b = _add_rank_labels(panel_ml_base, xgb_n_bins=B)
    _panel_xgb_by_bins[B] = panel_b
    return panel_b


def _get_or_train_model(panel_xgb_local: pd.DataFrame, as_of_dt: pd.Timestamp, xgb_n_bins: int) -> XGBRanker:
    key = (int(xgb_n_bins), _month_key(as_of_dt))
    cached = _model_by_bins_and_month.get(key)
    if cached is not None:
        return cached

    train_dates = sorted([d for d in panel_xgb_local["date"].unique() if pd.to_datetime(d) < pd.to_datetime(as_of_dt)])
    train_x = panel_xgb_local[panel_xgb_local["date"].isin(train_dates)].copy().sort_values(["date", "ticker"])
    if len(train_x) == 0:
        raise HTTPException(status_code=400, detail="Not enough history to train a model for the requested as_of_month.")

    group_train = train_x.groupby("date").size().to_numpy()
    model = XGBRanker(**xgb_model_params)
    model.fit(
        train_x[XGB_FEATURE_COLS].values,
        train_x["rel"].values,
        group=group_train,
    )

    _model_by_bins_and_month[key] = model
    return model


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="FF5 XGB Top-K API")


@app.on_event("startup")
def _startup():
    global panel_ml_base, panel_xgb_default
    panel_ml_base = build_panel_ml_base()
    panel_xgb_default = _add_rank_labels(panel_ml_base, xgb_n_bins=int(XGB_N_BINS_DEFAULT))
    _panel_xgb_by_bins[int(XGB_N_BINS_DEFAULT)] = panel_xgb_default

    last_dt = pd.to_datetime(sorted(panel_xgb_default["date"].unique())[-1]).to_period("M").to_timestamp()
    _ = _get_or_train_model(panel_xgb_default, as_of_dt=last_dt, xgb_n_bins=int(XGB_N_BINS_DEFAULT))
    print("Loaded panel_ml_base. Last month:", last_dt.date(), "| rows:", len(panel_ml_base))


@app.get("/health")
def health():
    return {"ok": True, "has_panel": panel_ml_base is not None}


@app.get("/topk")
def topk(
    k: int = Query(50, ge=1, le=500),
    n_bins: int = Query(XGB_N_BINS_DEFAULT, ge=3, le=10),
    as_of_month: str | None = Query(None),
):
    panel_xgb_local = _get_panel_for_bins(int(n_bins))
    as_of_dt = _parse_as_of_month(as_of_month, panel_xgb_local["date"].unique())
    if as_of_dt not in set(pd.to_datetime(panel_xgb_local["date"]).to_period("M").to_timestamp()):
        raise HTTPException(status_code=400, detail="Requested as_of_month is not available in the dataset.")

    model = _get_or_train_model(panel_xgb_local, as_of_dt=as_of_dt, xgb_n_bins=int(n_bins))

    test_x = panel_xgb_local[pd.to_datetime(panel_xgb_local["date"]).to_period("M").to_timestamp() == as_of_dt].copy().sort_values(["date", "ticker"])
    scores = model.predict(test_x[XGB_FEATURE_COLS].values)
    ranked = test_x.assign(score=scores).sort_values("score", ascending=False)

    top = ranked.head(min(int(k), len(ranked)))[["ticker", "score"]].to_dict(orient="records")

    return {
        "as_of_month": str(as_of_dt.date()),
        "k": int(k),
        "n_bins": int(n_bins),
        "feature_cols": XGB_FEATURE_COLS,
        "topk": top,
    }
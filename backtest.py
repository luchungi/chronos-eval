"""
Chronos vs. simple baselines on financial time series — rolling-origin backtest.

  * Data:        VIXCLS
  * Target:      h = 21 business days ahead
  * Baselines:   Naive, AR(1) AR-EGARCH, HAR (refit at every origin)
  * Chronos:     zero-shot, context = trailing 512 obs on log-VIX.
                 chronos-raw variant on raw VIX values (potentially negative)
  * Validation:  rolling forecast origins (daily / week-end / month-end via
                 origin_frequency), context strictly <= origin date -> no
                 look-ahead.
                 Optional test_start restricts the evaluation window (e.g. to
                 origins after Chronos's pretraining-data cutoff).
  * Metrics:     MAE/MASE on the median forecast, RMSE/RMSSE on the mean
                 forecast, WQL, 90% interval coverage, PIT (all
                 sample-based models), directional accuracy, regime split
                 (volmageddon / COVID / tariffs / none of the above).

Usage
-----
Edit CONFIG in config.py first

Failure policy: raise exceptions on non-finite forecast values.
Collapsed, explosive, degenerate forecasts and implausibly wide intervals are
LOGGED as warnings and scored as-is (Chronos can be explosive due to
discretisaion in log space).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from tqdm.auto import tqdm
from chronos import ChronosPipeline

import arch_fit
from arma_egarch import AREGARCH
from har import HAR

# ----------------------------------------------------------------------------
# Run logging
# ----------------------------------------------------------------------------

# parent of "backtest.arch_fit" (ladder retries) and "backtest.har" (iid-dist
# fallback): one handler set catches everything
logger = logging.getLogger("backtest")


def setup_run_logger(cfg: dict, tag: str = "backtest") -> str:
    """
    Per-run file logger: {out_dir}/{tag}_<run start datetime>.log, opening
    with the full config. Returns the log file path.
    """
    os.makedirs(cfg["out_dir"], exist_ok=True)
    start = datetime.now()
    path = os.path.join(cfg["out_dir"], f"{tag}_{start:%Y%m%d_%H%M%S}.log")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    fh = logging.FileHandler(path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    sh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(sh)
    logger.info("run started %s", start.isoformat(timespec="seconds"))
    logger.info("config:\n%s", json.dumps(cfg, indent=2, default=str))
    return path


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def load_csv(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=['observation_date'], index_col='observation_date')
    df.dropna(inplace=True)
    return df[str(df.columns[0])].sort_index()


def effective_context_len(cfg: dict) -> int:
    """
    Longest context any model uses: origins must have this much history so
    every model shares identical origins (per-model overrides, e.g. HAR).
    NOTE: all models used same context len as Chronos in the end i.e. 512.
    """
    return max(cfg["context_len"], cfg.get("har_context_len") or 0)


def select_origins(index: pd.DatetimeIndex, cfg: dict) -> list[pd.Timestamp]:
    """
    Forecast origins: last available trading day per context within the
    evaluation window [test_start, test_end], keeping only origins with a
    full context behind them and a full horizon ahead.

    test_start=None means "as early as the data allows".
    Set a date to shrink the evaluation window, e.g. to only score
    origins after Chronos's pretraining-data cutoff.
    """
    h, C = cfg["horizon"], effective_context_len(cfg)
    start = pd.Timestamp(cfg["test_start"]) if cfg.get("test_start") else index[0]
    end = pd.Timestamp(cfg["test_end"]) if cfg.get("test_end") else index[-1]
    if cfg["origin_frequency"] not in ("D", "M", "W"):
        raise ValueError(f"origin_frequency must be 'D', 'M' or 'W', got {cfg['origin_frequency']!r}")

    s = pd.Series(np.arange(len(index)), index=index)  # position lookup
    last_per_period = s.groupby(index.to_period(cfg["origin_frequency"])).apply(
        lambda x: x.index[-1])

    origins = []
    for d in last_per_period:
        pos = s[d]
        if d < start or d > end:
            continue
        if pos + 1 < C: # not enough history for the context window
            continue
        if pos + h >= len(index): # not enough future data to score the forecast
            continue
        origins.append(d)
    return origins


# ----------------------------------------------------------------------------
# Forecasting models. Each returns a dict:
#   point      : np.ndarray (h,)                 median path forecast (MAE/MASE)
#   point_mean : np.ndarray (h,) [optional]      mean path forecast (RMSE/RMSSE);
#   q          : dict{level: np.ndarray (h,)} | None  per-step quantiles
#   samples_h  : np.ndarray (n,) [optional]      step-h samples
#   samples    : np.ndarray (n, h) [optional]    full sample paths (RAW scale),
# All models see ONLY `context` (values up to and including the origin date).
# ----------------------------------------------------------------------------

def _samples_output(out: dict) -> dict:
    """
    Adapt a model-class forecast dict (point/quantiles/samples) to the
    harness convention, deriving the mean forecast and step-h samples.
    """
    samples = out["samples"]
    return dict(
        point=out["point"],
        point_mean=out.get("point_mean", samples.mean(axis=0)),
        q=out["quantiles"],
        samples_h=samples[:, -1],
        samples=samples, # save full paths for rescore_wql can remove if not needed
    )


def forecast_naive(context: np.ndarray, h: int) -> dict:
    return dict(point=np.full(h, context[-1]), q=None)


def forecast_ar1(context: np.ndarray, h: int, cfg: dict) -> dict:
    """
    AR(1) with constant and Gaussian residuals, fit by OLS on the context.
    Minimal mean-reversion benchmark.
    Ignores models_apply_diff (diff handling in AREGARCH/HAR only).
    """
    apply_log = cfg.get("models_apply_log", True)
    z = np.log(context) if apply_log else context
    x, y = z[:-1], z[1:]
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    c, phi = beta
    sigma = np.std(y - X @ beta, ddof=2)
    rng = np.random.default_rng(cfg["seed"])
    paths = np.empty((cfg["num_samples"], h))
    last = np.full(cfg["num_samples"], z[-1])
    for i in range(h):
        last = c + phi * last + sigma * rng.standard_normal(len(last))
        paths[:, i] = last
    samples = np.exp(paths) if apply_log else paths
    return _samples_output(dict(
        point=np.median(samples, axis=0),
        quantiles={lv: np.quantile(samples, lv, axis=0)
                   for lv in cfg["quantile_levels"]},
        samples=samples,
    ))


def forecast_aregarch(context: np.ndarray, h: int, cfg: dict) -> dict:
    """
    AR(p)-EGARCH baseline (arma_egarch.py), refit at each origin.
    (apply_log=True: log-VIX inside, raw VIX out).
    """
    model = AREGARCH(
        ar_order=cfg.get("aregarch_ar_order", 2),
        vol=cfg.get("aregarch_vol", "EGARCH"),
        egarch_order=cfg.get("aregarch_order", (1, 1, 1)),
        dist=cfg.get("aregarch_dist", "normal"),
        apply_log=cfg.get("models_apply_log", True),
        apply_diff=cfg.get("models_apply_diff", False),
        random_state=cfg["seed"],
    ).fit(context)
    out = model.forecast(h, num_samples=cfg["num_samples"],
                            quantile_levels=cfg["quantile_levels"])
    return _samples_output(out)


def forecast_har(context: np.ndarray, h: int, cfg: dict) -> dict:
    """
    HAR(1,5,22) baseline (har.py), refit at each origin.
    Residual distribution and the optional EGARCH layer come from cfg.
    The harness may pass a longer context than global one(cfg['har_context_len']).
    """
    model = HAR(
        dist=cfg.get("har_dist", "normal"),
        egarch=cfg.get("har_egarch", False),
        egarch_order=cfg.get("har_egarch_order", (1, 1, 1)),
        apply_log=cfg.get("models_apply_log", True),
        apply_diff=cfg.get("models_apply_diff", False),
        random_state=cfg["seed"],
    ).fit(context)
    out = model.forecast(h, num_samples=cfg["num_samples"],
                            quantile_levels=cfg["quantile_levels"])
    return _samples_output(out)


class ChronosForecaster:
    def __init__(self, cfg: dict):
        """ChronosPipeline wrapper for the backtest harness. Each call reseeds"""
        if cfg["device"] is not None:
            device = cfg["device"]
        elif torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.dtype = torch.bfloat16
        print(f"[chronos] loading {cfg['chronos_model']} on {device} ({self.dtype})")
        self.pipeline = ChronosPipeline.from_pretrained(
            cfg["chronos_model"], device_map=device, dtype=self.dtype
        )
        self.num_samples = cfg["num_samples"]
        self.qlevels = cfg["quantile_levels"]
        self.temperature = cfg["temperature"]
        if cfg["seed"] is not None:
            torch.manual_seed(cfg["seed"])

    def __call__(self, context: np.ndarray, h: int, log_transform: bool = True) -> dict:
        """Uses ChronosPipeline.predict() to generate sample paths and quantiles."""
        ctx = torch.tensor(context, dtype=self.dtype)
        ctx = ctx.log() if log_transform else ctx
        samples = self.pipeline.predict(
            inputs=ctx,
            prediction_length=h,
            num_samples=self.num_samples,
            temperature=self.temperature
        )                                    # shape [1, num_samples, h]
        samples = samples[0].cpu().numpy()   # (num_samples, h)
        # Sample mean AFTER exp() -> correct raw-scale mean forecast (RMSE/RMSSE).
        samples = np.exp(samples) if log_transform else samples
        q = {lv: np.quantile(samples, lv, axis=0) for lv in self.qlevels}
        return dict(point=np.median(samples, axis=0),
                    point_mean=samples.mean(axis=0),
                    q=q,
                    samples_h=samples[:, -1],
                    samples=samples)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def pinball_loss(y: float, q: float, level: float) -> float:
    return (level * (y - q)) if y >= q else ((1.0 - level) * (q - y))


def assign_regime(date: pd.Timestamp, regimes: dict) -> str:
    for name, (a, b) in regimes.items():
        if pd.Timestamp(a) <= date <= pd.Timestamp(b):
            return name
    return "others"


# ----------------------------------------------------------------------------
# Backtest loop
# ----------------------------------------------------------------------------

@dataclass
class OriginResult:
    origin: pd.Timestamp
    model: str
    y_origin: float          # last observed value at origin (transformed scale)
    y_h: float               # realized value at t+h
    yhat_h: float            # median point forecast at t+h (MAE/MASE)
    yhat_mean_h: float       # mean point forecast at t+h (RMSE/RMSSE)
    q_lo: float = np.nan
    q_hi: float = np.nan
    wql_h: float = np.nan    # weighted quantile loss at step h
    regime: str = "others"


def check_forecast_sane(name: str, origin: pd.Timestamp,
                        out: dict, context: np.ndarray,
                        interval: tuple[float, float] = (0.1, 0.9)) -> None:
    """
    Sanity-check on forecasts
    RAISES on non-finite values (although har and aregarch have their own internal checks).
    LOGS a warning and scores as-is for collapsed, explosive, degenerate forecasts and
    implausibly wide intervals
    """
    checks = [("point", out["point"])]
    if out.get("point_mean") is not None:
        checks.append(("mean point", out["point_mean"]))

    # raise on non-finite values, log warnings on collapsed/explosive
    for what, point in checks:
        if not np.isfinite(point).all():
            raise RuntimeError(f"{name} produced a non-finite {what} forecast "
                               f"at origin {origin.date()}")
        if np.abs(point).max() > 1e3 * max(1.0, np.abs(context).max()):
            logger.warning("%s produced an explosive %s forecast at origin %s "
                           "(max |forecast| = %.3g, context max = %.3g) — scored as-is",
                           name, what, origin.date(),
                           np.abs(point).max(), np.abs(context).max())
        # collapse check only where ~0 cannot be a legitimate forecast
        if np.all(context > 0) and point.min() < context.min() / 1e3:
            logger.warning(f"{name} produced a collapsed {what} forecast at origin "
                               f"{origin.date()} (min forecast = {point.min():.3g} vs "
                               f"context min = {context.min():.3g})")

    # log warnings on degenerate predictive distributions and implausibly wide intervals
    q = out.get("q")
    if q is not None and out.get("samples_h") is not None:
        lo_lv, hi_lv = interval
        if lo_lv in q and hi_lv in q:
            q_lo, q_hi = q[lo_lv][-1], q[hi_lv][-1]
            # degenerate predictive distribution where q10 == q90 (unless num_samples too low)
            if q_lo == q_hi:
                logger.warning(
                    f"{name} produced a degenerate predictive distribution at origin "
                    f"{origin.date()} (q{lo_lv:g} == q{hi_lv:g} == {q_lo:.3g} at step h)."
                    "Check the model's residual distribution and/or the num_samples.")
            # dispersion sanity: an interval vastly wider than everything
            # the model saw in its context suggests a broken predictive distribution
            if np.all(context > 0) and q_lo > 0:
                width = np.log(q_hi / q_lo) # log scale for +ve series
                ctx_range = np.log(context.max() / context.min())
            else:
                width = q_hi - q_lo                       # linear fallback (yields etc.)
                ctx_range = context.max() - context.min()
            if ctx_range > 0 and width > 4.0 * ctx_range:
                logger.warning("%s produced an implausibly wide predictive distribution "
                               "at origin %s (q%g/q%g = %.3g/%.3g at step h spans %.1fx "
                               "the context range; bound 4) — scored as-is",
                               name, origin.date(), lo_lv, hi_lv, q_lo, q_hi,
                               width / ctx_range)


def _init_samples_dir(cfg: dict, log_path: str) -> str | None:
    """
    Per-run sample-dump directory: {out_dir}/samples_<run stamp>/
    (stamp shared with the run log) holding one NPZ per origin and metadata.json
    with the full generating config. None when save_samples is off.
    """
    if not cfg.get("save_samples", True):
        return None
    stamp = os.path.basename(log_path).removesuffix(".log").split("_", 1)[1]
    samples_dir = os.path.join(cfg["out_dir"], f"samples_{stamp}")
    os.makedirs(samples_dir, exist_ok=True)
    meta = dict(
        schema="samples-v1",
        arrays="one (num_samples, horizon) float32 sample-path array per model "
               "(RAW scale); y_future = realized (horizon,) path after the "
               "origin; y_origin = last observation at the origin",
        config=cfg,
    )
    with open(os.path.join(samples_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return samples_dir


def _save_origin_samples(samples_dir: str, origin: pd.Timestamp,
                         model_outputs: dict, y_future: np.ndarray,
                         y_origin: float) -> None:
    """
    One compressed NPZ per origin, written as soon as the origin is scored
    so RAM stays flat across the run: full float32 sample paths per model.
    Models without samples (naive) are skipped.
    """
    arrays = {name: out["samples"].astype(np.float32)
              for name, out in model_outputs.items()
              if out.get("samples") is not None}
    arrays["y_future"] = np.asarray(y_future, dtype=np.float64)
    arrays["y_origin"] = np.float64(y_origin)
    np.savez_compressed(os.path.join(samples_dir, f"{origin.date()}.npz"),
                        **arrays)


def run_backtest(cfg: dict) -> tuple[pd.DataFrame, dict]:
    log_path = setup_run_logger(cfg)
    print(f"[backtest] logging to {log_path}")
    np.random.seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    samples_dir = _init_samples_dir(cfg, log_path)
    if samples_dir is not None:
        print(f"[backtest] saving sample paths to {samples_dir}")
        logger.info("sample dump dir: %s", samples_dir)

    s = load_csv(cfg["csv_path"])
    # no transformations here as forecasting functions handle them internally
    values = s.values.astype(np.float64)
    index = s.index
    pos = pd.Series(np.arange(len(index)), index=index)
    h, C = cfg["horizon"], cfg["context_len"]
    lo_lv, hi_lv = cfg["interval"]

    # get all end points of the context windows (origins) that have enough history
    # and horizon ahead
    origins = select_origins(index, cfg)
    msg = (f"{len(origins)} forecast origins "
           f"({origins[0].date()} .. {origins[-1].date()}), h={h}, context={C}")
    print(f"[backtest] {msg}")
    logger.info(msg)

    chronos = ChronosForecaster(cfg)
    baselines = {
        "naive": forecast_naive,
        "ar1": partial(forecast_ar1, cfg=cfg),
        "aregarch": partial(forecast_aregarch, cfg=cfg),
        "har": partial(forecast_har, cfg=cfg),
    }
    # per-model context-length overrides (not used in the end, all models used 512)
    context_lens = {"har": cfg.get("har_context_len") or C}

    rows: list[OriginResult] = []
    for i, origin in tqdm(enumerate(origins), total=len(origins), desc="[backtest] origins"):
        p = pos[origin]
        context = values[p - C + 1: p + 1] # includes the origin observation; nothing after it
        y_future = values[p + 1: p + 1 + h] # used ONLY for scoring
        regime = assign_regime(origin, cfg["regimes"])
        # to trace when a rescaling or seeding fit was required
        arch_fit.set_fit_context(f"origin {origin.date()}")

        model_outputs = {}
        # get baseline forecasts
        for name, fn in baselines.items():
            try:
                model_outputs[name] = fn(values[p - context_lens.get(name, C) + 1: p + 1], h)
            except Exception:
                logger.exception("%s raised at origin %s — aborting", name, origin.date())
                raise
        try:
            # chronos forecast with a raw variant if log transform is applied
            # each run gets reseeded
            if cfg["seed"] is not None:
                torch.manual_seed(cfg["seed"] + i)
            model_outputs["chronos"] = chronos(context, h, log_transform=cfg["models_apply_log"])
            if cfg["models_apply_log"]:
                if cfg["seed"] is not None:
                    torch.manual_seed(cfg["seed"] + i)
                model_outputs["chronos-raw"] = chronos(context, h, log_transform=False)
        except Exception:
            logger.exception("chronos raised at origin %s — aborting", origin.date())
            raise

        for name, out in model_outputs.items():
            try:
                check_forecast_sane(name, origin, out, context, cfg["interval"])
            except Exception:
                logger.exception("sanity check failed for %s at origin %s — aborting",
                                 name, origin.date())
                raise
            point_mean = out.get("point_mean")
            if point_mean is None:
                point_mean = np.zeros_like(out["point"])
            r = OriginResult(
                origin=origin, model=name,
                y_origin=context[-1], y_h=y_future[-1], yhat_h=out["point"][-1],
                yhat_mean_h=point_mean[-1],
                regime=regime,
            )
            if out["q"] is not None:
                if lo_lv in out["q"] and hi_lv in out["q"]:
                    r.q_lo, r.q_hi = out["q"][lo_lv][-1], out["q"][hi_lv][-1]
                # WQL at step h across all available quantile levels
                # normalized later by mean |y_h| over the test set
                r.wql_h = np.mean([pinball_loss(y_future[-1], out["q"][lv][-1], lv)
                                   for lv in out["q"]])
            rows.append(r)

        if samples_dir is not None:
            _save_origin_samples(samples_dir, origin, model_outputs,
                                 y_future, context[-1])

    # reset logging context in arch_fit
    arch_fit.set_fit_context("")
    df = pd.DataFrame([r.__dict__ for r in rows])
    logger.info("backtest complete: %d origins x %d models = %d rows",
                len(origins), df["model"].nunique(), len(df))
    return df


# ----------------------------------------------------------------------------
# Evaluation / reporting
# ----------------------------------------------------------------------------

def evaluate(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate metrics at step h across origins, overall and per regime.
    Returns (summary table, per-origin scored DataFrame).

    target_mode='change': score y_h - y_origin against yhat_h - y_origin
    (feed levels to models, evaluate the change; naive change forecast = 0).

    Point-forecast pairing: MAE/MASE use the MEDIAN forecast (yhat_h), RMSE/
    RMSSE use the MEAN forecast (yhat_mean_h) — each metric is scored against
    the point forecast that optimises it. Direction stays median-based.

    NOTE on MASE/RMSSE: the denominators are the ONE-step in-context naive
    MAE/RMSE (standard definitions), so the h=21 value of a random walk sits
    near sqrt(21) ~ 4.6, not 1. Compare models against each other (and against
    naive), not against 1.0. For a "1.0 = ties naive at this horizon" number
    read rel_MAE / rel_RMSE (model error / naive error at the SAME horizon).
    """
    mode = cfg["target_mode"]
    df = df.copy()
    if mode == "change":
        df["actual"] = df["y_h"] - df["y_origin"]
        df["pred"] = df["yhat_h"] - df["y_origin"]
        df["pred_mean"] = df["yhat_mean_h"] - df["y_origin"]
        df["q_lo_t"] = df["q_lo"] - df["y_origin"]
        df["q_hi_t"] = df["q_hi"] - df["y_origin"]
    else:
        df["actual"], df["pred"] = df["y_h"], df["yhat_h"]
        df["pred_mean"] = df["yhat_mean_h"]
        df["q_lo_t"], df["q_hi_t"] = df["q_lo"], df["q_hi"]
    df["abs_err"] = (df["pred"] - df["actual"]).abs()
    df["sq_err"] = (df["pred_mean"] - df["actual"]) ** 2
    df["covered"] = ((df["actual"] >= df["q_lo_t"]) & (df["actual"] <= df["q_hi_t"])).where(
        df["q_lo"].notna())
    # direction of the h-step CHANGE (economically the relevant sign in both modes)
    df["dir_actual"] = (df["y_h"] > df["y_origin"]).astype(int)
    df["dir_pred"] = (df["yhat_h"] > df["y_origin"]).astype(int)

    naive = df[df.model == "naive"].set_index("origin")

    def block(sub: pd.DataFrame, label: str) -> list[dict]:
        out = []
        # WQL normalizer: mean |actual| over the slice's origins ("actual" is
        # identical across models at a given origin, so dedupe by origin)
        mean_abs_y = sub.drop_duplicates("origin")["actual"].abs().mean()
        for m, g in sub.groupby("model"):
            g = g.set_index("origin").sort_index()
            nb = naive.loc[g.index]
            out.append(dict(
                slice=label, model=m, n=len(g),
                MAE=g["abs_err"].mean(),
                RMSE=np.sqrt(g["sq_err"].mean()),
                rel_MAE=g["abs_err"].mean() / nb["abs_err"].mean(),
                rel_RMSE=np.sqrt(g["sq_err"].mean() / nb["sq_err"].mean()),
                WQL=(g["wql_h"].mean() / mean_abs_y) if g["wql_h"].notna().any() else np.nan,
                coverage90=g["covered"].mean() if g["covered"].notna().any() else np.nan,
                dir_acc=(g["dir_actual"] == g["dir_pred"]).mean(),
            ))
        return out

    results = block(df, "ALL")
    for reg, g in df.groupby("regime"):
        results += block(g, reg)
    summary = pd.DataFrame(results).set_index(["slice", "model"]).round(4)
    return summary, df


def plot_regimes(cfg: dict, model_names: list[str], regimes: dict, figsize: tuple = (16, 12)) -> None:
    """
    Visualize the forecast of each model on each regime window.
    """
    if "chronos" in model_names or "chronos-raw" in model_names:
        model = ChronosForecaster(cfg)
    os.makedirs("./figures", exist_ok=True)
    df = load_csv(cfg["csv_path"])
    C, h = cfg["context_len"], cfg["horizon"]
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    interval = round((max(cfg["quantile_levels"]) - min(cfg["quantile_levels"])) * 100)
    for i, regime in enumerate(regimes.keys()):
        period_df = df[regimes[regime][0]:regimes[regime][1]]
        data = period_df.values.squeeze()[:C+h]
        context = data[:C]

        # adapters take RAW VIX and log internally (models_apply_log); all outputs are raw-VIX scale
        results_list = []
        if "har" in model_names: results_list.append(forecast_har(context, h, cfg))
        if "aregarch" in model_names: results_list.append(forecast_aregarch(context, h, cfg))
        if "ar1" in model_names: results_list.append(forecast_ar1(context, h, cfg))
        if "chronos" in model_names: results_list.append(model(context, h))
        if "chronos-raw" in model_names: results_list.append(model(context, h, log_transform=False))
        colors = ["tomato", "green", "orange"]

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        period_df.iloc[C-20:C+h].plot(color="royalblue", label="VIX", ax=ax)
        for j, results in enumerate(results_list):
            # visualize the forecast
            low, median, high = results['q'][cfg["quantile_levels"][0]], results['point'], results['q'][cfg["quantile_levels"][-1]]
            median_series = pd.Series(median, index=period_df.index[C:C+h])
            median_series.plot(color=colors[j], label=f"{model_names[j]} median forecast", ax=ax)
            mean_series = pd.Series(results['point_mean'], index=period_df.index[C:C+h])
            if mean_series.max() < 500:  # avoid plotting crazy mean forecasts from chronos
                mean_series.plot(color=colors[j], label=f"{model_names[j]} mean forecast", ax=ax, linestyle=':')
            # ax[i].plot(period_df.index[C:C+h].values, median, color=colors[j], label=f"{model_names[j]} median forecast")
            ax.fill_between(period_df.index[C:C+h], low, high, color=colors[j], alpha=0.3, label=f"{model_names[j]} {interval}% prediction interval")
        ax.set_title(f"Forecasting on VIX with {regime} window as context", fontsize=18)
        ax.set_xlabel("Date", fontsize=14)
        ax.legend(fontsize=14)
        ax.grid()
        model_names_str = "_".join(model_names)
        plt.savefig(f"./figures/forecast_{regime}_{model_names_str}.png", dpi=300)

    plt.tight_layout()
    plt.show()


def rescore_wql(samples_dir: str, quantile_levels) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recompute step-h WQL from a run's sample dump based on quantile_levels.
    Mirrors evaluate's convention: mean pinball loss across levels per origin,
    normalized per slice by mean |actual| ("actual" respects the dump config's target_mode)
    slices = ALL + regime + others. Returns (summary, per-origin DataFrame).
    """
    with open(os.path.join(samples_dir, "metadata.json")) as f:
        cfg = json.load(f)["config"]
    levels = sorted(float(lv) for lv in quantile_levels)
    rows = []
    for fname in sorted(os.listdir(samples_dir)):
        if not fname.endswith(".npz"):
            continue
        origin = pd.Timestamp(fname[:-len(".npz")])
        with np.load(os.path.join(samples_dir, fname)) as z:
            y_h = float(z["y_future"][-1])
            y_origin = float(z["y_origin"])
            for model in z.files:
                if model in ("y_future", "y_origin"):
                    continue
                q = np.quantile(z[model][:, -1].astype(np.float64), levels)
                rows.append(dict(
                    origin=origin, model=model,
                    wql_h=np.mean([pinball_loss(y_h, qv, lv)
                                   for qv, lv in zip(q, levels)]),
                    y_h=y_h, y_origin=y_origin,
                    regime=assign_regime(origin, cfg["regimes"]),
                ))
    if not rows:
        raise ValueError(f"no origin NPZ files found in {samples_dir}")
    df = pd.DataFrame(rows)
    df["actual"] = (df["y_h"] - df["y_origin"]) if cfg["target_mode"] == "change" \
        else df["y_h"]

    def block(sub: pd.DataFrame, label: str) -> list[dict]:
        mean_abs_y = sub.drop_duplicates("origin")["actual"].abs().mean()
        return [dict(slice=label, model=m, n=len(g),
                     WQL=g["wql_h"].mean() / mean_abs_y)
                for m, g in sub.groupby("model")]

    results = block(df, "ALL")
    for reg, g in df.groupby("regime"):
        results += block(g, reg)
    summary = pd.DataFrame(results).set_index(["slice", "model"]).round(4)
    return summary, df


# ----------------------------------------------------------------------------
# Residual analysis
#
# A residual is the out-of-sample forecast error on the MODEL scale
# (log if models_apply_log):
#     e_{t+k} = x_{t+k} - E[x_{t+k} | x_{<=t}],   k = `step` (1 = daily
#     one-step-ahead, cfg["horizon"] = end-of-horizon)
# Forecasts are made EVERY day; resid_refit_frequency only sets how often the
# baselines' parameters are re-estimated (Chronos is zero-shot and refits nothing).
# ----------------------------------------------------------------------------

RESID_MODELS = ("chronos-raw", "chronos", "aregarch", "har")


def _model_scale_series(cfg: dict) -> pd.Series:
    """The series on the scale the models operate on (log if models_apply_log)."""
    if cfg.get("models_apply_diff", False):
        # NOTE: diff transform is backtest-only not used in residual analysis
        raise ValueError("models_apply_diff=True is supported by run_backtest "
                         "only, not by the residual/optimise machinery (R49)")
    s = load_csv(cfg["csv_path"])
    if cfg.get("models_apply_log", True):
        if (s <= 0).any():
            raise ValueError("models_apply_log=True but the series has non-positive values")
        s = np.log(s)
    return s


def _har_cond_mean(beta: np.ndarray, hist: np.ndarray, step: int) -> float:
    """
    step-ahead conditional mean of the HAR(1,5,22) recursion (linear in the
    lags, so plugging conditional means in for future values is exact).
    `hist` = trailing >= 22 model-scale values up to and including the origin.
    """
    b0, bd, bw, bm = beta
    h = list(hist[-22:])
    for _ in range(step):
        h.append(b0 + bd * h[-1] + bw * np.mean(h[-5:]) + bm * np.mean(h[-22:]))
        h.pop(0)
    return h[-1]


def _ar_cond_mean(const: float, ar: list[float], hist: np.ndarray, step: int) -> float:
    """
    step-ahead conditional mean of an AR(p): x_{t+1} = c + sum_i a_i x_{t+1-i}.
    """
    h = list(hist[-len(ar):])
    for _ in range(step):
        h.append(const + sum(a * h[-i] for i, a in enumerate(ar, start=1)))
        h.pop(0)
    return h[-1]


def _refit_positions(index: pd.DatetimeIndex, cfg: dict, step: int) -> list[int]:
    """
    Positions of the period-end refit days (same convention as
    select_origins): full effective context behind, >= 1 scorable day ahead.
    Ccfg["resid_refit_frequency"] determines refit frequency (M or W).
    """
    freq = cfg.get("resid_refit_frequency", "W")
    if freq not in ("M", "W"):
        raise ValueError(f"resid_refit_frequency must be 'M' or 'W', got {freq!r}")
    C = effective_context_len(cfg)
    s = pd.Series(np.arange(len(index)), index=index)
    last_per_period = s.groupby(index.to_period(freq)).apply(
        lambda x: x.iloc[-1])
    return [int(p) for p in last_per_period if p + 1 >= C and p + step < len(index)]


def _resid_forecast_days(index: pd.DatetimeIndex, cfg: dict, step: int
                         ) -> tuple[list[int], dict[int, int]]:
    """
    Forecast-origin positions t (daily) and the refit position serving each:
    the most recent period-end <= t. Bounded by resid_start/resid_end (on the
    forecast day) and by data availability for the target x_{t+step}.
    """
    refits = _refit_positions(index, cfg, step)
    if not refits:
        raise ValueError("no valid refit day: series shorter than the context length?")
    start = pd.Timestamp(cfg["resid_start"]) if cfg.get("resid_start") else None
    end = pd.Timestamp(cfg["resid_end"]) if cfg.get("resid_end") else None
    days, refit_of = [], {}
    for j, p_o in enumerate(refits):
        p_next = refits[j + 1] if j + 1 < len(refits) else len(index)
        for t in range(p_o, p_next):
            if t + step >= len(index):
                break
            if (start is not None and index[t] < start) or \
               (end is not None and index[t] > end):
                continue
            days.append(t)
            refit_of[t] = p_o
    if not days:
        raise ValueError("resid_start/resid_end leave no forecast days")
    return days, refit_of


def _har_residuals(x: np.ndarray, days: list[int], refit_of: dict[int, int],
                   step: int, cfg: dict) -> np.ndarray:
    """
    HAR conditional-mean errors. Only the OLS betas enter the conditional
    mean, and they are identical across dist/EGARCH variants so fit the
    cheap iid-normal variant at each refit while reusing the same model class.

    NOTE: apply_log=False: `x` comes from _model_scale_series, which ALREADY
    applied the log when cfg["models_apply_log"] — the model must not log it
    a second time.
    """
    C = cfg.get("har_context_len") or cfg["context_len"]
    betas: dict[int, np.ndarray] = {}
    out = np.empty(len(days))
    for i, t in enumerate(tqdm(days, desc="residuals[har]", unit="day")):
        p_o = refit_of[t]
        if p_o not in betas:
            betas[p_o] = HAR(dist="normal", egarch=False, apply_log=False,
                             random_state=cfg["seed"]).fit(x[p_o - C + 1: p_o + 1]).beta_
        out[i] = x[t + step] - _har_cond_mean(betas[p_o], x[t - 21: t + 1], step)
    return out


def _aregarch_residuals(x: np.ndarray, days: list[int], refit_of: dict[int, int],
                        step: int, cfg: dict) -> np.ndarray:
    """
    AR(p)-EGARCH conditional-mean errors. The EGARCH layer shapes the MLE
    of the AR coefficients (joint fit with cfg's vol/dist) but the conditional
    mean is the plain AR recursion — analytic, no simulation.

    NOTE: apply_log=False: `x` comes from _model_scale_series, which ALREADY
    applied the log when cfg["models_apply_log"] — the model must not log it
    a second time.
    """
    C = cfg["context_len"]
    p_ar = cfg["aregarch_ar_order"]
    params: dict[int, tuple[float, list[float]]] = {}
    out = np.empty(len(days))
    for i, t in enumerate(tqdm(days, desc="residuals[aregarch]", unit="day")):
        p_o = refit_of[t]
        if p_o not in params:
            model = AREGARCH(
                ar_order=p_ar,
                vol=cfg.get("aregarch_vol", "EGARCH"),
                egarch_order=cfg.get("aregarch_order", (1, 1, 1)),
                dist=cfg.get("aregarch_dist", "normal"),
                apply_log=False,
                random_state=cfg["seed"],
            ).fit(x[p_o - C + 1: p_o + 1])
            # mean_params_ is on the UNIT scale regardless of the fit ladder's
            # winning rung (res_.params would be on fit_scale_ * x, R34)
            pr = model.mean_params_
            params[p_o] = (float(pr["Const"]),
                           [float(pr[f"y[{k}]"]) for k in range(1, p_ar + 1)])
        const, ar = params[p_o]
        out[i] = x[t + step] - _ar_cond_mean(const, ar, x[t - p_ar + 1: t + 1], step)
    return out


@torch.no_grad()
def _chronos_step1_means(fc: "ChronosForecaster", ctx: torch.Tensor) -> np.ndarray:
    """
    EXACT 1-step-ahead conditional mean: Chronos's next-value predictive
    distribution is categorical over its bin centers, so
        E[x_{t+1}] = sum_i softmax(logits / T)_i * value(token_i) * scale
    from ONE encoder-decoder forward pass — the quantity the 256-sample mean
    only estimates, with zero Monte-Carlo noise and no `generate` batch
    expansion. value(token) reproduces tokenizer.output_transform exactly
    (clamped indexing maps special/edge tokens like sampling does).
    """
    pipe = fc.pipeline
    token_ids, attention_mask, scale = pipe.tokenizer.context_input_transform(ctx)
    inner = pipe.model.model  # the underlying T5ForConditionalGeneration
    dec_start = inner.config.decoder_start_token_id
    dec = torch.full((token_ids.shape[0], 1), dec_start,
                     dtype=torch.long, device=pipe.model.device)
    logits = inner(input_ids=token_ids.to(pipe.model.device),
                   attention_mask=attention_mask.to(pipe.model.device),
                   decoder_input_ids=dec).logits[:, 0, :].float()
    probs = torch.softmax(logits / fc.temperature, dim=-1).cpu()
    n_special = pipe.model.config.n_special_tokens
    centers = pipe.tokenizer.centers
    tok2val = centers[torch.clamp(
        torch.arange(probs.shape[-1]) - n_special - 1, 0, len(centers) - 1)]
    return ((probs * tok2val).sum(dim=-1) * scale).numpy().astype(np.float64)


def _chronos_sample_means(fc: "ChronosForecaster", ctx: torch.Tensor,
                          step: int, max_seqs: int) -> np.ndarray:
    """
    step-ahead conditional mean as the average of num_samples simulated path
    ENDPOINTS (multi-step means have no closed form — paths must be rolled
    forward). Same estimator as before; the samples are just split across
    predict() calls so at most `max_seqs` decoder sequences are in flight at
    once (generate expands the batch to n_days * n_samples; each sequence
    carries its own KV cache over the 512-token context -> memory cap).
    """
    per_call = max(1, max_seqs // ctx.shape[0])
    acc, got = np.zeros(ctx.shape[0]), 0
    while got < fc.num_samples:
        k = min(per_call, fc.num_samples - got)
        samples = fc.pipeline.predict(
            inputs=ctx, prediction_length=step,
            num_samples=k, temperature=fc.temperature,
        ) # (n_days, k, step)
        acc += samples[:, :, -1].numpy().astype(np.float64).sum(axis=1)
        got += k
    return acc / fc.num_samples


def _chronos_residuals(x: np.ndarray, days: list[int], step: int, cfg: dict) -> np.ndarray:
    """
    Chronos conditional-mean errors on the MODEL scale. Zero-shot -> no refit;
    step == 1 -> exact analytic mean; step > 1 -> capped sample mean (faster).
    NOTE: apply_log=False: `x` comes from _model_scale_series, which ALREADY
    applied the log when cfg["models_apply_log"] — the model must not log it
    a second time.
    """
    fc = ChronosForecaster(cfg)
    C = cfg["context_len"]
    bs = int(cfg.get("resid_batch_size", 8))
    max_seqs = int(cfg.get("resid_max_seqs", 512))
    out = np.empty(len(days))
    for lo in tqdm(range(0, len(days), bs), desc="residuals[chronos]", unit="batch"):
        chunk = days[lo: lo + bs]
        ctx = torch.stack(
            [torch.tensor(x[t - C + 1: t + 1], dtype=torch.float32) for t in chunk])
        if step == 1:
            # analytic mean is exact and faster than sampling
            means = _chronos_step1_means(fc, ctx)
        else:
            # sample mean is a Monte-Carlo estimate of the step-ahead conditional mean
            means = _chronos_sample_means(fc, ctx, step, max_seqs)
        out[lo: lo + len(chunk)] = x[[t + step for t in chunk]] - means
    return out


def _resid_config_stamp(cfg: dict, step: int) -> dict:
    """
    The cfg subset that determines the residual series; JSON-safe.
    """
    return dict(
        csv_path=cfg["csv_path"],
        models_apply_log=cfg.get("models_apply_log", True),
        context_len=cfg["context_len"],
        har_context_len=cfg.get("har_context_len"),
        origin_frequency=cfg.get("resid_refit_frequency", "W"),
        step=step,
        resid_start=cfg.get("resid_start"),
        resid_end=cfg.get("resid_end"),
        chronos_model=cfg["chronos_model"],
        chronos_mean="analytic" if step == 1 else "sample",
        num_samples=cfg["num_samples"],
        temperature=cfg["temperature"],
        seed=cfg["seed"],
        aregarch_ar_order=cfg.get("aregarch_ar_order", 2),
        aregarch_vol=cfg.get("aregarch_vol", "EGARCH"),
        aregarch_order=list(cfg.get("aregarch_order", (1, 1, 1))),
        aregarch_dist=cfg.get("aregarch_dist", "normal"),
    )


def compute_residuals(cfg: dict, models: tuple[str, ...] = RESID_MODELS,
                      step: int = 1, force: bool = False) -> pd.DataFrame:
    """
    Daily out-of-sample step-`step` forecast residuals for `models`, on the
    model scale (log if models_apply_log). Cached in
    {out_dir}/residuals_step{step}.csv (index = TARGET date t+step, one column
    per model) with a JSON sidecar of the generating config; a stale sidecar
    or force=True regenerates, missing model columns are computed and merged.
    """
    bad = [m for m in models if m not in RESID_MODELS]
    if bad:
        raise ValueError(f"unknown residual model(s) {bad}; choose from {RESID_MODELS}")
    os.makedirs(cfg["out_dir"], exist_ok=True)
    csv_path = f"{cfg['out_dir']}/residuals_step{step}.csv"
    meta_path = f"{cfg['out_dir']}/residuals_step{step}.json"
    stamp = _resid_config_stamp(cfg, step)

    cached = None
    if not force and os.path.exists(csv_path) and os.path.exists(meta_path):
        with open(meta_path) as f:
            if json.load(f) == stamp:
                cached = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
            else:
                print("[residuals] cached config differs -> regenerating")
    todo = [m for m in models if cached is None or m not in cached.columns]
    if not todo:
        return cached[list(models)]

    s = _model_scale_series(cfg)
    x, index = s.values, s.index
    days, refit_of = _resid_forecast_days(index, cfg, step)
    print(f"[residuals] step={step}: {len(days)} forecast days "
          f"({index[days[0]].date()}, {index[days[1]].date()} .. {index[days[-2]].date()}, {index[days[-1]].date()}), "
          f"computing {todo}")

    generators = {
        "har": lambda: _har_residuals(x, days, refit_of, step, cfg),
        "aregarch": lambda: _aregarch_residuals(x, days, refit_of, step, cfg),
        "chronos": lambda: _chronos_residuals(x, days, step, cfg),
        "chronos-raw": lambda: _chronos_residuals(np.exp(x) if cfg["models_apply_log"] else x, days, step, {**cfg, "models_apply_log": False}),
    }
    new = pd.DataFrame({m: generators[m]() for m in todo},
                       index=index[[t + step for t in days]])
    new.index.name = "date"
    df = new if cached is None else pd.concat([cached, new], axis=1)
    df.to_csv(csv_path)
    with open(meta_path, "w") as f:
        json.dump(stamp, f, indent=2)
    return df[list(models)]


def fit_residual_distribution(e: np.ndarray, dist: str = "normal") -> dict:
    """
    Fit one candidate to a residual series with dist : 'normal' | 't' | 'ged'
    """
    if dist == "normal":
        loc, scale = stats.norm.fit(e)
        fitted = stats.norm(loc, scale)
    elif dist == "t":
        df_, loc, scale = stats.t.fit(e)
        fitted = stats.t(df_, loc, scale)
    elif dist == "ged":
        b, loc, scale = stats.gennorm.fit(e)
        fitted = stats.gennorm(b, loc, scale)
    else:
        raise ValueError(f"dist must be 'normal', 't' or 'ged', got {dist!r}")
    return fitted


def plot_residuals(resid: pd.DataFrame, cfg: dict):
    """
    Plot residual diagnostics for each model in resid: histogram with fitted
    """
    os.makedirs("./figures", exist_ok=True)
    for model in resid.columns:
        e = resid.loc[:, model].dropna()
        if len(e) < 30:
            continue
        arr = e.values
        fits = {name: fit_residual_distribution(arr, dist=name)
                for name in ("normal", "t", "ged")}
        # fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
        fig, axes = plt.subplots(2, 2, figsize=(8, 6.))

        # ax = axes[0, 0]
        # ax.hist(arr, bins=50, density=True, alpha=0.6, edgecolor="k", lw=0.3)
        # grid = np.linspace(arr.min(), arr.max(), 400)
        # for name, f in fits.items():
        #     ax.plot(grid, f.pdf(grid), lw=1.2, label=name)
        # ax.legend(fontsize=8)
        # ax.set_title(f"{model} residuals + fitted densities", fontsize=9)
        # for ax, (name, f) in zip(axes[0, 1:], fits.items()):
        #     stats.probplot(arr, dist=f, plot=ax)
        #     ax.get_lines()[0].set(markersize=2.5)
        #     ax.set_title(f"QQ vs fitted {name}", fontsize=9)

        ax = axes[0, 0]
        stats.probplot(arr, dist=fits['normal'], plot=ax)
        ax.get_lines()[0].set(markersize=2.5)
        ax.set_title(f"QQ vs fitted normal", fontsize=12)
        ax = axes[0, 1]
        stats.probplot(arr, dist=fits['t'], plot=ax)
        ax.get_lines()[0].set(markersize=2.5)
        ax.set_title(f"QQ vs fitted t", fontsize=12)

        # ax = axes[0, 0]
        # ax.plot(e.index, arr, lw=0.5)
        # ax.axhline(0, color="k", lw=0.5)
        # ax.set_title("residuals over time", fontsize=12)
        plot_acf(arr, ax=axes[1, 0], lags=min(40, len(arr) // 3), title="ACF(e)")
        plot_acf(arr**2, ax=axes[1, 1], lags=min(40, len(arr) // 3), title="ACF(e²)")
        # ax = axes[1, 3]
        # ax.plot(e.index, e.rolling(63, min_periods=21).std(), lw=0.8)
        # ax.set_title("rolling std (63d)", fontsize=12)

        fig.suptitle(f"{model}", fontsize=12)
        fig.tight_layout()
        fig.savefig(f"figures/fig_resid_{model}.png", dpi=150)
        plt.close(fig)
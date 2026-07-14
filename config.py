config = dict(
    # ---- data ----
    csv_path="./data/VIXCLS.csv",

    # ---- forecasting problem ----
    horizon=21,                     # business days (~1 month), well inside Chronos's 64-step training length
    context_len=512,                # Chronos T5 default/max context; ~2 years of daily data
    target_mode="level",            # "level" (VIX) or "change"
    temperature=1.0,                # Chronos sampling temperature set to 1.0 to match implicit distribution
                                    # during pre-training
    models_apply_log=True,          # AR(1), AR-EGARCH, HAR + Chronos model log(y) internally,
                                    # forecast on raw scale; naive is transform-invariant.
    models_apply_diff=False,        # True -> the BASELINES (AR-EGARCH, HAR) fit first differences and
                                    # integrate forecasts back to levels; Chronos always sees levels

    # ---- evaluation window ----
    test_start="2017-12-29",        # Day after M4 dataset (used for Chronos pre-training)
                                    # every model refits on its rolling context.
    test_end=None,                  # None -> use all available data
    origin_frequency="D",           # forecast-origin grid, backtest only
                                    # "D" = every trading day
                                    # "W"/"M" = last trading day of each week/month

    # ---- Chronos ----
    chronos_model="amazon/chronos-t5-base",
    num_samples=1024,                # sample paths for Chronos AND the simulation baselines
                                     # 0.05/0.95 MUST be present for coverage90 interval
    quantile_levels=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    device='mps',                    # None -> auto (cuda > mps > cpu)
    seed=42,

    # ---- baselines ----
    aregarch_ar_order=2,
    aregarch_vol='EGARCH',
    aregarch_order=(1, 1, 1),
    aregarch_dist="normal",
    har_dist="normal",
    har_egarch=True,
    har_egarch_order=(1, 1, 1),
    har_context_len=None,           # None -> global context_len

    # ---- inference ----
    interval=(0.05, 0.95),            # nominal 90% interval for coverage

    # ---- residual analysis (compute_residuals / analyse_residuals) ----
    resid_refit_frequency="W",       # baseline REFIT cadence ("W"|"M") for the residual computation
                                    # residuals are computed daily regardless.
                                    # Split from origin_frequency so the daily backtest
                                    # grid doesn't break/invalidate the residual caches
                                    # (cache stamp keeps the field name origin_frequency).
    resid_start='2017-12-29',       # bound the daily forecast days (None -> full history);
    resid_end=None,                 # changing these invalidates the residual cache
    resid_batch_size=8,             # number of forecast days per Chronos forward/predict call.
    resid_max_seqs=512,             # step>1 only: max decoder sequences in flight per predict
                                    # (resid_batch_size x resid_max_seqs) to limit memory

    # ---- regimes (dates refer to the FORECAST ORIGIN) ----
    regimes={
        "volmageddon": ("2018-01-02", "2018-02-02"),
        "covid": ("2020-01-22", "2020-07-01"),
        "tariffs": ("2025-03-04", "2025-04-10"),
        # anything else -> "others"
    },

    # ---- output ----
    save_samples=True,              # persist full (num_samples, h) float32 sample paths per
                                    # origin to {out_dir}/samples_<run stamp>/
    out_dir="backtest_output",
)
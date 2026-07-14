"""
AR(p)-EGARCH baseline.

    x_t   = c + sum_{i=1..p} a_i x_{t-i} + e_t,      e_t = sigma_t z_t
    log sigma_t^2 = omega + alpha(|z_{t-1}| - E|z|) + gamma z_{t-1}
                          + beta log sigma_{t-1}^2

Fit jointly by MLE with the `arch` package.

  * Residual distribution default to Normal. Student-t was the original choice
    but AR-EGARCH convergence is numerically fragile on log-VIX.
  * Forecast    : Monte-Carlo simulation (EGARCH has no closed-form multi-step
                  variance); predictive quantiles are empirical sample quantiles,
                  point forecast is the median.
  * Robust MLE  : uses arch_fit.fit_arch_robust for health check on fit

Usage
-----
    model = AREGARCH(ar_order=2).fit(data)
    out = model.forecast(h=21, num_samples=256)
"""

from __future__ import annotations
import warnings
from typing import Optional, Sequence, Union

import numpy as np
from arch import arch_model
from arch_fit import fit_arch_robust

DEFAULT_QUANTILES = (0.05, 0.5, 0.95)


class AREGARCH:
    """AR(p)-EGARCH baseline.

    Parameters
    ----------
    ar_order : int, default 2
        Order p of the AR conditional mean.
    vol : {'EGARCH', 'GARCH'}, default 'EGARCH'
        Volatility model. 'GARCH' with `o>0` is GJR-GARCH (a stable, fat-tail
        friendly asymmetric alternative to EGARCH).
    egarch_order : tuple(int, int, int), default (1, 1, 1)
        The (p, o, q): : symmetric error lag, asymmetry error lags, cond. var lags.
    dist : {'normal', 't', 'skewt', 'ged'}, default 'normal'
        Residual distribution. Normal is the stable default (see module note).
    apply_log : bool, default True
        If True, work internally on log transformation and return raw values.
        Set False when the caller already supplies the model-scale series.
    apply_diff : bool, default False
        If True, fit on FIRST DIFFERENCES of the (log-)series and integrate
        forecast paths back to levels (last observed level + cumulative sum)
        before returning.
    random_state : int or np.random.Generator, optional
        Seed / generator for the forecast simulation.
    """

    def __init__(
        self,
        ar_order: int = 2,
        vol: str = "EGARCH",
        egarch_order: tuple = (1, 1, 1),
        dist: str = "normal",
        apply_log: bool = True,
        apply_diff: bool = False,
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ):
        self.ar_order = ar_order
        self.vol = vol
        self.egarch_order = egarch_order
        self.dist = dist
        self.apply_log = apply_log
        self.apply_diff = apply_diff
        self.random_state = random_state
        self._np_seed = self._resolve_seed(random_state)

        self.res_ = None
        self.model_ = None
        self.last_value_ = None
        self.last_level_ = None   # integration anchor (model scale, apply_diff)
        self.fit_scale_ = None    # rescale-ladder rung that produced res_ (1.0 = native)
        self.mean_params_ = None  # unit-scale mean params (Const, y[1..p]) regardless of rung

    @staticmethod
    def _resolve_seed(random_state) -> Optional[int]:
        if random_state is None:
            return None
        if isinstance(random_state, np.random.Generator):
            return int(random_state.integers(0, 2**31 - 1))
        return int(random_state)

    def _to_model_scale(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float).ravel()
        if self.apply_log:
            if np.any(y <= 0):
                raise ValueError("apply_log=True but series has non-positive values")
            return np.log(y)
        return y

    def _from_model_scale(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x) if self.apply_log else x

    def _build_model(self, x: np.ndarray):
        # use arch fully for joint MLE
        p, o, q = self.egarch_order
        # rescaling done in fit_arch_robust; arch_model() does not rescale the data
        kwargs = dict(mean="ARX", lags=self.ar_order, dist=self.dist, rescale=False)
        if self.vol.upper() == "EGARCH":
            kwargs.update(vol="EGARCH", p=p, o=o, q=q)
        else:  # GARCH / GJR-GARCH
            kwargs.update(vol="GARCH", p=p, o=o, q=q)
        return arch_model(x, **kwargs)

    def fit(self, y: Union[np.ndarray, Sequence[float]]) -> "AREGARCH":
        """
        Fit the AR(p)-EGARCH by joint MLE.
        Fitting goes through the rescale-retry ladder using arch_fit.fit_arch_robust
        """
        x = self._to_model_scale(y)
        self.last_level_ = float(x[-1])
        if self.apply_diff:
            x = np.diff(x)
        self.last_value_ = float(x[-1])

        # Fit with joint MLE
        def build(data):
            m = self._build_model(data)
            # seed the distribution so simulation forecasts are reproducible
            # AND drawn from the fitted distribution (normal/ged/t)
            m.distribution = type(m.distribution)(seed=self._np_seed)
            return m

        # benchmark to check if fit is healthy
        def benchmark(data):
            return arch_model(data, mean="ARX", lags=self.ar_order,
                              vol="Constant", dist=self.dist, rescale=False)

        self.res_, self.fit_scale_ = fit_arch_robust(
            x, build, benchmark,
            label=f"AR({self.ar_order})-{self.vol} (dist={self.dist!r})")
        self.model_ = build(x * self.fit_scale_ if self.fit_scale_ != 1.0 else x)
        self.mean_params_ = self.res_.params.iloc[: self.ar_order + 1] / np.concatenate(
            [[self.fit_scale_], np.ones(self.ar_order)])
        return self

    def _check_fitted(self):
        if self.res_ is None:
            raise RuntimeError("Call fit() before forecast()/simulate().")

    def forecast(
        self,
        h: int,
        num_samples: int = 256,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
    ) -> dict:
        """
        Simulation-based predictive distribution for the next `h` steps.

        Returns a dict with:
          point      : (h,) median forecast on the output scale
          quantiles  : {level: (h,)} predictive quantiles
          samples    : (n_finite, h) simulated paths on the output scale
        """
        self._check_fitted()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self.res_.forecast(
                horizon=h,
                method="simulation",
                simulations=num_samples,
                reindex=False,
            )
        # (1, num_samples, h) -> (num_samples, h); undo the fit-ladder scale
        # first (paths live on fit_scale_ * x), then integrate diffs back to
        # levels (apply_diff), then the model (log) scale.
        paths = np.asarray(fc.simulations.values)[-1] / self.fit_scale_
        if self.apply_diff:
            paths = self.last_level_ + np.cumsum(paths, axis=1)
        paths = self._from_model_scale(paths)

        # raise if any paths are non-finite
        finite = np.isfinite(paths).all(axis=1)
        if finite.sum() < num_samples:
            raise RuntimeError(
                "AR-EGARCH forecast produced non-finite paths "
                f"{num_samples - finite.sum()} are non-finite."
                "try dist='normal' or vol='GARCH'."
            )
        # paths = paths[finite]

        quantiles = {
            float(qq): np.quantile(paths, qq, axis=0) for qq in quantile_levels
        }
        point = np.median(paths, axis=0)
        return {"point": point, "quantiles": quantiles, "samples": paths}

    def simulate(self, n_steps: int, num_paths: int = 1, burn_in: int = 500) -> np.ndarray:
        """
        Generate UNCONDITIONAL sample path(s) from the fitted model: arch's
        model.simulate() draws from the stationary process after `burn_in`
        steps, ignoring the observed history and using its own RNG stream.

        NOTE: this is NOT the generator behind forecast(), which simulates
        paths CONDITIONAL on the fitted history via res_.forecast(
        method='simulation') — unlike har.py whose simulate() continues
        from the data and therefore backs forecast().

        Returns an array of shape (num_paths, n_steps) on the output scale.
        """
        self._check_fitted()
        out = np.empty((num_paths, n_steps), dtype=float)
        for i in range(num_paths):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sim = self.model_.simulate(
                    self.res_.params, nobs=n_steps, burn=burn_in
                )
            out[i] = sim["data"].values
        out = out / self.fit_scale_
        if self.apply_diff:
            # unconditional Δ paths, integrated from the last observed level
            out = self.last_level_ + np.cumsum(out, axis=1)
        return self._from_model_scale(out)


def _demo():  # pragma: no cover - manual smoke test
    import pandas as pd

    s = pd.read_csv(
        "data/VIXCLS.csv", parse_dates=[0], index_col=0
    ).iloc[:, 0].dropna()
    context = s.values[-512:]
    model = AREGARCH(ar_order=2, random_state=0).fit(context)
    print(model.res_.params.round(4).to_dict())
    out = model.forecast(h=21, num_samples=256)
    print(f"last VIX={context[-1]:.2f}  median h=21={out['point'][-1]:.2f}"
          f"  q10/q90={out['quantiles'][0.1][-1]:.2f}/{out['quantiles'][0.9][-1]:.2f}"
          f"  finite_paths={out['samples'].shape[0]}")
    path = model.simulate(n_steps=252, num_paths=1)[0]
    print("simulated 1y path VIX range:", np.round(path.min(), 2), "..", np.round(path.max(), 2))


if __name__ == "__main__":
    _demo()

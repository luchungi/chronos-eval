"""
HAR baseline for VIX forecasting.

Heterogeneous Autoregressive model (Corsi, 2009) on x = log-VIX with the standard
daily / weekly / monthly regressors:

    x_t = b0 + b1 x_{t-1} + b2 mean(x_{t-5..t-1}) + b3 mean(x_{t-22..t-1}) + e_t

  * Two-step estimation: OLS for the mean then a residual model:
      - `dist` in {'normal' (default), 'ged', 't'}: iid MLE with mean fixed
        at 0 (iid norm / gennorm / t) when EGARCH is off;
      - `egarch=True`: zero-mean EGARCH (order configurable) on the OLS
        residuals via `arch`, with the same `dist` choices.
        Uses arch_fit.fit_arch_robust for health check on fit
  * h-step forecast: iterated Monte Carlo. The residual process does not feed
    back into the mean recursion, so eps paths are pre-simulated (iid draws or
    arch simulation for EGARCH) and the HAR recursion is then iterated per
    path, recomputing the 1/5/22-day regressors from the simulated history.
    Non-finite paths raise an exception.

Usage
-----
    model = HAR(dist="t", egarch=True).fit(data)
    out = model.forecast(h=21, num_samples=256)
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Sequence, Union

import numpy as np
from scipy import stats
from arch import arch_model

from arch_fit import fit_arch_robust

# child of the "backtest" run logger: fit-time fallbacks land in the per-run
# log file when one is configured (backtest.setup_run_logger)
logger = logging.getLogger("backtest.har")

DEFAULT_QUANTILES = (0.05, 0.5, 0.95)
_LAGS = (1, 5, 22)  # daily / weekly / monthly averaging windows


class HAR:
    """HAR(1, 5, 22) baseline.

    Parameters
    ----------
    dist : {'normal', 'ged', 't'}, default 'normal'
        Residual distribution. Fit by MLE with loc=0 on the OLS residuals
        (iid case) or used as the EGARCH residual distribution.
    egarch : bool, default False
        If True, layer a zero-mean EGARCH on the OLS residuals (volatility
        clustering in the residuals) instead of an iid residual model.
    egarch_order : tuple(int, int, int), default (1, 1, 1)
        (p, o, q): symmetric error lag, asymmetry error lags, cond. var lags.
    apply_log : bool, default True
        If True, work internally on log transformation and return raw values.
        Set False when the caller already supplies the model-scale series.
    apply_diff : bool, default False
        If True, fit the HAR recursion on FIRST DIFFERENCES of the
        (log-)series and integrate forecast paths back to levels (last
        observed level + cumulative sum) before returning.
    random_state : int or np.random.Generator, optional
        Seed / generator for the forecast simulation.
    """

    def __init__(
        self,
        dist: str = "normal",
        egarch: bool = False,
        egarch_order: tuple = (1, 1, 1),
        apply_log: bool = True,
        apply_diff: bool = False,
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ):
        dist = dist.lower()
        if dist not in ("normal", "ged", "t"):
            raise ValueError(f"dist must be 'normal', 'ged' or 't', got {dist}")
        self.dist = dist
        self.egarch = egarch
        self.egarch_order = egarch_order
        self.apply_log = apply_log
        self.apply_diff = apply_diff
        self.random_state = random_state
        self._np_seed = self._resolve_seed(random_state)

        self.beta_ = None          # (4,) OLS coefficients [const, d, w, m]
        self.resid_ = None
        self.resid_dist_ = None    # frozen scipy distribution (iid case)
        self.egarch_res_ = None    # arch fit result (egarch case), on fit_scale_ * resid
        self.fit_scale_ = None     # rescale-ladder rung of the EGARCH fit (1.0 = native)
        self.x_tail_ = None        # last 22 model-scale values, recursion seed
        self.last_level_ = None    # integration anchor (model scale, apply_diff)

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

    @staticmethod
    def _design(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute regressor matrix [1, x_{t-1}, mean_5, mean_22] and target x_t."""
        m = max(_LAGS)
        n = len(x)
        t = np.arange(m, n)
        csum = np.concatenate([[0.0], np.cumsum(x)])
        cols = [np.ones(len(t))]
        for lag in _LAGS:
            # t is up to origin due to additional 0. at start of csum
            cols.append((csum[t] - csum[t - lag]) / lag)
        return np.column_stack(cols), x[t]

    def fit(self, y: Union[np.ndarray, Sequence[float]]) -> "HAR":
        """Fit HAR by OLS, then the residual model."""
        x = self._to_model_scale(y)
        self.last_level_ = float(x[-1])
        if self.apply_diff:
            x = np.diff(x)
        if len(x) < max(_LAGS) + 30:
            raise ValueError(f"need at least {max(_LAGS) + 30} observations, got {len(x)}")
        X, target = self._design(x) # get regressors and target for OLS
        self.beta_, *_ = np.linalg.lstsq(X, target, rcond=None) # OLS coefficients
        self.resid_ = target - X @ self.beta_
        self.x_tail_ = x[-max(_LAGS):].copy()

        if self.egarch:
            # Fit EGARCH on the OLS residuals (zero-mean, dist=dist)
            def build(data):
                p, o, q = self.egarch_order
                # rescaling done in fit_arch_robust; arch_model() does not rescale the data
                am = arch_model(data, mean="Zero", vol="EGARCH",
                                p=p, o=o, q=q, dist=self.dist, rescale=False)
                # seed the distribution so simulation forecasts are reproducible
                # AND drawn from the fitted distribution (normal/ged/t)
                am.distribution = type(am.distribution)(seed=self._np_seed)
                return am

            # benchmark to check if the EGARCH fit is healthy
            def benchmark(data):
                return arch_model(data, mean="Zero", vol="Constant",
                                  dist=self.dist, rescale=False)
            self.egarch_res_, self.fit_scale_ = fit_arch_robust(
                self.resid_, build, benchmark,
                label=f"HAR residual EGARCH (dist={self.dist})")
        else:
            self.fit_scale_ = 1.0  # no arch fit in the iid branch
            self.resid_dist_ = self._fit_iid_dist(self.resid_)
        return self

    def _fit_iid_dist(self, resid: np.ndarray):
        """MLE of the residual distribution with location fixed at 0."""
        try:
            if self.dist == "t":
                df, loc, scale = stats.t.fit(resid, floc=0.0)
                return stats.t(df, loc=loc, scale=scale)
            if self.dist == "ged":
                b, loc, scale = stats.gennorm.fit(resid, floc=0.0)
                return stats.gennorm(b, loc=loc, scale=scale)
        except Exception:
            logger.warning("MLE fit of %r residual distribution failed; "
                           "falling back to normal", self.dist)
        return stats.norm(loc=0.0, scale=float(np.sqrt(np.mean(resid**2))))

    def _check_fitted(self):
        if self.beta_ is None:
            raise RuntimeError("Call fit() before forecast()/simulate().")

    def _simulate_eps(self, h: int, num_paths: int) -> np.ndarray:
        """Pre-simulated residual paths, shape (num_paths, h), model scale."""
        if self.egarch:
            # use arch forecast to simulate the residuals
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = self.egarch_res_.forecast(
                    horizon=h, method="simulation",
                    simulations=num_paths, reindex=False,
                )
            # zero-mean model -> the simulated values ARE the eps paths
            # (on fit_scale_ * resid -> undo the fit-ladder scale)
            return np.asarray(fc.simulations.values)[-1] / self.fit_scale_
        else:
            # use the iid distribution to draw eps paths if not EGARCH
            rng = np.random.default_rng(self._np_seed)
            return self.resid_dist_.rvs(size=(num_paths, h), random_state=rng)

    def _iterate_paths(self, eps: np.ndarray) -> np.ndarray:
        """Iterate the HAR recursion per path, model scale, shape like eps."""
        num_paths, h = eps.shape
        m = max(_LAGS)
        hist = np.tile(self.x_tail_, (num_paths, 1))   # (num_paths, m)
        out = np.empty((num_paths, h), dtype=float)
        b0, bd, bw, bm = self.beta_
        for k in range(h):
            mean = (b0
                    + bd * hist[:, -1]
                    + bw * hist[:, -_LAGS[1]:].mean(axis=1)
                    + bm * hist[:, -m:].mean(axis=1))
            x_new = mean + eps[:, k] # add the residual to the mean to get the new value
            out[:, k] = x_new
            hist = np.column_stack([hist[:, 1:], x_new]) # shift in the new value
        return out

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
          point_mean : (h,) mean forecast on the output scale
          quantiles  : {level: (h,)} predictive quantiles
          samples    : (n_finite, h) simulated paths on the output scale
        """
        # forecast paths ARE conditional simulations continuing from the fit
        paths = self.simulate(n_steps=h, num_paths=num_samples)

        # raise if any paths are non-finite
        finite = np.isfinite(paths).all(axis=1)
        if finite.sum() < num_samples:
            raise RuntimeError(
                "HAR forecast produced non-finite paths"
                f"{num_samples - finite.sum()} are non-finite."
                "try egarch=False or dist='normal'."
            )
        # paths = paths[finite]

        quantiles = {float(qq): np.quantile(paths, qq, axis=0) for qq in quantile_levels}
        return {
            "point": np.median(paths, axis=0),
            "point_mean": paths.mean(axis=0),
            "quantiles": quantiles,
            "std": paths.std(axis=0),
            "samples": paths,
        }

    def simulate(self, n_steps: int, num_paths: int = 1) -> np.ndarray:
        """
        Conditional sample path(s) continuing from the end of the fit data.

        Returns an array of shape (num_paths, n_steps) on the output scale.
        """
        self._check_fitted()
        eps = self._simulate_eps(n_steps, num_paths)
        paths = self._iterate_paths(eps)          # model scale (Δ if apply_diff)
        if self.apply_diff:
            paths = self.last_level_ + np.cumsum(paths, axis=1)
        return self._from_model_scale(paths)


def _demo():  # pragma: no cover - manual smoke test
    import pandas as pd

    s = pd.read_csv(
        "data/VIXCLS.csv", parse_dates=[0], index_col=0
    ).iloc[:, 0].dropna()
    context = s.values[-512:]
    for dist, egarch in [("normal", False), ("t", False), ("ged", False),
                         ("normal", True), ("t", True)]:
        model = HAR(dist=dist, egarch=egarch, random_state=0).fit(context)
        out = model.forecast(h=21, num_samples=256)
        print(f"dist={dist:6s} egarch={egarch!s:5s}  beta={np.round(model.beta_, 3)}  "
              f"last={context[-1]:.2f}  med={out['point'][-1]:.2f}  "
              f"mean={out['point_mean'][-1]:.2f}  "
              f"q10/q90={out['quantiles'][0.1][-1]:.2f}/{out['quantiles'][0.9][-1]:.2f}  "
              f"finite={out['samples'].shape[0]}")
    path = HAR(random_state=0).fit(context).simulate(n_steps=252, num_paths=1)[0]
    print("simulated 1y path VIX range:", np.round(path.min(), 2), "..", np.round(path.max(), 2))


if __name__ == "__main__":
    _demo()

"""
Robust MLE fitting for `arch` models.

Why this exists: on log-VIX-scale data (variance ~0.005, below the 1..1000
range arch's optimiser SLSQP heuristics are tuned for), the joint EGARCH
MLE intermittently fails on volatility-regime-shift windows. Failures
come in two flavours, both of which leave arch holding a garbage iterate:

  * convergence_flag == 4 ("Inequality constraints incompatible"): SLSQP's QP
    subproblem went infeasible and the optimizer aborted mid-run;
  * convergence_flag == 0 but the solution is far from optimal

Refitting on scaled data (x10 / x100 / x1000) rescues every observed failure/
Fit at native scale, health-check, rescale and retry until healthy.
Health check based on loglikelihood better than constant vol benchmark and
iid-normal floor within the same scale.

All rescaled attempts fails results in final refit with seeded starting values
from the nested constant-vol fit's parameters plus near-constant volatility dynamics.
Raise exception only if that also fails.
"""

from __future__ import annotations

import logging
import warnings
from typing import Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger("backtest.arch_fit")

DEFAULT_SCALES = (1.0, 10.0, 100.0, 1000.0)

_FIT_OPTIONS = {"maxiter": 2000}

# slight tolerance on the health check with iid-normal
_FLOOR_SLACK = 5.0

_fit_context: str = ""

def set_fit_context(ctx: str) -> None:
    """
    Free-text tag (e.g. 'origin 2018-01-31') prepended to every ladder log
    line, so retries in a long run can be traced back to their window.
    """
    global _fit_context
    _fit_context = ctx


def _ctx() -> str:
    return f"[{_fit_context}] " if _fit_context else ""


def _iid_normal_floor(data: np.ndarray, nobs: int) -> float:
    """
    Closed-form loglik of the constant-mean iid-normal MLE on the last
    `nobs` points of `data`.
    Returns -inf (check disabled) for degenerate data (zero variance).
    """
    x = np.asarray(data[-nobs:], dtype=float)
    var = float(np.var(x))
    if not np.isfinite(var) or var <= 0.0:
        return -np.inf
    return -0.5 * nobs * (1.0 + np.log(2.0 * np.pi * var))


def _seeded_starting_values(model, bench_res) -> np.ndarray:
    """
    Starting values for a GARCH-family fit seeded from the nested
    constant-vol benchmark fit on the same data. Benchmark mean and
    shape parameters, near-constant volatility dynamics. arch's parameter
    order is [mean..., vol..., dist...]; the benchmark's single vol
    parameter is sigma2.
    """
    n_dist = int(model.distribution.num_params)
    bench = np.asarray(bench_res.params, dtype=float)
    n_mean = len(bench) - 1 - n_dist
    mean_p = bench[:n_mean]
    sigma2 = max(float(bench[n_mean]), 1e-12)
    dist_p = bench[len(bench) - n_dist:] if n_dist else np.empty(0)

    vol = model.volatility
    p = int(getattr(vol, "p", 0) or 0)
    o = int(getattr(vol, "o", 0) or 0)
    q = int(getattr(vol, "q", 0) or 0)
    if type(vol).__name__.upper() == "EGARCH":
        # stationary level: log sigma2 = omega / (1 - sum(beta)), beta -> 0.9
        vol_p = np.concatenate([
            [0.1 * np.log(sigma2)],
            np.full(p, 0.2 / max(p, 1)),
            np.zeros(o),
            np.full(q, 0.9 / max(q, 1)),
        ])
    else:  # GARCH / GJR-GARCH
        vol_p = np.concatenate([
            [0.05 * sigma2],
            np.full(p, 0.05 / max(p, 1)),
            np.full(o, 0.05 / max(o, 1)),
            np.full(q, 0.85 / max(q, 1)),
        ])
    return np.concatenate([mean_p, vol_p, dist_p])


def fit_arch_robust(
    x: np.ndarray,
    build: Callable[[np.ndarray], "object"],
    benchmark_build: Optional[Callable[[np.ndarray], "object"]] = None,
    scales: Sequence[float] = DEFAULT_SCALES,
    label: str = "arch model",
):
    """
    Fit an arch model with the rescale-retry ladder.

    Parameters
    ----------
    x : 1D data on the caller's model scale (e.g. log-VIX).
    build : callable(data) -> arch model
        Builds the model to estimate on (possibly scaled) data. Called once
        per attempted rung, so per-fit setup (e.g. seeding the distribution)
        belongs inside it.
    benchmark_build : callable(data) -> arch model, optional
        Builds the NESTED benchmark (same mean, constant variance, same dist)
        used purely as a garbage detector: the model's loglik must be >= the
        benchmark's on the same scaled data. Also the parameter source for
        the R52 seeded second pass. None -> no benchmark/seeded pass; the
        flag, floor and shape checks still apply (use when the model has no
        strictly nested constant-vol special case, e.g. when it IS the
        constant-variance model).
    scales : ladder rungs, tried in order; first healthy fit wins.
    label : name used in the failure message.

    Returns
    -------
    (result, scale) : the healthy arch fit result (estimated on `x * scale`)
        and the winning scale. scale == 1.0 from the FIRST pass means the fit
        is bit-identical to a plain fit on `x`. Parameters/forecasts of a
        scale>1 result live on the scaled data; the caller owns mapping them
        back (R34).

    Raises
    ------
    RuntimeError if every rung of both passes fails the health check (R33).
    """
    x = np.asarray(x, dtype=float)
    attempts = []
    bench_cache: dict[float, object] = {}
    for seeded in (False, True):
        # start with no seeding then try seeded pass if failed
        if seeded and benchmark_build is None:
            # no benchmark -> no seeded pass
            break
        for scale in scales:
            data = x * scale if scale != 1.0 else x
            # silence convergence warnings from arch's SLSQP optimizer
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = build(data)
                bench_res = None
                # if a benchmark is provided, cache it per scale so the seeded pass
                if benchmark_build is not None:
                    if scale not in bench_cache:
                        bench_cache[scale] = benchmark_build(data).fit(
                            disp="off", show_warning=False)
                    bench_res = bench_cache[scale]
                sv = None
                if seeded:
                    try:
                        # get seeded starting parameters from the benchmark fit
                        sv = _seeded_starting_values(model, bench_res)
                    except Exception as exc:
                        logger.info("%s%s: cannot build seeded starting values "
                                    "(%s) — skipping the seeded pass",
                                    label, exc)
                        break
                # fit the requested residual model
                res = model.fit(disp="off", show_warning=False,
                                options=_FIT_OPTIONS, starting_values=sv)
            ll = float(res.loglikelihood) # fitted loglikelihood
            nobs = int(getattr(res, "nobs", len(data)) or len(data))
            floor = _iid_normal_floor(data, nobs) # min. ll of iid-normal
            problems = []
            if res.convergence_flag != 0:
                # did not converge to a local optimum
                problems.append(f"flag={res.convergence_flag}")
            if not np.isfinite(ll):
                problems.append(f"loglik={ll}")
            else:
                if bench_res is not None:
                    # compare with benchmark fit on the same scale
                    bench_ll = float(bench_res.loglikelihood)
                    if ll < bench_ll:
                        problems.append(f"loglik={ll:.6g} < constant-vol "
                                        f"benchmark={bench_ll:.6g}")
                if ll < floor - _FLOOR_SLACK:
                    problems.append(f"loglik={ll:.6g} < iid-normal "
                                    f"floor={floor:.6g}")
            if not problems:
                if scale != 1.0 or seeded:
                    # successful fit on a rescaled rung or seeded pass
                    logger.info("%s%s: rescued at rung x%g (loglik=%.6g)",
                            _ctx(), label, scale, ll)
                return res, float(scale)
            attempt = (f"x{scale:g}{' seeded' if seeded else ''}: "
                       + ", ".join(problems))
            attempts.append(attempt)
            logger.info("%s%s: rung %s unhealthy — retrying at the next scale",
                    _ctx(), label, attempt)
    msg = (f"{_ctx()}{label}: MLE unhealthy at every rescale rung (plain and "
           "seeded passes) — the fit is not a usable optimum at this window; "
           "try dist='normal' or vol='GARCH'.\n  " + "\n  ".join(attempts))
    logger.error(msg)
    raise RuntimeError(msg)

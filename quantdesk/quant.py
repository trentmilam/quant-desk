"""Self-contained deterministic quantitative-finance math.

No scipy, no pandas: Black-Scholes pricing + greeks use ``math.erf`` for the normal
CDF; portfolio stats and Monte-Carlo VaR use numpy with a FIXED seed. Every result
is exact/closed-form or reproducible bit-for-bit, so a downstream agent never has to
let an LLM do the arithmetic.
"""
from __future__ import annotations

import math

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT2PI


def _validate_bs(spot, strike, t, vol, kind, r):
    """Reject out-of-domain inputs loudly instead of returning a guessed number."""
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    if not spot > 0:
        raise ValueError(f"spot must be > 0, got {spot!r}")
    if not strike > 0:
        raise ValueError(f"strike must be > 0, got {strike!r}")
    if not t > 0:
        raise ValueError(f"t (time to expiry) must be > 0, got {t!r}")
    if not vol > 0:
        raise ValueError(f"vol must be > 0, got {vol!r}")
    if not math.isfinite(r):
        raise ValueError(f"r (risk-free rate) must be finite, got {r!r}")


def _d1_d2(spot, strike, t, r, vol):
    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    return d1, d1 - vol * math.sqrt(t)


def black_scholes(spot, strike, t, r, vol, kind="call"):
    """European option price (closed form)."""
    _validate_bs(spot, strike, t, vol, kind, r)
    d1, d2 = _d1_d2(spot, strike, t, r, vol)
    if kind == "call":
        return spot * _cdf(d1) - strike * math.exp(-r * t) * _cdf(d2)
    return strike * math.exp(-r * t) * _cdf(-d2) - spot * _cdf(-d1)


def greeks(spot, strike, t, r, vol, kind="call"):
    """delta (per $1), gamma (per $1), vega (per 1% vol), theta (per day), rho (per 1% rate)."""
    _validate_bs(spot, strike, t, vol, kind, r)
    d1, d2 = _d1_d2(spot, strike, t, r, vol)
    gamma = _pdf(d1) / (spot * vol * math.sqrt(t))
    vega = spot * _pdf(d1) * math.sqrt(t) / 100.0
    if kind == "call":
        delta = _cdf(d1)
        theta = (-spot * _pdf(d1) * vol / (2 * math.sqrt(t)) - r * strike * math.exp(-r * t) * _cdf(d2)) / 365.0
        rho = strike * t * math.exp(-r * t) * _cdf(d2) / 100.0
    else:
        delta = _cdf(d1) - 1.0
        theta = (-spot * _pdf(d1) * vol / (2 * math.sqrt(t)) + r * strike * math.exp(-r * t) * _cdf(-d2)) / 365.0
        rho = -strike * t * math.exp(-r * t) * _cdf(-d2) / 100.0
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(price, spot, strike, t, r, kind="call", tol=1e-8, iters=200):
    """Back out implied volatility by bisection (BS price is monotone-increasing in vol).

    Raises ValueError if the observed price violates the no-arbitrage bounds for the
    option kind, or if the bisection does not actually converge to within ``tol`` -- so
    a search-bound is never returned as if it were a genuine implied vol (fail-loud, not
    fail-open).
    """
    _validate_bs(spot, strike, t, 1.0, kind, r)  # reuse spot/strike/t/kind/r guards (vol unused here)
    if not price > 0:
        raise ValueError(f"option price must be > 0, got {price!r}")
    disc_strike = strike * math.exp(-r * t)
    if kind == "call":
        intrinsic, upper = max(spot - disc_strike, 0.0), spot
    else:
        intrinsic, upper = max(disc_strike - spot, 0.0), disc_strike
    if not (intrinsic < price < upper):
        raise ValueError(
            f"{kind} price {price!r} violates no-arbitrage bounds "
            f"({intrinsic:.6g}, {upper:.6g}); implied vol is undefined"
        )
    lo, hi = 1e-6, 5.0
    mid = 0.5 * (lo + hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        diff = black_scholes(spot, strike, t, r, mid, kind) - price
        if abs(diff) < tol:
            return mid
        if diff > 0:
            hi = mid
        else:
            lo = mid
    residual = abs(black_scholes(spot, strike, t, r, mid, kind) - price)
    if residual > tol:
        raise ValueError(
            f"implied_vol failed to converge in {iters} iters "
            f"(residual {residual:.3g} > tol {tol:g})"
        )
    return mid


_MAX_PRICES = 1_000_000
_MAX_SIMS = 1_000_000
_MAX_HORIZON_DAYS = 10_000
# rng.normal(mean, vol, (sims, horizon_days)) allocates sims * horizon_days floats, so
# the two per-parameter caps above multiply rather than add -- both individually valid
# at their max (1,000,000 x 10,000) would allocate ~74.5GB. Bound the product directly.
# 10,000,000 elements is ~80MB for a float64 array: a safe absolute memory budget.
_MAX_PATH_ELEMENTS = 10_000_000


def portfolio_stats(prices):
    """Annualized return/vol (252d), Sharpe (rf=0, None if volatility is degenerate),
    max drawdown (peak-to-trough)."""
    import numpy as np

    if len(prices) < 2:
        raise ValueError("need at least 2 price points")
    if len(prices) > _MAX_PRICES:
        raise ValueError(f"prices length must be <= {_MAX_PRICES}, got {len(prices)}")
    p = np.asarray(prices, dtype=float)
    if not np.all(np.isfinite(p)):
        raise ValueError("prices must all be finite")
    if not np.all(p > 0):
        raise ValueError("prices must all be > 0 (a non-positive price yields Inf/NaN stats)")
    rets = np.diff(p) / p[:-1]
    daily_vol = float(np.std(rets))
    ann_return = float(np.mean(rets)) * 252
    ann_vol = daily_vol * math.sqrt(252)
    # A return series with (near-)zero true variance lands `np.std` on floating-point
    # noise (e.g. ~1e-16) rather than an exact 0.0, so an exact `!= 0` guard never
    # fires and the code would silently divide by that noise -- finite but nonsense.
    # Treat volatility as degenerate when it's within a relative epsilon of the scale
    # of the returns that produced it (mean absolute daily return), not just zero.
    ret_scale = float(np.mean(np.abs(rets)))
    vol_epsilon = ret_scale * 1e-8
    sharpe = float(ann_return / ann_vol) if daily_vol > vol_epsilon else None
    running_max = np.maximum.accumulate(p)
    max_dd = float(np.min(p / running_max - 1.0))
    return {"annual_return": ann_return, "annual_volatility": ann_vol,
            "sharpe_ratio": sharpe, "max_drawdown": max_dd}


def monte_carlo_var(value, mean, vol, horizon_days=1, confidence=0.95, sims=10000, seed=12345):
    """Monte-Carlo VaR + Expected Shortfall (CVaR), positive = potential loss.

    Simulates ``horizon_days`` INDEPENDENT daily shocks per path and compounds them --
    not one shock raised to a power, which would model perfect autocorrelation (the
    same return recurring every day of the horizon). Deterministic given a fixed seed.

    VaR/CVaR can legitimately come back NEGATIVE: when the expected return (``mean``)
    dominates ``vol`` at the requested ``confidence``, the loss distribution's tail is
    itself a gain, and a negative figure means "projected gain," not an error -- this is
    standard risk-practice behavior, not a bug, and is not rejected here.
    """
    import numpy as np

    if not math.isfinite(mean):
        raise ValueError(f"mean must be finite, got {mean!r}")
    if not math.isfinite(value):
        raise ValueError(f"value must be finite, got {value!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    if not vol > 0:
        raise ValueError(f"vol must be > 0, got {vol!r}")
    if not isinstance(horizon_days, int) or horizon_days < 1:
        raise ValueError(f"horizon_days must be a positive integer, got {horizon_days!r}")
    if horizon_days > _MAX_HORIZON_DAYS:
        raise ValueError(f"horizon_days must be <= {_MAX_HORIZON_DAYS}, got {horizon_days!r}")
    if not isinstance(sims, int) or sims <= 0:
        raise ValueError(f"sims must be a positive integer, got {sims!r}")
    if sims > _MAX_SIMS:
        raise ValueError(f"sims must be <= {_MAX_SIMS}, got {sims!r}")
    if sims * horizon_days > _MAX_PATH_ELEMENTS:
        raise ValueError(
            f"sims * horizon_days must be <= {_MAX_PATH_ELEMENTS}, "
            f"got {sims!r} * {horizon_days!r} = {sims * horizon_days!r}"
        )
    rng = np.random.default_rng(seed)
    daily = rng.normal(mean, vol, (sims, horizon_days))
    horizon_returns = np.prod(1 + daily, axis=1) - 1
    pnl = float(value) * horizon_returns
    var_value = float(np.percentile(pnl, (1.0 - confidence) * 100))
    tail = pnl[pnl <= var_value]
    cvar_value = float(np.mean(tail)) if tail.size else var_value
    return {"var": -var_value, "cvar": -cvar_value, "confidence": confidence}

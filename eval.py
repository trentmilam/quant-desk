"""quant-desk eval — correctness + determinism, exits 0 on all checks.

    python eval.py

Checks the math against KNOWN closed-form values and identities (not against
itself), plus determinism and the agent's fail-loud contract.
"""
import math
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from quantdesk import quant                     # noqa: E402
from quantdesk.agent import dispatch            # noqa: E402


def approx(a, b, tol=1e-3):
    return abs(a - b) < tol


# --- independent closed-form Gaussian VaR/CVaR oracle (validation baseline) ---
# Used ONLY by the eval to grade quant.monte_carlo_var against a NON-Monte-Carlo
# reference. This is not imported by the package; it exists so the MC engine is
# checked against external math, not against itself.
def _norm_ppf(p):
    """Inverse standard-normal CDF (Acklam's rational approximation, ~1e-9)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= 1 - plow:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def _gaussian_var_cvar(value, mean, vol, confidence):
    """Closed-form 1-day Gaussian VaR + Expected Shortfall (positive = loss)."""
    alpha = 1.0 - confidence
    z = _norm_ppf(alpha)
    npdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    var = value * (-mean - vol * z)
    cvar = value * (-mean + vol * npdf / alpha)
    return var, cvar


def main() -> int:
    checks = {}

    # 1) Black-Scholes vs a known textbook value (S=K=100, t=1, r=5%, vol=20%, call ~= 10.4506)
    call = quant.black_scholes(100, 100, 1.0, 0.05, 0.20, "call")
    checks["bs_call_known_value"] = approx(call, 10.4506, 1e-3)

    # 2) put-call parity: C - P == S - K*e^{-rt}
    put = quant.black_scholes(100, 100, 1.0, 0.05, 0.20, "put")
    parity_lhs = call - put
    parity_rhs = 100 - 100 * math.exp(-0.05 * 1.0)
    checks["put_call_parity"] = approx(parity_lhs, parity_rhs, 1e-6)

    # 3) greeks sanity: 0<delta_call<1, gamma>0, delta_call - delta_put == 1
    gc = quant.greeks(100, 100, 1.0, 0.05, 0.20, "call")
    gp = quant.greeks(100, 100, 1.0, 0.05, 0.20, "put")
    checks["delta_call_range"] = 0.0 < gc["delta"] < 1.0
    checks["gamma_positive"] = gc["gamma"] > 0.0
    checks["delta_parity"] = approx(gc["delta"] - gp["delta"], 1.0, 1e-6)

    # 4) implied-vol round trip: price at vol=0.2, recover ~0.2
    iv = quant.implied_vol(call, 100, 100, 1.0, 0.05, "call")
    checks["implied_vol_roundtrip"] = approx(iv, 0.20, 1e-4)

    # 5) portfolio stats on a known monoter series (all up 1%/step -> positive return, ~0 drawdown)
    prices = [100 * (1.01 ** i) for i in range(30)]
    ps = quant.portfolio_stats(prices)
    checks["portfolio_return_positive"] = ps["annual_return"] > 0
    checks["portfolio_no_drawdown"] = approx(ps["max_drawdown"], 0.0, 1e-9)
    # Regression guard: a perfectly steady series has (near-)zero true return
    # variance, so np.std lands on floating-point noise (~1e-16) rather than an exact
    # 0.0. An exact `!= 0` guard never fires on that noise and used to silently divide
    # by it, returning sharpe_ratio ~1.6e15. The fix must either flag this as
    # degenerate (sharpe_ratio is None) or keep the value sanely bounded -- never that
    # silent astronomical number.
    checks["portfolio_sharpe_not_silently_wrong"] = (
        ps["sharpe_ratio"] is None or abs(ps["sharpe_ratio"]) < 1000
    )

    # 6) VaR determinism (two runs identical) + positive + CVaR >= VaR
    v1 = quant.monte_carlo_var(1_000_000, 0.0, 0.02, 1, 0.95)
    v2 = quant.monte_carlo_var(1_000_000, 0.0, 0.02, 1, 0.95)
    checks["var_deterministic"] = v1 == v2
    checks["var_positive"] = v1["var"] > 0
    checks["cvar_ge_var"] = v1["cvar"] >= v1["var"]

    # Regression guard: a multi-day horizon must simulate `horizon_days`
    # INDEPENDENT daily shocks and compound them, not raise a single shock to a power
    # (perfect autocorrelation), which inflated 10-day VaR ~9x instead of the correct
    # ~3.16x = sqrt(10) for i.i.d. daily shocks.
    mc_1d = quant.monte_carlo_var(1_000_000, 0.0, 0.02, horizon_days=1, confidence=0.95, sims=200_000)
    mc_10d = quant.monte_carlo_var(1_000_000, 0.0, 0.02, horizon_days=10, confidence=0.95, sims=200_000)
    horizon_ratio = mc_10d["var"] / mc_1d["var"]
    checks["var_horizon_scales_like_sqrt_t"] = approx(horizon_ratio, math.sqrt(10), 0.3)

    # 7) agent fail-loud contract: never guesses on bad input
    checks["agent_unknown_tool"] = dispatch({"tool": "nope"}).get("ok") is False
    checks["agent_missing_param"] = dispatch({"tool": "black_scholes", "params": {"spot": 100}}).get("ok") is False
    checks["agent_unknown_param"] = dispatch({"tool": "black_scholes", "params": {"spot": 100, "strike": 100, "t": 1, "r": 0.05, "vol": 0.2, "wat": 1}}).get("ok") is False
    good = dispatch({"tool": "black_scholes", "params": {"spot": 100, "strike": 100, "t": 1, "r": 0.05, "vol": 0.2}})
    checks["agent_good_request"] = good.get("ok") is True and approx(good["result"], 10.4506, 1e-3) and good["deterministic"] is True

    # 8) value-domain fail-loud contract: bad values MUST return ok:False (never a guessed number)
    def _bad(tool, **params):
        return dispatch({"tool": tool, "params": params}).get("ok") is False

    base = {"spot": 100, "strike": 100, "t": 1, "r": 0.05, "vol": 0.2}
    checks["reject_bad_kind"] = _bad("black_scholes", **{**base, "kind": "straddle"})
    checks["reject_neg_vol"] = _bad("black_scholes", **{**base, "vol": -0.2})
    checks["reject_zero_vol"] = _bad("black_scholes", **{**base, "vol": 0})
    checks["reject_nonpos_t"] = _bad("black_scholes", **{**base, "t": 0})
    checks["reject_nonpos_spot"] = _bad("black_scholes", **{**base, "spot": 0})
    checks["reject_nonpos_strike"] = _bad("black_scholes", **{**base, "strike": -100})
    checks["reject_neg_vol_greeks"] = _bad("greeks", **{**base, "vol": -0.2})
    checks["reject_bad_confidence"] = _bad("monte_carlo_var", value=1e6, mean=0.0, vol=0.02, confidence=1.5)
    # Validation guard: horizon_days/sims are on the untrusted dispatch surface
    # and must fail loud, never silently return a zero/negative-horizon "result".
    checks["reject_bad_horizon_zero"] = _bad("monte_carlo_var", value=1e6, mean=0.0, vol=0.02, horizon_days=0)
    checks["reject_bad_horizon_negative"] = _bad("monte_carlo_var", value=1e6, mean=0.0, vol=0.02, horizon_days=-1)
    checks["reject_bad_sims_zero"] = _bad("monte_carlo_var", value=1e6, mean=0.0, vol=0.02, sims=0)
    checks["reject_bad_sims_negative"] = _bad("monte_carlo_var", value=1e6, mean=0.0, vol=0.02, sims=-5)
    checks["reject_oversized_sims"] = _bad("monte_carlo_var", value=1e6, mean=0.0, vol=0.02, sims=10_000_000)
    checks["reject_oversized_horizon"] = _bad("monte_carlo_var", value=1e6, mean=0.0, vol=0.02, horizon_days=100_000)
    checks["reject_degenerate_prices"] = _bad("portfolio_stats", prices=[100.0, 0.0, 100.0])
    checks["reject_negative_prices"] = _bad("portfolio_stats", prices=[100.0, -50.0])
    checks["reject_too_few_prices"] = _bad("portfolio_stats", prices=[100.0])
    # green: no bad-value guard may reject a clean request
    checks["accept_valid_put"] = dispatch({"tool": "black_scholes", "params": {**base, "kind": "put"}}).get("ok") is True

    # 9) implied_vol no-arbitrage guard: an unrecoverable observed price MUST return
    #    ok:False -- never a silent search-bound handed back as if it were a real vol.
    ivbase = {"spot": 100, "strike": 100, "t": 1, "r": 0.05}
    checks["iv_reject_zero_price"] = _bad("implied_vol", price=0.0, **ivbase)
    checks["iv_reject_neg_price"] = _bad("implied_vol", price=-5.0, **ivbase)
    checks["iv_reject_call_above_spot"] = _bad("implied_vol", price=150.0, **ivbase)   # call > spot
    checks["iv_reject_below_intrinsic"] = _bad("implied_vol", price=2.0, **ivbase)     # call < intrinsic (~4.88)
    checks["iv_reject_put_above_bound"] = _bad("implied_vol", price=200.0, kind="put", **ivbase)
    # green: a legitimate observed price still recovers ok:True (round-trip preserved)
    goodiv = dispatch({"tool": "implied_vol", "params": {"price": call, **ivbase}})
    checks["iv_accept_valid_roundtrip"] = goodiv.get("ok") is True and approx(goodiv["result"], 0.20, 1e-4)

    # 10) Validation baseline: compare the Monte-Carlo VaR/CVaR engine against an
    #     independent closed-form Gaussian VaR/CVaR oracle, and against a naive
    #     VaR-only report.
    #
    #     The incumbent baseline is standard risk-desk practice: (a) validate the MC
    #     estimate by its own determinism and CVaR>=VaR self-consistency, and (b) report
    #     VaR alone, the single-quantile figure that was the regulatory standard for
    #     decades (Basel II).
    #
    #     Measured here:
    #       * the Monte-Carlo VaR/CVaR converge to an independent non-MC closed form,
    #         validating the estimate against external math rather than itself; and
    #       * CVaR is materially larger than VaR at these parameters, i.e. a VaR-only
    #         report would miss real tail loss that CVaR captures.
    cf_var, cf_cvar = _gaussian_var_cvar(1_000_000, 0.0, 0.02, 0.95)

    mc_hi = quant.monte_carlo_var(1_000_000, 0.0, 0.02, 1, 0.95, sims=200_000)
    var_err_hi = abs(mc_hi["var"] - cf_var) / cf_var
    cvar_err_hi = abs(mc_hi["cvar"] - cf_cvar) / cf_cvar

    mc_lo = quant.monte_carlo_var(1_000_000, 0.0, 0.02, 1, 0.95, sims=10_000)
    var_err_lo = abs(mc_lo["var"] - cf_var) / cf_var

    # (a) the MC engine converges to the independent closed form as sims grow
    checks["mc_var_matches_closed_form"] = var_err_hi < 1e-3        # measured ~0.010%
    checks["mc_cvar_matches_closed_form"] = cvar_err_hi < 4e-3      # measured ~0.214%
    checks["mc_converges_with_sims"] = var_err_hi < var_err_lo      # 200k tighter than 10k
    checks["mc_default_within_mc_error"] = var_err_lo < 3e-2        # measured ~2.345%

    # (b) naive-VaR-vs-CVaR contrast: VaR under-reports the expected tail loss
    cvar_var_ratio = cf_cvar / cf_var
    naive_var_shortfall = (cf_cvar - cf_var) / cf_cvar
    checks["cvar_exceeds_var_materially"] = cvar_var_ratio > 1.20   # measured 1.254
    checks["naive_var_underreports_tail"] = naive_var_shortfall > 0.15  # measured 0.203
    checks["tool_cvar_ge_var_at_scale"] = mc_hi["cvar"] > mc_hi["var"]

    # (c) the naive self-consistency validation is INSUFFICIENT: an "ES := VaR"
    #     shortcut (some desks quote VaR as the tail number) passes deterministic +
    #     CVaR>=VaR, yet the closed-form oracle rejects it by the measured margin.
    shortcut_es = mc_hi["var"]                     # naive: report VaR as expected shortfall
    passes_naive_selfcheck = (shortcut_es >= mc_hi["var"])          # True (equal) -> accepted
    shortcut_cf_err = abs(shortcut_es - cf_cvar) / cf_cvar
    checks["naive_selfcheck_accepts_wrong_es"] = passes_naive_selfcheck
    checks["closed_form_rejects_wrong_es"] = shortcut_cf_err > 0.15  # measured ~0.20

    print("=== quant-desk checks (measured) ===")
    for k, v in checks.items():
        print(f"{'OK  ' if v else 'FAIL'} {k}")
    print(f"\nreference: BS call={call:.4f} put={put:.4f} iv={iv:.4f} "
          f"VaR={v1['var']:.2f} CVaR={v1['cvar']:.2f}")
    print("--- closed-form validation baseline (measured) ---")
    print(f"MC vs closed-form Gaussian VaR:  err {var_err_lo*100:.3f}% (10k sims) "
          f"-> {var_err_hi*100:.3f}% (200k sims)")
    print(f"MC vs closed-form Gaussian CVaR: err {cvar_err_hi*100:.3f}% (200k sims)")
    print(f"naive VaR-only vs CVaR: CVaR/VaR = {cvar_var_ratio:.4f}  "
          f"(VaR under-reports the expected tail loss by {naive_var_shortfall*100:.1f}%)")
    passed = all(checks.values())
    print("RESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

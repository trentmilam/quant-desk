"""The no-arithmetic-by-LLM dispatch.

An LLM (or any caller) selects a tool and extracts its parameters; the NUMBER is
produced only by the exact, deterministic code in :mod:`quant`. A malformed request
returns an explicit error -- never a guessed value. Every successful result is
audited: it echoes the inputs, the method used, and a determinism flag, so an
advisor can reproduce and defend it.
"""
from __future__ import annotations

import math

from . import quant


def _is_finite(x) -> bool:
    """A result is trustworthy only if every number in it is finite (no NaN/Inf)."""
    if isinstance(x, dict):
        return all(_is_finite(v) for v in x.values())
    if isinstance(x, (int, float)):
        return math.isfinite(x)
    return True

TOOLS = {
    "black_scholes": {
        "fn": quant.black_scholes,
        "required": ["spot", "strike", "t", "r", "vol"],
        "optional": {"kind": "call"},
        "method": "closed-form Black-Scholes (normal CDF via math.erf)",
    },
    "greeks": {
        "fn": quant.greeks,
        "required": ["spot", "strike", "t", "r", "vol"],
        "optional": {"kind": "call"},
        "method": "closed-form Black-Scholes greeks",
    },
    "implied_vol": {
        "fn": quant.implied_vol,
        "required": ["price", "spot", "strike", "t", "r"],
        "optional": {"kind": "call"},
        "method": "bisection on the closed-form Black-Scholes price",
    },
    "portfolio_stats": {
        "fn": quant.portfolio_stats,
        "required": ["prices"],
        "optional": {},
        "method": "annualized (252d) return/vol, Sharpe, max drawdown",
    },
    "monte_carlo_var": {
        "fn": quant.monte_carlo_var,
        "required": ["value", "mean", "vol"],
        "optional": {"horizon_days": 1, "confidence": 0.95, "sims": 10000, "seed": 12345},
        "method": "seeded Monte-Carlo VaR + CVaR (deterministic given seed)",
    },
}


def _round(x, nd=6):
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, float):
        return round(x, nd)
    return x


def dispatch(request: dict) -> dict:
    """request = {"tool": <name>, "params": {...}} -> audited result dict."""
    tool = request.get("tool")
    spec = TOOLS.get(tool)
    if spec is None:
        return {"ok": False, "error": f"unknown tool: {tool!r}", "available": sorted(TOOLS)}

    params = request.get("params", {}) or {}
    if not isinstance(params, dict):
        return {"ok": False, "error": "params must be an object", "tool": tool}

    missing = [k for k in spec["required"] if k not in params]
    if missing:
        return {"ok": False, "error": f"missing required params: {missing}", "tool": tool}

    allowed = set(spec["required"]) | set(spec["optional"])
    unknown = [k for k in params if k not in allowed]
    if unknown:
        return {"ok": False, "error": f"unknown params: {unknown}", "tool": tool, "allowed": sorted(allowed)}

    kwargs = {**spec["optional"], **{k: params[k] for k in params}}
    try:
        result = spec["fn"](**kwargs)
    except Exception as e:  # bad numeric input (e.g. vol<=0) -> explicit, never a guess
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "tool": tool, "inputs": kwargs}

    if not _is_finite(result):  # never hand back NaN/Inf as if it were trustworthy
        return {"ok": False, "error": "non-finite result (NaN/Inf)", "tool": tool, "inputs": kwargs}

    return {
        "ok": True,
        "tool": tool,
        "inputs": kwargs,
        "method": spec["method"],
        "deterministic": True,
        "result": _round(result),
    }

# quant-desk — deterministic, auditable quant math (no-arithmetic-by-LLM)

A finance-math layer an LLM **calls** instead of doing the arithmetic itself. The
model selects a tool and extracts parameters; the number is produced only by exact,
deterministic code and returned **audited** (inputs + method + determinism flag), so
an analyst or advisor can reproduce and defend it. A malformed request returns an
explicit error — never a guessed value.

Built for the person who cannot ship a wrong number: analysts, wealth managers,
prop-desk and advisory tooling that (rightly) distrusts LLM arithmetic.

## Quickstart

```
python eval.py
```

Self-contained: Black-Scholes pricing + greeks use `math.erf` (no scipy); portfolio
stats and Monte-Carlo VaR use numpy with a fixed seed. Deterministic and offline.

## Measured (eval.py, exit 0 — 42/42 checks)

Math checked against KNOWN values/identities, not against itself:

- **Black-Scholes call = 10.4506** for S=K=100, t=1, r=5%, vol=20% (textbook value);
- **put-call parity** `C - P == S - K·e^(-rt)` holds;
- greeks sanity: `0 < delta_call < 1`, `gamma > 0`, `delta_call - delta_put == 1`;
- **implied vol round-trips** to 0.2000;
- portfolio stats: positive annual return + ~0 drawdown on a monotone-up series;
- **VaR is deterministic** (two runs identical), positive, and `CVaR ≥ VaR`;
- **agent fail-loud contract**: unknown tool / missing param / unknown param all
  return `ok:false`; only a valid request yields a number.
- **value-domain fail-loud**: bad `kind`, `vol<=0`, `t<=0`, `spot<=0`, `strike<=0`,
  `confidence` outside `(0,1)`, and degenerate/non-positive/too-few price series all
  return `ok:false`; no `NaN`/`Inf` is ever returned as `ok:true`.
- **implied-vol no-arbitrage guard**: an observed option price that is non-positive or
  violates its no-arbitrage bounds (e.g. a call `> spot` or `< intrinsic`) returns
  `ok:false` — the bisection never hands back a search-bound as if it were a real vol,
  and a returned vol is reported only after the price is verified reproduced within tol.

### Validation baseline: closed-form vs Monte-Carlo, and naive-VaR vs CVaR

The Monte-Carlo VaR/CVaR engine is graded against an **independent closed-form Gaussian
VaR/CVaR oracle** (Acklam inverse-normal-CDF, no scipy) — external math, not itself:

- **MC converges to the closed form.** VaR error `2.345%` at the 10k-sim default →
  `0.010%` at 200k sims; CVaR error `0.214%` at 200k [measured]. The estimator is right,
  not just deterministic.
- **naive VaR-only under-reports the tail.** VaR was the industry's single-quantile
  standard for decades, but here `CVaR/VaR = 1.254` — reporting VaR as the tail figure
  **under-reserves the expected shortfall by 20.3%** [measured]. The tool ships CVaR
  (coherent expected shortfall), which the naive VaR-only report misses.
- **the naive self-check is insufficient.** An "ES := VaR" shortcut passes the common
  `deterministic + CVaR≥VaR` self-consistency check yet the closed-form oracle rejects it
  (`20.3%` off) — validation must be against external math, which is what this eval does.

## API

```python
from quantdesk import quant, dispatch

quant.black_scholes(100, 100, 1.0, 0.05, 0.20, "call")   # -> 10.4506...
dispatch({"tool": "black_scholes",
          "params": {"spot":100,"strike":100,"t":1,"r":0.05,"vol":0.2}})
# -> {"ok":True,"tool":"black_scholes","inputs":{...},
#     "method":"closed-form Black-Scholes (normal CDF via math.erf)",
#     "deterministic":True,"result":10.4506}
```

Tools: `black_scholes`, `greeks`, `implied_vol`, `portfolio_stats`, `monte_carlo_var`.

## Honest scope

Deterministic math, not investment advice. Assumptions: European options, flat
constant volatility, no dividends; Gaussian Monte-Carlo VaR (no fat tails / no
historical bootstrap); 252-trading-day annualization; risk-free rate 0 for Sharpe.
Word-for-word an implementation of standard textbook formulas — the value is the
**auditable, deterministic, LLM-doesn't-do-the-math** delivery, not new math.

## Where it fits

A **compute module** for the Consilium switchboard — the v3
step of heterogeneous modules that mix retrieval (cite a source) with computation
(return an audited number). Part of the wealth-tech portfolio track.

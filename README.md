# quant-desk — deterministic finance math for LLM agents

LLMs are confidently wrong at arithmetic. quant-desk does the finance math in exact,
testable code instead, so a model never has to guess a number: it picks a tool,
supplies the inputs, and gets back an answer with its inputs and method attached so
anyone can check it. A malformed request returns an explicit error, never a guessed
value.

Built for people who cannot afford to ship a wrong number: analysts, wealth managers,
and trading or advisory tools that don't trust an LLM to do arithmetic on its own.

## Setup

```
pip install -e .
```

Installs the one runtime dependency, `numpy`. Tested against Python 3.12 + numpy 2.5;
`pyproject.toml` declares `numpy>=1.22`, `requires-python>=3.9`.

For development (running the `pytest` suite in addition to `eval.py`), install the
`dev` extra instead: `pip install -e '.[dev]'`. See `CONTRIBUTING.md`.

## Quickstart

```
python eval.py
```

Black-Scholes pricing (the standard option-pricing formula) and greeks (numbers
describing how sensitive an option's price is) use `math.erf` (no scipy); portfolio
stats and Monte-Carlo VaR (a worst-case-loss estimate built from simulated price
paths) use numpy with a fixed seed. Deterministic and offline, with one declared
dependency (numpy).

## Measured (eval.py, exit 0 — 50/50 checks)

Math checked against known values and identities, not against itself:

- Black-Scholes call = 10.4506 for S=K=100, t=1, r=5%, vol=20% (textbook value);
- put-call parity `C - P == S - K·e^(-rt)` holds;
- greeks sanity: `0 < delta_call < 1`, `gamma > 0`, `delta_call - delta_put == 1`;
- implied vol (volatility backed out from an observed price) round-trips to 0.2000;
- portfolio stats: positive annual return and ~0 drawdown on a monotone-up series; a
  series with (near-)zero true return variance reports `sharpe_ratio: None` (degenerate
  volatility) rather than dividing by floating-point noise;
- VaR is deterministic (two runs identical) and `CVaR ≥ VaR` (CVaR is the average loss
  in the worst-case tail). VaR and CVaR are usually positive (a potential loss) but can
  legitimately come back negative when the expected return dominates volatility at the
  chosen confidence — that means "projected gain," not a bug (see "Honest scope"
  below);
- VaR horizon scaling tracks `sqrt(t)`: a 10-day horizon simulates 10 independent daily
  shocks and compounds them, so VaR scales `~sqrt(10) ≈ 3.16x` the 1-day figure, not
  `(1+r)^10` (perfect-autocorrelation) growth;
- agent fail-loud contract: unknown tool, missing param, unknown param, and a
  non-object top-level request all return `ok:false`; only a valid request yields a
  number.
- value-domain fail-loud: bad `kind`, `vol<=0`, `t<=0`, `spot<=0`, `strike<=0`,
  `confidence` outside `(0,1)`, non-positive/oversized `sims`, non-positive/oversized
  `horizon_days`, and degenerate/non-positive/too-few price series all return
  `ok:false`; no `NaN`/`Inf` is ever returned as `ok:true`.
- implied-vol no-arbitrage guard: an observed option price that is non-positive or
  violates its no-arbitrage bounds (e.g. a call `> spot` or `< intrinsic`) returns
  `ok:false` — the bisection never hands back a search-bound as if it were a real vol,
  and a returned vol is reported only after the price is verified reproduced within
  tol.

### Validation baseline: closed-form vs Monte-Carlo, and naive-VaR vs CVaR

The Monte-Carlo VaR/CVaR engine is graded against an independent closed-form Gaussian
VaR/CVaR oracle (Acklam inverse-normal-CDF, no scipy): external math, not itself.

- MC converges to the closed form. VaR error `2.345%` at the 10k-sim default →
  `0.010%` at 200k sims; CVaR error `0.214%` at 200k [measured]. The estimator is
  right, not just deterministic.
- Naive VaR-only under-reports the tail. VaR was the industry's single-quantile
  standard for decades, but here `CVaR/VaR = 1.254`: reporting VaR as the tail figure
  under-reserves the expected shortfall by 20.3% [measured]. The tool ships CVaR (the
  coherent version of expected shortfall), which the naive VaR-only report misses.
- The naive self-check is insufficient. An "ES := VaR" shortcut passes the common
  `deterministic + CVaR≥VaR` self-consistency check, yet the closed-form oracle rejects
  it (`20.3%` off). Validation has to be against external math, which is what this eval
  does.

## API

```python
from quantdesk import quant, dispatch

quant.black_scholes(100, 100, 1.0, 0.05, 0.20, "call")   # -> 10.4506...
dispatch({"tool": "black_scholes",
          "params": {"spot":100,"strike":100,"t":1,"r":0.05,"vol":0.2}})
# -> {"ok":True,"tool":"black_scholes","inputs":{...},
#     "method":"closed-form Black-Scholes (normal CDF via math.erf)",
#     "deterministic":True,"result":10.450584}
```

Tools: `black_scholes`, `greeks`, `implied_vol`, `portfolio_stats`, `monte_carlo_var`.

## Honest scope

Deterministic math, not investment advice. Assumptions: European options, flat
constant volatility, no dividends; Gaussian Monte-Carlo VaR (no fat tails, no
historical bootstrap); 252-trading-day annualization; risk-free rate 0 for Sharpe (a
risk-adjusted-return measure). This is a word-for-word implementation of standard
textbook formulas. The value isn't new math: it's that the math is deterministic,
auditable, and never left to the LLM to compute.

Numeric inputs reachable from an untrusted LLM caller are guarded, not blanket-capped.
Here is exactly what's checked: `black_scholes`/`greeks`/`implied_vol` require finite
`r` and a positive `spot`/`strike`/`t`/`vol`; `monte_carlo_var` requires finite
`mean`/`value`, a positive `vol`, `confidence` in `(0, 1)`, `sims <= 1,000,000` and
`horizon_days <= 10,000` individually and their product `sims * horizon_days <=
10,000,000` (the two caps multiply, since the simulation allocates a `sims x
horizon_days` array); `portfolio_stats` rejects `prices` longer than 1,000,000 points
or containing a non-finite/non-positive value. All guards raise a clean `ValueError`
rather than a raw internal traceback or an open-ended resource cost.

Negative VaR/CVaR is intentional, not a bug: when the expected return (`mean`)
dominates volatility at the requested `confidence`, the simulated loss distribution's
tail is itself a gain, and `monte_carlo_var` reports that as a negative number rather
than rejecting it or clamping it to zero. A downstream integrator should treat a
negative figure as "projected gain," not an error.

## Where it fits

Designed to compose with other audited compute modules in a larger agent system:
retrieval components cite a source, this kind of module returns an audited number, and
neither lets an LLM guess at the underlying fact.

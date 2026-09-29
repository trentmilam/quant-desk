# quant-desk (archived)

This repo is archived. The code lives in [wealth-guardrails/quant-desk](https://github.com/trentmilam/wealth-guardrails/tree/main/quant-desk), and that is where new work happens.

quant-desk does finance math in exact, testable code so an AI agent never has to guess a number.
It returns the answer with its inputs and method attached, or a clear error.

It moved because it is the arithmetic layer that fee-forensics, tail-risk and mandate-monitor
reason about, and it reads better next to them. The commit history moved with it, and that repo's
CI runs it on every push.

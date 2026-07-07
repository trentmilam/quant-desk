# Contributing

## Development setup

```
pip install -e '.[dev]'
```

This installs the runtime dependency (`numpy`) plus the `dev` extra (`pytest`).

## Running the checks

```
python eval.py   # correctness/determinism suite, exits 0 on all checks
pytest           # same suite via pytest, for standard CI test reporting
```

Both must pass before a change is merged. If you add or change a domain guard (a
`ValueError` raised on out-of-range input), add a corresponding regression case to
`eval.py` so the specific failure mode is locked in.

## Versioning and releases

`pyproject.toml`'s `version` is the single source of truth. A release tag must match it
exactly (e.g. `pyproject.toml` at `0.1.0` tags as `v0.1.0`); bump the version and create
the tag in the same commit so `pip show quant-desk` never contradicts the tag a user
just installed.

## Scope

This is a deterministic math library, not investment advice — see "Honest scope" in the
README for the assumptions each function makes. Keep additions in that spirit: closed-form
or seeded-and-reproducible, with an explicit `ValueError` for any input outside the
documented domain rather than a guessed or silently-wrong number.

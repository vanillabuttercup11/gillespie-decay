# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

A virtualenv lives in `.venv/` (gitignored). Use `.venv/bin/python` or activate it with `source .venv/bin/activate`.

```bash
.venv/bin/python -m pytest                                   # all tests
.venv/bin/python -m pytest tests/test_core.py::test_half_life  # single test
.venv/bin/python make_figures.py                             # regenerate figures/*.png and print the k' fit
.venv/bin/jupyter nbconvert --to notebook --execute --inplace gillespie_decay.ipynb   # re-run the notebook
```

Run everything from the repo root so `import gillespie_decay` resolves (there is no `setup.py`/`pyproject.toml`; the package is imported from the current directory). There is no linter configured.

## Architecture

A single small package, `gillespie_decay/`, with all logic in `core.py` (re-exported via `__init__.py`). It simulates the pseudo-first-order decay A + B → C + B (B in large excess, so k' = k[B]) with Gillespie's direct method. Since there is one reaction channel, no reaction-selection step is needed; each event is just N → N − 1.

`core.py` has two halves that must stay consistent:
- **Simulation:** `simulate_decay` produces one trajectory as a step function (`times`, `counts`, each of length n0 + 1). `counts_on_grid` resamples it onto a regular time grid, and `simulate_ensemble` stacks many runs into an `(n_runs, len(grid))` array.
- **Analytical reference:** `survival_probability`, `analytical_mean` and `analytical_std` give the exact solution. N(t) ~ Binomial(n0, e^(−k't)), so the mean is n0·p and the variance is n0·p(1−p).

Three consumers sit on top of the package and share the same parameters (k = 0.1, [B] = 0.5, k' = 0.05 s⁻¹, N₀ = 100): `tests/test_core.py`, `make_figures.py` (writes to `figures/`, which the README embeds), and `gillespie_decay.ipynb`. If you change a default parameter, update all of them and the README's quoted numbers (k' fit 0.0503, 12 tests).

## Testing conventions

Tests are statistical, not exact-value. They use fixed seeds and a 4-standard-error tolerance against the analytical binomial reference. Keep seeds fixed when adding tests, and derive tolerances from the standard error rather than hard-coding them. Changing how random numbers are drawn in `simulate_decay` changes every seeded result, so re-check the tests and regenerate the figures.

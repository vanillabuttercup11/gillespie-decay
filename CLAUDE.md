# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

A virtualenv lives in `.venv/` (gitignored). Use `.venv/bin/python` or activate it with `source .venv/bin/activate`.

```bash
.venv/bin/python -m pytest                                   # all tests (~10 s)
.venv/bin/python -m pytest tests/test_core.py::test_half_life  # single test
.venv/bin/python make_figures.py                             # regenerate figures/*.png, print the k' estimate; exits 1 if a validation check fails
.venv/bin/jupyter nbconvert --to notebook --execute --inplace gillespie_decay.ipynb   # re-run the notebook
```

Run everything from the repo root so `import gillespie_decay` resolves (there is no `setup.py`/`pyproject.toml`; the package is imported from the current directory). There is no linter configured. The notebook is generated: edit its source cells, then re-execute it, rather than relying on stale outputs.

## Architecture

A single small package, `gillespie_decay/`, with all logic in `core.py` (re-exported via `__init__.py`). It simulates the pseudo-first-order decay A + B → C (B consumed, in large excess, so k' = k[B]₀) with Gillespie's direct method.

`core.py` has three layers that must stay consistent:
- **General engine:** `gillespie_direct(x0, stoich, propensity_fn, ...)` draws two uniforms per event, `u1` for the waiting time and `u2` for event selection (reaction j with probability a_j/a_0, via `searchsorted` on the cumulative propensities). Every simulation goes through it.
- **Decay wrappers:** `simulate_decay` (one channel, rate k'N) and `simulate_explicit_b` (B simulated explicitly, c = k'/N_B0) return `(times, counts)`. `counts_on_grid` resamples a trajectory onto a time grid and raises on grid times before 0. `simulate_runs` and `simulate_ensemble` build many runs.
- **Analytical reference and estimators:** `analytical_mean`/`analytical_std` (N(t) ~ Binomial(n0, e^(−k't))), `standardized_residuals` (z-score of the ensemble mean against the exact mean), and `estimate_rate` (pooled-lifetime MLE with a standard error and CI; it is only valid for runs taken to completion).

Three consumers sit on top of the package and share the same parameters (k = 0.1, [B] = 0.5, k' = 0.05 s⁻¹, N₀ = 100): `tests/test_core.py`, `make_figures.py` (writes `figures/`, which the README embeds), and `gillespie_decay.ipynb`. If you change a default parameter, update all of them and the numbers quoted in the README (the k' estimate and CI, max|z|, test count).

## Testing conventions

Tests are statistical, not exact-value. They use fixed seeds and a tolerance of `Z_TOLERANCE` (4) standard errors against the analytical reference, and several are "can this fail?" tests (wrong rate, wrong selection, wrong CI). Keep seeds fixed when adding tests, derive tolerances from standard errors, and when adding a check also add a case showing it fails on wrong input. Changing how random numbers are drawn in `gillespie_direct` changes every seeded result, so re-check the tests and regenerate the figures and notebook.

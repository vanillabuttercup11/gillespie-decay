"""Gillespie kinetic Monte Carlo (kMC) for a pseudo-first-order decay.

The chemistry
-------------
Consider the bimolecular reaction

    A + B  --k-->  C            (B is consumed)

with B in large excess, so [B] barely changes while A decays. The rate law

    d[A]/dt = -k [A] [B]  ~  -k' [A],      k' = k [B]_0

is then *pseudo-first-order* in A, and deterministically [A](t) = [A]_0 exp(-k' t).
The approximation only holds while B is not noticeably depleted, i.e. N_B0 >> N_A0;
`simulate_explicit_b` simulates B explicitly so that claim can be tested.

Gillespie's direct method
-------------------------
`gillespie_direct` is the general algorithm. For a system with several reaction
channels j, each with propensity a_j(x) (event rate given the current state x):

    a_0 = sum_j a_j                       total event rate
    tau = -ln(u1) / a_0                   waiting time until the next event
    j   = first index with sum_{i<=j} a_i > u2 * a_0     WHICH reaction fires
    x  -> x + stoich[j]                   apply it

with u1, u2 independent uniform random numbers. Choosing j with probability
a_j / a_0 is the "event selection" step. For the decay below there is a single
channel, so selection is trivial there; it is exercised non-trivially by the
reversible A <-> C test in tests/test_core.py.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np

# Tolerance (in standard errors) used wherever a simulation is compared with theory.
Z_TOLERANCE = 4.0


class Trajectory(NamedTuple):
    times: np.ndarray       # shape (n_events + 1,); times[0] = 0
    states: np.ndarray      # shape (n_events + 1, n_species); state after each event
    reactions: np.ndarray   # shape (n_events,); index of the reaction fired at each event


def effective_rate(k: float, conc_b: float) -> float:
    """Pseudo-first-order rate constant k' = k * [B] (B in large excess)."""
    if k < 0 or conc_b < 0:
        raise ValueError("k and conc_b must be non-negative")
    return k * conc_b


# ---------------------------------------------------------------------------
# General engine
# ---------------------------------------------------------------------------

def gillespie_direct(
    x0,
    stoich,
    propensity_fn: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator | None = None,
    t_max: float = np.inf,
    max_events: int = 10_000_000,
) -> Trajectory:
    """Run one exact Gillespie (direct method) trajectory.

    Parameters
    ----------
    x0 : initial molecule counts, one per species.
    stoich : integer array (n_reactions, n_species); row j is the change in
        counts when reaction j fires.
    propensity_fn : maps the current counts x to an array of n_reactions
        non-negative propensities.
    t_max : stop at this time (the state at t_max is the last recorded state).
        With the default (inf) the run ends when every propensity is zero.
    max_events : safety cap; raises RuntimeError if reached.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.array(x0, dtype=np.int64)
    stoich = np.asarray(stoich, dtype=np.int64)
    if stoich.ndim != 2 or stoich.shape[1] != x.size:
        raise ValueError("stoich must have shape (n_reactions, n_species)")
    n_reactions = stoich.shape[0]

    t = 0.0
    times = [t]
    states = [x.copy()]
    fired: list[int] = []

    while len(fired) < max_events:
        a = np.asarray(propensity_fn(x), dtype=float)
        if a.shape != (n_reactions,) or np.any(a < 0):
            raise ValueError("propensity_fn must return n_reactions non-negative values")
        cum = np.cumsum(a)
        a0 = cum[-1]
        if a0 <= 0:                              # nothing can happen any more
            break
        u1 = 1.0 - rng.random()                  # in (0, 1], avoids log(0)
        u2 = rng.random()
        tau = -np.log(u1) / a0                   # exponential waiting time
        if t + tau > t_max:
            break
        t += tau
        # Event selection: reaction j with probability a_j / a_0.
        j = min(int(np.searchsorted(cum, u2 * a0, side="right")), n_reactions - 1)
        x = x + stoich[j]
        times.append(t)
        states.append(x)
        fired.append(j)
    else:
        raise RuntimeError(f"max_events ({max_events}) reached")

    return Trajectory(np.array(times), np.array(states), np.array(fired, dtype=np.int64))


# ---------------------------------------------------------------------------
# The pseudo-first-order decay
# ---------------------------------------------------------------------------

def simulate_decay(
    n0: int,
    k_eff: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ONE trajectory of A -> products (rate k' N) until all A is gone.

    Returns
    -------
    times : length n0 + 1. times[0] = 0, times[i] = time of the i-th event.
    counts : length n0 + 1. Number of A molecules after the i-th event.
    """
    if n0 <= 0 or int(n0) != n0:
        raise ValueError("n0 must be a positive integer")
    if k_eff <= 0:
        raise ValueError("k_eff must be positive")
    traj = gillespie_direct(
        [int(n0)], [[-1]], lambda x: np.array([k_eff * x[0]]), rng
    )
    return traj.times, traj.states[:, 0]


def simulate_explicit_b(
    n0: int,
    nb0: int,
    k_eff: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Trajectory of A + B -> C with B simulated explicitly (B is consumed).

    The per-pair stochastic rate constant is c = k_eff / nb0, so that
    c * N_B = k_eff at the start and the pseudo-first-order approximation
    becomes exact as nb0 / n0 -> infinity. Returns (times, counts of A).
    """
    if nb0 < n0:
        raise ValueError("nb0 must be >= n0 (otherwise B runs out before A)")
    c = k_eff / nb0
    traj = gillespie_direct(
        [int(n0), int(nb0)], [[-1, -1]], lambda x: np.array([c * x[0] * x[1]]), rng
    )
    return traj.times, traj.states[:, 0]


def counts_on_grid(
    times: np.ndarray, counts: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Evaluate a trajectory (a step function) on a time grid.

    The state at time t is the count after the last event with time <= t.
    Grid times before the start of the trajectory are an error: they would
    otherwise wrap around to the final count.
    """
    grid = np.asarray(grid, dtype=float)
    if np.any(grid < times[0]):
        raise ValueError("grid contains times before the start of the trajectory")
    idx = np.searchsorted(times, grid, side="right") - 1
    return counts[idx]


def simulate_runs(
    n0: int, k_eff: float, n_runs: int, seed: int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Independent decay trajectories sharing one random generator."""
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    rng = np.random.default_rng(seed)
    return [simulate_decay(n0, k_eff, rng) for _ in range(n_runs)]


def simulate_ensemble(
    n0: int,
    k_eff: float,
    n_runs: int,
    grid: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """Many trajectories sampled on one grid; shape (n_runs, len(grid))."""
    runs = simulate_runs(n0, k_eff, n_runs, seed)
    return np.array([counts_on_grid(t, c, grid) for t, c in runs])


# ---------------------------------------------------------------------------
# Analytical results to validate against
# ---------------------------------------------------------------------------
# Each molecule survives to time t with probability p(t) = exp(-k' t),
# independently, so N(t) ~ Binomial(n0, p(t)).

def survival_probability(t: np.ndarray, k_eff: float) -> np.ndarray:
    return np.exp(-k_eff * np.asarray(t, dtype=float))


def analytical_mean(t: np.ndarray, n0: int, k_eff: float) -> np.ndarray:
    """E[N(t)] = n0 exp(-k' t)  (the deterministic rate-law solution)."""
    return n0 * survival_probability(t, k_eff)


def analytical_std(t: np.ndarray, n0: int, k_eff: float) -> np.ndarray:
    """Std of N(t) for Binomial(n0, p): sqrt(n0 p (1 - p))."""
    p = survival_probability(t, k_eff)
    return np.sqrt(n0 * p * (1.0 - p))


def standardized_residuals(
    ensemble: np.ndarray, grid: np.ndarray, n0: int, k_eff: float
) -> np.ndarray:
    """z-score of the ensemble mean against the exact mean at each grid time.

    z = (sample mean - n0 e^{-k't}) / (binomial std / sqrt(n_runs)).
    Where the exact std is zero (t = 0) the residual is 0 by construction.
    A correct simulation gives |z| well below Z_TOLERANCE; a wrong rate or
    wrong algorithm does not (see the tests).
    """
    se = analytical_std(grid, n0, k_eff) / np.sqrt(ensemble.shape[0])
    diff = ensemble.mean(axis=0) - analytical_mean(grid, n0, k_eff)
    z = np.zeros_like(diff, dtype=float)
    ok = se > 0
    z[ok] = diff[ok] / se[ok]
    return z


class RateEstimate(NamedTuple):
    k_hat: float
    se: float
    ci_low: float
    ci_high: float
    n: int


def estimate_rate(lifetimes: np.ndarray, z: float = 1.96) -> RateEstimate:
    """Estimate k' (with uncertainty) from the event times of completed runs.

    In a decay run to completion, the n0 event times are the order statistics
    of n0 independent Exp(k') molecular lifetimes, so the event times from all
    runs can be pooled as lifetimes. For n exponential lifetimes the unbiased
    maximum-likelihood estimate is k_hat = (n - 1) / sum(lifetimes), with
    standard error k_hat / sqrt(n - 2). Returns a normal-approximation
    confidence interval (z = 1.96 -> 95 %).
    """
    x = np.asarray(lifetimes, dtype=float).ravel()
    if x.size < 3 or np.any(x <= 0):
        raise ValueError("need at least 3 positive lifetimes")
    n = x.size
    k_hat = (n - 1) / x.sum()
    se = k_hat / np.sqrt(n - 2)
    return RateEstimate(k_hat, se, k_hat - z * se, k_hat + z * se, n)

"""Gillespie kinetic Monte Carlo (kMC) for a pseudo-first-order decay.

The chemistry
-------------
Consider the bimolecular reaction

    A + B  --k-->  C + B        (B acts like a catalyst here: it is not consumed)

with B in large excess, so [B] stays effectively constant. The rate law

    d[A]/dt = -k [A] [B]  ~  -k' [A],      k' = k [B]

is then *pseudo-first-order* in A. Deterministically, [A](t) = [A]_0 exp(-k' t).

The stochastic picture
----------------------
With a small number N of A molecules, decay is a random process. Each A
molecule independently reacts with probability k' dt in a short time dt, so
the total "propensity" (event rate) is

    a(N) = k' * N

Gillespie's direct method says: the waiting time until the next event is
exponentially distributed with rate a(N),

    tau = -ln(u) / a(N),     u ~ Uniform(0, 1)

and (since there is only one reaction channel here) the event is always
"one A molecule decays", so N -> N - 1. Repeat until N = 0.
"""

from __future__ import annotations

import numpy as np


def effective_rate(k: float, conc_b: float) -> float:
    """Pseudo-first-order rate constant k' = k * [B] (B in large excess)."""
    if k < 0 or conc_b < 0:
        raise ValueError("k and conc_b must be non-negative")
    return k * conc_b


def simulate_decay(
    n0: int,
    k_eff: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ONE Gillespie trajectory of A -> products until all A is gone.

    Parameters
    ----------
    n0 : initial number of A molecules (positive integer).
    k_eff : pseudo-first-order rate constant k' (units of 1/time, > 0).
    rng : NumPy random Generator (pass one with a seed for reproducibility).

    Returns
    -------
    times : array of length n0 + 1. times[0] = 0, times[i] = time of i-th event.
    counts : array of length n0 + 1. counts[i] = number of A molecules
        immediately after the i-th event (counts[0] = n0, counts[-1] = 0).
    """
    if n0 <= 0 or int(n0) != n0:
        raise ValueError("n0 must be a positive integer")
    if k_eff <= 0:
        raise ValueError("k_eff must be positive")
    if rng is None:
        rng = np.random.default_rng()

    n0 = int(n0)
    times = np.empty(n0 + 1)
    counts = np.empty(n0 + 1, dtype=np.int64)
    t = 0.0
    n = n0
    times[0] = t
    counts[0] = n

    for step in range(1, n0 + 1):
        propensity = k_eff * n                 # total event rate a(N) = k' N
        u = 1.0 - rng.random()                 # in (0, 1], avoids log(0)
        t += -np.log(u) / propensity           # exponential waiting time
        n -= 1                                 # the only possible event
        times[step] = t
        counts[step] = n

    return times, counts


def counts_on_grid(
    times: np.ndarray, counts: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Evaluate a trajectory (a step function) on a regular time grid.

    The state at time t is the count after the last event with time <= t.
    """
    idx = np.searchsorted(times, grid, side="right") - 1
    return counts[idx]


def simulate_ensemble(
    n0: int,
    k_eff: float,
    n_runs: int,
    grid: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """Run many independent trajectories, all sampled on the same time grid.

    Returns an array of shape (n_runs, len(grid)).
    """
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    rng = np.random.default_rng(seed)
    out = np.empty((n_runs, len(grid)), dtype=np.int64)
    for i in range(n_runs):
        times, counts = simulate_decay(n0, k_eff, rng)
        out[i] = counts_on_grid(times, counts, grid)
    return out


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

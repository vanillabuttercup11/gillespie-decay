import numpy as np
import pytest

from gillespie_decay import (
    Z_TOLERANCE,
    analytical_mean,
    analytical_std,
    counts_on_grid,
    effective_rate,
    estimate_rate,
    gillespie_direct,
    simulate_decay,
    simulate_ensemble,
    simulate_explicit_b,
    simulate_runs,
    standardized_residuals,
)

K_EFF = 0.05
N0 = 200


# ---------------------------------------------------------------- basic structure
def test_effective_rate():
    assert effective_rate(0.1, 0.5) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        effective_rate(-1.0, 1.0)


def test_trajectory_structure():
    times, counts = simulate_decay(N0, K_EFF, np.random.default_rng(1))
    assert len(times) == len(counts) == N0 + 1
    assert times[0] == 0.0 and counts[0] == N0 and counts[-1] == 0
    assert np.all(np.diff(times) > 0)          # time strictly increases
    assert np.all(np.diff(counts) == -1)       # one molecule lost per event


def test_reproducible_with_seed():
    a = simulate_decay(50, K_EFF, np.random.default_rng(42))
    b = simulate_decay(50, K_EFF, np.random.default_rng(42))
    assert np.array_equal(a[0], b[0])


@pytest.mark.parametrize("bad", [0, -5, 2.5])
def test_invalid_n0(bad):
    with pytest.raises(ValueError):
        simulate_decay(bad, K_EFF)


def test_invalid_rate():
    with pytest.raises(ValueError):
        simulate_decay(10, 0.0)


def test_engine_rejects_bad_inputs():
    with pytest.raises(ValueError):                       # stoich/species mismatch
        gillespie_direct([1, 0], [[-1]], lambda x: np.array([1.0]))
    with pytest.raises(ValueError):                       # negative propensity
        gillespie_direct([1], [[-1]], lambda x: np.array([-1.0]))


# ---------------------------------------------------------------- grid sampling
def test_counts_on_grid_is_step_function():
    times = np.array([0.0, 1.0, 2.0])
    counts = np.array([2, 1, 0])
    grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 5.0])
    assert list(counts_on_grid(times, counts, grid)) == [2, 2, 1, 1, 0, 0]


def test_counts_on_grid_rejects_negative_times():
    """A negative time used to wrap (index -1) to the final count, 0."""
    times = np.array([0.0, 1.0, 2.0])
    counts = np.array([2, 1, 0])
    with pytest.raises(ValueError):
        counts_on_grid(times, counts, np.array([-0.5, 0.0]))


# ---------------------------------------------------------------- event selection
def test_event_selection_probabilities():
    """One A can become B (propensity 3) or C (propensity 1): P(B) must be 3/4.

    Also checks the waiting time ~ Exp(a_0 = 4), whose mean is 0.25.
    """
    rng = np.random.default_rng(11)
    stoich = [[-1, 1, 0], [-1, 0, 1]]
    prop = lambda x: np.array([3.0 * x[0], 1.0 * x[0]])
    n_runs = 6000
    first_reaction, first_time = [], []
    for _ in range(n_runs):
        tr = gillespie_direct([1, 0, 0], stoich, prop, rng)
        first_reaction.append(tr.reactions[0])
        first_time.append(tr.times[1])
    frac_b = np.mean(np.array(first_reaction) == 0)
    se_frac = np.sqrt(0.75 * 0.25 / n_runs)
    assert abs(frac_b - 0.75) < Z_TOLERANCE * se_frac
    assert abs(np.mean(first_time) - 0.25) < Z_TOLERANCE * 0.25 / np.sqrt(n_runs)


def test_reversible_reaction_reaches_correct_equilibrium():
    """A <-> C with kf = 0.3, kb = 0.1 settles at E[A] = N kb/(kf+kb) = N/4.

    Broken selection (always channel 0, or uniform choice) gives 0 or N/2.
    """
    kf, kb, n = 0.3, 0.1, 20
    stoich = [[-1, 1], [1, -1]]
    prop = lambda x: np.array([kf * x[0], kb * x[1]])
    rng = np.random.default_rng(3)
    n_runs = 1000
    final_a = np.array([
        gillespie_direct([n, 0], stoich, prop, rng, t_max=20.0).states[-1, 0]
        for _ in range(n_runs)
    ])
    p = kb / (kf + kb)
    se = np.sqrt(n * p * (1 - p) / n_runs)
    assert abs(final_a.mean() - n * p) < Z_TOLERANCE * se


def test_t_max_truncates_run():
    tr = gillespie_direct([100], [[-1]], lambda x: np.array([0.05 * x[0]]),
                          np.random.default_rng(0), t_max=5.0)
    assert tr.times[-1] <= 5.0
    assert tr.states[-1, 0] > 0                  # not yet fully decayed at t = 5


# ---------------------------------------------------------------- validation vs theory
def test_first_waiting_time_is_exponential():
    """The first event happens after tau ~ Exp(k' N0): mean 1/(k' N0)."""
    rng = np.random.default_rng(7)
    n_runs = 4000
    taus = np.array([simulate_decay(N0, K_EFF, rng)[0][1] for _ in range(n_runs)])
    expected_mean = 1.0 / (K_EFF * N0)
    std_error = expected_mean / np.sqrt(n_runs)   # std of Exp equals its mean
    assert abs(taus.mean() - expected_mean) < Z_TOLERANCE * std_error


GRID = np.linspace(0, 60, 31)


def test_ensemble_mean_passes_residual_check():
    ens = simulate_ensemble(N0, K_EFF, 1000, GRID, seed=123)
    z = standardized_residuals(ens, GRID, N0, K_EFF)
    assert np.max(np.abs(z)) < Z_TOLERANCE


def test_residual_check_can_fail():
    """The same check must reject data simulated with a rate 10 % too high."""
    ens = simulate_ensemble(N0, 1.10 * K_EFF, 1000, GRID, seed=123)
    z = standardized_residuals(ens, GRID, N0, K_EFF)
    assert np.max(np.abs(z)) > Z_TOLERANCE


def test_ensemble_variance_is_binomial():
    """N(t) is Binomial(n0, e^{-k't}), so Var = n0 p (1-p)."""
    grid = np.array([10.0, 20.0, 40.0])
    ens = simulate_ensemble(N0, K_EFF, 3000, grid, seed=5)
    expected_var = analytical_std(grid, N0, K_EFF) ** 2
    assert np.allclose(ens.var(axis=0, ddof=1), expected_var, rtol=0.15)


def test_half_life():
    """Mean count should be half of n0 at t = ln 2 / k'."""
    t_half = np.log(2) / K_EFF
    ens = simulate_ensemble(N0, K_EFF, 1000, np.array([t_half]), seed=9)
    std_error = analytical_std(np.array([t_half]), N0, K_EFF)[0] / np.sqrt(1000)
    assert abs(ens.mean() - N0 / 2) < Z_TOLERANCE * std_error


# ---------------------------------------------------------------- rate estimator
def _lifetimes(k_eff, n0, n_runs, seed):
    return np.concatenate([t[1:] for t, _ in simulate_runs(n0, k_eff, n_runs, seed)])


def test_estimate_rate_ci_coverage():
    """A 95 % CI should contain the true k' about 95 % of the time."""
    reps = 200
    hits = 0
    for seed in range(reps):
        est = estimate_rate(_lifetimes(K_EFF, 100, 5, seed))   # 500 lifetimes
        hits += est.ci_low <= K_EFF <= est.ci_high
    assert 0.90 <= hits / reps <= 0.99


def test_estimate_rate_detects_wrong_rate():
    """With 20 000 lifetimes (SE ~ 0.7 %), a 5 % error must fall outside the CI."""
    est = estimate_rate(_lifetimes(1.05 * K_EFF, 100, 200, seed=1))
    assert not (est.ci_low <= K_EFF <= est.ci_high)


def test_estimate_rate_rejects_bad_input():
    with pytest.raises(ValueError):
        estimate_rate(np.array([1.0, 2.0]))


# ---------------------------------------------------------------- pseudo-first-order limit
def _explicit_b_ensemble(nb0, n_runs, seed):
    rng = np.random.default_rng(seed)
    return np.array([
        counts_on_grid(*simulate_explicit_b(N0 // 2, nb0, K_EFF, rng), GRID)
        for _ in range(n_runs)
    ])


def test_large_b_excess_recovers_pseudo_first_order():
    n0 = N0 // 2
    ens = _explicit_b_ensemble(nb0=100 * n0, n_runs=1000, seed=21)
    z = standardized_residuals(ens, GRID, n0, K_EFF)
    assert np.max(np.abs(z)) < Z_TOLERANCE


def test_small_b_excess_breaks_pseudo_first_order():
    """With B = A initially, B is used up and decay is clearly NOT exp(-k't)."""
    n0 = N0 // 2
    ens = _explicit_b_ensemble(nb0=n0, n_runs=1000, seed=22)
    z = standardized_residuals(ens, GRID, n0, K_EFF)
    assert np.max(np.abs(z)) > Z_TOLERANCE

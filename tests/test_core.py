import numpy as np
import pytest

from gillespie_decay import (
    analytical_mean,
    analytical_std,
    counts_on_grid,
    effective_rate,
    simulate_decay,
    simulate_ensemble,
)

K_EFF = 0.05
N0 = 200


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


def test_counts_on_grid_is_step_function():
    times = np.array([0.0, 1.0, 2.0])
    counts = np.array([2, 1, 0])
    grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 5.0])
    assert list(counts_on_grid(times, counts, grid)) == [2, 2, 1, 1, 0, 0]


def test_first_waiting_time_is_exponential():
    """The first event happens after tau ~ Exp(k' N0): mean 1/(k' N0)."""
    rng = np.random.default_rng(7)
    n_runs = 4000
    taus = np.array([simulate_decay(N0, K_EFF, rng)[0][1] for _ in range(n_runs)])
    expected_mean = 1.0 / (K_EFF * N0)
    std_error = expected_mean / np.sqrt(n_runs)   # std of Exp equals its mean
    assert abs(taus.mean() - expected_mean) < 4 * std_error


def test_ensemble_mean_matches_exponential_decay():
    """Averaged over runs, N(t) must follow n0 * exp(-k' t) (within 4 sigma)."""
    grid = np.linspace(0, 60, 31)
    n_runs = 1000
    ens = simulate_ensemble(N0, K_EFF, n_runs, grid, seed=123)
    mean = ens.mean(axis=0)
    expected = analytical_mean(grid, N0, K_EFF)
    std_error = analytical_std(grid, N0, K_EFF) / np.sqrt(n_runs)
    # At t = 0 the std is zero, so allow a tiny floor to avoid 0 < 0 failing.
    assert np.all(np.abs(mean - expected) <= 4 * std_error + 1e-9)


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
    assert abs(ens.mean() - N0 / 2) < 4 * std_error

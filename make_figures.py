"""Generate the figures used in the README and run the validation checks.

Run: python make_figures.py   (exits with status 1 if a check fails)
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np

from gillespie_decay import (
    Z_TOLERANCE,
    analytical_mean,
    analytical_std,
    counts_on_grid,
    effective_rate,
    estimate_rate,
    simulate_decay,
    simulate_runs,
    standardized_residuals,
)

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# ---- Parameters -----------------------------------------------------------
K = 0.1           # bimolecular rate constant (1 / (M s))
CONC_B = 0.5      # [B], effectively constant because B is in large excess (M)
K_EFF = effective_rate(K, CONC_B)   # k' = 0.05 / s
N0 = 100          # initial number of A molecules
N_RUNS = 500
SEED = 2024
T_MAX = 5 / K_EFF                    # five lifetimes
grid = np.linspace(0, T_MAX, 300)
t_fine = np.linspace(0, T_MAX, 500)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})

# ---- Figure 1: individual trajectories vs the deterministic curve ---------
rng = np.random.default_rng(SEED)
fig, ax = plt.subplots(figsize=(7, 4.5))
for i in range(5):
    times, counts = simulate_decay(N0, K_EFF, rng)
    ax.step(times, counts, where="post", lw=1, alpha=0.8,
            label="Gillespie trajectories" if i == 0 else None)
ax.plot(t_fine, analytical_mean(t_fine, N0, K_EFF), "k--", lw=2,
        label=r"deterministic $N_0 e^{-k't}$")
ax.set_xlim(0, T_MAX)
ax.set_xlabel("time (s)")
ax.set_ylabel("number of A molecules")
ax.set_title(f"Individual kMC trajectories ($N_0$={N0}, $k'$={K_EFF:g} s$^{{-1}}$)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig1_trajectories.png", dpi=150)
plt.close(fig)

# ---- Ensemble used by figures 2 and 3 -------------------------------------
runs = simulate_runs(N0, K_EFF, N_RUNS, seed=SEED)
ens = np.array([counts_on_grid(t, c, grid) for t, c in runs])
mean = ens.mean(axis=0)
std = ens.std(axis=0, ddof=1)
z = standardized_residuals(ens, grid, N0, K_EFF)
max_abs_z = np.max(np.abs(z))

# ---- Figure 2: mean +/- std vs exact, with a quantitative pass/fail panel --
fig, (ax, axz) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.3]})
ax.fill_between(grid, mean - std, mean + std, alpha=0.3, color="C0",
                label=r"kMC mean $\pm$ 1 std")
ax.plot(grid, mean, color="C0", lw=2, label=f"kMC mean ({N_RUNS} runs)")
exact = analytical_mean(grid, N0, K_EFF)
sd = analytical_std(grid, N0, K_EFF)
ax.plot(grid, exact, "k--", lw=1.5, label=r"exact mean $N_0 e^{-k't}$")
ax.plot(grid, exact + sd, "k:", lw=1.2, label=r"exact $\pm$ 1 std (binomial)")
ax.plot(grid, np.clip(exact - sd, 0, None), "k:", lw=1.2)
ax.set_ylabel("number of A molecules")
ax.set_title("Ensemble average vs the exact rate law")
ax.legend()
axz.axhspan(-Z_TOLERANCE, Z_TOLERANCE, color="green", alpha=0.12,
            label=f"pass band ±{Z_TOLERANCE:g} s.e.")
axz.plot(grid, z, color="C0", lw=1)
axz.set_ylim(-2 * Z_TOLERANCE, 2 * Z_TOLERANCE)
axz.set_xlabel("time (s)")
axz.set_ylabel("z (mean − exact)\n/ std. error")
axz.set_title(f"max |z| = {max_abs_z:.2f}  "
              f"({'PASS' if max_abs_z < Z_TOLERANCE else 'FAIL'})", fontsize=10)
axz.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig2_ensemble.png", dpi=150)
plt.close(fig)

# ---- Headline number: k' estimated with an uncertainty --------------------
lifetimes = np.concatenate([t[1:] for t, _ in runs])
est = estimate_rate(lifetimes)
z_rate = (est.k_hat - K_EFF) / est.se

# ---- Figure 3: ln(N/N0) is linear with slope -k' (illustration only) ------
mask = mean >= 3   # log of tiny counts is too noisy to plot
slope, intercept = np.polyfit(grid[mask], np.log(mean[mask] / N0), 1)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(grid[mask], np.log(mean[mask] / N0), "o", ms=3, label="kMC ensemble mean")
ax.plot(grid[mask], slope * grid[mask] + intercept, "r-",
        label=f"line fit: slope = {slope:.4f} s$^{{-1}}$")
ax.plot(grid[mask], -K_EFF * grid[mask], "k--", label=f"true $-k'$ = {-K_EFF:g} s$^{{-1}}$")
ax.set_xlabel("time (s)")
ax.set_ylabel(r"$\ln\,(\langle N\rangle / N_0)$")
ax.set_title("Pseudo-first-order check: straight line on a log plot")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig3_log_linear.png", dpi=150)
plt.close(fig)

# ---- Report and pass/fail -------------------------------------------------
print(f"Figures saved to {OUT}")
print(f"Fig 2 check: max |z| over {len(grid)} grid points = {max_abs_z:.2f} "
      f"(tolerance {Z_TOLERANCE:g})")
print(f"k' estimate: {est.k_hat:.5f} ± {est.se:.5f} s^-1 (s.e., n = {est.n} lifetimes); "
      f"95% CI [{est.ci_low:.5f}, {est.ci_high:.5f}]; true {K_EFF:g}; "
      f"z = {z_rate:+.2f}")

failed = []
if max_abs_z >= Z_TOLERANCE:
    failed.append("ensemble mean disagrees with the exact rate law")
if abs(z_rate) >= Z_TOLERANCE:
    failed.append("estimated k' disagrees with the true k'")
if failed:
    print("CHECK FAILED: " + "; ".join(failed))
    sys.exit(1)
print("All checks passed.")

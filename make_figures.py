"""Generate the figures used in the README. Run: python make_figures.py"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np

from gillespie_decay import (
    analytical_mean,
    analytical_std,
    effective_rate,
    simulate_decay,
    simulate_ensemble,
)

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# ---- Parameters -----------------------------------------------------------
K = 0.1           # bimolecular rate constant (1 / (M s))
CONC_B = 0.5      # [B], held constant because B is in large excess (M)
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
ens = simulate_ensemble(N0, K_EFF, N_RUNS, grid, seed=SEED)
mean = ens.mean(axis=0)
std = ens.std(axis=0, ddof=1)

# ---- Figure 2: ensemble mean +/- std vs exact binomial prediction ---------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.fill_between(grid, mean - std, mean + std, alpha=0.3, color="C0",
                label=r"kMC mean $\pm$ 1 std")
ax.plot(grid, mean, color="C0", lw=2, label=f"kMC mean ({N_RUNS} runs)")
exact = analytical_mean(grid, N0, K_EFF)
sd = analytical_std(grid, N0, K_EFF)
ax.plot(grid, exact, "k--", lw=1.5, label=r"exact mean $N_0 e^{-k't}$")
ax.plot(grid, exact + sd, "k:", lw=1.2, label=r"exact $\pm$ 1 std (binomial)")
ax.plot(grid, np.clip(exact - sd, 0, None), "k:", lw=1.2)
ax.set_xlabel("time (s)")
ax.set_ylabel("number of A molecules")
ax.set_title("Ensemble average reproduces the rate law")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig2_ensemble.png", dpi=150)
plt.close(fig)

# ---- Figure 3: ln(N/N0) is linear with slope -k' --------------------------
# Only fit where the mean is still well above zero (log of tiny numbers is noisy).
mask = mean >= 3
slope, intercept = np.polyfit(grid[mask], np.log(mean[mask] / N0), 1)
k_fit = -slope
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(grid[mask], np.log(mean[mask] / N0), "o", ms=3, label="kMC ensemble mean")
ax.plot(grid[mask], slope * grid[mask] + intercept, "r-",
        label=f"fit: $k'$ = {k_fit:.4f} s$^{{-1}}$")
ax.plot(grid[mask], -K_EFF * grid[mask], "k--", label=f"true $k'$ = {K_EFF:g} s$^{{-1}}$")
ax.set_xlabel("time (s)")
ax.set_ylabel(r"$\ln\,(\langle N\rangle / N_0)$")
ax.set_title("Pseudo-first-order check: straight line on a log plot")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig3_log_linear.png", dpi=150)
plt.close(fig)

print(f"k' true = {K_EFF:.4f} /s, k' fitted = {k_fit:.4f} /s "
      f"({100 * abs(k_fit - K_EFF) / K_EFF:.1f}% error)")
print(f"Figures saved to {OUT}")

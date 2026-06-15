import stereographic_algs
import euclidean_algs
import sub_Cauchy_algs

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geomstats.geometry.hypersphere import Hypersphere
import pandas as pd


def log_density(x, loc=None, sigma=1.0):
    """
    Multivariate Cauchy log-density up to an additive constant.
    """
    x = np.atleast_1d(x)

    if loc is None:
        loc = np.zeros_like(x)
    else:
        loc = np.asarray(loc, dtype=float)

    if loc.shape != x.shape:
        raise ValueError("loc must have the same shape as x.")

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    d = x.shape[0]

    y = (x - loc) / sigma
    norm_sq = np.dot(y, y)

    return -0.5 * (d + 1) * np.log1p(norm_sq)


def scs_centered_to_euclidean(samples, R, o):
    """
    Convert SCS samples from the centered sphere BS^d to R^d.

    The SCS chain lives on the centered sphere.
    To apply the sub-Cauchy projection, we first shift back by e_{d+1}.
    """
    samples = np.asarray(samples)
    d = samples.shape[1] - 1

    e = np.zeros(d + 1)
    e[-1] = 1.0

    x_samples = np.array([
        sub_Cauchy_algs.sub_cauchy_projection(z + e, R, o)
        for z in samples
    ])

    return x_samples


d = 20
R = np.sqrt(d)
S = Hypersphere(dim=d)

loc_cauchy = 1 * np.ones(d)

def target_log_density(x):
    return log_density(x, loc=loc_cauchy, sigma=1.0)

n_samples = int(1e4)
n_rep = 1000

proposal_std_eu = 2.5
proposal_std_scs = 1

tol = 1e-12

# --------------------------------------------------
# Observer for SCS
# --------------------------------------------------
# Need ||o - e_{d+1}|| <= 1 and o_{d+1} > 0.
o = np.zeros(d + 1)
o[-1] = 1.5

e = np.zeros(d + 1)
e[-1] = 1.0

meeting_times_eu = []
meeting_times_scs = []

eu_accept_rates = []
scs_accept_rates = []

scs_stepout_rates = []

for rep in range(n_rep):

    # -----------------------------
    # Initial points in R^d
    # -----------------------------
    x = 10 * np.random.randn(d)
    y = 10 * np.random.randn(d)

    # ==================================================
    # 1. Euclidean coupling
    # ==================================================
    eu_sample1, eu_sample2, eu_acc1, eu_acc2 = euclidean_algs.MRCoupling_sampler(
        n_samples,
        proposal_std_eu,
        x.copy(),
        y.copy(),
        target_log_density,
        d
    )

    dist_eu = np.linalg.norm(eu_sample1 - eu_sample2, axis=1)
    meet_indices_eu = np.where(dist_eu <= tol)[0]

    if len(meet_indices_eu) > 0:
        meeting_times_eu.append(meet_indices_eu[0])
    else:
        meeting_times_eu.append(np.inf)

    eu_accept_rates.append((eu_acc1, eu_acc2))

    # ==================================================
    # 2. Sub-Cauchy coupling
    # ==================================================
    # inverse_sub_cauchy_projection maps R^d to the shifted sphere
    # \widetilde{S}^d, so subtract e_{d+1} to get centered S^d.
    z1_scs_tilde = sub_Cauchy_algs.inverse_sub_cauchy_projection(x, R, o)
    z2_scs_tilde = sub_Cauchy_algs.inverse_sub_cauchy_projection(y, R, o)

    z1_scs = z1_scs_tilde - e
    z2_scs = z2_scs_tilde - e

    (
        scs_sample1,
        scs_sample2,
        scs_acc1,
        scs_acc2,
        scs_stepout1,
        scs_stepout2,
    ) = sub_Cauchy_algs.SCS_MRCoupling_sampler(
        n_samples,
        proposal_std_scs,
        R,
        S,
        o,
        z1_scs.copy(),
        z2_scs.copy(),
        target_log_density,
        d
    )

    # Project SCS samples back to R^d
    scs_sample1 = scs_centered_to_euclidean(scs_sample1, R, o)
    scs_sample2 = scs_centered_to_euclidean(scs_sample2, R, o)

    dist_scs = np.linalg.norm(scs_sample1 - scs_sample2, axis=1)
    meet_indices_scs = np.where(dist_scs <= tol)[0]

    if len(meet_indices_scs) > 0:
        meeting_times_scs.append(meet_indices_scs[0])
    else:
        meeting_times_scs.append(np.inf)

    scs_accept_rates.append((scs_acc1, scs_acc2))
    scs_stepout_rates.append((scs_stepout1, scs_stepout2))

    if (rep + 1) % 50 == 0:
        print(f"Finished {rep + 1}/{n_rep} repetitions")


meeting_times_eu = np.array(meeting_times_eu)
meeting_times_scs = np.array(meeting_times_scs)

results = pd.DataFrame({
    "meeting_time_eu": meeting_times_eu,
    "meeting_time_scs": meeting_times_scs,

    "eu_acc1": [a[0] for a in eu_accept_rates],
    "eu_acc2": [a[1] for a in eu_accept_rates],

    "scs_acc1": [a[0] for a in scs_accept_rates],
    "scs_acc2": [a[1] for a in scs_accept_rates],

    "scs_stepout1": [a[0] for a in scs_stepout_rates],
    "scs_stepout2": [a[1] for a in scs_stepout_rates],
})

results.to_csv(
    "/Users/bsc944/Documents/Unbiased MCMC/cauchy_coupling_tau.csv",
    index=False
)




results = pd.read_csv("/Users/bsc944/Documents/Unbiased MCMC/cauchy_coupling_tau.csv")

tau_eu_raw = results["meeting_time_eu"].to_numpy()
tau_scs_raw = results["meeting_time_scs"].to_numpy()

# -------------------------------------------------
# Convert recorded sample indices to Markov-chain times
# -------------------------------------------------
# If recorded tau = 0 corresponds to meeting after the first transition,
# then the true meeting time should be tau + 1.
tau_eu = np.where(np.isfinite(tau_eu_raw), tau_eu_raw + 1, np.inf)
tau_scs = np.where(np.isfinite(tau_scs_raw), tau_scs_raw + 1, np.inf)

# -------------------------------------------------
# Basic summaries
# -------------------------------------------------
tau_eu_finite = tau_eu[np.isfinite(tau_eu)]
tau_scs_finite = tau_scs[np.isfinite(tau_scs)]

print(f"EU met:  {len(tau_eu_finite)} / {len(tau_eu)}")
print(f"SCS met: {len(tau_scs_finite)} / {len(tau_scs)}")

scs_acc_mean = results[["scs_acc1", "scs_acc2"]].to_numpy().mean()
eu_acc_mean = results[["eu_acc1", "eu_acc2"]].to_numpy().mean()

# -------------------------------------------------
# Histogram bins, using shifted positive meeting times
# -------------------------------------------------
all_tau = np.concatenate([tau_scs_finite, tau_eu_finite])

bins = np.logspace(
    np.log10(all_tau.min()),
    np.log10(all_tau.max()),
    30
)

# -------------------------------------------------
# Estimated TV upper bounds: P(tau > t)
# -------------------------------------------------
max_iter = int(1e4)
iters = np.arange(max_iter + 1)

tv_bound_scs = np.array([
    np.mean(tau_scs > t)
    for t in iters
])

tv_bound_eu = np.array([
    np.mean(tau_eu > t)
    for t in iters
])

print(f"tv_bound_scs[0] = {tv_bound_scs[0]:.3f}")
print(f"tv_bound_eu[0]  = {tv_bound_eu[0]:.3f}")

# -------------------------------------------------
# Colors
# -------------------------------------------------
color_scs = "tab:green"
color_eu = "tab:orange"

# -------------------------------------------------
# Plot
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: meeting time histogram
sns.histplot(
    tau_scs_finite,
    bins=bins,
    stat="probability",
    alpha=0.7,
    color=color_scs,
    label=f"Sub-Cauchy, acc={scs_acc_mean:.3f}",
    ax=axes[0]
)

sns.histplot(
    tau_eu_finite,
    bins=bins,
    stat="probability",
    alpha=0.7,
    color=color_eu,
    label=f"Euclidean, acc={eu_acc_mean:.3f}",
    ax=axes[0]
)

axes[0].set_xscale("log")
axes[0].set_xlabel("Meeting time")
axes[0].set_ylabel("Probability")
axes[0].set_title("Distribution of meeting times, log scale")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: TV upper bound
axes[1].plot(
    iters,
    tv_bound_scs,
    linewidth=1.5,
    color=color_scs,
    label="Sub-Cauchy"
)

axes[1].plot(
    iters,
    tv_bound_eu,
    linewidth=1.5,
    color=color_eu,
    label="Euclidean"
)

axes[1].set_xlabel("Iteration $t$")
axes[1].set_ylabel(r"Estimated upper bound $\widehat{\mathbb{P}}(\tau > t)$")
axes[1].set_title("Coupling upper bound on TV distance")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "/Users/bsc944/Documents/Unbiased MCMC/cauchy_coupling_tau_logscale.pdf"
)
plt.show()
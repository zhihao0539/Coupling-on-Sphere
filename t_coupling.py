import stereographic_algs
import euclidean_algs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geomstats.geometry.hypersphere import Hypersphere
from scipy.stats import t
import pandas as pd
from sklearn.preprocessing import StandardScaler

np.random.seed(42)


df = pd.read_csv('/Users/bsc944/Downloads/BostonHousing.csv')

y_raw = df['medv'].values
X_raw = df.drop(columns='medv').values

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X = X_scaler.fit_transform(X_raw)
y = y_scaler.fit_transform(y_raw.reshape(-1, 1)).ravel()

d = X.shape[1] + 1



def log_density(theta, X=X, y=y, nu=4):
    beta = theta[:-1]
    u = theta[-1]

    r = y - X @ beta  # residuals
    exp2u = np.exp(2.0 * u)

    # log-likelihood term
    loglik = -0.5 * (nu + 1.0) * np.sum(
        np.log1p(r ** 2 / (nu * exp2u))
    )

    # Jeffreys prior + Jacobian
    logprior = -y.size * u

    return loglik + logprior


# '''untuned'''
R = np.sqrt(d)
S = Hypersphere(dim=d)

n_samples = int(2e4)
n_rep = 1000

# proposal_std_sp = 0.006
# proposal_std_eu = 0.05
proposal_std_sp = 0.007
proposal_std_eu = 0.03

tol = 1e-12

meeting_times_sp = []
meeting_times_eu = []

sp_accept_rates = []
eu_accept_rates = []

for rep in range(n_rep):

    # -----------------------------
    # Initial points
    # -----------------------------
    # x = 10 + np.random.randn(d)
    # y = -10 + np.random.randn(d)

    x = 10 * np.random.randn(d)
    y = 10 * np.random.randn(d)

    z1 = stereographic_algs.inverse_stereographic_projection(x, R)
    z2 = stereographic_algs.inverse_stereographic_projection(y, R)

    # -----------------------------
    # Stereographic coupling
    # -----------------------------
    sp_sample1, sp_sample2, sp_acc1, sp_acc2 = stereographic_algs.MRCoupling_sampler(
        n_samples,
        proposal_std_sp,
        R,
        S,
        z1.copy(),
        z2.copy(),
        log_density,
        d
    )

    sp_sample1 = np.array([
        stereographic_algs.stereographic_projection(z, R)
        for z in sp_sample1
    ])

    sp_sample2 = np.array([
        stereographic_algs.stereographic_projection(z, R)
        for z in sp_sample2
    ])

    dist_sp = np.linalg.norm(sp_sample1 - sp_sample2, axis=1)

    meet_indices_sp = np.where(dist_sp <= tol)[0]

    if len(meet_indices_sp) > 0:
        meeting_times_sp.append(meet_indices_sp[0])
    else:
        meeting_times_sp.append(np.inf)

    sp_accept_rates.append((sp_acc1, sp_acc2))

    # -----------------------------
    # Euclidean coupling
    # -----------------------------
    eu_sample1, eu_sample2, eu_acc1, eu_acc2 = euclidean_algs.MRCoupling_sampler(
        n_samples,
        proposal_std_eu,
        x.copy(),
        y.copy(),
        log_density,
        d
    )

    dist_eu = np.linalg.norm(eu_sample1 - eu_sample2, axis=1)

    meet_indices_eu = np.where(dist_eu <= tol)[0]

    if len(meet_indices_eu) > 0:
        meeting_times_eu.append(meet_indices_eu[0])
    else:
        meeting_times_eu.append(np.inf)

    eu_accept_rates.append((eu_acc1, eu_acc2))

    if (rep + 1) % 50 == 0:
        print(f"Finished {rep + 1}/{n_rep} repetitions")


meeting_times_sp = np.array(meeting_times_sp)
meeting_times_eu = np.array(meeting_times_eu)

meeting_times_sp = np.array(meeting_times_sp)
meeting_times_eu = np.array(meeting_times_eu)

results = pd.DataFrame({
    "meeting_time_sp": meeting_times_sp,
    "meeting_time_eu": meeting_times_eu,
    "sp_acc1": [a[0] for a in sp_accept_rates],
    "sp_acc2": [a[1] for a in sp_accept_rates],
    "eu_acc1": [a[0] for a in eu_accept_rates],
    "eu_acc2": [a[1] for a in eu_accept_rates],
})

results.to_csv("/Users/bsc944/Documents/Unbiased MCMC/t_coupling_tau2.csv", index=False)

df = pd.read_csv("/Users/bsc944/Documents/Unbiased MCMC/t_coupling_tau2.csv")

tau_sp = np.asarray(df["meeting_time_sp"], dtype=float)
tau_eu = np.asarray(df["meeting_time_eu"], dtype=float)

print(np.quantile(tau_eu, 0.95))
print(np.quantile(tau_sp, 0.95))

# Keep only finite meeting times for the histogram
tau_sp_finite = tau_sp[np.isfinite(tau_sp)]
tau_eu_finite = tau_eu[np.isfinite(tau_eu)]

print(f"SP: {len(tau_sp_finite)} / {len(tau_sp)} chains met")
print(f"Euclidean: {len(tau_eu_finite)} / {len(tau_eu)} chains met")

sp_acc_mean = df[["sp_acc1", "sp_acc2"]].to_numpy().mean()
eu_acc_mean = df[["eu_acc1", "eu_acc2"]].to_numpy().mean()

# Common bins for both histograms
all_tau = np.concatenate([tau_sp_finite, tau_eu_finite])
bins = np.linspace(all_tau.min(), all_tau.max(), 51)

# TV upper bound: keep infinite values here
meeting_times_sp = tau_sp
meeting_times_eu = tau_eu

max_iter = int(2e4)
iters = np.arange(max_iter + 1)

tv_bound_sp = np.array([
    np.mean(meeting_times_sp > t)
    for t in iters
])

tv_bound_eu = np.array([
    np.mean(meeting_times_eu > t)
    for t in iters
])

# -----------------------------
# Subplots
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# -----------------------------
# Left: meeting time histogram
# -----------------------------
sns.histplot(
    tau_sp_finite,
    bins=bins,
    stat="probability",
    alpha=0.7,
    label=f"Stereographic, acc={sp_acc_mean:.3f}",
    ax=axes[0]
)

sns.histplot(
    tau_eu_finite,
    bins=bins,
    stat="probability",
    alpha=0.7,
    label=f"Euclidean, acc={eu_acc_mean:.3f}",
    ax=axes[0]
)

axes[0].set_xlabel("Meeting time")
axes[0].set_ylabel("Probability")
axes[0].set_title("Distribution of meeting times")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# -----------------------------
# Right: TV upper bound
# -----------------------------
axes[1].plot(
    iters,
    tv_bound_sp,
    linewidth=1.5,
    label="Stereographic"
)

axes[1].plot(
    iters,
    tv_bound_eu,
    linewidth=1.5,
    label="Euclidean"
)

axes[1].set_xlabel("Iteration $t$")
axes[1].set_ylabel(r"Estimated upper bound $\widehat{\mathbb{P}}(\tau > t)$")
axes[1].set_title("Coupling upper bound on TV distance")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "/Users/bsc944/Documents/Unbiased MCMC/t_coupling_hist_tv_subplots.pdf",
    bbox_inches="tight"
)

plt.show()


'''tuned'''
# m = 10
# n_samples = [int(200 * k**0.8) for k in range(1, m + 1)]

# n_rep = 1000
# tol = 1e-12

# beta = 5
# target_acc = 0.27

# meeting_times_sp = []
# mean_acc1_per_rep = []
# mean_acc2_per_rep = []

# for rep in range(n_rep):

#     # Initial tuning parameters
#     R = np.sqrt(d)
#     S = Hypersphere(dim=d)
#     c = np.zeros(d)

#     proposal_std_sp = 0.006

#     # Initial Euclidean states
#     x = 10 + np.random.randn(d)
#     y = -10 + np.random.randn(d)

#     # Initial shifted sphere states
#     shift_z1 = stereographic_algs.inverse_stereographic_projection(x - c, R)
#     shift_z2 = stereographic_algs.inverse_stereographic_projection(y - c, R)

#     all_samples_sp = []

#     total_steps = 0
#     meeting_time = np.inf

#     acc1_list = []
#     acc2_list = []

#     for i, n_i in enumerate(n_samples):

#         shift_sp_sample1, shift_sp_sample2, sp_acc1, sp_acc2 = (
#             stereographic_algs.MRCoupling_sampler_tuned(
#                 n_i,
#                 proposal_std_sp,
#                 R,
#                 S,
#                 shift_z1,
#                 shift_z2,
#                 log_density,
#                 d,
#                 c,
#             )
#         )

#         acc1_list.append(sp_acc1)
#         acc2_list.append(sp_acc2)

#         # Map samples back to Euclidean coordinates
#         sp_sample1 = np.array([
#             stereographic_algs.stereographic_projection(z, R) + c
#             for z in shift_sp_sample1
#         ])

#         sp_sample2 = np.array([
#             stereographic_algs.stereographic_projection(z, R) + c
#             for z in shift_sp_sample2
#         ])

#         # Check meeting within this segment
#         dist_sp = np.linalg.norm(sp_sample1 - sp_sample2, axis=1)
#         meet_indices_sp = np.where(dist_sp <= tol)[0]

#         if len(meet_indices_sp) > 0:
#             meeting_time = total_steps + meet_indices_sp[0]
#             break

#         total_steps += n_i

#         # Store samples for adaptation
#         all_samples_sp.append(sp_sample1)
#         all_samples_sp.append(sp_sample2)

#         np_all_samples_sp = np.vstack(all_samples_sp)

#         # Adapt proposal scale using acceptance rate
#         sp_acc_mean = 0.5 * (sp_acc1 + sp_acc2)
#         adaptation_factor = np.exp(beta * (sp_acc_mean - target_acc))
#         proposal_std_sp *= adaptation_factor

#         # Adapt c and R
#         samples_mean = np.mean(np_all_samples_sp, axis=0)
#         samples_var = np.var(np_all_samples_sp, axis=0, ddof=1)

#         x_last = sp_sample1[-1]
#         y_last = sp_sample2[-1]

#         c = samples_mean
#         R = np.sqrt(np.sum(samples_var))

#         # Re-project current Euclidean states using the new c and R
#         shift_z1 = stereographic_algs.inverse_stereographic_projection(x_last - c, R)
#         shift_z2 = stereographic_algs.inverse_stereographic_projection(y_last - c, R)

#     meeting_times_sp.append(meeting_time)
#     mean_acc1_per_rep.append(np.mean(acc1_list))
#     mean_acc2_per_rep.append(np.mean(acc2_list))

#     if (rep + 1) % 50 == 0:
#         print(f"Finished {rep + 1}/{n_rep} repetitions")


# results = pd.DataFrame({
#     "meeting_time_sp": meeting_times_sp,
#     "sp_acc1_mean": mean_acc1_per_rep,
#     "sp_acc2_mean": mean_acc2_per_rep,
# })

# results.to_csv(
#     "/Users/bsc944/Downloads/meeting_times_results_tuned.csv",
#     index=False
# )



# df = pd.read_csv("/Users/bsc944/Downloads/meeting_times_results.csv")
# tau_sp = df['meeting_time_sp']
# tau_eu = df['meeting_time_eu']
# print(tau_sp.mean())

# df_tuned = pd.read_csv("/Users/bsc944/Downloads/meeting_times_results_tuned.csv")
# tau_sp_tuned = df_tuned['meeting_time_sp']
# print(tau_sp_tuned.mean())

# tau_sp = np.asarray(tau_sp, dtype=float)
# tau_eu = np.asarray(tau_eu, dtype=float)
# tau_sp_tuned = np.asarray(tau_sp_tuned, dtype=float)

# # Keep only finite meeting times
# tau_sp_finite = tau_sp[np.isfinite(tau_sp)]
# tau_sp_tuned_finite = tau_sp_tuned[np.isfinite(tau_sp_tuned)]
# tau_eu_finite = tau_eu[np.isfinite(tau_eu)]

# print(f"SP: {len(tau_sp_finite)} / {len(tau_sp)} chains met")
# print(f"SP_tuned: {len(tau_sp_tuned_finite)} / {len(tau_sp_tuned)} chains met")
# print(f"Euclidean: {len(tau_eu_finite)} / {len(tau_eu)} chains met")

# sp_acc_mean = df[["sp_acc1", "sp_acc2"]].to_numpy().mean()
# sp_tuned_acc_mean = df_tuned[["sp_acc1_mean", "sp_acc2_mean"]].to_numpy().mean()
# eu_acc_mean = df[["eu_acc1", "eu_acc2"]].to_numpy().mean()


# all_tau = np.concatenate([tau_sp_finite, tau_eu_finite, tau_sp_tuned_finite])

# bins = np.linspace(all_tau.min(), all_tau.max(), 51)

# plt.figure(figsize=(8, 5))

# sns.histplot(
#     tau_sp_finite,
#     bins=bins,
#     stat="probability",
#     alpha=0.7,
#     label=f"Stereographic meeting times untuned, acc={sp_acc_mean:.3f}"
# )

# sns.histplot(
#     tau_sp_tuned_finite,
#     bins=bins,
#     stat="probability",
#     alpha=0.7,
#     label=f"Stereographic meeting times tuned, acc={sp_tuned_acc_mean:.3f}"
# )

# sns.histplot(
#     tau_eu_finite,
#     bins=bins,
#     stat="probability",
#     alpha=0.7,
#     label=f"Euclidean meeting times, acc={eu_acc_mean:.3f}"
# )

# # plt.yscale("log")
# plt.xlabel("Meeting time")
# plt.ylabel("Probability")
# plt.legend()
# # plt.savefig("/Users/bsc944/Documents/Unbiased MCMC/t_coupling_tau_23.pdf")
# plt.show()











# fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# coord = 0

# # =========================
# # Row 1: stereographic chains
# # =========================

# axs[0, 0].plot(dist_sp)
# axs[0, 0].set_title("Distance between SP chains")
# axs[0, 0].set_xlabel("Iteration")
# axs[0, 0].set_ylabel(r"$\|X_t - Y_t\|$")

# sns.histplot(
#     sp_sample1[:, coord],
#     bins=50,
#     stat="probability",
#     alpha=0.7,
#     ax=axs[0, 1],
#     label="sp_sample1"
# )
# axs[0, 1].set_title("Histogram of sp_sample1")
# axs[0, 1].set_xlabel(f"Coordinate {coord}")
# axs[0, 1].set_ylabel("Probability")
# axs[0, 1].legend()

# sns.histplot(
#     sp_sample2[:, coord],
#     bins=50,
#     stat="probability",
#     alpha=0.7,
#     ax=axs[0, 2],
#     label="sp_sample2"
# )
# axs[0, 2].set_title("Histogram of sp_sample2")
# axs[0, 2].set_xlabel(f"Coordinate {coord}")
# axs[0, 2].set_ylabel("Probability")
# axs[0, 2].legend()


# # =========================
# # Row 2: Euclidean chains
# # =========================

# axs[1, 0].plot(dist_eu)
# axs[1, 0].set_title("Distance between Euclidean chains")
# axs[1, 0].set_xlabel("Iteration")
# axs[1, 0].set_ylabel(r"$\|X_t - Y_t\|$")

# sns.histplot(
#     eu_sample1[:, coord],
#     bins=50,
#     stat="probability",
#     alpha=0.7,
#     ax=axs[1, 1],
#     label="eu_sample1"
# )
# axs[1, 1].set_title("Histogram of eu_sample1")
# axs[1, 1].set_xlabel(f"Coordinate {coord}")
# axs[1, 1].set_ylabel("Probability")
# axs[1, 1].legend()

# sns.histplot(
#     eu_sample2[:, coord],
#     bins=50,
#     stat="probability",
#     alpha=0.7,
#     ax=axs[1, 2],
#     label="eu_sample2"
# )
# axs[1, 2].set_title("Histogram of eu_sample2")
# axs[1, 2].set_xlabel(f"Coordinate {coord}")
# axs[1, 2].set_ylabel("Probability")
# axs[1, 2].legend()

# plt.tight_layout()
# plt.show()
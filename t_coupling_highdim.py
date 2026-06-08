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


df = pd.read_csv('/Users/bsc944/Downloads/nba_salary.csv')

y_raw = df['sqrt_salary_million'].values
X_raw = df.drop(columns='sqrt_salary_million').values

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


R = np.sqrt(d)
S = Hypersphere(dim=d)

n_samples = int(1.5e6)
n_rep = 1

proposal_std_sp = 1e-3
proposal_std_eu = 1e-2

tol = 1e-12

meeting_times_sp = []
meeting_times_eu = []

sp_accept_rates = []
eu_accept_rates = []

for rep in range(n_rep):

    # -----------------------------
    # Initial points
    # -----------------------------

    x = 0.1 * np.random.randn(d)
    y = 0.1 * np.random.randn(d)

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
    print(sp_acc1, sp_acc2)

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

    print(eu_acc1, eu_acc2)

    dist_eu = np.linalg.norm(eu_sample1 - eu_sample2, axis=1)

    meet_indices_eu = np.where(dist_eu <= tol)[0]

    if len(meet_indices_eu) > 0:
        meeting_times_eu.append(meet_indices_eu[0])
    else:
        meeting_times_eu.append(np.inf)

    eu_accept_rates.append((eu_acc1, eu_acc2))

    if (rep + 1) % 50 == 0:
        print(f"Finished {rep + 1}/{n_rep} repetitions")


fig, axs = plt.subplots(2, 3, figsize=(18, 10))

coord = 7

# =========================
# Row 1: stereographic chains
# =========================

axs[0, 0].plot(dist_sp)
axs[0, 0].set_title("Distance between SP chains")
axs[0, 0].set_xlabel("Iteration")
axs[0, 0].set_ylabel(r"$\|X_t - Y_t\|$")

sns.histplot(
    sp_sample1[:, coord],
    bins=50,
    stat="probability",
    alpha=0.7,
    ax=axs[0, 1],
    label="sp_sample1"
)
axs[0, 1].set_title("Histogram of sp_sample1")
axs[0, 1].set_xlabel(f"Coordinate {coord}")
axs[0, 1].set_ylabel("Probability")
axs[0, 1].legend()

sns.histplot(
    sp_sample2[:, coord],
    bins=50,
    stat="probability",
    alpha=0.7,
    ax=axs[0, 2],
    label="sp_sample2"
)
axs[0, 2].set_title("Histogram of sp_sample2")
axs[0, 2].set_xlabel(f"Coordinate {coord}")
axs[0, 2].set_ylabel("Probability")
axs[0, 2].legend()


# =========================
# Row 2: Euclidean chains
# =========================

axs[1, 0].plot(dist_eu)
axs[1, 0].set_title("Distance between Euclidean chains")
axs[1, 0].set_xlabel("Iteration")
axs[1, 0].set_ylabel(r"$\|X_t - Y_t\|$")

sns.histplot(
    eu_sample1[:, coord],
    bins=50,
    stat="probability",
    alpha=0.7,
    ax=axs[1, 1],
    label="eu_sample1"
)
axs[1, 1].set_title("Histogram of eu_sample1")
axs[1, 1].set_xlabel(f"Coordinate {coord}")
axs[1, 1].set_ylabel("Probability")
axs[1, 1].legend()

sns.histplot(
    eu_sample2[:, coord],
    bins=50,
    stat="probability",
    alpha=0.7,
    ax=axs[1, 2],
    label="eu_sample2"
)
axs[1, 2].set_title("Histogram of eu_sample2")
axs[1, 2].set_xlabel(f"Coordinate {coord}")
axs[1, 2].set_ylabel("Probability")
axs[1, 2].legend()

plt.tight_layout()
plt.show()
import stereographic_tuned_algs
import stereographic_algs
import euclidean_tuned_algs
import sub_Cauchy_algs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geomstats.geometry.hypersphere import Hypersphere
from scipy.stats import t
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/Users/bsc944/Downloads/nba_salary.csv')

y_raw = df['sqrt_salary_million'].values
X_raw = df.drop(columns='sqrt_salary_million').values

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X = X_scaler.fit_transform(X_raw)
y = y_scaler.fit_transform(y_raw.reshape(-1, 1)).ravel()

d = X.shape[1] + 1

### flat prior
# def log_density(theta, X=X, y=y, nu=4):
#     beta = theta[:-1]
#     u = theta[-1]

#     r = y - X @ beta
#     exp2u = np.exp(2.0 * u)

#     loglik = -0.5 * (nu + 1.0) * np.sum(
#         np.log1p(r ** 2 / (nu * exp2u))
#     )

#     # Jeffreys prior + Jacobian
#     logprior = -y.size * u

#     return loglik + logprior


### normal prior
def log_density(theta, X=X, y=y, nu=4, prior_var=1.0):
    beta = theta[:-1]
    u = theta[-1]

    r = y - X @ beta
    exp2u = np.exp(2.0 * u)

    # Likelihood (Now correctly including the scale normalization)
    loglik = -0.5 * (nu + 1.0) * np.sum(np.log1p(r ** 2 / (nu * exp2u))) - y.size * u

    # Gaussian prior on beta: sum(-0.5 * beta^2 / prior_var)
    logprior_beta = -0.5 * np.sum(beta ** 2) / prior_var

    # Weak normal prior on u (e.g., N(0, variance=1)) for MCMC stability
    logprior_u = -0.5 * (u ** 2) 
    return loglik + logprior_beta + logprior_u


### student t prior
# def log_density(theta, X=X, y=y, nu_data=4, nu_prior=3, scale_prior=1.0):
#     beta = theta[:-1]
#     u = theta[-1]

#     r = y - X @ beta
#     exp2u = np.exp(2.0 * u)

#     # Likelihood
#     loglik = -0.5 * (nu_data + 1.0) * np.sum(np.log1p(r ** 2 / (nu_data * exp2u))) - y.size * u

#     # Student's t-prior on beta
#     # Note: We omit constants like Gamma functions that don't depend on theta
#     logprior_beta = -0.5 * (nu_prior + 1.0) * np.sum(
#         np.log1p((beta ** 2) / (nu_prior * scale_prior ** 2))
#     )

#     # Weak normal prior on u
#     logprior_u = -0.5 * (u ** 2)

#     return loglik + logprior_beta + logprior_u


### normal likelihood, normal prior
# def log_density(theta, X=X, y=y, prior_var=1.0):
#     beta = theta[:-1]
#     u = theta[-1]                

#     r = y - X @ beta
#     exp2u = np.exp(2.0 * u)

#     # Normal log-likelihood (constant terms omitted)
#     loglik = -0.5 * np.sum(r ** 2) / exp2u - y.size * u

#     # Gaussian prior on beta
#     logprior_beta = -0.5 * np.sum(beta ** 2) / prior_var

#     # Weak normal prior on log(sigma)
#     logprior_u = -0.5 * (u ** 2)

#     return loglik + logprior_beta + logprior_u



np.random.seed(42)

m = 30
n_samples = [int(4000 * k**0.8) for k in range(1, m + 1)]

n_rep = 1
tol = 1e-12

beta = 5
target_acc = 0.24

S = Hypersphere(dim=d)

meeting_times_sp = []
mean_acc1_per_rep = []
mean_acc2_per_rep = []

trace_chain1 = []
trace_chain2 = []
dist_sp_all = []

for rep in range(n_rep):

    c = np.zeros(d)
    Sigma = d * np.eye(d)
    proposal_std_sp = 1e-3

    # Initial Euclidean states
    x = np.random.randn(d)
    y = np.random.randn(d)

    # Initial shifted sphere states
    shift_z1 = stereographic_tuned_algs.inverse_stereographic_projection_general(
        x.copy(), c, Sigma
    )
    shift_z2 = stereographic_tuned_algs.inverse_stereographic_projection_general(
        y.copy(), c, Sigma
    )

    all_samples_sp = []

    total_steps = 0
    meeting_time = np.inf

    for i, n_i in enumerate(n_samples):

        shift_sp_sample1, shift_sp_sample2, sp_acc1, sp_acc2 = (
            stereographic_tuned_algs.MRCoupling_sampler(
                n_i,
                proposal_std_sp,
                c,
                Sigma,
                S,
                shift_z1,
                shift_z2,
                log_density,
                d
            )
        )


        trace_chain1.append(shift_sp_sample1)
        trace_chain2.append(shift_sp_sample2)

        dist_sp = np.array([S.metric.dist(z1, z2)for z1, z2 in zip(shift_sp_sample1, shift_sp_sample2)])
        meet_indices_sp = np.where(dist_sp <= tol)[0]

        dist_sp_all.extend(dist_sp)

        if len(meet_indices_sp) > 0:
            meeting_time = total_steps + meet_indices_sp[0]
            # break

        total_steps += n_i

        # Map samples back to Euclidean coordinates
        sp_sample1 = np.array([
            stereographic_tuned_algs.stereographic_projection_general(z, c, Sigma)
            for z in shift_sp_sample1
        ])

        sp_sample2 = np.array([
            stereographic_tuned_algs.stereographic_projection_general(z, c, Sigma)
            for z in shift_sp_sample2
        ])

        # Store samples for adaptation
        np_all_samples_sp = np.vstack([sp_sample1, sp_sample2])

        # Adapt proposal scale using acceptance rate
        sp_acc_mean = 0.5 * (sp_acc1 + sp_acc2)

        adaptation_factor = np.exp(beta * (sp_acc_mean - target_acc))
        proposal_std_sp *= adaptation_factor

        c, Sigma = stereographic_tuned_algs.estimate_stereographic_parameters(
            np_all_samples_sp
        )

        # Re-project current Euclidean states using the new c and R
        shift_z1 = stereographic_tuned_algs.inverse_stereographic_projection_general(sp_sample1[-1], c, Sigma)
        shift_z2 = stereographic_tuned_algs.inverse_stereographic_projection_general(sp_sample2[-1], c, Sigma)

    meeting_times_sp.append(meeting_time)

print(meeting_times_sp)

col_names = [f"dim_{i}" for i in range(d)]
df_c = pd.DataFrame([c], columns=col_names)
df_Sigma = pd.DataFrame(Sigma, columns=col_names)
full_df = pd.concat([df_c, df_Sigma], ignore_index=True)
full_df.to_csv("/Users/bsc944/Documents/Unbiased MCMC/highdim_t_tuned_params.csv", index=False)


trace_chain1 = np.vstack(trace_chain1)
trace_chain2 = np.vstack(trace_chain2)
dist_sp_all = np.asarray(dist_sp_all)

coord = -1
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

adapt_times = np.cumsum(n_samples)
max_len = min(len(dist_sp), len(trace_chain1), len(trace_chain2))

for x in adapt_times:
    for ax in axes:
        ax.axvline(
            x=x,
            linestyle="--",
            alpha=0.5,
            color="tab:gray",
            linewidth=0.5
        )

axes[0].plot(dist_sp_all)
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Spherical distance")
axes[0].set_title("Distance between coupled chains")


axes[1].plot(trace_chain1[:,coord])
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Latitude")
axes[1].set_title("Trace plot of latitude of the first chain")


axes[2].plot(trace_chain2[:,coord])
axes[2].set_xlabel("Iteration")
axes[2].set_ylabel("Latitude")
axes[2].set_title("Trace plot of latitude of the second chain")

plt.tight_layout()
plt.savefig(
    "/Users/bsc944/Documents/Unbiased MCMC/t_coupling_nba_tuning.pdf",
    bbox_inches="tight"
)
plt.show()







'''stereograhic'''
S = Hypersphere(dim=d)

n_samples = int(2e6)
n_rep = 20

proposal_std_sp = 1.5e-2

tol = 1e-12

meeting_times_sp = []
sp_accept_rates = []

df = pd.read_csv("/Users/bsc944/Documents/Unbiased MCMC/highdim_t_tuned_params.csv")

c = df.iloc[0].values.astype(float)
Sigma = df.iloc[1:].values.astype(float)

for rep in range(n_rep):

    x0 = np.random.randn(d)
    y0 = np.random.randn(d)

    # Initial sphere states under current c and Sigma
    z1 = stereographic_tuned_algs.inverse_stereographic_projection_general(
        x0, c, Sigma
    )
    z2 = stereographic_tuned_algs.inverse_stereographic_projection_general(
        y0, c, Sigma
    )

    sp_sample_z1, sp_sample_z2, sp_acc1, sp_acc2 = (
        stereographic_tuned_algs.MRCoupling_sampler(
            n_samples,
            proposal_std_sp,
            c,
            Sigma,
            S,
            z1,
            z2,
            log_density,
            d
        )
    )

    sp_accept_rates.append((sp_acc1, sp_acc2))

    dist_sp = np.array([
        S.metric.dist(z_1, z_2)
        for z_1, z_2 in zip(sp_sample_z1, sp_sample_z2)
    ])

    meet_indices_sp = np.where(dist_sp <= tol)[0]

    if len(meet_indices_sp) > 0:
        meeting_times_sp.append(meet_indices_sp[0])
        continue
    else:
        meeting_times_sp.append(np.inf)

    
results = pd.DataFrame({
    "meeting_time_sp": meeting_times_sp,
    "sp_acc1": [a[0] for a in sp_accept_rates],
    "sp_acc2": [a[1] for a in sp_accept_rates],
})

results.to_csv("/Users/bsc944/Documents/Unbiased MCMC/t_highdim_coupling_tau6.csv", index=False)

print(meeting_times_sp)
print(sp_accept_rates)



df = pd.read_csv("/Users/bsc944/Documents/Unbiased MCMC/t_highdim_coupling_tau2.csv")
tau_sp = np.asarray(df["meeting_time_sp"], dtype=float)
tau_sp_finite = tau_sp[np.isfinite(tau_sp)]
print(f"SP: {len(tau_sp_finite)} / {len(tau_sp)} chains met")
bins = np.linspace(tau_sp_finite.min(), tau_sp_finite.max(), 11)
sp_acc_mean = df[["sp_acc1", "sp_acc2"]].to_numpy().mean()

sns.histplot(
    tau_sp_finite,
    bins=bins,
    stat="probability",
    alpha=0.7,
    label=f"Stereographic, acc={sp_acc_mean:.3f}",
)

plt.legend()
plt.savefig(
    "/Users/bsc944/Documents/Unbiased MCMC/t_coupling_nba_sp.pdf",
    bbox_inches="tight"
)
plt.show()












'''Euclidean'''
df = pd.read_csv("/Users/bsc944/Documents/Unbiased MCMC/highdim_t_tuned_params.csv")
Sigma = df.iloc[1:].values.astype(float)
Sigma = Sigma / d
n_samples = int(2e6)
proposal_std_eu = 1e-1

tol = 1e-12
meeting_times_eu = []
eu_accept_rates = []

n_rep = 1
for rep in range(n_rep):

    x = np.random.randn(d)
    y = np.random.randn(d)

    eu_sample1, eu_sample2, eu_acc1, eu_acc2 = euclidean_tuned_algs.MRCoupling_sampler(
        n_samples,
        proposal_std_eu,
        Sigma,
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

print(meeting_times_eu)
print(eu_accept_rates)

coord = 1
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(dist_eu)
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Euclidean distance")
axes[0].set_title("Distance between coupled chains")


axes[1].plot(eu_sample1[:,coord])
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel(rf"$x_{{{coord}}}$")
axes[1].set_title(rf"Trace plot of first chain, coordinate $x_{{{coord}}}$")


axes[2].plot(eu_sample2[:,coord])
axes[2].set_xlabel("Iteration")
axes[2].set_ylabel(rf"$x_{{{coord}}}$")
axes[2].set_title(rf"Trace plot of second chain, coordinate $x_{{{coord}}}$")

plt.tight_layout()
plt.show()

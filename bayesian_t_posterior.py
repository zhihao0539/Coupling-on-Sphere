import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from geomstats.geometry.hypersphere import Hypersphere
import stereographic_algs
import euclidean_algs
import os


np.random.seed(42)

df = pd.read_csv('/Users/bsc944/Downloads/BostonHousing.csv')
y = df['medv'].values
X = df.drop(columns='medv').values
d = X.shape[1] + 1
X = StandardScaler().fit_transform(X)


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
num_trials = 100  
max_iterations = int(1e6)  

sigma_lst = [1,2,3,4,5,6,7,8,9,10]
step_sizes = np.logspace(-2, 0, num=10)

# Trackers for the parameters
recorded_sigmas = []
recorded_stds = []

# Lists for metrics
meeting_times_sp = []
meeting_times_sp_75 = []
failures_sp = []

meeting_times_eu = []
meeting_times_eu_75 = []
failures_eu = []

for sigma in sigma_lst:
    for std in step_sizes:
        print(f"Running Sigma: {sigma} | Step Size: {std:.4f}")
        
        # Record the parameters for this iteration
        recorded_sigmas.append(sigma)
        recorded_stds.append(std)

        trials_tau_sp = []
        fail_count_sp = 0 
        trials_tau_eu = []
        fail_count_eu = 0 

        for _ in range(num_trials):
            x1, x2 = sigma * np.random.randn(2, d)
            x1_copy = x1.copy()
            x2_copy = x2.copy()
            
            # --- Stereographic ---
            z1 = stereographic_algs.inverse_stereographic_projection(x1, R)
            z2 = stereographic_algs.inverse_stereographic_projection(x2, R)

            tau_sp = 0
            while not np.array_equal(z1, z2) and tau_sp < max_iterations:
                z1_proposal, z2_proposal = stereographic_algs.maximal_reflection_coupling(S, z1, z2, std, d)
                z1, z2 = stereographic_algs.accept_reject(log_density, z1, z2, z1_proposal, z2_proposal, R, d)
                tau_sp += 1

            if np.array_equal(z1, z2):
                trials_tau_sp.append(tau_sp)
            else:
                fail_count_sp += 1

            # --- Euclidean ---
            x1 = x1_copy
            x2 = x2_copy

            tau_eu = 0
            while not np.array_equal(x1, x2) and tau_eu < max_iterations:
                x1_proposal, x2_proposal = euclidean_algs.maximal_reflection_coupling(x1, x2, std, d)
                x1, x2 = euclidean_algs.accept_reject(log_density, x1, x2, x1_proposal, x2_proposal)
                tau_eu += 1

            if np.array_equal(x1, x2):
                trials_tau_eu.append(tau_eu)
            else:
                fail_count_eu += 1

        # Calculate and record metrics
        failures_sp.append(fail_count_sp)
        if trials_tau_sp:
            meeting_times_sp.append(np.mean(trials_tau_sp))
            meeting_times_sp_75.append(np.percentile(trials_tau_sp, 75))
        else:
            meeting_times_sp.append(np.nan)
            meeting_times_sp_75.append(np.nan)
        
        failures_eu.append(fail_count_eu)
        if trials_tau_eu:
            meeting_times_eu.append(np.mean(trials_tau_eu))
            meeting_times_eu_75.append(np.percentile(trials_tau_eu, 75))
        else:
            meeting_times_eu.append(np.nan)
            meeting_times_eu_75.append(np.nan)

# Create a single comprehensive DataFrame
results_df = pd.DataFrame({
    'Sigma': recorded_sigmas,
    'Step_Size': recorded_stds,
    'Mean_Tau_SP': meeting_times_sp,
    '75_Quantile_Tau_SP': meeting_times_sp_75,
    'Failures_SP': failures_sp,
    'Mean_Tau_EU': meeting_times_eu,
    '75_Quantile_Tau_EU': meeting_times_eu_75,
    'Failures_EU': failures_eu
})

# Save the DataFrame to a CSV file
results_df.to_csv('/Users/bsc944/Documents/Unbiased MCMC/t_meeting_times(sigma_std).csv', index=False)












import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import numpy as np


df = pd.read_csv('/Users/bsc944/Documents/Unbiased MCMC/t_meeting_times(sigma_std).csv')

# matrix_sp = df.pivot(index='Sigma', columns='Step_Size', values='Mean_Tau_SP')
# matrix_eu = df.pivot(index='Sigma', columns='Step_Size', values='Mean_Tau_EU')

matrix_sp = df.pivot(index='Sigma', columns='Step_Size', values='75_Quantile_Tau_SP')
matrix_eu = df.pivot(index='Sigma', columns='Step_Size', values='75_Quantile_Tau_EU')

X = matrix_sp.columns.values
Y = matrix_sp.index.values

global_vmin = max(1, min(np.nanmin(matrix_sp.values), np.nanmin(matrix_eu.values)))
global_vmax = max(np.nanmax(matrix_sp.values), np.nanmax(matrix_eu.values))

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

im1 = axes[0].pcolormesh(X, Y, matrix_sp.values, shading='nearest', cmap='inferno', 
                         norm=LogNorm(vmin=global_vmin, vmax=global_vmax))
axes[0].set_xscale('log') 
axes[0].set_title('75 Quantile Meeting Time - Stereographic')
axes[0].set_xlabel('Step Size (log scale)')
axes[0].set_ylabel('Sigma')

im2 = axes[1].pcolormesh(X, Y, matrix_eu.values, shading='nearest', cmap='inferno', 
                         norm=LogNorm(vmin=global_vmin, vmax=global_vmax))
axes[1].set_xscale('log') 
axes[1].set_title('75 Quantile Meeting Time - Euclidean')
axes[1].set_xlabel('Step Size (log scale)')

fig.colorbar(im1, ax=axes, label='Mean Meeting Time (Tau) - Log Scale')
plt.savefig('/Users/bsc944/Documents/Unbiased MCMC/t_heatmap.pdf')
plt.show()
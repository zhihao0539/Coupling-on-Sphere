import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from geomstats.geometry.hypersphere import Hypersphere
import stereographic_algs
from scipy.stats import t
from scipy.special import gamma
import matplotlib.pyplot as plt


def log_density(x): #(product student t)
    x = np.atleast_1d(x)
    d = x.shape[0]
    return np.sum(t.logpdf(x, df=d + 1))


# def log_density(x, alpha=1): #(Product exponential-power)
#     scale = np.sqrt(gamma(1/alpha) / gamma(3/alpha))
#     return -np.sum((np.abs(x) / scale) ** alpha, axis=-1)

# def log_density(x): #(spherical laplace)
#     x = np.asarray(x)
#     d = x.shape[-1]
#     return -np.sqrt(d + 1) * np.linalg.norm(x, axis=-1)


np.random.seed(41)
d = 100
# x1_initial = np.zeros(d)
x1_initial = [20] * d
x2_initial = [0] * d
std = 0.5/d
R = np.sqrt(d)
S = Hypersphere(dim=d)
n_samples = int(2e4)

z1_initial = stereographic_algs.inverse_stereographic_projection(x1_initial, R)
z2_initial = stereographic_algs.inverse_stereographic_projection(x2_initial, R)

samples1_cp, samples2_cp = stereographic_algs.MRCoupling_sampler(n_samples, std, R, S, z1_initial, z2_initial, log_density, d)
dist = np.array([S.metric.dist(samples1_cp[i], samples2_cp[i]) for i in range(n_samples)])
dist[np.abs(dist) < 1e-7] = 0

fig, ax1 = plt.subplots()

# First plot (left y-axis)
ax1.plot(samples1_cp[:, -1], label='samples_SRWM_1')
ax1.plot(samples2_cp[:, -1], label='samples_SRWM_2')
ax1.set_ylabel('latitude')
ax1.set_ylim(-1, 1)
ax1.set_xlabel('iterations')
# Create second y-axis sharing the same x-axis
ax2 = ax1.twinx()

# Second plot (right y-axis)
ax2.plot(dist, color='orange', label='distance')
ax2.set_ylabel('distance')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)

# plt.savefig('/Users/bsc944/Documents/Unbiased MCMC/meeting_mixing1.pdf')
plt.show()







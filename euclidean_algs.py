from scipy.linalg import sqrtm
import numpy as np

def maximal_reflection_coupling(mu1, mu2, proposal_std, d):
    'reflection coupling on Euclidean space'

    Sigma = (proposal_std**2) * np.identity(d)
    Sigma_sqrt = sqrtm(Sigma)
    Sigma_invsqrt = np.linalg.inv(Sigma_sqrt)

    z = Sigma_invsqrt @ (mu1 - mu2)
    e = z / np.linalg.norm(z)

    xdot = np.random.multivariate_normal(np.zeros(d), np.identity(d))
    w = np.random.rand()

    log_ratio = -0.5 * (np.linalg.norm(xdot + z)**2 - np.linalg.norm(xdot)**2)
    if np.log(w) <= log_ratio:
        ydot = xdot + z

    else:
        ydot = xdot - 2 * (e @ xdot) * e

    x = Sigma_sqrt @ xdot + mu1
    y = Sigma_sqrt @ ydot + mu2

    return x, y


def accept_reject(target_distribution, x1, x2, x1_proposal, x2_proposal):

    log_ratio1 = target_distribution(x1_proposal) - target_distribution(x1)
    log_ratio2 = target_distribution(x2_proposal) - target_distribution(x2)

    log_ratio1 = min(0, log_ratio1)
    log_ratio2 = min(0, log_ratio2)

    unirand = np.random.rand()
    if np.log(unirand) <= log_ratio1:
        x1 = x1_proposal
    if np.log(unirand) <= log_ratio2:
        x2 = x2_proposal

    return x1, x2
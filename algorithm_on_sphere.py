from geomstats.geometry.hypersphere import Hypersphere
import numpy as np
import math
from scipy.stats import norm
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.stats import multivariate_normal
from scipy.stats import multivariate_t
import pandas as pd
from scipy.stats import linregress
from scipy.special import kv, gammaln, gamma


def stereographic_projection(point, R):
    'SP: z to x'

    point = np.asarray(point)

    # Extract the last coordinate (z_{d+1})
    p_last = point[-1]

    # Compute scaling factor (1 - z_{d+1})
    scale = 1.0 - p_last

    # Project onto equatorial plane
    projected_points = R * point[:-1] / scale

    return projected_points

def inverse_stereographic_projection(point, R):
    'SP^{-1}: x to z'

    point = np.asarray(point)

    # Compute squared norms of the points
    q_squared_norms = np.linalg.norm(point)**2

    # Compute scaling factor
    denominator = R ** 2 + q_squared_norms

    # Project back to the sphere
    p_components = (2 * R * point) / denominator
    p_last = (q_squared_norms - R ** 2) / denominator

    # Combine into (d+1)-dimensional points
    projected_point = np.append(p_components, p_last)

    return projected_point

def sphere_random_walk(z, proposal_std, d):
    'random walk proposal on sphere'

    cov = (proposal_std**2) * np.identity(d+1)
    tilde_z = np.random.multivariate_normal(np.zeros(d+1), cov)
    # tangent vector
    dz = tilde_z - (z @ tilde_z / np.linalg.norm(z) ** 2) * z

    # proposal on sphere
    hat_z = (z + dz) / np.linalg.norm(z + dz)
    return hat_z, dz

def accept_rate(target_distribution, Sigma, z, z_proposal, R, d):
    'compute acceptance rate, the target distribution should be in log form'

    x = stereographic_projection(z, R)
    x_proposal = stereographic_projection(z_proposal, R)
    log_acceptance_ratio = target_distribution(x_proposal, Sigma, d) - target_distribution(x, Sigma, d) + d * (np.log(R**2 + np.linalg.norm(x_proposal)**2) - np.log(R**2 + np.linalg.norm(x)**2))
    # log_acceptance_ratio = min(0, log_acceptance_ratio)
    return log_acceptance_ratio


def reflection_coupling(S, z1, z2, proposal_std, d):
    'construct the contractive coupling by parallel transport along the great circle'

    z1_proposal, tangent_vec = sphere_random_walk(z1, proposal_std, d)

    # parallel transport
    v_transported = S.metric.parallel_transport(
        tangent_vec=tangent_vec, # vector to be transported
        base_point=z1, # starting point
        end_point=z2 #end point
    )

    v_flip = reflection(S, v_transported, z2, z1)
    z2_proposal = (z2 + v_flip) / np.linalg.norm(z2 + v_flip)

    # z2_proposal = (z2 + v_transported) / np.linalg.norm(z2 + v_transported)

    return np.array(z1_proposal), np.array(z2_proposal)


def parallel_coupling(S, z1, z2, proposal_std, d):
    'construct the contractive coupling by parallel transport along the great circle'

    z1_proposal, tangent_vec = sphere_random_walk(z1, proposal_std, d)

    # parallel transport
    v_transported = S.metric.parallel_transport(
        tangent_vec=tangent_vec, # vector to be transported
        base_point=z1, # starting point
        end_point=z2 #end point
    )

    z2_proposal = (z2 + v_transported) / np.linalg.norm(z2 + v_transported)

    return np.array(z1_proposal), np.array(z2_proposal)



def reflection(S,p,z1,z2):
    'reflect p w.r.t. the hyperplane perpendicular to the tangent vector z1-z2 and starts at z1'

    # v = z1 - z2
    # tangent_vec = S.to_tangent(vector=v, base_point=z1)

    tangent_vec = S.metric.log(z2, z1)
    tangent_vec = tangent_vec / np.linalg.norm(tangent_vec)

    ref = p - 2 * (p @ tangent_vec) * tangent_vec

    return ref


def arc_vec_plus(z2, p, z1):
    'a point z2 plus an arc p-z1 (Move point p along the same geodesic displacement that maps z1 to z2)'

    # Angle between z1 and z2
    cos_theta = np.clip(np.dot(z1, z2), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-15:  # z1 and z2 are the same
        return p.copy()

    # Orthonormal basis of the plane spanned by z1 and z2
    u = z1
    v = z2 - cos_theta * z1
    v /= np.linalg.norm(v)

    # Coordinates of z1 and z2 in this basis
    # z1 = 1 * u + 0 * v
    # z2 = cos(theta) * u + sin(theta) * v

    # Project p into this basis
    pu = np.dot(p, u)
    pv = np.dot(p, v)
    p_orth = p - pu * u - pv * v  # orthogonal component

    # Rotate (pu, pv) by +theta in the u-v plane
    pu_new = pu * np.cos(theta) - pv * np.sin(theta)
    pv_new = pu * np.sin(theta) + pv * np.cos(theta)

    # Recombine
    p_rot = pu_new * u + pv_new * v + p_orth
    # Normalize to unit sphere
    p_rot /= np.linalg.norm(p_rot)

    return p_rot


def maximal_reflection_coupling(S, z1, z2, proposal_std, d):
    'maximal reflection coupling'

    z1_star,_ = sphere_random_walk(z1, proposal_std, d)
    dist_1 = S.metric.dist(z1, z1_star)
    dist_2 = S.metric.dist(z2, z1_star)

    eps = 1e-12  # small tolerance
    cos1 = np.clip(np.cos(dist_1), eps, 1.0)
    cos2 = np.clip(np.cos(dist_2), eps, 1.0)

    tan1 = np.tan(np.clip(dist_1, None, np.pi/2 - eps))
    tan2 = np.tan(np.clip(dist_2, None, np.pi/2 - eps))

    log_trans_func_1 = - (d+1) * np.log(cos1) - (tan1**2) / (2 * proposal_std**2)
    log_trans_func_2 = - (d+1) * np.log(cos2) - (tan2**2) / (2 * proposal_std**2)

    if np.log(np.random.rand()) <= log_trans_func_2 - log_trans_func_1:
        z2_star = z1_star
    else:
        ref = reflection(S,z1_star,z1,z2)
        z2_star = arc_vec_plus(z2, ref, z1)

    return np.array(z1_star), np.array(z2_star)
    


def accept_reject(target_distribution, z1, z2, z1_proposal, z2_proposal, R, d):
    x1 = stereographic_projection(z1, R)
    x2 = stereographic_projection(z2, R)
    x1_proposal = stereographic_projection(z1_proposal, R)
    x2_proposal = stereographic_projection(z2_proposal, R)

    log_ratio1 = target_distribution(x1_proposal) - target_distribution(x1) + d * (np.log(R**2 + np.linalg.norm(x1_proposal)**2) - np.log(R**2 + np.linalg.norm(x1)**2))
    log_ratio2 = target_distribution(x2_proposal) - target_distribution(x2) + d * (np.log(R**2 + np.linalg.norm(x2_proposal)**2) - np.log(R**2 + np.linalg.norm(x2)**2))

    log_ratio1 = min(0, log_ratio1)
    log_ratio2 = min(0, log_ratio2)

    ratio1 = np.exp(log_ratio1)
    ratio2 = np.exp(log_ratio2)

    unirand = np.random.rand()
    accpt1, accpt2 = False, False
    if np.log(unirand) <= log_ratio1:
        z1 = z1_proposal
        accpt1 = True
    if np.log(unirand) <= log_ratio2:
        z2 = z2_proposal 
        accpt2 = True

    if [accpt1, accpt2] == [True, True]:
        a = 2
    elif [accpt1, accpt2] == [False, False]:
        a = 0
    else:
        a = 1 
    return z1, z2, a




def contract_and_maximal_reflection_coupling(target_distribution, S, z1, z2, proposal_std, d, threshold, n_samples=10000):

    R = np.sqrt(d)

    switch = False

    dist_lst = []

    for n in range(n_samples):
        dist = S.metric.dist(z1, z2)
        dist_lst.append(dist)

        if not switch and dist < threshold:
            switch = True

        if not switch:
            z1_proposal,z2_proposal = parallel_coupling(S, z1, z2, proposal_std, d)
            z1, z2, _, _ = accept_reject(target_distribution, z1, z2, z1_proposal, z2_proposal, R, d)
        else:
            z1_proposal,z2_proposal = maximal_reflection_coupling(S, z1, z2, proposal_std, d)
            z1, z2, _, _ = accept_reject(target_distribution, z1, z2, z1_proposal, z2_proposal, R, d)

    return dist_lst



def adap_maximal_reflection_coupling(target_distribution, target_accept, S, z1, z2, R, d, n_samples):

    S = Hypersphere(dim=d)
    sigma = 1.0  # 初始 step size
    gamma = 0.6
    t0 = 10

    accepted = 0
    acc_rates = []

    dist_lst_ref = [S.metric.dist(z1, z2)]

    for i in range(n_samples):
        x1 = stereographic_projection(z1, R)
        log_px1 = target_distribution(x1)

        x2 = stereographic_projection(z2, R)
        log_px2 = target_distribution(x2)

        z1_prop, z2_prop = maximal_reflection_coupling(S, z1, z2, sigma, d)

        x1_prop = stereographic_projection(z1_prop, R)
        x2_prop = stereographic_projection(z2_prop, R)

        log_p_prop1 = target_distribution(x1_prop)
        log_p_prop2 = target_distribution(x2_prop)

        log_alpha1 = log_p_prop1 - log_px1 + d * (np.log(R**2 + np.linalg.norm(x1_prop)**2) - np.log(R**2 + np.linalg.norm(x1)**2))
        log_alpha2 = log_p_prop2 - log_px2 + d * (np.log(R**2 + np.linalg.norm(x1_prop)**2) - np.log(R**2 + np.linalg.norm(x1)**2))

        log_alpha1 = min(0, log_alpha1)
        log_alpha2 = min(0, log_alpha2)

        log_w = np.log(np.random.rand())

        if log_w < log_alpha1:
            z1 = z1_prop
            log_px1 = log_p_prop1
            acc = 1
        else:
            acc = 0

        if log_w < log_alpha2:
            z2 = z2_prop
            log_px2 = log_p_prop2

        dist_lst_ref.append(S.metric.dist(z1, z2))
        accepted += acc

        # Robbins–Monro 更新
        eta = 1.0 / ((i+1) + t0) ** gamma
        sigma = np.exp(np.log(sigma) + eta * (acc - target_accept))

        acc_rates.append(accepted / (i+1))

    return np.array(acc_rates), sigma, dist_lst_ref




def new_reflection_coupling(S, z1, z2, proposal_std, d):
    z1_proposal, tangent_vec = sphere_random_walk(z1, proposal_std, d)
    ref = reflection(S,tangent_vec,z1,z2)
    v_transported = S.metric.parallel_transport(
        tangent_vec=ref, # vector to be transported
        base_point=z1, # starting point
        end_point=z2 #end point
    )

    z2_proposal = (z2 + v_transported) / np.linalg.norm(z2 + v_transported)

    return np.array(z1_proposal), np.array(z2_proposal)

'''show contraction'''

# d = 100
# S = Hypersphere(dim=d)
# z1, z2 = S.random_point(n_samples=2)
# R = np.sqrt(d)
# std = 1/d

# _, std, _ = adap_maximal_reflection_coupling(log_density, 0.23, S, z1, z2, R, d, 10000)

# print(std)

# z1_init, z2_init = z1.copy(), z2.copy()

# dist_lst_maxref = [S.metric.dist(z1, z2)]
# dist_lst_maxpar = [S.metric.dist(z1, z2)]
# dist_lst_con_noflip = [S.metric.dist(z1, z2)]
# dist_lst_con = [S.metric.dist(z1, z2)]

# acc_lst_maxref = []
# acc_lst_maxpar = []

# with accept reject

# for i in range(1000):
#     z1_proposal,z2_proposal = reflection_coupling(S, z1, z2, std, d)
#     z1, z2, ratio1, ratio2 = accept_reject(log_density, z1, z2, z1_proposal, z2_proposal, R, d)
#     dist_lst_con.append(S.metric.dist(z1, z2))
#     ar1.append(ratio1)
#     ar2.append(ratio2)

# z1, z2 = z1_init.copy(), z2_init.copy()
# for i in range(1000):
#     z1_proposal,z2_proposal = maximal_reflection_coupling(S, z1, z2, std, d)
#     z1, z2, a = accept_reject(log_density, z1, z2, z1_proposal, z2_proposal, R, d)
#     dist_lst_maxref.append(S.metric.dist(z1, z2))
#     acc_lst_maxref.append(a)


# z1, z2 = z1_init.copy(), z2_init.copy()
# for i in range(1000):
#     z1_proposal,z2_proposal = maximal_parallel_coupling(S, z1, z2, std, d)
#     z1, z2, a = accept_reject(log_density, z1, z2, z1_proposal, z2_proposal, R, d)
#     dist_lst_maxpar.append(S.metric.dist(z1, z2))
#     acc_lst_maxpar.append(a)

# z1, z2 = z1_init.copy(), z2_init.copy()
# for i in range(1000):
#     z1_proposal,z2_proposal = parallel_coupling(S, z1, z2, std, d)
#     z1, z2, ratio1, ratio2 = accept_reject(log_density, z1, z2, z1_proposal, z2_proposal, R, d)
#     dist_lst_con_noflip.append(S.metric.dist(z1, z2))
#     ar1.append(ratio1)
#     ar2.append(ratio2)

# plt.plot(dist_lst_con, label='contractive coupling with flip')
# plt.plot(dist_lst_con_noflip, label='contractive coupling without flip')
# plt.plot(dist_lst_maxref, label='maximal reflection coupling')
# plt.plot(dist_lst_maxpar, label='maximal parallel coupling')
# plt.legend()
# plt.show()



# # without accept reject
# z1, z2 = z1_init.copy(), z2_init.copy()
# for i in range(10000):
#     z1,z2 = reflection_coupling(S, z1, z2, std, d)
#     dist_lst_con.append(S.metric.dist(z1, z2))
# z1, z2 = z1_init.copy(), z2_init.copy()
# for i in range(10000):
#     z1,z2 = maximal_reflection_coupling(S, z1, z2, std, d)
#     dist_lst_ref.append(S.metric.dist(z1, z2))
# z1, z2 = z1_init.copy(), z2_init.copy()
# for i in range(10000):
#     z1,z2 = parallel_coupling(S, z1, z2, std, d)
#     dist_lst_con_noflip.append(S.metric.dist(z1, z2))

# plt.plot(dist_lst_con, label='contractive coupling with flip')
# plt.plot(dist_lst_con_noflip, label='contractive coupling without flip')
# plt.plot(dist_lst_ref, label='maximal reflection coupling')
# plt.xlim(0,300)
# plt.xlabel('iterations')
# plt.ylabel('geodesic distance')
# plt.legend()
# plt.show()


'''histogram of meeting times'''

# d = 70
# S = Hypersphere(dim=d)
# R = np.sqrt(d)
# z1, z2 = S.random_point(n_samples=2)
# # acc_rates, std, _ = adap_maximal_reflection_coupling(log_density, 0.4, S, z1, z2, R, d, 10000)

# # print(acc_rates[-1])
# # print(std)
# std = 1/np.sqrt(d)
# meeting_times = []

# acc_lst = []

# for i in range(1000):
#     z1, z2 = S.random_point(n_samples=2)
#     tau = 0
#     while S.metric.dist(z1, z2) > 0:
#         z1_proposal,z2_proposal = maximal_reflection_coupling(S, z1, z2, std, d)
#         z1, z2, a = accept_reject(log_density, z1, z2, z1_proposal, z2_proposal, R, d)
#         tau = tau + 1
#         acc_lst.append(a)
#     print(i)
#     meeting_times.append(tau)

# print(np.mean(meeting_times))

# plt.hist(meeting_times)
# plt.yscale('log')
# plt.show()




'''meeting times vs dimensions'''

def log_density(x, mu=0):
    x = np.asarray(x)
    var = 1 - mu**2
    const = -0.5 * np.log(2 * np.pi * var)
    logp = np.sum(const - 0.5 * (x - mu)**2 / var)
    return logp

# def log_density(x):
#     x = np.atleast_1d(x)
#     d = x.shape[0]
#     return np.sum(t.logpdf(x, df=d + 1))

# def log_density(x):
#     x = np.atleast_1d(x)
#     d = x.shape[0]
#     dist = multivariate_t(loc=np.zeros(d), shape=np.eye(d), df=d+1)
#     return dist.logpdf(x)


# def log_density(x):
#     x = np.asarray(x)
#     d = x.shape[-1]

#     norm_x = np.linalg.norm(x, axis=-1) + 1e-15
    
#     # nu is the order: (dimension / 2) - 1
#     nu = (d / 2.0) - 1.0
#     log_density = (nu * -np.log(norm_x)) + np.log(kv(nu, np.sqrt(2) * norm_x))
    
#     return log_density

# def log_density(x):
#     x = np.asarray(x)
#     d = x.shape[-1]

#     a = (gamma((d+2)/4) / (d * gamma(d/4)))**2
#     return -a * np.linalg.norm(x)**4

dlist = [10, 20, 50, 75, 100, 120]
taulist = []

for d in dlist:

    R = np.sqrt(d)

    S = Hypersphere(dim=d)
    z1, z2 = S.random_point(n_samples=2)

    std = 1/np.sqrt(d)
    # std = 1/(d**0.25)
    # std = 1/d
    meeting_times = []


    for i in range(1000):
        z1, z2 = S.random_point(n_samples=2)
        tau = 0
        while S.metric.dist(z1, z2) > 0:
            z1_proposal,z2_proposal = maximal_reflection_coupling(S, z1, z2, std, d)
            z1, z2, a = accept_reject(log_density, z1, z2, z1_proposal, z2_proposal, R, d)
            tau = tau + 1
        meeting_times.append(tau)

    
    taulist.append(np.mean(meeting_times))
    print(np.mean(meeting_times))
    print(d)

plt.plot(dlist, taulist, 'o-')
# plt.xscale('log')
plt.xlabel('dimensions')
plt.ylabel('meeting times')
plt.show()

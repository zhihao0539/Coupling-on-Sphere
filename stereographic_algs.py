import numpy as np

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


def SRWM_sampler(n_samples, proposal_std, R, initial_x, log_density, c=None):
    d = len(initial_x)
    samples = np.zeros((n_samples, d))
    acc = 0

    if c is None:
        c = np.zeros(d)

    log_pi_x = log_density(initial_x)
    shift_x = initial_x - c

    for i in range(n_samples):
        # Projection x onto sphere
        z = inverse_stereographic_projection(shift_x, R)
        tilde_z = np.random.multivariate_normal(np.zeros(d+1), (proposal_std**2) * np.identity(d+1))
        dz = tilde_z - (z @ tilde_z / np.linalg.norm(z)**2) * z
        hat_z = (z + dz) / np.linalg.norm(z + dz)

        # Proposal
        shift_x_proposal = stereographic_projection(hat_z, R)
        x_proposal = shift_x_proposal + c
        log_pi_y = log_density(x_proposal)

        # Compute acceptance ratio
        log_acceptance_ratio = log_pi_y - log_pi_x + d * (np.log(R**2 + np.linalg.norm(shift_x_proposal)**2) - np.log(R**2 + np.linalg.norm(shift_x)**2))
        log_acceptance_ratio = min(0, log_acceptance_ratio)

        # Accept or reject
        if np.log(np.random.rand()) < log_acceptance_ratio:
            shift_x = shift_x_proposal
            log_pi_x = log_pi_y
            acc = acc + 1

        samples[i] = shift_x + c

    acceptance_rate = acc / n_samples

    return samples, acceptance_rate


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

    unirand = np.random.rand()
    if np.log(unirand) <= log_ratio1:
        z1 = z1_proposal
    if np.log(unirand) <= log_ratio2:
        z2 = z2_proposal

    return z1, z2


def MRCoupling_sampler(n_samples, proposal_std, R, S, initial_z1, initial_z2, log_density, d):
    samples1 = np.zeros((n_samples, d+1))
    samples2 = np.zeros((n_samples, d+1))

    z1 = initial_z1
    z2 = initial_z2

    for i in range(n_samples):
        z1_proposal,z2_proposal = maximal_reflection_coupling(S, z1, z2, proposal_std, d)
        z1, z2 = accept_reject(log_density, z1, z2, z1_proposal, z2_proposal, R, d)

        samples1[i] = z1
        samples2[i] = z2

    return samples1, samples2


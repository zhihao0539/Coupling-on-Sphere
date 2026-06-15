import numpy as np
import stereographic_algs


def _check_observer(o, tol=1e-12):
    """
    Check that o is a valid observer in R^{d+1} satisfying
        ||o - e_{d+1}|| <= 1.
    """
    o = np.asarray(o, dtype=float)

    if o.ndim != 1:
        raise ValueError("o must be a one-dimensional array of shape (d+1,).")

    d = len(o) - 1
    if d < 1:
        raise ValueError("o must have length at least 2.")

    e = np.zeros(d + 1)
    e[-1] = 1.0

    if np.linalg.norm(o - e) > 1.0 + tol:
        raise ValueError("Observer must satisfy ||o - e_{d+1}|| <= 1.")

    if abs(o[-1]) <= tol:
        raise ValueError(
            "o_{d+1} is too close to 0. "
            "The Jacobian formula contains division by o_{d+1}."
        )

    return o, d


def _as_batch(x, dim):
    """
    Convert input of shape (dim,) or (n, dim) to shape (n, dim).
    Return whether the original input was one-dimensional.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        if x.shape[0] != dim:
            raise ValueError(f"Expected shape ({dim},), got {x.shape}.")
        return x[None, :], True

    if x.ndim == 2:
        if x.shape[1] != dim:
            raise ValueError(f"Expected shape (n, {dim}), got {x.shape}.")
        return x, False

    raise ValueError("Input must have shape (dim,) or (n, dim).")


def _M_xo(x, R, o, clip_discriminant=True):
    """
    Compute M_{x,o} appearing in the inverse sub-Cauchy projection.
    """
    if R <= 0:
        raise ValueError("R must be positive.")

    o, d = _check_observer(o)
    x, was_1d = _as_batch(x, d)

    o_head = o[:-1]
    o_last = o[-1]

    u = x / R - o_head

    A = np.sum(u**2, axis=1) + o_last**2

    B = (
        np.sum((x / R) * o_head - o_head**2, axis=1)
        - o_last * (o_last - 1.0)
    )

    C = np.dot(o, o) - 2.0 * o_last

    discriminant = B**2 - A * C

    # Avoid tiny negative values caused by floating-point error.
    if clip_discriminant:
        discriminant = np.maximum(discriminant, 0.0)

    M = (-B + np.sqrt(discriminant)) / A

    return M[0] if was_1d else M


def sub_cauchy_projection(z, R, o, check_bright_side=True, tol=1e-12):
    """
    Sub-Cauchy projection SCP_o(z).

    Parameters
    ----------
    z : array_like, shape (d+1,) or (n, d+1)
        Point(s) on the sphere centered at e_{d+1}.
    R : float
        Positive scaling parameter.
    o : array_like, shape (d+1,)
        Observer satisfying ||o - e_{d+1}|| <= 1.
    check_bright_side : bool
        If True, checks z_{d+1} < o_{d+1}.

    Returns
    -------
    x : ndarray, shape (d,) or (n, d)
        Point(s) in R^d.
    """
    if R <= 0:
        raise ValueError("R must be positive.")

    o, d = _check_observer(o)
    z, was_1d = _as_batch(z, d + 1)

    o_head = o[:-1]
    o_last = o[-1]

    z_head = z[:, :-1]
    z_last = z[:, -1]

    if check_bright_side and np.any(z_last >= o_last - tol):
        raise ValueError("z must satisfy z_{d+1} < o_{d+1}, i.e. lie on the bright side.")

    denom = o_last - z_last

    if np.any(np.abs(denom) <= tol):
        raise ValueError("Projection denominator o_{d+1} - z_{d+1} is too close to zero.")

    x = R * (o_last * z_head - z_last[:, None] * o_head) / denom[:, None]

    return x[0] if was_1d else x


def inverse_sub_cauchy_projection(x, R, o):
    """
    Inverse sub-Cauchy projection SCP_o^{-1}(x).

    Parameters
    ----------
    x : array_like, shape (d,) or (n, d)
        Point(s) in R^d.
    R : float
        Positive scaling parameter.
    o : array_like, shape (d+1,)
        Observer satisfying ||o - e_{d+1}|| <= 1.

    Returns
    -------
    z : ndarray, shape (d+1,) or (n, d+1)
        Point(s) on the sphere centered at e_{d+1}.
    """
    if R <= 0:
        raise ValueError("R must be positive.")

    o, d = _check_observer(o)
    x, was_1d = _as_batch(x, d)

    o_head = o[:-1]
    o_last = o[-1]

    M = _M_xo(x, R, o)
    M_col = np.asarray(M)[:, None] if np.ndim(M) > 0 else np.array([[M]])

    z_head = M_col * (x / R) + (1.0 - M_col) * o_head
    z_last = (1.0 - np.asarray(M)) * o_last

    z = np.concatenate([z_head, z_last.reshape(-1, 1)], axis=1)

    return z[0] if was_1d else z


def jacobian(x, R, o, log=False):
    """
    Jacobian J_o(x) of the sub-Cauchy projection.

    Parameters
    ----------
    x : array_like, shape (d,) or (n, d)
        Point(s) in R^d.
    R : float
        Positive scaling parameter.
    o : array_like, shape (d+1,)
        Observer satisfying ||o - e_{d+1}|| <= 1.
    log : bool
        If True, return log J_o(x), useful for MCMC log densities.

    Returns
    -------
    J : float or ndarray
        Jacobian value(s), or log-Jacobian value(s) if log=True.
    """
    if R <= 0:
        raise ValueError("R must be positive.")

    o, d = _check_observer(o)
    x, was_1d = _as_batch(x, d)

    o_head = o[:-1]
    o_last = o[-1]

    M = _M_xo(x, R, o)

    u = x / R - o_head

    sum1 = np.sum(u**2, axis=1)

    sum2 = np.sum((x / R) * o_head - o_head**2, axis=1)

    numerator = (
        M * sum1
        + sum2
        + o_last
        - o_last**2 * (1.0 - M)
    )

    denominator = (M**d) * o_last

    if log:
        J = d * np.log(R) + np.log(numerator) - np.log(denominator)
    else:
        J = (R**d) * numerator / denominator

    return J[0] if was_1d else J

def is_bright_side(z, o, tol=1e-12):
    """
    Bright side condition for centered sphere.

    Since z_tilde = z + e_{d+1}, the shifted-sphere condition
        z_tilde[-1] < o[-1]
    becomes
        z[-1] < o[-1] - 1.
    """
    z = np.asarray(z, dtype=float)
    o = np.asarray(o, dtype=float)

    return z[-1] < o[-1] - 1.0 + tol


def stepping_out(z, z_prop, o, tol=1e-12, check_result=True):
    """
    Stepping-out map S_z(z_prop) for SCS on the centered sphere.

    Parameters
    ----------
    z : ndarray, shape (d+1,)
        Current state on the centered unit sphere S^d.
        It should be in the bright side BS^d.

    z_prop : ndarray, shape (d+1,)
        Raw proposal on S^d. Usually this lies in the dark side.

    o : ndarray, shape (d+1,)
        Observer in the shifted sphere convention.

    tol : float
        Numerical tolerance.

    check_result : bool
        If True, check that the stepped-out point is in the bright side.

    Returns
    -------
    z_hat : ndarray, shape (d+1,)
        Relocated proposal in the bright side BS^d.
    """
    z = np.asarray(z, dtype=float)
    z_prop = np.asarray(z_prop, dtype=float)
    o = np.asarray(o, dtype=float)

    if z.shape != z_prop.shape:
        raise ValueError("z and z_prop must have the same shape.")

    if o.shape != z.shape:
        raise ValueError("o must have the same shape as z.")

    # Normalize defensively
    z = z / np.linalg.norm(z)
    z_prop = z_prop / np.linalg.norm(z_prop)

    if not is_bright_side(z, o, tol=tol):
        raise ValueError("Current state z must be in the bright side.")

    # If z_prop is already bright, no stepping-out is needed
    if is_bright_side(z_prop, o, tol=tol):
        return z_prop.copy()

    # Boundary latitude for the centered sphere
    lat = o[-1] - 1.0

    inner = np.clip(np.dot(z, z_prop), -1.0, 1.0)

    alpha = np.arccos(inner)

    if alpha <= tol:
        raise ValueError("z and z_prop are too close; stepping-out is not well-defined.")

    u = z_prop - inner * z
    u_norm = np.linalg.norm(u)

    if u_norm <= tol:
        raise ValueError("The tangent direction is numerically degenerate.")

    u = u / u_norm

    radius = np.sqrt(z[-1]**2 + u[-1]**2)

    if radius <= tol:
        raise ValueError("Degenerate latitude geometry: radius is too small.")

    gamma_arg = np.clip(lat / radius, -1.0, 1.0)
    phi_arg = np.clip(z[-1] / radius, -1.0, 1.0)

    gamma = np.arccos(gamma_arg)
    phi = np.arccos(phi_arg)

    K = int(np.floor((phi + gamma) / alpha) + 1)

    theta = K * alpha

    z_hat = z * np.cos(theta) + u * np.sin(theta)
    z_hat = z_hat / np.linalg.norm(z_hat)

    if check_result and not is_bright_side(z_hat, o, tol=10 * tol):
        raise RuntimeError(
            "Stepping-out failed: output is not in the bright side. "
            f"z_hat[-1]={z_hat[-1]}, boundary={lat}, K={K}, alpha={alpha}"
        )

    return z_hat

def log_sc_density_centered(z, R, o, log_density):
    """
    Log transformed target density on the centered bright side BS^d.

    Parameters
    ----------
    z : ndarray, shape (d+1,)
        Point on the centered sphere S^d, assumed to be in BS^d.

    R : float
        Projection parameter.

    o : ndarray, shape (d+1,)
        Observer.

    log_density : callable
        Original target log-density on R^d.
        It should take x of shape (d,) and return log pi(x).

    Returns
    -------
    log_pi_sc : float
        Log transformed density at z + e_{d+1}.
    """
    z = np.asarray(z, dtype=float)
    o = np.asarray(o, dtype=float)

    d = len(z) - 1

    e = np.zeros(d + 1)
    e[-1] = 1.0

    # Move from centered sphere S^d to shifted sphere \widetilde{S}^d
    z_tilde = z + e

    x = sub_cauchy_projection(z_tilde, R, o)

    return log_density(x) + jacobian(x, R, o, log=True)


def scs_step(z, h, R, o, log_density, rng=None, tol=1e-12):
    """
    One-step Sub-Cauchy Sampler transition.

    Parameters
    ----------
    z : ndarray, shape (d+1,)
        Current state on the centered sphere S^d.
        Must lie in BS^d.

    h : float
        Proposal step size.

    R : float
        Projection parameter in the sub-Cauchy projection.

    o : ndarray, shape (d+1,)
        Observer.

    log_density : callable
        Original target log-density on R^d.

    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    z_next : ndarray, shape (d+1,)
        Next state on BS^d.

    accepted : bool
        Whether the proposal was accepted.

    stepped_out : bool
        Whether the raw proposal landed in the dark side and was relocated.
    """
    if rng is None:
        rng = np.random.default_rng()

    if h <= 0:
        raise ValueError("h must be positive.")

    z = np.asarray(z, dtype=float)
    o = np.asarray(o, dtype=float)

    if z.shape != o.shape:
        raise ValueError("z and o must have the same shape.")

    z = z / np.linalg.norm(z)

    if not is_bright_side(z, o, tol=tol):
        raise ValueError("Initial state z must be in the bright side.")

    d_plus_1 = len(z)

    # --------------------------------------------------
    # Random walk in tangent space
    # --------------------------------------------------
    eps_tilde = rng.normal(loc=0.0, scale=h, size=d_plus_1)

    # Project Gaussian noise onto tangent space T_z S^d
    eps = eps_tilde - np.dot(z, eps_tilde) * z

    # Project back to the sphere
    z_prop = z + eps
    z_prop = z_prop / np.linalg.norm(z_prop)

    # --------------------------------------------------
    # Stepping-out if proposal is in the dark side
    # --------------------------------------------------
    if is_bright_side(z_prop, o, tol=tol):
        z_hat = z_prop
        stepped_out = False
    else:
        z_hat = stepping_out(z, z_prop, o, tol=tol)
        stepped_out = True

    # Defensive check
    if not is_bright_side(z_hat, o, tol=tol):
        raise RuntimeError("Stepping-out failed: relocated proposal is not in the bright side.")

    # --------------------------------------------------
    # Metropolis accept/reject step
    # --------------------------------------------------
    log_pi_current = log_sc_density_centered(z, R, o, log_density)
    log_pi_proposal = log_sc_density_centered(z_hat, R, o, log_density)

    log_accept_ratio = log_pi_proposal - log_pi_current

    if np.log(rng.uniform()) <= min(0.0, log_accept_ratio):
        return z_hat, True, stepped_out
    else:
        return z, False, stepped_out


def scs_accept_reject_coupled(
    log_density,
    z1,
    z2,
    z1_proposal,
    z2_proposal,
    R,
    o,
):
    """
    Coupled Metropolis accept/reject step for SCS.

    Uses the same uniform random variable for both chains.
    """
    log_pi_z1 = log_sc_density_centered(z1, R, o, log_density)
    log_pi_z2 = log_sc_density_centered(z2, R, o, log_density)

    log_pi_z1_prop = log_sc_density_centered(z1_proposal, R, o, log_density)
    log_pi_z2_prop = log_sc_density_centered(z2_proposal, R, o, log_density)

    log_alpha1 = min(0.0, log_pi_z1_prop - log_pi_z1)
    log_alpha2 = min(0.0, log_pi_z2_prop - log_pi_z2)

    log_u = np.log(np.random.uniform())

    accept1 = log_u <= log_alpha1
    accept2 = log_u <= log_alpha2

    if accept1:
        z1_next = z1_proposal
    else:
        z1_next = z1

    if accept2:
        z2_next = z2_proposal
    else:
        z2_next = z2

    return z1_next, z2_next, accept1, accept2


def SCS_MRCoupling_sampler(
    n_samples,
    proposal_std,
    R,
    S,
    o,
    initial_z1,
    initial_z2,
    log_density,
    d,
    tol=1e-12,
):
    """
    Coupling of the Sub-Cauchy Sampler.

    Parameters
    ----------
    n_samples : int
        Number of samples.

    proposal_std : float
        Step size h.

    R : float
        Sub-Cauchy projection parameter.

    S :
        Sphere object, e.g. geomstats Hypersphere(dim=d).

    o : ndarray, shape (d+1,)
        Observer.

    initial_z1, initial_z2 : ndarray, shape (d+1,)
        Initial points on the centered sphere S^d, assumed to be in BS^d.

    log_density : callable
        Original target log-density on R^d.

    d : int
        Dimension of the Euclidean target.

    Returns
    -------
    samples1, samples2 : ndarray
        Coupled SCS samples on the centered bright side BS^d.

    acc_rate1, acc_rate2 : float
        Acceptance rates of the two chains.

    stepout_rate1, stepout_rate2 : float
        Frequencies with which the raw proposals entered the dark side.
    """
    samples1 = np.zeros((n_samples, d + 1))
    samples2 = np.zeros((n_samples, d + 1))

    z1 = np.asarray(initial_z1, dtype=float)
    z2 = np.asarray(initial_z2, dtype=float)

    z1 = z1 / np.linalg.norm(z1)
    z2 = z2 / np.linalg.norm(z2)

    if not is_bright_side(z1, o, tol=tol):
        raise ValueError("initial_z1 must lie in the bright side.")

    if not is_bright_side(z2, o, tol=tol):
        raise ValueError("initial_z2 must lie in the bright side.")

    a1 = 0
    a2 = 0

    n_stepout1 = 0
    n_stepout2 = 0

    for i in range(n_samples):

        # --------------------------------------------------
        # 1. Maximal-reflection coupling of proposal kernels
        # --------------------------------------------------
        z1_raw, z2_raw = stereographic_algs.maximal_reflection_coupling(
            S,
            z1,
            z2,
            proposal_std,
            d,
        )

        z1_raw = z1_raw / np.linalg.norm(z1_raw)
        z2_raw = z2_raw / np.linalg.norm(z2_raw)

        # --------------------------------------------------
        # 2. Stepping-out if raw proposals are in the dark side
        # --------------------------------------------------
        if is_bright_side(z1_raw, o, tol=tol):
            z1_proposal = z1_raw
            stepped_out1 = False
        else:
            z1_proposal = stepping_out(z1, z1_raw, o, tol=tol)
            stepped_out1 = True

        if is_bright_side(z2_raw, o, tol=tol):
            z2_proposal = z2_raw
            stepped_out2 = False
        else:
            z2_proposal = stepping_out(z2, z2_raw, o, tol=tol)
            stepped_out2 = True

        n_stepout1 += int(stepped_out1)
        n_stepout2 += int(stepped_out2)

        # Defensive checks
        if not is_bright_side(z1_proposal, o, tol=tol):
            raise RuntimeError("z1 proposal is not in the bright side after stepping-out.")

        if not is_bright_side(z2_proposal, o, tol=tol):
            raise RuntimeError("z2 proposal is not in the bright side after stepping-out.")

        # --------------------------------------------------
        # 3. Coupled Metropolis accept/reject step
        # --------------------------------------------------
        z1, z2, accept1, accept2 = scs_accept_reject_coupled(
            log_density=log_density,
            z1=z1,
            z2=z2,
            z1_proposal=z1_proposal,
            z2_proposal=z2_proposal,
            R=R,
            o=o,
        )

        a1 += int(accept1)
        a2 += int(accept2)

        samples1[i] = z1
        samples2[i] = z2

    acc_rate1 = a1 / n_samples
    acc_rate2 = a2 / n_samples

    stepout_rate1 = n_stepout1 / n_samples
    stepout_rate2 = n_stepout2 / n_samples

    return samples1, samples2, acc_rate1, acc_rate2, stepout_rate1, stepout_rate2
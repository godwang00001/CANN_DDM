import brainpy as bp
import brainpy.math as bm
import numpy as np

from rate_model_core.math import edge_states, sigmoid

EDGE_KERNEL_BASE_EXC_SIGMA = 1.0
EDGE_KERNEL_INHIBITION_WIDTH_RATIO = 1.2
EDGE_TO_BUMP_KERNEL_MODE = 'simple'
EDGE_TO_BUMP_KERNEL_SIGMA = 0.5
EDGE_TO_BUMP_KERNEL_SHIFT = 1.0
EDGE_TO_BUMP_KERNEL_GAIN = 100.
BUMP_TO_EDGE_KERNEL_MODE = 'simple'
BUMP_TO_EDGE_KERNEL_SIGMA = 1.0
BUMP_TO_EDGE_KERNEL_GAIN = 1.0


def DoG_kernel(num, sigma_E, s, offset=0):
    assert offset < bm.pi / 2 and offset > -bm.pi / 2, "Offset must be between -pi/2 and pi/2."
    theta = bm.linspace(-bm.pi / 2, bm.pi / 2, num)
    sigma_I = s * sigma_E
    G_E = 1 / (bm.sqrt(2 * bm.pi) * sigma_E) * bm.exp(-0.5 * ((theta - offset) / sigma_E) ** 2)
    G_I = 1 / (bm.sqrt(2 * bm.pi) * sigma_I) * bm.exp(-0.5 * ((theta - offset) / sigma_I) ** 2)
    return G_E - 1 / s * G_I


def Gaussian_kernel(num, sigma_E, offset=0):
    assert offset < bm.pi / 2 and offset > -bm.pi / 2, "Offset must be between -pi/2 and pi/2."
    theta = bm.linspace(-bm.pi / 2, bm.pi / 2, num)
    return 1 / (bm.sqrt(2 * bm.pi) * sigma_E) * bm.exp(-0.5 * ((theta - offset) / sigma_E) ** 2)


def _reflect_indices(indices, num):
    indices = np.asarray(indices, dtype=int).copy()
    if num <= 1:
        return np.zeros_like(indices)
    while True:
        below = indices < 0
        above = indices >= num
        if not (np.any(below) or np.any(above)):
            break
        indices[below] = -indices[below] - 1
        indices[above] = 2 * num - 1 - indices[above]
    return indices


def make_conn_mat_from_kernel(num, kernel, normed=False, boundary_mode='truncate'):
    """
    Build an interaction matrix from a 1D kernel sampled on the neuron grid.
    """
    Nk = len(kernel)
    assert Nk % 2 == 1, "Kernel length must be odd."
    assert Nk <= num, "Kernel length must be less than or equal to N."
    if boundary_mode not in ('truncate', 'reflect'):
        raise ValueError("boundary_mode must be one of: truncate, reflect")

    m = (Nk - 1) // 2
    conn = np.zeros((num, num), dtype=float)
    rows = np.arange(num, dtype=int)
    offsets = np.arange(-m, m + 1, dtype=int)
    kernel_np = np.asarray(kernel, dtype=float)

    for offset, weight in zip(offsets, kernel_np):
        cols = rows + offset
        if boundary_mode == 'truncate':
            valid = (cols >= 0) & (cols < num)
            conn[rows[valid], cols[valid]] = weight
        else:
            cols = _reflect_indices(cols, num)
            conn[rows, cols] += weight

    if normed:
        row_sums = conn.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        conn = conn / row_sums
    return bm.asarray(conn)


def make_bump_conn_mat(
    num,
    sigma,
    beta,
    geometry,
    offset=0,
    kernel_mode='gaussian_cann',
    kernel_gain=2.0,
    kernel_sigma=None,
    kernel_normed=True,
):
    """
    Build the bump-population connectivity matrix.
    """
    clamp_frac = geometry.clamp_frac
    num_interior = int(num * (1 - clamp_frac))
    if num_interior % 2 == 0:
        num_interior = num_interior - 1
    kernel_sigma = sigma if kernel_sigma is None else kernel_sigma
    kernel = Gaussian_kernel(num_interior, kernel_sigma, offset=offset)
    if kernel_mode == 'gaussian_cann':
        return kernel_gain * make_conn_mat_from_kernel(num, kernel, normed=kernel_normed)

    raise ValueError(
        f"Unknown bump kernel_mode '{kernel_mode}'. Supported modes: gaussian_cann."
    )


def make_edge_to_bump_kernel(
    kernel_sigma=EDGE_TO_BUMP_KERNEL_SIGMA,
    kernel_shift=EDGE_TO_BUMP_KERNEL_SHIFT,
    support_radius=None,
    normalize_first_moment=True,
):
    """
    Build a smooth short-range asymmetric W_EB kernel on the discrete grid.

    The kernel is a difference of shifted Gaussians. Its discrete zeroth moment
    is enforced to vanish exactly after sampling. The first moment can be
    normalized so the overall strength is controlled by a separate gain.
    """
    if kernel_sigma <= 0:
        raise ValueError("kernel_sigma must be positive")
    if kernel_shift <= 0:
        raise ValueError("kernel_shift must be positive")

    if support_radius is None:
        support_radius = max(2, int(float(bm.ceil(kernel_shift + 4 * kernel_sigma))))

    offsets = bm.arange(-support_radius, support_radius + 1, dtype=float)
    G_left = bm.exp(-0.5 * ((offsets + kernel_shift) / kernel_sigma) ** 2)
    G_right = bm.exp(-0.5 * ((offsets - kernel_shift) / kernel_sigma) ** 2)
    kernel = G_left - G_right

    # Enforce vanishing discrete zeroth moment exactly.
    kernel = kernel - bm.mean(kernel)

    if normalize_first_moment:
        first_moment = bm.sum(offsets * kernel)
        if bm.abs(first_moment) < 1e-12:
            raise ValueError("edge-to-bump kernel first moment vanished after discretization")
        kernel = kernel / bm.abs(first_moment)

    return kernel


def make_edge_to_bump_conn_mat(
    num,
    kernel_mode=EDGE_TO_BUMP_KERNEL_MODE,
    kernel_sigma=EDGE_TO_BUMP_KERNEL_SIGMA,
    kernel_shift=EDGE_TO_BUMP_KERNEL_SHIFT,
    kernel_gain=EDGE_TO_BUMP_KERNEL_GAIN,
    support_radius=None,
):
    """
    Build the explicit W_EB matrix from a short-range asymmetric kernel.
    """
    if kernel_mode == 'simple':
        if num < 2:
            raise ValueError("num must be at least 2 for W_EB")

        W_EB = bm.zeros((num, num))

        # One-sided boundary rows matching the original local operator.
        W_EB[0, 0] = 1.0
        W_EB[0, 1] = -1.0
        W_EB[-1, -2] = 1.0
        W_EB[-1, -1] = -1.0

        # Interior rows use the original centered three-point stencil.
        if num > 2:
            idx = bm.arange(1, num - 1)
            W_EB[idx, idx - 1] = 0.5
            W_EB[idx, idx + 1] = -0.5

        return kernel_gain * W_EB
    elif kernel_mode == 'smooth_asymmetric':
        kernel = make_edge_to_bump_kernel(
            kernel_sigma=kernel_sigma,
            kernel_shift=kernel_shift,
            support_radius=support_radius,
            normalize_first_moment=True,
        )
    else:
        raise ValueError(
            f"Unknown eb_kernel_mode '{kernel_mode}'. Supported modes: simple, smooth_asymmetric."
        )

    return kernel_gain * make_conn_mat_from_kernel(num, kernel, normed=False)


def make_bump_to_edge_kernel(
    kernel_sigma=BUMP_TO_EDGE_KERNEL_SIGMA,
    support_radius=None,
    normalize_sum=True,
):
    """
    Build a smooth short-range symmetric W_BE kernel on the discrete grid.
    """
    if kernel_sigma <= 0:
        raise ValueError("kernel_sigma must be positive")

    if support_radius is None:
        support_radius = max(1, int(float(bm.ceil(4 * kernel_sigma))))

    offsets = bm.arange(-support_radius, support_radius + 1, dtype=float)
    kernel = bm.exp(-0.5 * (offsets / kernel_sigma) ** 2)

    if normalize_sum:
        kernel_sum = bm.sum(kernel)
        if bm.abs(kernel_sum) < 1e-12:
            raise ValueError("bump-to-edge kernel sum vanished after discretization")
        kernel = kernel / kernel_sum

    return kernel


def make_bump_to_edge_conn_mat(
    num,
    kernel_mode=BUMP_TO_EDGE_KERNEL_MODE,
    kernel_sigma=BUMP_TO_EDGE_KERNEL_SIGMA,
    kernel_gain=BUMP_TO_EDGE_KERNEL_GAIN,
    support_radius=None,
):
    """
    Build the explicit W_BE matrix from a short-range symmetric kernel.
    """
    if kernel_mode == 'simple':
        return kernel_gain * bm.eye(num)
    if kernel_mode == 'smooth_symmetric':
        kernel = make_bump_to_edge_kernel(
            kernel_sigma=kernel_sigma,
            support_radius=support_radius,
            normalize_sum=True,
        )
        return kernel_gain * make_conn_mat_from_kernel(num, kernel, normed=False)

    raise ValueError(
        f"Unknown be_kernel_mode '{kernel_mode}'. Supported modes: simple, smooth_symmetric."
    )


def make_edge_conn_mat(
    num,
    gamma,
    geometry,
    edge_type='tanh',
    offset=0,
    alpha=1.0,
    kernel_clamp_frac=None,
):
    """
    Build the edge-population connectivity matrix.

    The recurrent DoG kernel currently uses fixed implementation constants
    `EDGE_KERNEL_BASE_EXC_SIGMA` and `EDGE_KERNEL_INHIBITION_WIDTH_RATIO`.
    These set the numerical kernel shape used to construct `J_EE`; they are
    not the same as the model's public edge-profile/readout parameter `gamma`.
    Any automatic search over `offset` is handled by the caller; this function
    remains a pure builder for a single explicit `offset` value.
    """
    clamp_frac = geometry.clamp_frac if kernel_clamp_frac is None else float(kernel_clamp_frac)
    center_idx = num // 2

    gamma_effective = 4 * gamma / bm.exp(1) if edge_type == 'tanh' else gamma

    def _base_kernel_matrix(local_clamp_frac, local_offset):
        num_interior = int(num * (1 - local_clamp_frac))
        if num_interior % 2 == 0:
            num_interior = num_interior - 1
        kernel = DoG_kernel(
            num_interior,
            EDGE_KERNEL_BASE_EXC_SIGMA,
            EDGE_KERNEL_INHIBITION_WIDTH_RATIO,
            offset=local_offset,
        )
        return make_conn_mat_from_kernel(num, kernel, normed=True, boundary_mode='reflect')

    def _match_linear_drive(J_EE_local):
        theta = bm.linspace(geometry.theta_min, geometry.theta_max, num)
        real_r = edge_states(num, gamma, geometry=geometry, edge_type=edge_type, center_pos=0.0)
        U0 = J_EE_local @ real_r
        th = theta[center_idx - 10:center_idx + 10]
        y = U0[center_idx - 10:center_idx + 10]
        A, _ = bm.polyfit(th, y, deg=1)
        return -gamma_effective / A

    J_EE = _base_kernel_matrix(clamp_frac, offset)
    J0 = _match_linear_drive(J_EE)
    beta = -J0 / 2
    return J0 * J_EE, beta

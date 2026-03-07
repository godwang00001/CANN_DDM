import numpy as np
import matplotlib.pyplot as plt

def make_edge_conn_mat(num, sigma, alpha, clamp_frac, eps=1e-3, theta0 = 0):
    """
    Construct a Toeplitz Mexican-hat connectivity matrix W such that
    a sigmoid-shaped edge profile is a stable fixed point.

    Parameters
    ----------
    num : int
        Number of neurons.
    sigma : float
        Slope of the target edge (controls sharpness).
    alpha : float
        Gain of the sigmoid nonlinearity.
    clamp_frac : float in [0, 1]
        Fraction of neurons at each end used as clamp neurons.
        Mapped to theta_2 = (pi/2) * (1 - clamp_frac).
    eps : float
        Error tolerance and saturation threshold.

    Logic (high level)
    ------------------
    1. Define target edge r_target(theta) = sigmoid(gamma * theta).
    2. Automatically determine interior region from eps-crossing.
    3. Construct and scale a Mexican-hat kernel to match linear drive in interior.
    4. Apply clamp inputs near boundaries.
    5. Validate clamp extent and roll back if boundary error exceeds eps.
    """

    N = int(num)

    # -----------------------------
    # Helper functions
    # -----------------------------
    def sigmoid(x):
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))

    def logit(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return np.log(p / (1 - p))

    # -----------------------------
    # Neural coordinate
    # -----------------------------
    k = np.arange(N)
    theta = -np.pi / 2 + (k / N) * np.pi

    # -----------------------------
    # Target edge profile
    # -----------------------------
    real_r = sigmoid(sigma * (theta - theta0))

    # -----------------------------
    # Automatically determine theta_1 (interior boundary)
    # -----------------------------
    interior_mask = (real_r > eps) & (real_r < 1 - eps)
    if interior_mask.sum() < 10:
        raise ValueError("Interior region too small after automatic detection.")

    theta_1 = np.max(np.abs(theta[interior_mask]))

    # -----------------------------
    # Mexican-hat kernel (DoG)
    # -----------------------------
    sigma_E = 0.2
    sigma_I = 0.4
    J_E = 0.4
    J_I = 0.2

    dtheta = theta[:, None] - theta[None, :]
    G_E = np.exp(-0.5 * (dtheta / sigma_E) ** 2)
    G_I = np.exp(-0.5 * (dtheta / sigma_I) ** 2)
    W0 = J_E * G_E - J_I * G_I

    # Remove DC component
    W0 = W0 - W0.mean(axis=1, keepdims=True)

    # -----------------------------
    # Match linear drive in interior
    # -----------------------------
    U0 = W0 @ real_r
    th = theta[interior_mask]
    y = U0[interior_mask]

    A, B = np.polyfit(th, y, deg=1)
    if A < 0:
        raise ValueError("Interior regression slope is negative. Please adjust sigma_E/sigma_I or J_E/J_I.")

    s = sigma / (alpha * A)
    conn_mat = s * W0

    beta = -alpha * (s * B)

    # -----------------------------
    # Clamp region defined by fraction
    # -----------------------------
    if not (0.0 <= clamp_frac <= 1.0):
        raise ValueError("clamp_frac must be in [0, 1].")

    theta_2 = (np.pi / 2) * (1 - clamp_frac)

    r_des = np.full(N, np.nan)
    r_des[theta <= -theta_2] = eps
    r_des[theta >= +theta_2] = 1.0 - eps

    z_des = np.zeros(N)
    z_des[theta <= -theta_2] = logit(eps)
    z_des[theta >= +theta_2] = logit(1.0 - eps)

    drive = conn_mat @ real_r
    I_clamp = np.zeros(N)
    I_clamp[theta <= -theta_2] = (z_des[theta <= -theta_2] - beta) / alpha - drive[theta <= -theta_2]
    I_clamp[theta >= +theta_2] = (z_des[theta >= +theta_2] - beta) / alpha - drive[theta >= +theta_2]

    # -----------------------------
    # Simulate dynamics
    # -----------------------------
    r = np.clip(real_r + 0.02 * np.random.randn(N), eps, 1 - eps)
    dt = 0.2
    tol = 1e-8

    for _ in range(5000):
        u = conn_mat @ r + I_clamp
        r_new = sigmoid(alpha * u + beta)

        r_new[theta <= -theta_2] = eps
        r_new[theta >= +theta_2] = 1.0 - eps

        if np.max(np.abs(r_new - r)) < tol:
            break
        r = (1 - dt) * r + dt * r_new

    pred_r = r.copy()

    # -----------------------------
    # Validate clamp extent and roll back
    # -----------------------------
    error = np.abs(pred_r - real_r)
    valid_mask = (theta > theta_1) & (error <= eps)

    if not valid_mask.any():
        raise ValueError("No valid clamp boundary satisfies error constraint.")

    theta2_star = np.max(theta[valid_mask])
    if theta2_star < theta_2:
        print(f"[clamp rollback] theta_2 reduced from {theta_2:.3f} to {theta2_star:.3f}")
        theta_2 = theta2_star

    # meta = dict(
    #     theta_1=float(theta_1),
    #     theta_2=float(theta_2),
    #     clamp_frac=float(clamp_frac),
    #     beta_used=float(beta_used),
    #     scale_s=float(s),
    # )

    return conn_mat, beta


def make_bump_conn_mat(num, sigma_B, alpha, clamp_frac, eps=1e-3):
    """
    Construct a *circular* (ring / circulant) connectivity matrix W using
    a classic "positive Gaussian - constant" kernel:
        W(Δθ) = J * exp(-Δθ^2/(2*sigma_W^2)) - g

    Parameters
    ----------
    num : int
        Number of neurons on the ring.
    sigma_B : float
        Width of the target Gaussian bump (in theta units).
    alpha : float
        Gain of the sigmoid nonlinearity.
    clamp_frac : float in [0, 1]
        Fraction of neurons at each end (in the linearized theta chart) treated as clamp band.
        Mapped to theta_2 = (pi/2) * (1 - clamp_frac).
    eps : float
        Error tolerance and saturation threshold (used as "0" for clamp).

    Logic (high level)
    ------------------
    1. Target bump r_target(theta) = r0 * exp(-theta^2/(2*sigma_B^2)), clipped to (eps,1-eps).
    2. Compute interior boundary theta_1 via eps-crossing of r_target.
    3. Build circular kernel W0(Δθ)=J*G(Δθ)-g (global inhibition) and fit scale/bias on interior:
           logit(r_target) ≈ alpha*(s*(W0 r_target)) + beta_used
    4. Hard clamp both ends (|theta|>=theta_2) to eps via I_clamp and overwrite during simulation.
    5. Validate theta_2 and roll back to theta_2* if boundary-induced error exceeds eps.
    """

    N = int(num)
    if N < 8:
        raise ValueError("num too small.")
    if not (0.0 <= clamp_frac <= 1.0):
        raise ValueError("clamp_frac must be in [0, 1].")
    if not (0.0 < eps < 0.5):
        raise ValueError("eps must be in (0, 0.5).")

    # -----------------------------
    # Helper functions
    # -----------------------------
    def sigmoid(x):
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))

    def logit(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return np.log(p / (1 - p))

    # -----------------------------
    # Ring coordinate (period = pi)
    # -----------------------------
    period = np.pi
    k = np.arange(N)
    theta = -np.pi / 2 + (k / N) * period  # periodic grid

    # -----------------------------
    # Target bump profile
    # -----------------------------
    r0 = 1.0 - eps
    real_r = r0 * np.exp(-(theta ** 2) / (2.0 * sigma_B ** 2))
    real_r = np.clip(real_r, eps, 1.0 - eps)

    # -----------------------------
    # Automatically determine theta_1 (interior boundary)
    # -----------------------------
    interior_mask = (real_r > eps) & (real_r < 1 - eps)
    if interior_mask.sum() < 10:
        raise ValueError("Interior region too small after automatic detection.")
    theta_1 = np.max(np.abs(theta[interior_mask]))

    # -----------------------------
    # Clamp band via fraction
    # theta_2 = pi/2 * (1 - clamp_frac)
    # -----------------------------
    theta_2 = (np.pi / 2) * (1 - clamp_frac)

    # Check theta_1 not inside clamp band
    if theta_1 >= theta_2:
        raise ValueError(
            f"Interior overlaps clamp band: theta_1={theta_1:.3f} >= theta_2={theta_2:.3f}. "
            "Reduce clamp_frac or adjust sigma_B/eps."
        )

    clamp_mask = np.abs(theta) >= theta_2

    # -----------------------------
    # Positive Gaussian - constant kernel (circulant)
    # -----------------------------
    # Choose sigma_W relative to target bump width (reasonable default)
    sigma_W = max(2 * sigma_B, 0.03)
    J = 1.0

    dtheta = theta[:, None] - theta[None, :]
    dtheta = (dtheta + period / 2) % period - period / 2  # wrap to [-period/2, period/2)

    G = np.exp(-0.5 * (dtheta / sigma_W) ** 2)

    # Global inhibition constant: set relative to mean of Gaussian row
    # This keeps net drive balanced and helps avoid uniform high state.
    g = 5 * G.mean()

    W0 = J * G - g

    # -----------------------------
    # Fit scale and bias on interior:
    # logit(r_target) ≈ alpha*(s*(W0 r_target)) + beta_used
    # -----------------------------
    U0 = W0 @ real_r
    z = logit(real_r)

    U_i = U0[interior_mask]
    z_i = z[interior_mask]

    varU = np.var(U_i)
    if varU < 1e-12:
        raise ValueError("Interior variance of drive is too small; kernel choice failed.")

    covUz = np.mean((U_i - np.mean(U_i)) * (z_i - np.mean(z_i)))
    s = covUz / (alpha * varU)
    beta_used = np.mean(z_i) - alpha * s * np.mean(U_i)

    # Base connectivity (no beta yet)
    conn_mat = s * W0

    # -----------------------------
    # ABSORB beta_used into conn_mat via rank-1 DC term:
    # Want: alpha*(conn_mat' @ real_r) ≈ alpha*(conn_mat @ real_r) + beta_used
    # Add: c * 11^T so that (c*11^T @ real_r) = c*sum(real_r) = beta_used/alpha
    # -----------------------------
    Rsum = float(np.sum(real_r))
    if Rsum < 1e-12:
        raise ValueError("Sum(real_r) too small; cannot absorb beta into W.")
    c = (beta_used / alpha) / Rsum
    conn_mat = conn_mat + c * np.ones((N, N))

    # -----------------------------
    # Clamp input: both ends -> eps (hard clamp)
    # -----------------------------
    drive = conn_mat @ real_r
    z_eps = logit(eps)

    I_clamp = np.zeros(N)
    if clamp_mask.any():
        I_clamp[clamp_mask] = (z_eps - beta_used) / alpha - drive[clamp_mask]

    # -----------------------------
    # Simulate dynamics (with hard clamp overwrite)
    # -----------------------------
    r = np.clip(real_r + 0.02 * np.random.randn(N), eps, 1 - eps)
    dt = 0.2
    tol = 1e-8

    for _ in range(5000):
        u = conn_mat @ r + I_clamp
        r_new = sigmoid(alpha * u)

        # Hard clamp both ends to (approximately) 0
        r_new[clamp_mask] = eps

        if np.max(np.abs(r_new - r)) < tol:
            break
        r = (1 - dt) * r + dt * r_new

    pred_r = r.copy()

    # -----------------------------
    # Validate theta_2 and roll back if necessary
    # theta2_star: largest |theta| (outside theta_1) with error <= eps
    # -----------------------------
    error = np.abs(pred_r - real_r)
    valid_mask = (np.abs(theta) > theta_1) & (error <= eps)

    if not valid_mask.any():
        raise ValueError("No valid clamp boundary satisfies error constraint.")

    theta2_star = np.max(np.abs(theta[valid_mask]))
    if theta2_star < theta_2:
        print(f"[clamp rollback] theta_2 reduced from {theta_2:.3f} to {theta2_star:.3f}")
        theta_2 = theta2_star
        clamp_mask = np.abs(theta) >= theta_2

    # meta = dict(
    #     theta_1=float(theta_1),
    #     theta_2=float(theta_2),
    #     clamp_frac=float(clamp_frac),
    #     beta_used=float(beta_used),
    #     scale_s=float(s),
    #     sigma_B=float(sigma_B),
    #     sigma_W=float(sigma_W),
    #     J=float(J),
    #     g=float(g),
    # )

    return conn_mat
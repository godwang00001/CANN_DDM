import numpy as np
import brainpy.math as bm


def idx_to_pos(idx, geometry):
    idx = bm.asarray(idx)
    if geometry.num_units <= 1:
        return bm.asarray(geometry.theta_min)
    return geometry.theta_min + idx / (geometry.num_units - 1) * (geometry.theta_max - geometry.theta_min)


def pos_to_idx(pos, geometry):
    pos = bm.asarray(pos)
    pos = bm.clip(pos, geometry.theta_min, geometry.theta_max)
    idx = (
        (pos - geometry.theta_min)
        / (geometry.theta_max - geometry.theta_min)
        * (geometry.num_units - 1)
    )
    return idx


def pos_to_evidence(pos, boundary, theta_max, s):
    pos = bm.asarray(pos)
    theta_min = -theta_max
    interval = theta_max - theta_min
    normalization = 1.0 - bm.exp(-s * interval)
    interior = boundary * (1.0 - bm.exp(-s * (pos - theta_min))) / normalization
    return bm.where(
        pos <= theta_min,
        0.0,
        bm.where(pos >= theta_max, boundary, interior),
    )


def evidence_to_pos(evidence, boundary, theta_max, s):
    evidence = bm.asarray(evidence)
    theta_min = -theta_max
    interval = theta_max - theta_min
    normalization = 1.0 - bm.exp(-s * interval)
    clipped = bm.clip(evidence, 0.0, boundary)
    interior = theta_min - bm.log(1.0 - (clipped / boundary) * normalization) / s
    return bm.where(
        evidence <= 0,
        theta_min,
        bm.where(evidence >= boundary, theta_max, interior),
    )


def get_RT(prep_time, hit_boundary_trace):
    true_indices = np.where(hit_boundary_trace)[0]
    if len(true_indices) > 0:
        return true_indices[0] - prep_time
    return None


def generate_cues_input(dur, dt_DDM, p, t_start, seed=None):
    num_steps = int(dur / dt_DDM)
    rng = np.random.default_rng(seed)
    cue_R = rng.binomial(1, p, num_steps)
    cue_R[:int(t_start / dt_DDM)] = 0
    cue_L = np.zeros_like(cue_R)
    cue_L = 1 - cue_R
    cue_L[:int(t_start / dt_DDM)] = 0
    cue_R_all = np.zeros(dur)
    cue_R_all[0::int(dt_DDM)] = cue_R
    cue_L_all = np.zeros(dur)
    cue_L_all[0::int(dt_DDM)] = cue_L
    return cue_L_all, cue_R_all


def generate_click_inputs(dur, lambda_click_L, lambda_click_R, t_start, seed=None):
    if not (0.0 <= float(lambda_click_L) <= 1.0):
        raise ValueError("lambda_click_L must be in [0, 1] for the 1 ms Bernoulli approximation")
    if not (0.0 <= float(lambda_click_R) <= 1.0):
        raise ValueError("lambda_click_R must be in [0, 1] for the 1 ms Bernoulli approximation")

    rng = np.random.default_rng(seed)
    click_L_all = np.zeros(int(dur), dtype=float)
    click_R_all = np.zeros(int(dur), dtype=float)
    active_slice = slice(int(t_start), int(dur))
    active_len = max(int(dur) - int(t_start), 0)
    if active_len > 0:
        click_L_all[active_slice] = rng.binomial(1, float(lambda_click_L), active_len)
        click_R_all[active_slice] = rng.binomial(1, float(lambda_click_R), active_len)
    return click_L_all, click_R_all


def build_discrete_click_drift(click_R_all, click_L_all, t_start, dt_DDM, delta_click_x, drive_x_speed_unit):
    assert len(click_R_all) == len(click_L_all), "click_R_all and click_L_all must have the same length"

    step_ms = int(round(float(dt_DDM)))
    if step_ms <= 0 or not np.isclose(float(dt_DDM), float(step_ms)):
        raise ValueError("dt_DDM must be a positive integer number of milliseconds")
    if float(drive_x_speed_unit) <= 0:
        raise ValueError("drive_x_speed_unit must be positive")

    T = len(click_R_all)
    v_drift_all = np.zeros(T, dtype=float)
    start_idx = int(t_start)
    if start_idx >= T:
        return v_drift_all

    for start in range(start_idx, T, step_ms):
        stop = min(start + step_ms, T)
        window_width = stop - start
        if window_width <= 0:
            continue
        net_clicks = float(np.sum(click_R_all[start:stop] - click_L_all[start:stop]))
        held_value = float(delta_click_x) * net_clicks / (float(drive_x_speed_unit) * float(window_width))
        v_drift_all[start:stop] = held_value
    return v_drift_all


def get_x_traj(t_start, dx, dt_DDM, x0, cue_R_all, cue_L_all, boundary):
    assert len(cue_R_all) == len(cue_L_all), "cue_R_all and cue_L_all must have the same length"
    T = len(cue_R_all)
    x_traj = np.full(T, x0)
    delta = (cue_R_all - cue_L_all) * (dx / dt_DDM)
    delta[:t_start] = 0
    x_cumsum = np.cumsum(delta)
    x_traj = x0 + x_cumsum
    x_traj[:t_start] = x0
    above = x_traj >= boundary
    below = x_traj <= 0
    cross = above | below
    if np.any(cross):
        first_cross = np.argmax(cross)
        if x_traj[first_cross] >= boundary:
            x_traj[first_cross] = boundary
        else:
            x_traj[first_cross] = 0.0
        x_traj[first_cross + 1:] = x_traj[first_cross]
    return x_traj


def get_x_traj_discrete(t_start, delta_click_x, x0, click_R_all, click_L_all, dW_all, noise_scale, boundary):
    assert len(click_R_all) == len(click_L_all), "click_R_all and click_L_all must have the same length"
    assert len(click_R_all) == len(dW_all), "click traces and dW_all must have the same length"

    T = len(click_R_all)
    x_traj = np.full(T, float(x0), dtype=float)
    x_curr = float(x0)
    absorbed = False
    for t in range(T):
        if t < int(t_start):
            x_traj[t] = float(x0)
            continue
        if absorbed:
            x_traj[t] = x_curr
            continue
        x_curr += float(delta_click_x) * float(click_R_all[t] - click_L_all[t])
        x_curr += float(noise_scale) * float(dW_all[t])
        if x_curr >= boundary:
            x_curr = float(boundary)
            absorbed = True
        elif x_curr <= 0.0:
            x_curr = 0.0
            absorbed = True
        x_traj[t] = x_curr
    return x_traj

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
    interior = boundary * bm.exp(s * (pos - theta_max))
    return bm.where(
        pos <= theta_min,
        0.0,
        bm.where(pos >= theta_max, boundary, interior),
    )


def evidence_to_pos(evidence, boundary, theta_max, s):
    evidence = bm.asarray(evidence)
    theta_min = -theta_max
    clipped = bm.clip(evidence, 1e-12, boundary)
    interior = theta_max + bm.log(clipped / boundary) / s
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
        x_traj[first_cross + 1:] = x_traj[first_cross]
    return x_traj

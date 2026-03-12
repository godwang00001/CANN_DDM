import numpy as np
import brainpy.math as bm


def idx_to_pos(idx, geometry):
    return bm.where(
        idx < geometry.k1,
        geometry.theta_min,
        bm.where(
            idx > geometry.k2,
            geometry.theta_max,
            geometry.theta_min
            + (idx - geometry.k1) / (geometry.k2 - geometry.k1) * (geometry.theta_max - geometry.theta_min),
        ),
    )


def pos_to_idx(pos, geometry):
    pos = bm.asarray(pos)
    pos = bm.clip(pos, geometry.theta_min, geometry.theta_max)
    idx = geometry.k1 + (
        (pos - geometry.theta_min)
        / (geometry.theta_max - geometry.theta_min)
        * (geometry.k2 - geometry.k1)
    )
    return idx


def pos_to_evidence(pos, boundary, theta_max, s):
    return boundary * bm.exp(s * (pos - theta_max))


def evidence_to_pos(evidence, boundary, theta_max, s):
    return theta_max + bm.log(evidence / boundary) / s


def get_RT(prep_time, hit_boundary_trace):
    true_indices = np.where(hit_boundary_trace)[0]
    if len(true_indices) > 0:
        return true_indices[0] - prep_time
    return None


def generate_cues_input(dur1, dur2, dt_DDM, p, t_start, seed=None):
    num1 = int(dur1 / dt_DDM)
    num2 = int(dur2 / dt_DDM)
    rng = np.random.default_rng(seed)
    cue_R = rng.binomial(1, p, num1 + num2)
    cue_R[:int(t_start / dt_DDM)] = 0
    cue_L = np.zeros_like(cue_R)
    cue_L = 1 - cue_R
    cue_L[:int(t_start / dt_DDM)] = 0
    cue_R_all = np.zeros(dur1 + dur2)
    cue_R_all[0::int(dt_DDM)] = cue_R
    cue_L_all = np.zeros(dur1 + dur2)
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

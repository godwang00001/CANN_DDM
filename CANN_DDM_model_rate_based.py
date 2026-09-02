import numpy as np
import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import warnings
from jax import lax
from scipy.linalg import circulant
import scipy
from tqdm import tqdm

from rate_model_core.connectivity import (
    make_bump_to_edge_conn_mat,
    make_edge_to_bump_conn_mat,
    make_edge_conn_mat,
    make_bump_conn_mat,
)
from rate_model_core.config import (
    EdgePopConfig,
    BumpPopConfig,
    DecisionSpaceConfig,
    NeuralGeometryConfig,
    build_geometry,
    parse_bump_config,
    parse_decision_space_config,
    parse_edge_config,
    parse_geometry_config,
)
from rate_model_core.math import (
    bump_states as canonical_bump_states,
    edge_states as canonical_edge_states,
    sigmoid as canonical_sigmoid,
)
from rate_model_core.utils import (
    build_discrete_click_drift as util_build_discrete_click_drift,
    evidence_to_pos as util_evidence_to_pos,
    generate_click_inputs as util_generate_click_inputs,
    generate_cues_input as util_generate_cues_input,
    get_RT as util_get_RT,
    get_x_traj_discrete as util_get_x_traj_discrete,
    idx_to_pos as util_idx_to_pos,
    pos_to_evidence as util_pos_to_evidence,
    pos_to_idx as util_pos_to_idx,
)


class CANN_DDM_model(bp.dyn.NeuDyn):
    # ------------------------------------------------------------------
    # Construction and parameter initialization
    # ------------------------------------------------------------------
    def __init__(self, CANN_params=None, **kwargs):
        if CANN_params is not None:
            edge_config = parse_edge_config(CANN_params.get('edge_pop', {}))
            bump_config = parse_bump_config(CANN_params.get('bump_pop', {}))
            decision_space_config = parse_decision_space_config(
                CANN_params.get('decision_space_params', {})
            )
            geometry_config = parse_geometry_config(
                CANN_params.get('geometry'),
                edge_config,
                bump_config,
            )

            # initialize edge and bump using provided sub-dictionaries
            self._init_geometry(geometry_config)
            self._init_edge_pop(edge_config)
            self._init_bump_pop(bump_config)
            self._init_decision_space(decision_space_config)
            self.c_BE_func = self.c_BE_theta(self.c_BE_params)
        else:
            raise ValueError("CANN_params is required")

        # Initialize integration function
        assert self.num_E == self.num_B, "The exposed edge and bump population sizes must be the same"
        super(CANN_DDM_model, self).__init__(size=self.num_E, **kwargs)
        self.integral = bp.odeint(self.derivative)

    def _init_geometry(self, config: NeuralGeometryConfig):
        self.geometry_config = config
        self.geometry = build_geometry(config)
        self.coding_limit = self.geometry.coding_limit
        self.coding_frac = self.geometry.coding_frac
        self.clamp_frac = self.geometry.clamp_frac
        self.theta_min = self.geometry.theta_min
        self.theta_max = self.geometry.theta_max
        self.coding_theta_min = self.geometry.coding_theta_min
        self.coding_theta_max = self.geometry.coding_theta_max
        self.k1 = self.geometry.k1
        self.k2 = self.geometry.k2

    def _init_edge_pop(self, config: EdgePopConfig):
        """
        Initialize edge-population parameters and state variables.
        """
        # Raw parameters
        self.num_E = self.geometry.num_units
        self.tau_E = config.tau_E
        self.alpha_E = config.alpha_E
        self.gamma_E = config.gamma_E
        self.noise_scale_edge = config.noise_scale_edge
        self.clamp_frac_E = self.geometry.clamp_frac
        self.edge_type = config.edge_type
        self.c_EB = config.c_EB
        assert self.edge_type in ['Laplace', 'tanh'], "Edge type should be either Laplace or tanh."
        self.offset = config.offset
        self.eb_kernel_mode = config.eb_kernel_mode
        self.eb_kernel_sigma = config.eb_kernel_sigma
        self.eb_kernel_shift = config.eb_kernel_shift
        self.eb_kernel_gain = config.eb_kernel_gain

        # Derived geometry
        self.edge_geometry = self.geometry
        self.edge_k1 = self.edge_geometry.k1
        self.edge_k2 = self.edge_geometry.k2
        self.clamp_width = int(self.clamp_frac_E * self.num_E)

        # Static precomputations
        self.J_EE, self.beta_E = make_edge_conn_mat(
            self.num_E,
            self.gamma_E,
            geometry=self.edge_geometry,
            edge_type=self.edge_type,
            offset=self.offset,
            alpha=self.alpha_E,
        )
        self.r_E0 = self.edge_states(self.num_E, self.gamma_E, self.edge_geometry, self.edge_type)
        if self.eb_kernel_mode == 'edge_readout_bump':
            self.W_EB = None
            reference_w_eb = make_edge_to_bump_conn_mat(
                self.num_E,
                kernel_mode='simple',
                kernel_gain=self.eb_kernel_gain,
            )
            reference_i_eb = reference_w_eb @ self.r_E0
            self.eb_readout_bump_peak = float(bm.max(bm.abs(reference_i_eb)))
        else:
            self.W_EB = make_edge_to_bump_conn_mat(
                self.num_E,
                kernel_mode=self.eb_kernel_mode,
                kernel_sigma=self.eb_kernel_sigma,
                kernel_shift=self.eb_kernel_shift,
                kernel_gain=self.eb_kernel_gain,
            )
            self.eb_readout_bump_peak = 0.0

        # Dynamic state
        self.c_EB_dym = bm.Variable(bm.zeros(1))
        self.I_EE = bm.Variable(bm.zeros(self.num_E))
        self.I_EB = bm.Variable(bm.zeros(self.num_E))
        self.I_clamp_E = bm.Variable(self.get_I_clamp_edge(self.num_E, self.clamp_frac_E))
        self.Iext_E = bm.Variable(bm.zeros(self.num_E))
        self.x_E = bm.Variable(bm.zeros(1))  # decision variable read out from the edge population
        self.theta_E = bm.Variable(bm.zeros(1))
        self.r_E = bm.Variable(self.r_E0)
        self.I_E_noise = bm.Variable(bm.zeros(self.num_E))

    def _init_bump_pop(self, config: BumpPopConfig):
        """
        Initialize bump-population parameters and state variables.
        """
        # Raw parameters
        self.num_B = self.geometry.num_units
        self.tau_B = config.tau_B
        self.c_BE = config.c_BE
        self.c_BE_params = dict(config.c_BE_params)
        self.noise_scale_bump = config.noise_scale_bump
        self.sigma_B = config.sigma_B
        self.sigma_I_BE = config.sigma_I_BE
        self.beta_B = config.beta_B
        self.clamp_frac_B = self.geometry.clamp_frac
        self.bump_kernel_mode = config.kernel_mode
        self.bump_kernel_gain = config.kernel_gain
        self.bump_kernel_sigma = config.kernel_sigma
        self.bump_kernel_normed = config.kernel_normed
        self.be_kernel_mode = config.be_kernel_mode
        self.be_kernel_sigma = config.be_kernel_sigma
        self.be_kernel_gain = config.be_kernel_gain

        # Derived geometry
        self.bump_geometry = self.geometry
        self.bump_k1 = self.bump_geometry.k1
        self.bump_k2 = self.bump_geometry.k2
        self.rho_B = bm.pi / self.num_B

        # Static precomputations
        self.J_BB = make_bump_conn_mat(
            self.num_B,
            self.sigma_B,
            self.beta_B,
            geometry=self.bump_geometry,
            kernel_mode=self.bump_kernel_mode,
            kernel_gain=self.bump_kernel_gain,
            kernel_sigma=self.bump_kernel_sigma,
            kernel_normed=self.bump_kernel_normed,
        )
        self.W_BE = make_bump_to_edge_conn_mat(
            self.num_B,
            kernel_mode=self.be_kernel_mode,
            kernel_sigma=self.be_kernel_sigma,
            kernel_gain=self.be_kernel_gain,
        )
        self.r_B0 = self.bump_states(self.num_B, self.sigma_B, self.bump_geometry)

        # Dynamic state
        self.I_BB = bm.Variable(bm.zeros(self.num_B))
        self.I_BE = bm.Variable(bm.zeros(self.num_B))
        self.I_clamp_B = 0
        self.Iext_B = bm.Variable(bm.zeros(self.num_B))
        self.x_B = bm.Variable(bm.zeros(1))  # decision variable read out from the bump population
        self.theta_B = bm.Variable(bm.zeros(1))
        self.c_BE_dyn = bm.Variable(bm.zeros(1))
        self.r_B = bm.Variable(self.r_B0)
        self.I_B_noise = bm.Variable(bm.zeros(self.num_B))

    def _init_decision_space(self, config: DecisionSpaceConfig):
        """
        Initialize decision-space parameters, cue streams, and reference trajectory.
        """
        # Raw parameters
        self.decision_mode = config.decision_mode
        self.decision_paradigm = config.decision_paradigm
        self.t_start = config.t_start  # unit: ms, time to start encoding the evidence since the simulation begin
        self.dur = config.dur  # unit: ms, external evidence duration
        self.max_time = config.max_time if config.max_time is not None else config.dur  # unit: ms, total trial horizon
        self.boundary = config.boundary
        self.mar = float(config.mar)
        self.drift_rate = config.drift_rate
        self.noise_scale = config.noise_scale
        self.dt_DDM = config.dt_DDM
        self.dt_DDM_ms = self._decision_step_ms(self.dt_DDM)
        self.dt_DDM_s = self.dt_DDM * 1e-3
        self.lambda_click_L = float(config.lambda_click_L)
        self.lambda_click_R = float(config.lambda_click_R)
        self.delta_click_x = float(config.delta_click_x)
        if self.delta_click_x < 0:
            raise ValueError("delta_click_x must be non-negative")
        self.dx = self.noise_scale * np.sqrt(self.dt_DDM_s)
        self.x0 = config.x0 if config.x0 is not None else self.boundary / 2
        assert self.x0 >= 0 and self.x0 <= self.boundary, "x0 should be between 0 and boundary"
        if not (0.0 <= self.mar < 0.5):
            raise ValueError("mar must satisfy 0 <= mar < 0.5")
        self.left_decision_boundary = self.mar * float(self.boundary)
        self.right_decision_boundary = (1.0 - self.mar) * float(self.boundary)
        self.seed = config.seed
        self.drive_x_speed_unit = float(self.c_BE_params.get('drive_x_speed_unit', 3.0e-4))
        if self.drive_x_speed_unit <= 0:
            raise ValueError("c_BE_params['drive_x_speed_unit'] must be positive")
        if self.decision_paradigm not in ('free_response', 'interrogation'):
            raise ValueError(
                f"Unknown decision_paradigm '{self.decision_paradigm}'. "
                "Supported paradigms: free_response, interrogation."
            )
        if int(self.t_start) < 0:
            raise ValueError("t_start must be non-negative")
        if int(self.dur) <= 0:
            raise ValueError("dur must be positive")
        if int(self.max_time) < int(self.dur):
            raise ValueError("max_time must be greater than or equal to dur")
        if int(self.t_start) > int(self.dur):
            raise ValueError("t_start must satisfy 0 <= t_start <= dur")
        horizon = int(self.max_time)

        # Static precomputations
        if self.decision_mode == 'continuous':
            self.p = None
            self.click_R_all = np.zeros(horizon, dtype=float)
            self.click_L_all = np.zeros(horizon, dtype=float)
            self.dW_all = self.generate_continuous_noise_input(
                horizon,
                self.dt_DDM,
                self.t_start,
                active_stop=self.dur,
                seed=self.seed,
            )
            step_starts = np.arange(0, horizon, self.dt_DDM_ms, dtype=int)
            dW_steps = np.asarray(self.dW_all[step_starts], dtype=float)
            active_step_mask = (step_starts >= int(self.t_start)) & (step_starts < int(self.dur))
            v_drift_steps = np.zeros_like(dW_steps)
            v_drift_steps[active_step_mask] = (
                float(self.drift_rate) * float(self.dt_DDM_s) / (self.drive_x_speed_unit * float(self.dt_DDM_ms))
            )
            v_noise_steps = np.zeros_like(dW_steps)
            v_noise_steps[active_step_mask] = (
                float(self.noise_scale) * dW_steps[active_step_mask] / (self.drive_x_speed_unit * float(self.dt_DDM_ms))
            )
            self.v_drift_all = self.expand_decision_step_values(
                horizon,
                self.dt_DDM,
                v_drift_steps,
            )
            self.v_noise_all = self.expand_decision_step_values(
                horizon,
                self.dt_DDM,
                v_noise_steps,
            )
            self.v_drive_all = self.v_drift_all + self.v_noise_all
            self.x_traj = self.get_x_traj_continuous(
                self.t_start,
                self.dur,
                self.drift_rate,
                self.noise_scale,
                self.dt_DDM,
                self.x0,
                self.dW_all,
                self.boundary,
            )
        elif self.decision_mode == 'discrete':
            self.p = None
            self.click_L_all, self.click_R_all = self.generate_click_inputs(
                horizon,
                self.lambda_click_L,
                self.lambda_click_R,
                self.t_start,
                active_stop=self.dur,
                seed=self.seed,
            )
            noise_seed = None if self.seed is None else int(self.seed) + 1
            self.dW_all = self.generate_continuous_noise_input(
                horizon,
                self.dt_DDM,
                self.t_start,
                active_stop=self.dur,
                seed=noise_seed,
            )
            step_starts = np.arange(0, horizon, self.dt_DDM_ms, dtype=int)
            dW_steps = np.asarray(self.dW_all[step_starts], dtype=float)
            active_step_mask = (step_starts >= int(self.t_start)) & (step_starts < int(self.dur))
            v_noise_steps = np.zeros_like(dW_steps)
            v_noise_steps[active_step_mask] = (
                float(self.noise_scale) * dW_steps[active_step_mask] / (self.drive_x_speed_unit * float(self.dt_DDM_ms))
            )
            self.v_noise_all = self.expand_decision_step_values(
                horizon,
                self.dt_DDM,
                v_noise_steps,
            )
            self.v_drift_all = self.build_discrete_click_drift(
                self.click_R_all,
                self.click_L_all,
                self.t_start,
                self.dt_DDM,
                self.delta_click_x,
                self.drive_x_speed_unit,
            )
            self.v_drive_all = self.v_drift_all + self.v_noise_all
            self.x_traj = self.get_x_traj_discrete(
                self.t_start,
                self.dur,
                self.delta_click_x,
                self.x0,
                self.click_R_all,
                self.click_L_all,
                self.dW_all,
                self.noise_scale,
                self.boundary,
            )
        else:
            raise ValueError(
                f"Unknown decision_mode '{self.decision_mode}'. Supported modes: continuous, discrete."
            )

        # Legacy cue buffers are retained only for compatibility with
        # exploratory scripts that still mutate them directly. The active
        # continuous/discrete model path uses v_drift/v_noise/v_drive instead.
        self.cue_R_all = np.zeros(horizon, dtype=float)
        self.cue_L_all = np.zeros(horizon, dtype=float)

        # Dynamic state
        # Legacy placeholders kept so compatibility scripts can still bind cue
        # inputs through DSRunner. The main model path ignores them.
        self.cue_R = bm.Variable(bm.zeros(1))
        self.cue_L = bm.Variable(bm.zeros(1))
        self.v_drift = bm.Variable(bm.zeros(1))
        self.v_noise = bm.Variable(bm.zeros(1))
        self.v_drive = bm.Variable(bm.zeros(1))
        self.hit_boundary = bm.Variable(bm.zeros(1, dtype=bool))

    # ------------------------------------------------------------------
    # Canonical state/profile builders
    # ------------------------------------------------------------------
    def edge_states(self, num, gamma, geometry, type='Laplace', center_pos=0.0):
        """
        Return the canonical edge profile in theta space.
        """
        return canonical_edge_states(num, gamma, geometry, edge_type=type, center_pos=center_pos)

    def bump_states(self, num, sigma, geometry, center_pos=0.0):
        return canonical_bump_states(num, sigma, geometry, center_pos=center_pos)

    def get_edge_stimulus_by_pos(self, pos, edge_type='Laplace'):
        return self.edge_states(self.num_E, self.gamma_E, self.edge_geometry, edge_type, center_pos=pos)

    def initialize_state(self, pos_B, pos_E=None):
        if pos_E is None:
            pos_E = pos_B
        self.r_B[:] = self.bump_states(self.num_B, self.sigma_B, self.bump_geometry, center_pos=pos_B)
        self.r_E[:] = self.get_edge_stimulus_by_pos(pos_E, self.edge_type)
        self.theta_B[:] = self.find_current_bump_location(self.r_B)
        self.theta_E[:] = self.find_current_edge_location(self.r_E)
        self.x_B[:] = self.pos_to_evidence(self.theta_B, self.gamma_E)
        self.x_E[:] = self.pos_to_evidence(self.theta_E, self.gamma_E)
        self.I_BB[:] = 0.
        self.I_EE[:] = 0.
        self.I_BE[:] = 0.
        self.I_EB[:] = 0.
        self.I_B_noise[:] = 0.
        self.I_E_noise[:] = 0.
        self.Iext_B[:] = 0.
        self.Iext_E[:] = 0.
        self.c_BE_dyn[:] = 0.
        self.c_EB_dym[:] = 0.
        self.cue_R[:] = 0.
        self.cue_L[:] = 0.
        self.v_drift[:] = 0.
        self.v_noise[:] = 0.
        self.v_drive[:] = 0.
        self.hit_boundary[:] = False

    # ------------------------------------------------------------------
    # Nonlinearities and static helpers
    # ------------------------------------------------------------------
    def phi_B(self, x):
        return bm.square(x) / (self.beta_B * self.rho_B * (1 + bm.sum(bm.square(x))))

    def phi_E(self, x):
        return self.sigmoid(self.alpha_E * x + self.beta_E)

    def sigmoid(self, x):
        return canonical_sigmoid(x)

    def get_I_clamp_edge(self, num_E, clamp_frac):
        I_clamp = bm.zeros(num_E)
        I_clamp[:int(num_E * clamp_frac / 2)] = 100
        I_clamp[-int(num_E * clamp_frac / 2):] = -100
        return I_clamp

    def get_I_clamp_bump(self, num_B, clamp_frac):
        I_clamp = bm.zeros(num_B)
        I_clamp[:int(num_B * clamp_frac / 2)] = -100
        I_clamp[-int(num_B * clamp_frac / 2):] = -100
        return I_clamp

    def make_mask(self, num, fixed_end_width):
        mask = bm.zeros(num, dtype=bool)
        mask = mask.at[fixed_end_width + 50:-fixed_end_width - 50].set(True)
        return mask

    # ------------------------------------------------------------------
    # Coordinate transforms and readout mappings
    # ------------------------------------------------------------------
    def idx_to_pos(self, idx):
        return util_idx_to_pos(idx, self.geometry)

    def pos_to_idx(self, pos):
        return util_pos_to_idx(pos, self.geometry)

    def pos_to_evidence(self, pos, s):
        return util_pos_to_evidence(pos, self.boundary, self.geometry.coding_theta_max, s)

    def evidence_to_pos(self, evidence, s):
        return util_evidence_to_pos(evidence, self.boundary, self.geometry.coding_theta_max, s)

    def theta_req(self):
        decision_dt_ms = float(self.dt_DDM)
        delta_x = self.delta_click_x if self.decision_mode == 'discrete' else self.dx
        theta_req_pos = lambda theta: 1 / (self.gamma_E * decision_dt_ms) * bm.log(1 + delta_x / self.pos_to_evidence(theta, self.gamma_E))
        theta_req_neg = lambda theta: -1 / (self.gamma_E * decision_dt_ms) * bm.log(1 - delta_x / self.pos_to_evidence(theta, self.gamma_E))
        theta_req = lambda theta: 0.5 * (theta_req_pos(theta) + theta_req_neg(theta))
        return theta_req

    def c_BE_theta(self, c_BE_params):
        """
        Return a callable function that maps `theta` -> c_BE value.

        Supported modes:
        - 'const': returns a constant value; parameter: `value` (defaults to `self.c_BE`).
        - 'linear': returns theta_req(theta) / k_linear; parameter: `k_linear` (or `k`), default 1.
        """
        theta_req_func = self.theta_req()
        mode = c_BE_params.get('mode', 'const')
        if mode == 'const':
            value = self.c_BE
            return lambda theta: value
        elif mode == 'linear':
            k_linear = float(c_BE_params.get('k_linear', 1.0))
            if k_linear == 0:
                raise ValueError('k_linear must be non-zero for linear mode')

            return lambda theta: theta_req_func(theta) / k_linear
        elif mode == 'quadratic':
            a = float(c_BE_params.get('a', 1.0))
            b = float(c_BE_params.get('b', 1.0))

            delta = lambda theta: bm.sqrt(b ** 2 + 4 * a * theta_req_func(theta))
            return lambda theta: (-b + delta(theta)) / (2 * a)
        elif mode == 'target_diffusion':
            if 'kappa' not in c_BE_params:
                def _unprepared_target_diffusion(theta):
                    raise ValueError(
                        "target_diffusion mode requires calibration. Call prepare_target_diffusion_mode() first."
                    )
                return _unprepared_target_diffusion
            kappa = float(c_BE_params['kappa'])
            if np.isclose(kappa, 0.0):
                raise ValueError("kappa must be non-zero for target_diffusion mode")

            theta_margin = float(c_BE_params.get('theta_margin', 0.02))
            drive_x_speed_unit = float(c_BE_params.get('drive_x_speed_unit', 3.0e-4))
            if drive_x_speed_unit <= 0:
                raise ValueError("drive_x_speed_unit must be positive for target_diffusion mode")
            theta_min = float(self.geometry.coding_theta_min)
            theta_max = float(self.geometry.coding_theta_max)
            interval = theta_max - theta_min
            normalization = 1.0 - bm.exp(-self.gamma_E * interval)
            eval_theta = lambda theta: bm.clip(theta, theta_min + theta_margin, theta_max - theta_margin)
            dx_dtheta = lambda theta: (
                self.boundary
                * self.gamma_E
                * bm.exp(-self.gamma_E * (eval_theta(theta) - theta_min))
                / normalization
            )
            return lambda theta: drive_x_speed_unit / (kappa * dx_dtheta(theta))
        else:
            raise ValueError(f"Unknown c_BE_mode '{mode}'. Supported modes: const, linear, quadratic, target_diffusion.")

    def prepare_target_diffusion_mode(self, **overrides):
        params = dict(self.c_BE_params)
        params.update(overrides)
        if params.get('mode') != 'target_diffusion':
            raise ValueError("prepare_target_diffusion_mode() requires c_BE_params['mode'] == 'target_diffusion'.")

        from rate_model_core.calibration import calibrate_target_diffusion_profile

        result = calibrate_target_diffusion_profile(
            self,
            calibration_x0=params.get('calibration_x0'),
            c_be_sweep=params.get('c_be_sweep'),
            min_accumulation_samples=int(params.get('min_accumulation_samples', 200)),
            mean_abs_tol=params.get('mean_abs_tol'),
            std_abs_tol=params.get('std_abs_tol'),
            theta_margin=float(params.get('theta_margin', 0.02)),
            sweep_expand_factor=float(params.get('sweep_expand_factor', 1.1)),
            kappa_tol=float(params.get('kappa_tol', 0.1)),
        )
        self.c_BE_params.update(params)
        self.c_BE_params['kappa'] = float(result['kappa'])
        self.c_BE_func = self.c_BE_theta(self.c_BE_params)
        if not result['certificate_passed']:
            warnings.warn(
                "target_diffusion calibration did not pass the alignment/kappa consistency certificate; continuing anyway.",
                stacklevel=2,
            )
        return result

    # ------------------------------------------------------------------
    # State localization and readout helpers
    # ------------------------------------------------------------------
    def _interpolated_peak_idx(self, values):
        peak_idx = bm.argmax(values)
        left_idx = bm.maximum(peak_idx - 1, 0)
        right_idx = bm.minimum(peak_idx + 1, len(values) - 1)

        left_val = values[left_idx]
        center_val = values[peak_idx]
        right_val = values[right_idx]
        denominator = left_val - 2 * center_val + right_val
        offset = bm.where(
            bm.abs(denominator) > 1e-10,
            0.5 * (left_val - right_val) / denominator,
            0.0
        )
        return peak_idx + offset

    def _interpolated_level_crossing_idx(self, values, level):
        below = values <= level
        has_crossing = bm.any(below)
        right_idx = bm.where(has_crossing, bm.argmax(below.astype(int)), len(values) - 1)
        left_idx = bm.maximum(right_idx - 1, 0)

        left_val = values[left_idx]
        right_val = values[right_idx]
        denominator = right_val - left_val
        frac = bm.where(
            bm.abs(denominator) > 1e-10,
            (level - left_val) / denominator,
            0.0,
        )
        return bm.clip(left_idx + frac, 0.0, float(len(values) - 1))

    def find_current_edge_location(self, r_E):
        if self.edge_type == 'tanh':
            interpolated_idx = self._interpolated_level_crossing_idx(r_E, 0.5)
            return self.idx_to_pos(interpolated_idx)

        # Fallback for non-tanh edges until their crossing level is calibrated.
        diff_r_E = bm.abs(bm.diff(r_E))
        interpolated_idx = self._interpolated_peak_idx(diff_r_E)
        return self.idx_to_pos(interpolated_idx)

    def find_current_bump_location(self, r_B):
        # Find the peak using argmax
        interpolated_idx = self._interpolated_peak_idx(r_B)
        return self.idx_to_pos(interpolated_idx)

    def get_RT(self, prep_time, hit_boudary_trace):
        """
        Return the first boundary-hitting timestep after the preparation period.
        """
        return util_get_RT(prep_time, hit_boudary_trace)

    # ------------------------------------------------------------------
    # Cue-generation and reference-trajectory utilities
    # ------------------------------------------------------------------
    def _decision_step_ms(self, dt_DDM):
        step_ms = int(round(float(dt_DDM)))
        if step_ms <= 0 or not np.isclose(float(dt_DDM), float(step_ms)):
            raise ValueError("dt_DDM must be a positive integer number of milliseconds")
        return step_ms

    def generate_cues_input(self, dur, dt_DDM, p, t_start, seed=None):
        return util_generate_cues_input(dur, dt_DDM, p, t_start, seed=seed)

    def generate_click_inputs(self, dur, lambda_click_L, lambda_click_R, t_start, active_stop=None, seed=None):
        total_dur = int(dur)
        active_stop = total_dur if active_stop is None else int(active_stop)
        click_L_all = np.zeros(total_dur, dtype=float)
        click_R_all = np.zeros(total_dur, dtype=float)
        active_start = int(t_start)
        active_stop = max(active_start, min(active_stop, total_dur))
        active_len = max(active_stop - active_start, 0)
        if active_len > 0:
            rng = np.random.default_rng(seed)
            click_L_all[active_start:active_stop] = rng.binomial(1, float(lambda_click_L), active_len)
            click_R_all[active_start:active_stop] = rng.binomial(1, float(lambda_click_R), active_len)
        return click_L_all, click_R_all

    def build_discrete_click_drift(self, click_R_all, click_L_all, t_start, dt_DDM, delta_click_x, drive_x_speed_unit):
        return util_build_discrete_click_drift(
            click_R_all,
            click_L_all,
            t_start,
            dt_DDM,
            delta_click_x,
            drive_x_speed_unit,
        )

    def generate_continuous_noise_input(self, dur, dt_DDM, t_start, active_stop=None, seed=None):
        step_ms = self._decision_step_ms(dt_DDM)
        dt_s = float(dt_DDM) * 1e-3
        dW_all = np.zeros(int(dur), dtype=float)
        step_times = np.arange(0, int(dur), step_ms, dtype=int)
        active_stop = int(dur) if active_stop is None else int(active_stop)
        active_mask = (step_times >= int(t_start)) & (step_times < active_stop)
        if np.any(active_mask):
            rng = np.random.default_rng(seed)
            dW_all[step_times[active_mask]] = np.sqrt(dt_s) * rng.standard_normal(np.count_nonzero(active_mask))
        return dW_all

    def expand_decision_step_values(self, dur, dt_DDM, step_values):
        step_ms = self._decision_step_ms(dt_DDM)
        expanded = np.zeros(int(dur), dtype=float)
        step_starts = np.arange(0, int(dur), step_ms, dtype=int)
        if len(step_values) != len(step_starts):
            raise ValueError("step_values must have the same length as the decision-step grid")
        for start, value in zip(step_starts, np.asarray(step_values, dtype=float)):
            stop = min(start + step_ms, int(dur))
            expanded[start:stop] = value
        return expanded

    def get_x_traj_discrete(self, t_start, dur, delta_click_x, x0,
                            click_R_all, click_L_all, dW_all, noise_scale, boundary):
        return util_get_x_traj_discrete(
            t_start,
            delta_click_x,
            x0,
            click_R_all,
            click_L_all,
            dW_all,
            noise_scale,
            boundary,
        )

    def get_x_traj_continuous(self, t_start, dur, drift_rate, noise_scale, dt_DDM, x0, dW_all, boundary):
        T = len(dW_all)
        x_traj = np.full(T, float(x0), dtype=float)
        step_ms = self._decision_step_ms(dt_DDM)
        dt_s = float(dt_DDM) * 1e-3
        delta = noise_scale * np.asarray(dW_all, dtype=float)
        step_times = np.arange(0, T, step_ms, dtype=int)
        active_step_times = step_times[(step_times >= int(t_start)) & (step_times < int(dur))]
        delta[active_step_times] += drift_rate * dt_s

        x_curr = float(x0)
        absorbed = False
        for t in range(T):
            if t < int(t_start):
                x_traj[t] = float(x0)
                continue
            if absorbed:
                x_traj[t] = x_curr
                continue
            x_curr += float(delta[t])
            if x_curr >= boundary:
                x_curr = float(boundary)
                absorbed = True
            elif x_curr <= 0.0:
                x_curr = 0.0
                absorbed = True
            x_traj[t] = x_curr
        return x_traj

    def get_pos_offset(self, x0, gamma_E, tol=1e-4, max_iter=50, progress=False):
        """
        Find the position offset for the edge population using a continuous ternary search
        on a unimodal objective f(offset) = (theta_E - theta_B)^2. This is more efficient
        than bisection for unimodal functions and does not require integer offsets.
        """
        pos = self.evidence_to_pos(x0, gamma_E)
        left = -np.pi / 2 - pos
        right = np.pi / 2 - pos

        def f(offset):
            # Run a short simulation at the given positional offset and return the squared
            # difference between theta_E and theta_B at t_start. We must reset the model
            # state after the run so that subsequent simulations are independent.
            runner = self.run_simulation(mon_vars=['theta_E', 'theta_B'], pos_offset=offset, progress_bar=False, dt=1.)
            try:
                theta_E = float(runner.mon.theta_E[int(self.t_start), 0])
            except Exception:
                # fallback conversion if needed
                theta_E = float(bm.to_numpy(runner.mon.theta_E[int(self.t_start), 0]))
            try:
                theta_B = float(runner.mon.theta_B[int(self.t_start), 0])
            except Exception:
                theta_B = float(bm.to_numpy(runner.mon.theta_B[int(self.t_start), 0]))
            # reset model state so next call to run_simulation starts from the same initial state
            try:
                bp.reset_state(self)
            except Exception:
                # if reset_state is not available or fails silently continue (best-effort)
                pass
            return (theta_E - theta_B) ** 2

        it = 0
        pbar = None
        if progress:
            pbar = tqdm(total=max_iter, desc='get_pos_offset')

        while (right - left) > tol and it < max_iter:
            m1 = left + (right - left) / 3.0
            m2 = right - (right - left) / 3.0
            f1 = f(m1)
            f2 = f(m2)

            if progress and pbar is not None:
                pbar.set_postfix({'left': f"{left:.3f}", 'right': f"{right:.3f}", 'f1': f"{f1:.3e}", 'f2': f"{f2:.3e}"})
                pbar.update(1)

            # Standard ternary reduction for unimodal functions
            if f1 > f2:
                left = m1
            else:
                right = m2
            it += 1

        if pbar is not None:
            pbar.close()

        # return midpoint as best estimate
        return (left + right) / 2

    # ------------------------------------------------------------------
    # Coupling current helpers
    # ------------------------------------------------------------------
    def get_current_I_BE(self, cue_R, cue_L, r_B, c_BE):
        """
        Build the bump-to-edge input for the current cue state.
        """
        filtered_r_B = self.W_BE @ r_B
        if self.decision_mode in ('continuous', 'discrete'):
            return c_BE * self.v_drive * filtered_r_B
        I_BE = c_BE * (cue_R * filtered_r_B + cue_L * (-filtered_r_B))
        return I_BE

    def get_current_I_EB(self, r_E, c_EB):
        """
        Build the edge-to-bump input from the current edge profile.
        """
        pos = self.find_current_edge_location(r_E)
        if self.eb_kernel_mode == 'edge_readout_bump':
            stimulus = self.eb_readout_bump_peak * self.bump_states(
                self.num_B,
                self.sigma_B,
                self.bump_geometry,
                center_pos=pos,
            )
        else:
            stimulus = (self.W_EB @ r_E)
        zeros = bm.zeros_like(stimulus)
        return bm.where(
            (pos >= self.geometry.coding_theta_min) & (pos <= self.geometry.coding_theta_max),
            c_EB * stimulus,
            zeros,
        )

    # ------------------------------------------------------------------
    # Runtime dynamics and simulation entrypoints
    # ------------------------------------------------------------------
    @property
    def derivative(self):
        dr_B = lambda r_B, t, Iext_B: (-r_B + self.phi_B(self.I_BB + self.I_EB + Iext_B + self.I_clamp_B + self.I_B_noise)) / self.tau_B
        dr_E = lambda r_E, t, Iext_E: (-r_E + self.phi_E(self.I_EE + self.I_BE + Iext_E + self.I_clamp_E + self.I_E_noise)) / self.tau_E
        return bp.JointEq([dr_B, dr_E])

    def update(self, x=None):
        _t = bp.share['t']

        self.I_BB[:] = self.J_BB @ self.r_B
        self.I_EE[:] = self.J_EE @ self.r_E
        self.c_BE_dyn[:] = self.c_BE_func(self.theta_E)
        if self.decision_mode in ('continuous', 'discrete'):
            self.v_drive[:] = self.v_drift + self.v_noise
        else:
            self.v_drive[:] = 0.0
        self.I_E_noise[:] = self.noise_scale_edge * bm.random.normal(0, 1, self.num_E)
        self.I_B_noise[:] = self.noise_scale_bump * bm.random.normal(0, 1, self.num_B)
        self.I_BE[:] = ~self.hit_boundary * bm.where(
            _t < self.t_start,
            0,
            self.get_current_I_BE(self.cue_R, self.cue_L, self.r_B, self.c_BE_dyn),
        )
        self.I_EB[:] = ~self.hit_boundary * bm.where(
            _t < self.t_start,
            0,
            self.get_current_I_EB(self.r_E, self.c_EB),
        )

        r_B, r_E = self.integral(self.r_B, self.r_E, _t, self.Iext_B, self.Iext_E)
        self.r_B[:] = r_B
        self.r_E[:] = r_E
        self.theta_B[:] = self.find_current_bump_location(self.r_B)
        self.theta_E[:] = self.find_current_edge_location(self.r_E)
        self.x_B[:] = self.pos_to_evidence(self.theta_B, self.gamma_E)
        self.x_E[:] = self.pos_to_evidence(self.theta_E, self.gamma_E)
        hit_boundary = bm.logical_or(
            self.x_E >= self.right_decision_boundary,
            self.x_E <= self.left_decision_boundary,
        )
        # assign boolean array directly (shapes should match: both length 1)
        self.hit_boundary[:] = bm.where(self.hit_boundary, self.hit_boundary, hit_boundary)

        # Update the external input
        self.Iext_B[:] = 0.
        self.Iext_E[:] = 0.

    def build_runner(self, mon_vars, pos_offset=0, progress_bar=True, dt=1.):
        pos_init = self.evidence_to_pos(self.x0, self.gamma_E)
        self.initialize_state(pos_init, pos_init + pos_offset)

        return bp.DSRunner(
            self,
            inputs=[('cue_R', self.cue_R_all, 'iter', '='),
                    ('cue_L', self.cue_L_all, 'iter', '='),
                    ('v_drift', self.v_drift_all, 'iter', '='),
                    ('v_noise', self.v_noise_all, 'iter', '=')],
            monitors=mon_vars,
            dyn_vars=self.vars(),
            progress_bar=progress_bar,
            dt=dt,
        )

    def run_simulation(self, mon_vars, pos_offset=0, progress_bar=True, dt=1., get_RT=False):
        t_start = self.t_start
        runner = self.build_runner(mon_vars, pos_offset=pos_offset, progress_bar=progress_bar, dt=dt)
        runner.run(self.max_time)
        RT = self.get_RT(t_start, runner.mon.hit_boundary) 
        if RT:
            runner.rt_ms = RT / dt
        else:
            runner.rt_ms = None
        # if get_RT:
        #     assert 'hit_boundary' in mon_vars, "hit_boundary must be in mon_vars when get_RT is True"
        #     RT = self.get_RT(t_start, runner.mon.hit_boundary)
        #     if RT:
        #         return runner, RT / dt
        #     else:
        #         return runner, None
        # else:
        #    return runner
        return runner

import numpy as np
import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from jax import lax
from scipy.linalg import circulant
import scipy
from tqdm import tqdm
from make_conn_mat_updated import make_edge_conn_mat, make_bump_conn_mat


class CANN_DDM_model(bp.dyn.NeuDyn):
    def __init__(self, CANN_params=None, **kwargs):
    
        if CANN_params is not None:
            edge_params = CANN_params.get('edge_pop', {})
            bump_params = CANN_params.get('bump_pop', {})
            neural_space_params = CANN_params.get('neural_space_params', {})
            decision_space_params = CANN_params.get('decision_space_params', {})

            # initialize edge and bump using provided sub-dictionaries
            self._init_neural_space(**neural_space_params)
            self._init_edge_pop(**edge_params)
            self._init_bump_pop(**bump_params)
            self._init_decision_space(**decision_space_params)

        else:
            raise ValueError("CANN_params is required")
         
        # Initialize integration function
        assert self.num_E == self.num_B, "The number of neurons in the edge and bump population must be the same"
        super(CANN_DDM_model, self).__init__(size=self.num_E, **kwargs)
        self.integral = bp.odeint(self.derivative)
    
        
    
  
    def _init_bump_pop(self, **kwargs):
        """
        Initialize the parameters for the bump population 
        and define the variables that will be used to track the dynamics of the bump attractor 
        """
        self.num_B = kwargs.get('num_B', 1024)
        self.tau_B = kwargs.get('tau_B', 1)
        self.c_BE = kwargs.get('c_BE', 1)
        # c_BE_params contains both mode and parameters, e.g. {'mode': 'linear', 'k_linear': 2.0}
        self.c_BE_params = kwargs.get('c_BE_params', {'mode': 'const'})
        self.noise_scale_bump = kwargs.get('noise_scale_bump', 0.1)
        self.c_BE_func = self.c_BE_theta(self.c_BE_params)
        self.sigma_B = kwargs.get('sigma_B', 0.25)
        # Optional width control for I_BE (Gaussian filter applied in get_current_I_BE)
        self.sigma_I_BE = kwargs.get('sigma_I_BE', None)
        self.beta_B = kwargs.get('beta_B', 1.8)
        self.clamp_frac_B = kwargs.get('clamp_frac_B', 0.15)
        self.rho_B = bm.pi / self.num_B
        self.J_BB = make_bump_conn_mat(self.num_B, self.sigma_B, self.beta_B,
                                       clamp_frac=self.clamp_frac_B)
        self.I_BB = bm.Variable(bm.zeros(self.num_B))
        self.I_BE = bm.Variable(bm.zeros(self.num_B))
        #self.I_clamp_B = bm.Variable(self.get_I_clamp_bump(self.num_B, self.clamp_frac_B))
        self.I_clamp_B = 0
        self.Iext_B = bm.Variable(bm.zeros(self.num_B))
        self.x_B = bm.Variable(bm.zeros(1)) # decision variable read out from the bump population
        self.theta_B = bm.Variable(bm.zeros(1))
        self.c_BE_dyn = bm.Variable(bm.zeros(1))
        self.r_B0 = self.bump_states(self.num_B, self.sigma_B, self.theta_0, self.enc_range)
        self.r_B = bm.Variable(self.r_B0)
        self.I_B_noise = bm.Variable(bm.zeros(self.num_B))




    def _init_edge_pop(self, **kwargs):
        """
        Initialize the parameters for the edge population 
        and define the variables that will be used to track the dynamics of the edge attractor 
        """

        # Initialize the parameters for the edge population
        self.num_E = kwargs.get('num_E', 1024)
        self.tau_E = kwargs.get('tau_E', 1)
        self.alpha_E = kwargs.get('alpha_E', 1)
        self.sigma_E = kwargs.get('sigma_E', 1)
        self.noise_scale_edge = kwargs.get('noise_scale_edge', 0.1)
        self.clamp_frac_E = kwargs.get('clamp_frac_E', 0.15)
        self.edge_type = kwargs.get('edge_type', 'Laplace')
        assert self.edge_type in ['Laplace', 'tanh'], "Edge type should be either Laplace or tanh."
        #self.optimize_offset = kwargs.get('optimize_offset', False)
        #if not self.optimize_offset:
        #    self.offset = kwargs.get('offset', 0)
       # else:
        self.offset = 0
        self.c_EB = kwargs.get('c_EB', 1.)
        self.c_EB_dym = bm.Variable(bm.zeros(1))
        self.clamp_width = int(self.clamp_frac_E * self.num_E)
        self.J_EE, self.beta_E = make_edge_conn_mat(self.num_E, self.sigma_E, self.theta_0, self.enc_range,
                                       edge_type = self.edge_type, clamp_frac=self.clamp_frac_E, 
                                       offset=self.offset)
        self.I_EE = bm.Variable(bm.zeros(self.num_E))
        self.I_EB = bm.Variable(bm.zeros(self.num_E))
        self.I_clamp_E = bm.Variable(self.get_I_clamp_edge(self.num_E, self.clamp_frac_E))    
        self.Iext_E = bm.Variable(bm.zeros(self.num_E))
        self.x_E = bm.Variable(bm.zeros(1)) # decision variable read out from the edge population
        self.theta_E = bm.Variable(bm.zeros(1))
        self.r_E0 = self.edge_states(self.num_E, self.sigma_E, self.theta_0, self.enc_range, self.edge_type)
        self.r_E = bm.Variable(self.r_E0)
        self.I_E_noise = bm.Variable(bm.zeros(self.num_E))

    def _init_decision_space(self, **kwargs):
        """
        Initialize the decision space 
        """
        self.t_start = kwargs.get('t_start', 200) # unit: ms, time to start encoding the evidence since the simulation begin 
        self.dur1 = kwargs.get('dur1', 100) # unit: ms, duration of the period to initiate the network states
        self.dur2 = kwargs.get('dur2', 1000) # unit: ms, remove any external input after the network states are initiated
        self.boundary = kwargs.get('boundary', 1)
        self.drift_rate = kwargs.get('drift_rate', 0.5)
        self.noise_scale = kwargs.get('noise_scale', 0.5)
        self.dt_DDM = kwargs.get('dt_DDM', 25)
        #self.dx = np.sqrt(self.noise_scale**2 * self.dt_DDM*1e-3 + (self.drift_rate * self.dt_DDM*1e-3)**2)
        self.dx = self.noise_scale * np.sqrt(self.dt_DDM*1e-3)
        self.p = 0.5 * (1 + (self.drift_rate * np.sqrt(self.dt_DDM*1e-3))/self.noise_scale)
        self.x0 = kwargs.get('x0', self.boundary/2)
        assert self.x0 >= 0 and self.x0 <= self.boundary, "x0 should be between 0 and boundary"
        self.cue_R = bm.Variable(bm.zeros(1))
        self.cue_L = bm.Variable(bm.zeros(1))
        self.seed = kwargs.get('seed', None)
        self.cue_L_all, self.cue_R_all = self.generate_cues_input(self.dur1, self.dur2, self.dt_DDM, 
                                                                   self.p, self.t_start, seed=self.seed)
        self.x_traj = self.get_x_traj(self.t_start, self.dx, self.dt_DDM, self.x0, 
                                      self.cue_R_all, self.cue_L_all, self.boundary)

        # store hit boundary as a 1-D boolean array so it can be indexed/monitored per timestep
        self.hit_boundary = bm.Variable(bm.zeros(1, dtype=bool))  

    def _init_neural_space(self, **kwargs):
        """
        Initialize the neural space mapping rule
        """
        self.theta_0 = kwargs.get('theta_0', 0)
        self.enc_range = kwargs.get('enc_range', bm.pi/3)
    

    def edge_states(self, num, sigma, theta_0, enc_range, type='Laplace'):
        """
        Laplace solution for the edge attractor dynamics (synaptic input)
        return the ideal solution of the edge attractor dynamics on theta grid.
        """
        theta = -bm.pi / 2 + (bm.pi / num) * bm.arange(num)
        x = bm.pi / enc_range * (theta - theta_0)
        if type == 'Laplace':
            return bm.exp(-bm.exp(sigma * x))
        elif type == 'tanh':
            sigma_prime = 4*sigma / bm.exp(1)
            return self.sigmoid(-sigma_prime * x)
        else:
            raise ValueError('Edge type should be either Laplace or tanh.')
        
    def bump_states(self, num, sigma, theta_0, enc_range):
        theta = -bm.pi / 2 + (bm.pi / num) * bm.arange(num)
        return bm.exp(-(bm.pi / (bm.sqrt(2) * sigma * enc_range))**2 * (theta - theta_0)**2)

    def phi_B(self, x):
        return bm.square(x) / (self.beta_B * self.rho_B *(1 + bm.sum(bm.square(x))))
    
    def phi_E(self, x):
        return self.sigmoid(self.alpha_E * x + self.beta_E)

    def sigmoid(self, x):
        return 1/(1 + bm.exp(-x))
    
    def get_I_clamp_edge(self, num_E, clamp_frac):
        I_clamp = bm.zeros(num_E)
        I_clamp[:int(num_E * clamp_frac/2)] = 100
        I_clamp[-int(num_E * clamp_frac/2):] = -100
        return I_clamp
    
    def get_I_clamp_bump(self, num_B, clamp_frac):
        I_clamp = bm.zeros(num_B)
        I_clamp[:int(num_B * clamp_frac/2)] = -100
        I_clamp[-int(num_B * clamp_frac/2):] = -100
        return I_clamp

    def theta_req(self):
        theta_req_pos = lambda theta: 1/(self.sigma_E*self.dt_DDM) * bm.log(1 + self.dx / self.pos_to_evidence(theta, self.sigma_E))
        theta_req_neg = lambda theta: -1/(self.sigma_E*self.dt_DDM) * bm.log(1 - self.dx / self.pos_to_evidence(theta, self.sigma_E))
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
        # Extract mode from kwargs if provided, otherwise use the mode parameter
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
            
            delta = lambda theta: bm.sqrt(b**2 + 4*a*theta_req_func(theta))
            return lambda theta: (-b + delta(theta)) / (2*a)

        else:
            raise ValueError(f"Unknown c_BE_mode '{mode}'. Supported modes: const, linear, quadratic.")
    

    def make_mask(self, num, fixed_end_width):
        mask = bm.zeros(num, dtype=bool)
        mask = mask.at[fixed_end_width+50:-fixed_end_width-50].set(True)
        return mask

    def _encoding_bounds(self, num):
        """
        Derive index-space encoding bounds from enc_range:
        k2 - k1 = enc_range * num / pi
        k1 = num//2 - 0.5 * (enc_range * num / pi)
        k2 = num//2 + 0.5 * (enc_range * num / pi)
        """
        center = num // 2
        half_width = self.enc_range * num / (2 * bm.pi)
        k1 = center - half_width
        k2 = center + half_width
        return k1, k2

    
    def idx_to_pos(self, idx):
        idx = bm.asarray(idx)
        k1, k2 = self._encoding_bounds(self.num_E)
        return bm.where(
            idx < k1,
            -bm.pi/2,
            bm.where(
                idx > k2,
                bm.pi/2,
                -bm.pi/2 + (idx - k1) / (k2 - k1) * bm.pi
            )
        )
    
    def pos_to_idx(self, pos):
        # Accept either python floats or brainpy/jax scalars/arrays.
        # Clamp to valid range and return a bm-compatible numeric index (not a Python int),
        # so it can be used inside JIT/array computations and downstream functions.
        pos = bm.asarray(pos)
        pos = bm.clip(pos, -bm.pi/2, bm.pi/2)
        k1, k2 = self._encoding_bounds(self.num_E)
        idx = k1 + (pos + bm.pi/2) / bm.pi * (k2 - k1)
        return idx
    
    def pos_to_evidence(self, pos, s):
        return self.boundary * bm.exp(s * (-0.5 * bm.pi + pos))
   
    def evidence_to_pos(self, evidence, s):
        return 0.5 * bm.pi + bm.log(evidence / self.boundary) / s
        

    def find_current_edge_location(self, r_E):
        # Find the maximum gradient using argmax of absolute difference
        diff_r_E = bm.abs(bm.diff(r_E))
        peak_idx = bm.argmax(diff_r_E)
        
        # Get neighboring indices for interpolation
        left_idx = bm.maximum(peak_idx - 1, 0)
        right_idx = bm.minimum(peak_idx + 1, len(diff_r_E) - 1)
        
        # Get the gradient values at neighboring points
        left_val = diff_r_E[left_idx]
        center_val = diff_r_E[peak_idx]
        right_val = diff_r_E[right_idx]
        
        # Parabolic interpolation to find sub-pixel peak
        # Using the formula: x_peak = x0 + 0.5 * (left - right) / (left - 2*center + right)
        denominator = left_val - 2 * center_val + right_val
        # Avoid division by zero
        offset = bm.where(
            bm.abs(denominator) > 1e-10,
            0.5 * (left_val - right_val) / denominator,
            0.0
        )
        
        # Calculate the interpolated index
        interpolated_idx = peak_idx + offset
        
        return self.idx_to_pos(interpolated_idx)

    def find_current_bump_location(self, r_B):
        # Find the peak using argmax
        peak_idx = bm.argmax(r_B)
        
        # Get neighboring indices for interpolation
        left_idx = bm.maximum(peak_idx - 1, 0)
        right_idx = bm.minimum(peak_idx + 1, len(r_B) - 1)
        
        # Get the activity values at neighboring points
        left_val = r_B[left_idx]
        center_val = r_B[peak_idx]
        right_val = r_B[right_idx]
        
        # Parabolic interpolation to find sub-pixel peak
        # Using the formula: x_peak = x0 + 0.5 * (left - right) / (left - 2*center + right)
        denominator = left_val - 2 * center_val + right_val
        # Avoid division by zero
        offset = bm.where(
            bm.abs(denominator) > 1e-10,
            0.5 * (left_val - right_val) / denominator,
            0.0
        )
        
        # Calculate the interpolated index
        interpolated_idx = peak_idx + offset
        
        return self.idx_to_pos(interpolated_idx)

    def get_bump_stimulus_by_pos(self, pos):
        # bump stimulus lives on the bump population size (num_B)
        return self.bump_states(self.num_B, self.sigma_B, pos, self.enc_range)
    
    def get_edge_stimulus_by_pos(self, pos, edge_type='Laplace'):
        return self.edge_states(self.num_E, self.sigma_E, pos, self.enc_range, edge_type)
        
    
    def get_RT(self, prep_time, hit_boudary_trace):
        """
        Find the index of the first True element in self.hit_boundary.
        Returns the index if found, otherwise returns None.
        """
        true_indices = np.where(hit_boudary_trace)[0]
        if len(true_indices) > 0:
            return true_indices[0] - prep_time
        else:
            return None
        

    
    def get_current_I_BE(self, cue_R, cue_L, r_B, c_BE):
        """
        Get the current I_BE for the edge population
        cue_R: the right cue input
        cue_L: the left cue input
        U_B: the current bump state
        c_BE: the constant c_BE value that controls the strength of the Bump -> Edge interaction
        return the current I_BE
        """
        center_pos = self.find_current_edge_location(self.r_E)
        r_B = self.bump_states(self.num_B, self.sigma_B, center_pos, self.enc_range)
        I_BE = c_BE * (cue_R * r_B + cue_L * (-r_B))
        # if self.sigma_I_BE is not None:
        #     # Larger c_BE -> narrower kernel (closer to delta); smaller c_BE -> wider kernel.
        #     sigma_eff = self.sigma_I_BE / (bm.abs(c_BE) + 1e-6)
        #     sigma_eff = bm.clip(sigma_eff, 1e-6, bm.pi / 2)
        #     idx = bm.arange(self.num_B)
        #     delta = (idx - center) * self.rho_B
        #     weight = bm.exp(-0.5 * (delta / sigma_eff) ** 2)
        #     weight = weight / bm.sum(weight)
        #     I_BE = 10 * I_BE * weight
        return I_BE
    
    def get_current_I_EB(self, r_E, c_EB, scale=100):
        """
        Get the current I_EB for the bump population
        U_E: the current edge state
        return the current I_EB
        """
        pos = self.find_current_edge_location(r_E)
        #stimulus = self.get_bump_stimulus_by_pos(pos)
        stimulus = scale * bm.abs(bm.diff(r_E, append=r_E[-1]))
        zeros = bm.zeros_like(stimulus)
        return bm.where((pos >= -bm.pi/2) & (pos <= bm.pi/2), c_EB * stimulus, zeros)
    
    # def generate_cues_input(self, dur1, dur2, dt_DDM, p, t_start, seed=None):
    #     num1 = int(dur1 / dt_DDM) 
    #     num2 = int(dur2 / dt_DDM)
    #     rng = np.random.default_rng(seed)
    #     cue_R_all = rng.binomial(1, p, num1+num2)
    #     cue_R_all[:int(t_start/dt_DDM)] = 0
    #     cue_L_all = np.zeros_like(cue_R_all)
    #     cue_L_all = 1 - cue_R_all
    #     cue_L_all[:int(t_start/dt_DDM)] = 0
    #     cue_R_all = np.repeat(cue_R_all, int(dt_DDM))
    #     cue_L_all = np.repeat(cue_L_all, int(dt_DDM))
        
    #     return cue_L_all, cue_R_all
    
    def generate_cues_input(self, dur1, dur2, dt_DDM, p, t_start, seed=None):
        num1 = int(dur1 / dt_DDM) 
        num2 = int(dur2 / dt_DDM)
        rng = np.random.default_rng(seed)
        cue_R = rng.binomial(1, p, num1+num2)
        cue_R[:int(t_start/dt_DDM)] = 0
        cue_L = np.zeros_like(cue_R)
        cue_L = 1 - cue_R
        cue_L[:int(t_start/dt_DDM)] = 0
        cue_R_all = np.zeros(dur1+dur2)
        cue_R_all[0::int(dt_DDM)] = cue_R
        cue_L_all = np.zeros(dur1+dur2)
        cue_L_all[0::int(dt_DDM)] = cue_L

        return cue_L_all, cue_R_all
    
    def get_x_traj(self, t_start, dx, dt_DDM,x0, 
                   cue_R_all, cue_L_all, boundary):
        """
        Compute the decision variable trajectory x_traj according to the DDM rule.
        Before t_start, x_traj is x0. After t_start, at each t, if cue_R_all[t]==1, x increases by dx;
        if cue_L_all[t]==1, x decreases by dx. Once x crosses boundary or 0, it stays at that value.
        No explicit for-loop is used.
        """
        assert len(cue_R_all) == len(cue_L_all), "cue_R_all and cue_L_all must have the same length"
        T = len(cue_R_all)
        x_traj = np.full(T, x0)
        delta = (cue_R_all - cue_L_all) * (dx/dt_DDM)
        delta[:t_start] = 0
        x_cumsum = np.cumsum(delta)
        x_traj = x0 + x_cumsum
        x_traj[:t_start] = x0
        above = x_traj >= boundary
        below = x_traj <= 0
        cross = above | below
        if np.any(cross):
            first_cross = np.argmax(cross)
            x_traj[first_cross+1:] = x_traj[first_cross]
        return x_traj
    
    def get_pos_offset(self, x0, s_E, tol=1e-4, max_iter=50, progress=False):
        """
        Find the position offset for the edge population using a continuous ternary search
        on a unimodal objective f(offset) = (theta_E - theta_B)^2. This is more efficient
        than bisection for unimodal functions and does not require integer offsets.
        Parameters:
        - x0: the initial position
        - s_E: the slope of the edge population
        - tol: stopping tolerance on the search interval width
        - max_iter: maximum iterations to prevent infinite loops
        - progress: if True, print progress each iteration
        Returns the position offset (float)
        """
        pos = self.evidence_to_pos(x0, s_E)
        left = -np.pi/2 - pos
        right = np.pi/2 - pos

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

            # # Terminate early if function value is below tolerance at either probe
            # if f1 <= tol:
            #     if pbar is not None:
            #         pbar.close()
            #     return m1
            # if f2 <= tol:
            #     if pbar is not None:
            #         pbar.close()
            #     return m2

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
        


    @property
    def derivative(self):
        dr_B = lambda r_B, t, Iext_B: (-r_B + self.phi_B(self.I_BB + self.I_EB + Iext_B + self.I_clamp_B+self.I_B_noise)) / self.tau_B
        dr_E = lambda r_E, t, Iext_E: (-r_E + self.phi_E(self.I_EE + self.I_BE + Iext_E + self.I_clamp_E+self.I_E_noise)) / self.tau_E
        return bp.JointEq([dr_B, dr_E])
    

    
    def update(self, x=None):
        _t = bp.share['t']

        self.I_BB[:] = self.J_BB @ self.r_B
        self.I_EE[:] = self.J_EE @ self.r_E
        # Use pre-created c_BE function (created during initialization)
        #self.c_BE_dyn[:] = self.c_BE_func(self.find_current_bump_location(self.r_B))
        self.c_BE_dyn[:] = 1 * self.c_BE
        self.I_E_noise[:] = self.noise_scale_edge * bm.random.normal(0, 1, self.num_E)
        self.I_B_noise[:] = self.noise_scale_bump * bm.random.normal(0, 1, self.num_B)
        self.I_BE[:] = ~self.hit_boundary * bm.where(_t < self.t_start, 0, self.get_current_I_BE(self.cue_R, self.cue_L, self.r_B, self.c_BE_dyn))
        self.I_EB[:] = ~self.hit_boundary * bm.where(_t < self.t_start, 0, self.get_current_I_EB(self.r_E, self.c_EB))
        #self.I_BE[:] = 0
        #self.I_EB[:] = 0
        
        r_B, r_E = self.integral(self.r_B, self.r_E, _t, self.Iext_B, self.Iext_E)
        self.r_B[:] = r_B
        self.r_E[:] = r_E
        self.theta_B[:] = self.find_current_bump_location(self.r_B)
        self.theta_E[:] = self.find_current_edge_location(self.r_E)
        self.x_B[:] = self.pos_to_evidence(self.theta_B, self.sigma_E)
        self.x_E[:] = self.pos_to_evidence(self.theta_E, self.sigma_E)
        hit_boundary = bm.logical_or(self.theta_E >= bm.pi/2, self.theta_E <= -bm.pi/2)
        # assign boolean array directly (shapes should match: both length 1)
        self.hit_boundary[:] = bm.where(self.hit_boundary, self.hit_boundary, hit_boundary)

        # Update the external input
        self.Iext_B[:] = 0.
        self.Iext_E[:] = 0.
    
        
    def run_simulation(self, mon_vars, pos_offset=0, progress_bar=True, dt=1., get_RT=False):
        pos_B = self.evidence_to_pos(self.x0, self.sigma_E)
        pos_E = self.evidence_to_pos(self.x0, self.sigma_E)
        I1 = self.get_bump_stimulus_by_pos(pos_B)
        I2 = self.get_edge_stimulus_by_pos(pos_E+pos_offset, self.edge_type)
        I1 = 0
        I2 = 0
        dur1 = self.dur1
        dur2 = self.dur2
        t_start = self.t_start

        Iext_B, duration = bp.inputs.section_input(values=[I1, 0], durations=[dur1/dt, dur2/dt], return_length=True, dt=dt)
        Iext_E, duration = bp.inputs.section_input(values=[I2, 0], durations=[dur1/dt, dur2/dt], return_length=True, dt=dt)
        runner=bp.DSRunner(self, inputs=[('Iext_B', Iext_B, 'iter'), ('Iext_E', Iext_E, 'iter'),
                                          ('cue_R', self.cue_R_all, 'iter', '='), 
                                          ('cue_L', self.cue_L_all, 'iter' ,'=')],
                                          monitors=mon_vars, dyn_vars=self.vars(), progress_bar=progress_bar, dt=dt)
        runner.run(dur1+dur2)
        if get_RT:
            assert 'hit_boundary' in mon_vars, "hit_boundary must be in mon_vars when get_RT is True"
            RT = self.get_RT(t_start, runner.mon.hit_boundary)
            if RT:
                return runner, RT/dt
            else:
                return runner, None
        
        else:
            return runner

       

        
    


            





import numpy as np
import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from jax import lax
from scipy.linalg import circulant
import scipy
from tqdm import tqdm 



class CANN_DDM_model(bp.dyn.NeuDyn):
    def __init__(self, CANN_params=None, **kwargs):
    
        if CANN_params is not None:
            edge_params = CANN_params.get('edge_pop', {})
            bump_params = CANN_params.get('bump_pop', {})
            decision_space_params = CANN_params.get('decision_space_params', {})

            # initialize edge and bump using provided sub-dictionaries
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
        self.k_B = kwargs.get('k_B', 1)
        self.num_B = kwargs.get('num_B', 1024)
        self.tau_B = kwargs.get('tau_B', 1)
        self.c_BE = kwargs.get('c_BE', 1)
        # c_BE_params contains both mode and parameters, e.g. {'mode': 'linear', 'k_linear': 2.0}
        self.c_BE_params = kwargs.get('c_BE_params', {'mode': 'const'})
        self.c_BE_func = self.c_BE_theta(self.c_BE_params)
        self.U_B = bm.Variable(bm.zeros(self.num_B))
        self.V_B = bm.Variable(bm.zeros(self.num_B))
        self.r_B = bm.Variable(bm.zeros(self.num_B))
        self.J0_B = kwargs.get('J0_B', 4)
        self.a_B = kwargs.get('a_B', 1)
        self.J_BB = self.make_bump_conn_mat(self.num_B, self.J0_B, self.a_B)
        self.I_BB = bm.Variable(bm.zeros(self.num_B))
        self.I_BE = bm.Variable(bm.zeros(self.num_B))
        self.Iext_B = bm.Variable(bm.zeros(self.num_B))
        self.x_B = bm.Variable(bm.zeros(1)) # decision variable read out from the bump population
        self.theta_B = bm.Variable(bm.zeros(1))
        self.m_B = kwargs.get('m_B', 1)
        self.tau_VB = kwargs.get('tau_VB', 10)
        self.c_BE_dyn = bm.Variable(bm.zeros(1))



    def _init_edge_pop(self, **kwargs):
        """
        Initialize the parameters for the edge population 
        and define the variables that will be used to track the dynamics of the edge attractor 
        """

        # Initialize the parameters for the edge population
        self.num_E = kwargs.get('num_E', 1024)
        self.tau_E = kwargs.get('tau_E', 1)
        self.c_EB = kwargs.get('c_EB', 1.)
        self.beta_E = kwargs.get('beta_E', 2.5)
        self.sigma_E = kwargs.get('sigma_E', 1)
        self.fixed_ratio = kwargs.get('fixed_ratio', 0.05)
        self.J0_E = kwargs.get('J0_E', 1)
        self.edge_type = kwargs.get('edge_type', 'Laplace')
        assert self.edge_type in ['Laplace', 'tanh'], "Edge type should be either Laplace or tanh."
        self.edge_offset = kwargs.get('edge_offset', None)
        if self.edge_offset is None:
            self.optimize_offset = True
        else:
            self.optimize_offset = False
        self.direction = kwargs.get('direction', 'left')
        self.s_E = kwargs.get('s_E', 1)
        self.k1 = kwargs.get('k1', int(0.3 * self.num_E))
        self.k2 = kwargs.get('k2',int(0.5 * self.num_E))
        self.k0 = kwargs.get('k0', int(self.k1+self.k2)/2)
        assert self.k1 < self.k2, "k1 should be greater than k2"
        assert self.k0 > self.k1 and self.k0 < self.k2, "k0 should be between k1 and k2"

        # Initialize the variables for the edge population
        self.U_E = bm.Variable(bm.zeros(kwargs.get('num_E', 1024)))
        self.r_E = bm.Variable(bm.zeros(kwargs.get('num_E', 1024)))
        self.c_EB_dym = bm.Variable(bm.zeros(1))
        self.fixed_end_width = int(self.fixed_ratio * self.num_E)
        self.mask = self.make_mask(self.num_E, self.fixed_end_width)
        self.U_E0 = self.edge_states(self.num_E, self.s_E, self.k0, self.k1, self.k2, self.edge_type)
    
        
        self.U_E = bm.Variable(self.U_E0)
        J0_E_min = self.get_min_J0_E(self.sigma_E, self.U_E0)
        if self.J0_E < J0_E_min:
            print(f"J0_E is smaller than the minimum value, set to {J0_E_min: .2f}")
            self.J0_E = J0_E_min
        self.J_EE = self.make_edge_conn_mat(self.num_E, self.s_E, self.k1, 
                                            self.k2, self.J0_E, direction=self.direction, 
                                            optimize_offset=self.optimize_offset, 
                                            edge_offset=self.edge_offset, edge_type=self.edge_type)
        self.I_EE = bm.Variable(bm.zeros(self.num_E))
        self.I_EB = bm.Variable(bm.zeros(self.num_E))
        self.Iext_E = bm.Variable(bm.zeros(self.num_E))
        self.x_E = bm.Variable(bm.zeros(1)) # decision variable read out from the edge population
        self.theta_E = bm.Variable(bm.zeros(1))

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


        
    def find_optimal_edge_offset(self, sigma_E, U_E0, J0_E, k1, k2, direction='right', edge_type='Laplace', tol=1e-5, max_iter=50, progress=False):
        """
        Find the optimal edge offset for the edge population by searching over a range of offsets.
        Use a continuous ternary search (efficient for unimodal functions) to find the real-valued
        offset that minimizes ||tanh(sigma_E * J_EE(offset) @ U_E0) - U_E0||.
        If progress=True, show a tqdm progress bar for iterations.
        """
        # Define search range (fraction of total neurons). Tunable if needed.
        max_shift = max(1, int(0.2 * self.num_E))
        left = -float(max_shift)
        right = float(max_shift)

        def err_at(offset):
            J_EE_temp = self.make_edge_conn_mat(self.num_E, self.s_E, k1, k2, J0_E,
                                                direction=direction, edge_type=edge_type,
                                                edge_offset=offset, optimize_offset=False)
            U_star = bm.tanh(sigma_E * (J_EE_temp @ U_E0))
            err = bm.sqrt(bm.sum(bm.square(U_star - U_E0)))
            try:
                return float(err)
            except Exception:
                return float(bm.to_numpy(err))

        it = 0
        pbar = None
        if progress:
            pbar = tqdm(total=max_iter, desc='find_optimal_edge_offset')

        # Continuous ternary search
        while (right - left) > tol and it < max_iter:
            m1 = left + (right - left) / 3.0
            m2 = right - (right - left) / 3.0
            err1 = err_at(m1)
            err2 = err_at(m2)
            if progress and pbar is not None:
                pbar.set_postfix({'left': left, 'right': right, 'err1': err1, 'err2': err2})
                pbar.update(1)
            if err1 > err2:
                left = m1
            else:
                right = m2
            it += 1

        if pbar is not None:
            pbar.close()

        best_offset = (left + right) / 2.0
        best_err = err_at(best_offset)
        print(f"find_optimal_edge_offset: chosen offset {best_offset:.6f} with error {best_err:.6f}")
        return best_offset
    
    def edge_states(self, num, s, k0, k1, k2, type='Laplace', direction='left'):
        """
        Laplace solution for the edge attractor dynamics (synaptic input)
        num: number of neurons
        s: the population parameter for the edge population
        k0: the center of the edge population
        k1: the left encoding boundary of the edge population
        k2: the right encoding boundary of the edge population
        t0: the time constant of the edge population
        k: the width of the edge population
        return the ideal solution of the edge attractor dynamics as a function of neuron index k
        """
        sign = -1 if direction == 'left' else 1
        if type == 'Laplace':
            k = bm.arange(num)
            return 2 * bm.exp(-bm.exp(sign*s*(bm.pi/(k2-k1))*(k-k0))) - 1
        elif type == 'tanh':
            s_prime = 2 * s / bm.exp(1)
            k = bm.arange(num)
            return bm.tanh(-sign * s_prime * bm.pi/(k2-k1) * (k-k0))
        else:
            raise ValueError('Edge type should be either Laplace or tanh.')
        
    def theta_req(self):
        theta_req_pos = lambda theta: 1/(self.s_E*self.dt_DDM) * bm.log(1 + self.dx / self.pos_to_evidence(theta, self.s_E))
        theta_req_neg = lambda theta: -1/(self.s_E*self.dt_DDM) * bm.log(1 - self.dx / self.pos_to_evidence(theta, self.s_E))
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
    
    
    def derivative_edge_states(self, num, s, k0, k1, k2, type='Laplace', direction='left'):
        sign = 1 if direction == 'left' else -1
        if type == 'Laplace':
            k = bm.arange(num)
            return -2 * s * bm.exp(sign*s*(bm.pi/(k2-k1))*(k-k0)) * bm.exp(-bm.exp(sign*s*(bm.pi/(k2-k1))*(k-k0))) * sign * bm.pi/(k2-k1)
        elif type == 'tanh':
            s_prime = 2 * s / bm.exp(1)
            k = bm.arange(num)
            return s_prime * sign * bm.pi/(k2-k1) * (1 - bm.square(bm.tanh(sign * s_prime * bm.pi/(k2-k1) * (k-k0))))
        else:
            raise ValueError('Edge type should be either Laplace or tanh.')
    

    def bump_states(self, num, k0, k1, k2, a):
        k = bm.arange(num)
        return bm.exp(-(bm.pi/(2*a*(k2-k1)))**2*(k-k0)**2) 
    
    def fr_E(self, U_E):
        return bm.tanh(self.sigma_E * U_E)

    def fr_B(self, U_B):
        return bm.square(U_B) / (1.0 + self.k_B * bm.sum(bm.square(U_B)))

     # Distance conversion to the range [-z_range/2, z_range/2)
    def dist(self, d):
        d = bm.remainder(d, self.z_range)
        d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d



    # Compute the connection matrix for the primary population
    def make_bump_conn_mat(self, num, J0_B, a_B):
        a_eff = bm.sqrt(2) * a_B * (self.k2-self.k1) / bm.pi
        idx = bm.arange(num)
        d0 = idx - idx[:, None]
        range = num
        d0 = bm.remainder(d0, range)
        d = bm.where(d0 > 0.5 * range, d0 - range, d0)
        Jxx = J0_B * bm.exp(-0.5 * bm.square(d / a_eff)) / (bm.sqrt(2 * bm.pi) * a_eff)
        return Jxx
    

    
    def make_edge_conn_mat(self, num, s, k1, k2,
                           J0_E, direction='left', edge_type='Laplace', 
                           edge_offset=None, optimize_offset=False):
        """
        Create edge connection matrix. If find_optimal_offset is True, automatically search
        for the optimal offset and use it (printing the chosen offset). Otherwise, use the
        provided edge_offset.
        """
        # If requested, find an optimal offset automatically
        if optimize_offset:
            if edge_offset is not None:
                print("make_edge_conn_mat: `edge_offset` provided but will be ignored because `optimize_offset=True`")
            optimal = self.find_optimal_edge_offset(self.sigma_E, self.U_E0, J0_E, k1, k2,
                                                    direction=direction, edge_type=edge_type)
            edge_offset = optimal
            print(f"make_edge_conn_mat: using optimal edge_offset {edge_offset}")
        else:
            if edge_offset is None:
                raise ValueError('edge_offset must be provided when optimize_offset is False')

        num_edge = int(0.1 * num)
        nvals = num - num_edge - bm.arange(num- num_edge + 1)
        k0 = (num - num_edge) / 2 - edge_offset
        kernel = self.derivative_edge_states(len(nvals), s, k0, k1, k2, type=edge_type)

        if direction == 'left':
            kernel = bm.flip(kernel)
        elif direction != 'right':
            raise ValueError('Direction should be either right or left.')
        self.kernel = kernel

        return J0_E * self._make_conn_mat_from_kernel(num, kernel, normed=True)


    def _make_conn_mat_from_kernel(self, num, kernel, normed=False):
        """
        Takes a 1D list specifying the shape of an interaction kernel and produces an
        interaction matrix for neurons lying along a 1D space.
        
        The kernel is assumed to be zero outside of the range [-N/2,N/2].

        
        num                  : the number of neurons 
        discreteKernel       : Should have an odd length <= N.  The kernel is centered on the
                                middle element (index (len(discreteKernel)-1)/2).
        N                    : The number of neurons (dimension of the output matrix is NxN)
        normed (True)        : If True, each row is normalized to sum to 1.
        """
        # Nk = len(Kernel)
        # assert(Nk%2==1) # discreteKernel should have odd length
        # assert(Nk <= self.num) # discreteKernel should have length less than or equal to N
        # mat = bm.zeros((self.num, self.num))
        
        # # copy in the appropriate part of the kernel for each row
        # for i in range(self.num):
        #     matIndexMin = max(0,i-(Nk-1)//2)
        #     matIndexMax = min(i+Nk-(Nk-1)//2, self.num)
        #     kIndexMin = 0 + max(0, matIndexMin - (i-(Nk-1)//2))
        #     kIndexMax = Nk - max(0,i+Nk-(Nk-1)//2 - self.num)
        #     mat[i,matIndexMin:matIndexMax] = Kernel[kIndexMin:kIndexMax]
        #     Nk = Kernel.size
        Nk = len(kernel)
        assert Nk % 2 == 1, "Kernel length must be odd."
        assert Nk <= num, "Kernel length must be less than or equal to N."
        
        m = (Nk - 1) // 2  # center index of the kernel
        # Create row and column indices for the N x N matrix.
        i = bm.arange(num).reshape(num, 1)
        j = bm.arange(num).reshape(1, num)
        
        # The relative offset from the kernel's center is (j-i)
        diff = j - i
        # The kernel index is diff shifted by m.
        indices = diff + m
        
        # Create a mask for valid indices (i.e., where the kernel is defined).
        valid = (indices >= 0) & (indices < Nk)
        
        # Build the connection matrix.
        conn = bm.zeros((num, num))
        conn[valid] = kernel[indices[valid]]
        if normed:
            # Normalize each row to sum to 1.
            row_sums = conn.sum(axis=1, keepdims=True)
            # Avoid division by zero.
            row_sums[row_sums == 0] = 1
            conn = conn / row_sums
        return conn
    
    def get_edge_clamped(self, state, direction='left'):
        state_clamped = bm.copy(state)
        if direction == 'left':
            state_clamped[:self.fixed_end_width] = 100
            state_clamped[-self.fixed_end_width:] = -100
            return state_clamped
        elif direction == 'right':
            state_clamped[:self.fixed_end_width] = -100
            state_clamped[-self.fixed_end_width:] = 100
            return state_clamped
        else:
            raise ValueError('Direction should be either right or left.')

    def make_mask(self, num, fixed_end_width):
        mask = bm.zeros(num, dtype=bool)
        mask = mask.at[fixed_end_width+50:-fixed_end_width-50].set(True)
        return mask
    
    def get_min_J0_E(self, sigma_E, U_E0, edge_type='Laplace'):
        J_EE = self.make_edge_conn_mat(self.num_E,  self.s_E, 
                                            self.k1, self.k2, 1, 
                                            direction='left', edge_offset=0, edge_type=edge_type)
        res = J_EE @ np.tanh(sigma_E * U_E0)
        return 1 / np.max(res) 


    
    def idx_to_pos(self, idx):
        return bm.where(
            idx < self.k1,
            -bm.pi/2,
            bm.where(
                idx > self.k2,
                bm.pi/2,
                -bm.pi/2 + (idx - self.k1) / (self.k2 - self.k1) * bm.pi
            )
        )
    
    def pos_to_idx(self, pos):
        # Accept either python floats or brainpy/jax scalars/arrays.
        # Clamp to valid range and return a bm-compatible numeric index (not a Python int),
        # so it can be used inside JIT/array computations and downstream functions.
        pos = bm.asarray(pos)
        pos = bm.clip(pos, -bm.pi/2, bm.pi/2)
        idx = self.k1 + (pos + bm.pi/2) / bm.pi * (self.k2 - self.k1)
        return idx
    
    def pos_to_evidence(self, pos, s, direction = 'left'):
        if direction == 'left':
            return self.boundary * bm.exp(s * (pos - 0.5 * bm.pi))
   
    def evidence_to_pos(self, evidence, s, direction='left'):
        if direction == 'left':
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
        k0 = self.pos_to_idx(pos)
        # bump stimulus lives on the bump population size (num_B)
        return self.bump_states(self.num_B, k0, self.k1, self.k2, self.a_B)
    
    def get_edge_stimulus_by_pos(self, pos, edge_type='Laplace'):
        k0 = self.pos_to_idx(pos)
        return self.edge_states(self.num_E, self.s_E, k0, self.k1, self.k2, edge_type)
        
    
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
        c_BE_theta: callable function that takes theta as input and returns the c_BE value
        return the current I_BE
        """

        I_BE = c_BE * (cue_R * (-r_B) + cue_L * (r_B))    
        return I_BE
    
    def get_current_I_EB(self, r_E, c_EB):
        """
        Get the current I_EB for the bump population
        U_E: the current edge state
        return the current I_EB
        """
        pos = self.find_current_edge_location(r_E)
        stimulus = self.get_bump_stimulus_by_pos(pos)
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
        dU_B = lambda U_B, t, Iext_B: (-U_B + self.I_BB + self.I_EB + Iext_B - self.V_B) / self.tau_B
        dV_B = lambda V_B, t, Iext_B: (-V_B + self.m_B * self.U_B) / self.tau_VB
        dU_E = lambda U_E, t, Iext_E: (-U_E + self.I_EE + self.I_BE + Iext_E) / self.tau_E
        return bp.JointEq([dU_B, dV_B, dU_E])
    
    def update(self, x=None):
        _t = bp.share['t']
        self.r_B = self.fr_B(self.U_B)
        self.r_E = self.fr_E(self.U_E)
        self.I_BB[:] = self.J_BB @ self.r_B
        self.I_EE[:] = self.J_EE @ self.r_E
        # Use pre-created c_BE function (created during initialization)
        self.c_BE_dyn[:] = self.c_BE_func(self.find_current_bump_location(self.r_B))
        self.I_BE[:] = ~self.hit_boundary * bm.where(_t < self.t_start, 0, self.get_current_I_BE(self.cue_R, self.cue_L, self.r_B, self.c_BE_dyn))
        self.I_EB[:] = ~self.hit_boundary * bm.where(_t < self.t_start, 0, self.get_current_I_EB(self.r_E, self.c_EB))
        U_B, V_B, U_E = self.integral(self.U_B, self.V_B, self.U_E, _t, self.Iext_B, self.Iext_E)
        self.U_B[:] = U_B
        self.U_E[:] = U_E
        self.V_B[:] = V_B
        self.theta_B[:] = self.find_current_bump_location(self.r_B)
        self.theta_E[:] = self.find_current_edge_location(self.r_E)
        self.x_B[:] = self.pos_to_evidence(self.theta_B, self.s_E)
        self.x_E[:] = self.pos_to_evidence(self.theta_E, self.s_E)
        hit_boundary = bm.logical_or(self.theta_E >= bm.pi/2, self.theta_E <= -bm.pi/2)
        # assign boolean array directly (shapes should match: both length 1)
        self.hit_boundary[:] = bm.where(self.hit_boundary, self.hit_boundary, hit_boundary)

        # Update the external input
        self.Iext_B[:] = 0.
        self.Iext_E[:] = 0.
    
        
    def run_simulation(self, mon_vars, pos_offset=0, progress_bar=True, dt=1., get_RT=False):
        pos_B = self.evidence_to_pos(self.x0, self.s_E)
        pos_E = self.evidence_to_pos(self.x0, self.s_E)
        I1 = self.get_bump_stimulus_by_pos(pos_B)
        I2 = self.get_edge_stimulus_by_pos(pos_E+pos_offset, self.edge_type)
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

       

        
    


            





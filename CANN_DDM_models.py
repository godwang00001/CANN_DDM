import numpy as np
import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from jax import lax
from scipy.linalg import circulant
import scipy
from tqdm import tqdm 
from jaxopt import Bisection, GaussNewton
path_dir = '/Users/wangchenyu/research/CANN/CANN_diffusion_model/'
bm.set_platform('gpu')

class CANN_DDM_base_model(bp.dyn.NeuDyn):
  def __init__(self, seed=None, num=1024, m=0, tau_bump=1., tau_v=10., k=8.1, a=0.5, A=10., J0_bump=4.,
              z_min=-bm.pi, z_max=bm.pi, c=1, boundary=1, t_prep=0,  **kwargs):
      super(CANN_DDM_base_model, self).__init__(size=num, **kwargs)

      # 0. Initialize the random seed
      if seed is not None:
        bm.random.seed(seed)
      # 1. Initialize parameters
      self.tau_bump = tau_bump
      self.tau_v = tau_v  # time constant of SFA
      self.k = k
      self.a = a
      self.A = A
      self.J0_bump = J0_bump
      self.m = m  # SFA strength
      self.c = c  # shifter strength
      self.boundary = boundary
      self.t_prep = t_prep

      # 2. Initialize feature space parameters
      self.z_min = z_min
      self.z_max = z_max
      self.z_range = z_max - z_min
      
      self.x = bm.linspace(z_min, z_max, num)
      self.n = bm.arange(self.num) # neuron index
      self.rho = num / self.z_range
      self.dx = self.z_range / num

      # 3. Initialize variables for the bump population
      self.u = bm.Variable(bm.zeros(num))
      self.v = bm.Variable(bm.zeros(num))  # SFA current
      self.r = bm.Variable(bm.zeros(num))  # output firing rate
      self.Irec = bm.Variable(bm.zeros(num))  # recurrent input
      self.u_pos = bm.Variable(0.0)  # record the position of the bump
      self.ul_pos = bm.Variable(0.0)  # record the position of the bump
      self.u_dpos =  bm.Variable(0.0)
      self.Ishift = bm.Variable(bm.zeros(num))  # shifted input from the evidence
      self.input = bm.Variable(bm.zeros(num))
      self.conn_mat = self.make_gaussian_conn_mat(self.x)  # connection matrix

      # 4. Initialize the dynamical variable to track different evidence

      self.clicks_left = bm.Variable(bm.zeros(1))  # left evidence
      self.clicks_right = bm.Variable(bm.zeros(1))  # right evidence

      # 5. Initialize integration function
      self.integral = bp.odeint(self.derivative)

      # 6. Initialize the assymetric connection configs
      if self.__class__ is CANN_DDM_base_model:  
        self.init_edge_pop()  

  # Initialize the assymetric connection from external source
  def init_edge_pop(self):
    """
    should be rewritten in the Laplace network
    """
    self.conn_left = self.make_conn_asym(self.x, direction='left')
    self.conn_right = self.make_conn_asym(self.x, direction='right')

  # Differential equations for the primary bump population
  
  def derivative(self):
    du = lambda u, t, I_ext: (-u + self.Irec + self.Ishift - self.v + I_ext) / self.tau_bump
    dv = lambda v, t: (-v + self.m * self.u) / self.tau_v
    return bp.JointEq([du, dv])
  
  def get_u_fr(self, u):
    return bm.square(u) / (1.0 + 8.1 * bm.sum(bm.square(u)))


  # Distance conversion to the range [-z_range/2, z_range/2)
  def dist(self, d):
    d = bm.remainder(d, self.z_range)
    d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
    return d

       

  # Compute the connection matrix for the primary population
  def make_gaussian_conn_mat(self, x):
    assert bm.ndim(x) == 1
    d = self.dist(x - x[:, None])  # distance matrix
    Jxx = self.J0_bump * bm.exp(-0.5 * bm.square(d / self.a)) / (bm.sqrt(2 * bm.pi) * self.a)
    return Jxx
  
  def make_conn_mat_from_kernel(self, kernel, normed=True):
    """
    Takes a 1D list specifying the shape of an interaction kernel and produces an
    interaction matrix for neurons lying along a 1D space.
    
    The kernel is assumed to be zero outside of the range [-N/2,N/2].
    
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
    assert Nk <= self.num, "Kernel length must be less than or equal to N."
    
    m = (Nk - 1) // 2  # center index of the kernel

    # Create row and column indices for the N x N matrix.
    i = bm.arange(self.num).reshape(self.num, 1)
    j = bm.arange(self.num).reshape(1, self.num)
    
    # The relative offset from the kernel's center is (j-i)
    diff = j - i
    # The kernel index is diff shifted by m.
    indices = diff + m
    
    # Create a mask for valid indices (i.e., where the kernel is defined).
    valid = (indices >= 0) & (indices < Nk)
    
    # Build the connection matrix.
    conn = bm.zeros((self.num, self.num))
    conn[valid] = kernel[indices[valid]]
    
    if normed:
        # Normalize each row to sum to 1.
        row_sums = conn.sum(axis=1, keepdims=True)
        # Avoid division by zero.
        row_sums[row_sums == 0] = 1
        conn = conn / row_sums

    return conn
  
  def find_bump_location(self, bump_state):
    """
    compute the bump location given the current bump state
    The location is normalized to lie between -1 and 1.
    """
    return 2 * bm.argmax(bump_state) / self.num - 1
    
      

  def make_conn_asym(self, x, length=20, direction='right'):
    assert bm.ndim(x) == 1
    d = self.dist(x - x[:, None])  # distance matrix
    Jxx = self.J0_edge * bm.exp(-0.5 * bm.square(d / self.a**3)) * bm.abs(d) / (bm.sqrt(2 * bm.pi) * self.a)
    if direction == 'left':
      Jxx = bm.triu(Jxx, k=1)
    elif direction == 'right':
      Jxx = bm.tril(Jxx, k=1)
    else:
      raise ValueError('Direction should be right or left')
    return Jxx

  # Get the stimulus input to each neuron from the neuron at position pos
  def get_stimulus_by_pos(self, pos):
    return self.A * bm.exp(-0.25 * bm.square(self.dist(self.x - pos) / self.a))

  # Update function for the primary bump population
  def update(self, x=None):
    # Update primary bump population
    _t = bp.share['t']
    u2 = bm.square(self.u)
    self.r.value = u2 / (1.0 + self.k * bm.sum(u2))
    self.Irec[:] = self.conn_mat @ self.r
    Ishift = self.c * (self.clicks_left * self.conn_left @ self.r + self.clicks_right * self.conn_right @ self.r)
    Ishift_pause_by_bound = bm.where(bm.sign(self.u_pos) * self.u_pos < self.boundary, Ishift, 0)
    self.Ishift = bm.where(_t<self.t_prep, 0, Ishift_pause_by_bound)
    u, v = self.integral(self.u, self.v, bp.share['t'], self.input)
    self.u[:] = bm.where(u > 0, u, 0)
    self.v[:] = v
    self.u_pos[:] = self.find_bump_location(self.u)
    self.input[:] = 0.  # reset external input current


class CANN_DDM_bump_edge_model(CANN_DDM_base_model):
  def __init__(self, num=1024, tau_edge=1, c1=1., c2=1., 
              boundary=1, beta=2.5, fixed_ratio=0.05,
              J0_edge=1, edge_type='tanh', edge_offset=0, 
              optimize_offset=False, delta_z=1, speed_mode='const', alpha=None, **kwargs):
      super().__init__(num=num, boundary=boundary, **kwargs)
      self.tau_edge = tau_edge
      self.edge_type = edge_type
      self.beta = beta
      #self.x0 = (self.z_max + self.z_min) / 2 # default starting pos for the edge
      self.n = bm.arange(self.num)
      self.n0 = 0.5 * self.num
      self.init_edge_pop(edge_type, edge_offset, optimize_offset,
                       J0_edge, delta_z, fixed_ratio, c1, c2, speed_mode, alpha)
      self.if_bump_hit = bm.Variable(bm.bool(False))
      self.if_edge_hit = bm.Variable(bm.bool(False))
      self.RT = bm.Variable(bm.inf)
      self.first_hit = bm.Variable(bm.bool(False))

      #self.x_pred = bm.Variable(bm.zeros(1)) # theoretical prediction of the decision variable


  def init_edge_pop(self, edge_type, edge_offset, optimize_offset, 
                    J0_edge, delta_z, fixed_ratio, c1, c2, speed_mode, alpha):
    self.edge_offset = edge_offset
    self.delta_z = delta_z
    self.J0_edge = J0_edge
    self.fixed_end_width = int(fixed_ratio * self.num)
    self.c1 = c1
    self.c2 = c2
    self.c1_dyn = bm.Variable(bm.zeros(1))
    self.c2_dyn = bm.Variable(bm.zeros(1))
    if speed_mode == 'log':
      self.alpha = alpha
    elif speed_mode == 'const':
      self.alpha = 1


    if edge_type == 'tanh':
        self.Fnt = lambda n,  n0=None: self.edge_states_tanh(n, -2*bm.exp(-1) * self.delta_z, 
                                                                    n0=self.n0 if n0 is None else n0)
        self.s0 = bm.flip(self.Fnt(self.n, n0=self.n0))
    elif edge_type == 'Laplace':
        self.Fnt =  lambda n, n0=None: self.edge_states_Laplace(n, t=1, n0=self.n0 if n0 is None else n0,
                                                                              delta_z=self.delta_z)
        self.s0 = bm.flip(2 * self.Fnt(self.n, n0=self.n0) - 1)
    self.dFnt = lambda n: self.Fnt(n+self.dx) - self.Fnt(n)
    
    self.s = bm.Variable(self.s0)
    self.Iss = bm.Variable(bm.zeros(self.num))
    mask = bm.zeros_like(self.s0, dtype=bool)
    self.mask = mask.at[self.fixed_end_width+50:-self.fixed_end_width-50].set(True) # mask to cut off the two ends of the edge for display
    if optimize_offset:
      self.edge_offset = self.find_zero_velocity_asymmetric_edge_shift()
    else:
      self.edge_offset = edge_offset
    self.conn_mat_ss = self.make_edge_conn_mat(self.edge_offset, direction = 'right')
    self.Ius = bm.Variable(bm.zeros(self.num))
    self.s_pos = bm.Variable(0.0)
    self.I_pause = bm.Variable(bm.zeros(self.num))
    self.f_kernel = lambda a, x0, x: 1/(bm.sqrt(2*bm.pi) * a) *  bm.exp(-(x-x0)**2/(2*a**2)) 


  def edge_states_Laplace(self, n, t, delta_z=1, n0=10, t0=1):
    #return 2 * bm.exp(-bm.log(2) * (t/t0) *(bm.exp(-delta_z*(n-n0)))) - 1
    return bm.exp( -(t/t0)*bm.exp(-delta_z*(n-n0)+bm.log(bm.log(2))) ) 

  def bump_states_Laplace(self, n, t, delta_z=1, n0=10, t0=1):
    return (t/t0) * self.edge_states_Laplace(
                n,t,delta_z=delta_z,n0=n0,t0=t0)*delta_z/np.exp(delta_z*(n-n0))

  def edge_states_tanh(self, n, beta, n0):
    return -bm.tanh(beta * (n - n0))

  def make_edge_conn_mat(self, edge_offset, direction='right'):
    t0 = 1 # this value shouldn't matter; just needs to be nonzero
    num_edge = int(0.1*self.num)
    nvals = self.num - num_edge - bm.arange(self.num-num_edge+1)
    n0 = (self.num-num_edge) / 2 - edge_offset
    if direction == 'right':
      self.kernel = self.bump_states_Laplace(nvals, t0, delta_z=self.delta_z, n0=n0, t0=t0)
    elif direction == 'left':
      self.kernel = bm.flip(self.bump_states_Laplace(nvals, t0, delta_z=self.delta_z, n0=n0, t0=t0))
    else:
      raise ValueError('Direction should be either right or left.')

    return self.J0_edge * self.make_conn_mat_from_kernel(self.kernel, self.num)

  def _get_edge_one_step_velocity(self, edge_offset):
    """
    Change in position of the asymmetric edge after going once through the synaptic
    nonlinearity and given interaction matrix.
    """
    conn_mat_srsr = self.make_edge_conn_mat(edge_offset)
    s_r0_init_state = self.find_fixed_point(conn_mat_srsr, self.s_r0) # compute the initial state 
    current_edge_loc = self.find_edge_location(s_r0_init_state)
    next_edge_state = conn_mat_srsr @ bm.tanh(self.beta * self.s_r0)
    next_edge_loc = self.find_edge_location(next_edge_state)
    return (next_edge_loc - current_edge_loc)**2

  def find_fixed_point(self, J, initial_guess):
    dyn = lambda x: -x + J @ bm.tanh(self.beta * x)
    #solver = GaussNewton(residual_fun=dyn)
    #solver = Broyden(fun=dyn)
    #solution = solver.run(initial_guess).params
    #return solution
    sol = scipy.optimize.root(dyn, initial_guess, tol=1e-10)
    return sol.x

  def get_s_fr(self, s):
    return bm.tanh(self.beta * s)


  def find_edge_location(self, edge_state):
    """
    compute the edge location given the current edge state.
    The location is normalized to lie between -1 and 1.
    """
    diff = bm.diff(bm.sign(edge_state))
    abs_diff = bm.abs(diff)
    i = bm.argmax(abs_diff) 
    # if abs_diff[i] == 0:
    #     raise ValueError("The edge neurons are not at a stable state.")
    n_star = self.n[i] - edge_state[i] * (self.n[i+1] - self.n[i]) / (edge_state[i+1] - edge_state[i]) 
    return 2 * ((n_star-self.fixed_end_width) / (self.num-2*self.fixed_end_width)) - 1

  def find_bump_location(self, bump_state):
    """
    compute the bump location given the current bump state
    The location is normalized to lie between -1 and 1.
    """
    return 2 * ((bm.argmax(bump_state)-self.fixed_end_width) / (self.num-2*self.fixed_end_width)) - 1



  def find_zero_velocity_asymmetric_edge_shift(self):
    """
    Find interaction matrix edge shift for an asymmetric edge with zero velocity under
    the dynamics with given sigma_edge.

    Note: sigma_edge = 0 should correspond to zero velocity with edge shift = 0.
    """
    vel_sq = lambda edge_offset: self._get_edge_one_step_velocity(edge_offset)
    solver = GaussNewton(fun=vel_sq, maxiter=50)
    result = solver.run(0)
    return result.params if result.converged else None
    #result = scipy.optimize.minimize_scalar(vel_sq) #,bracket=(-1,0,1))
    #assert(result['fun'] < 1e-10) # make sure we were successful
    #return result['x']

  def get_edge_clamped(self, state, ratio=0.01, direction='right'):
    state_clamped = bm.copy(state)
    if direction == 'right':
      state_clamped[:self.fixed_end_width] = 100
      state_clamped[-self.fixed_end_width:] = -100
      return state_clamped
    elif direction == 'left':
      state_clamped[:self.fixed_end_width] = -100
      state_clamped[-self.fixed_end_width:] = 100
      return state_clamped
    else:
      return ValueError('Direction should be either right or left.')



  @property
  def derivative(self):
      du = lambda u, t, I_ext: (-u + self.Irec + self.Ishift - self.v + I_ext) / self.tau_bump
      dv = lambda v, t: (-v + self.m * self.u) / self.tau_v
      ds = lambda s, t:(-s + self.Iss + self.Ius) / self.tau_edge

      return bp.JointEq([du, dv, ds])


  def update(self, x=None):
    _t = bp.share['t']
    # Compute the new hit condition
    if_bump_hit= bm.where(bm.sign(self.u_pos) * self.u_pos < self.boundary, bm.bool(False), bm.bool(True))
    if_edge_hit= bm.where(bm.sign(self.s_pos) * self.s_pos < self.boundary, bm.bool(False), bm.bool(True))
    # Once hit is True, keep it True for subsequent updates
    self.first_hit = bm.logical_and(if_bump_hit, bm.logical_not(self.if_bump_hit))
    self.RT = bm.where(self.first_hit, _t-self.t_prep, self.RT)
    self.if_bump_hit = bm.logical_or(self.if_bump_hit, if_bump_hit)
    self.if_edge_hit = bm.logical_or(self.if_edge_hit, if_edge_hit)
  

    u2 = bm.square(self.u)
    self.r.value = u2 / (1.0 + 8.1 * bm.sum(u2))
    self.Irec[:] = self.conn_mat @ self.r
    self.Iss[:] = self.conn_mat_ss @ bm.tanh(self.beta * self.s)
    
    Ius = self.clicks_right * self.u + self.clicks_left * (-self.u)
    #Ius = bm.where(_t<self.t_prep, 0, self.c1 * Ius)
    self.c1_dyn[:] = self.c1 * self.alpha ** (-3*(self.boundary-bm.abs(self.u_pos)))
    self.c2_dyn[:] = self.c2
    self.Ius[:] = bm.where(bm.logical_or(self.if_bump_hit, _t < self.t_prep), 0, self.c1_dyn * Ius)


    #Ishift = self.c2*(self.clicks_left * self.conn_left @ self.r + self.clicks_right * self.conn_right @ self.r)
    Ishift = bm.zeros_like(self.s0)
    Ishift[self.mask] = -self.c2_dyn * bm.diff(self.s[self.mask], prepend=self.s[self.mask][0])
    #Ishift = bm.where(bm.sign(self.u_pos) * self.u_pos < self.boundary, Ishift, 0)
    #self.Ishift[:] = bm.where(_t<self.t_prep, 0, Ishift)
    self.Ishift[:] = bm.where(bm.logical_or(self.if_bump_hit, _t < self.t_prep), 0, Ishift)
    #self.I_pause[:] = bm.where(bm.sign(self.u_pos) * self.u_pos < self.boundary, 0, -10)
    self.I_pause[:] = bm.where(self.if_bump_hit, -10, 0)
    u, v, s = self.integral(self.u, self.v, self.s, _t, self.input)
    self.u[:] = bm.where(u>0, u, 0)
    self.v[:] = v 
    self.s[:] = bm.where(_t<self.t_prep, self.get_edge_clamped(s, direction='right'), s)
    #self.s[:] = self.get_edge_clamped(s, direction='right')
    u_pos = self.find_bump_location(self.u)
    self.u_dpos = u_pos - self.u_pos
    self.u_pos = u_pos
    self.s_pos = self.find_edge_location(self.s)
    self.input[:] = 0.

class CANN_DDM_bump_edge_model_v2(CANN_DDM_bump_edge_model):
  def __init__(self, num=1024, tau_edge=1, c1=1., c2=1., 
              boundary=1, beta=2.5, fixed_ratio=0.05,
              J0_edge=1, edge_type='tanh', edge_offset=0, 
              optimize_offset=False, delta_z=1, speed_mode='const', alpha=None, **kwargs):
      super().__init__(num=num, boundary=boundary, **kwargs)
      self.tau_edge = tau_edge
      self.edge_type = edge_type
      self.beta = beta
      #self.x0 = (self.z_max + self.z_min) / 2 # default starting pos for the edge
      self.n = bm.arange(self.num)
      self.n0 = 0.5 * self.num
      #self.integral = bp.odeint(self.derivative)
      self.init_edge_pop(edge_type, edge_offset, optimize_offset,
                         J0_edge, delta_z, fixed_ratio, c1, c2, speed_mode, alpha)
      
      self.if_bump_hit = bm.Variable(bm.bool(False))
      self.if_edge_hit = bm.Variable(bm.bool(False))
      self.RT = bm.Variable(bm.inf)
      self.Iuu = bm.Variable(bm.zeros(self.num))
      self.Iulul = bm.Variable(bm.zeros(self.num))
      self.first_hit = bm.Variable(bm.bool(False))
      self.input1 = bm.Variable(bm.zeros(num))
      self.input2 = bm.Variable(bm.zeros(num))

  def init_edge_pop(self, edge_type, edge_offset, optimize_offset, 
                    J0_edge, delta_z, fixed_ratio, c1, c2, speed_mode, alpha):
    self.edge_offset = edge_offset
    self.delta_z = delta_z
    self.J0_edge = J0_edge
    self.fixed_end_width = int(fixed_ratio * self.num)
    self.c1 = c1
    self.c2 = c2
    self.c1_dyn = bm.Variable(bm.zeros(1))
    self.c1_dyn_left = bm.Variable(bm.zeros(1))
    self.c2_dyn = bm.Variable(bm.zeros(1))

    if speed_mode == 'log':
      self.alpha = alpha
    elif speed_mode == 'const':
      self.alpha = 1


    if edge_type == 'tanh':
        self.Fnt = lambda n,  n0=None: self.edge_states_tanh(n, -2*bm.exp(-1) * self.delta_z, 
                                                                    n0=self.n0 if n0 is None else n0)
        self.s0 = bm.flip(self.Fnt(self.n, n0=self.n0))
    elif edge_type == 'Laplace':
        self.Fnt =  lambda n, n0=None: self.edge_states_Laplace(n, t=1, n0=self.n0 if n0 is None else n0,
                                                                              delta_z=self.delta_z)
        self.s0 = bm.flip(2 * self.Fnt(self.n, n0=self.n0) - 1)
    self.dFnt = lambda n: self.Fnt(n+self.dx) - self.Fnt(n)
    
    self.s = bm.Variable(self.s0)
    self.s_l = bm.Variable(self.s0)
    self.u = bm.Variable(bm.zeros(self.num))
    self.u_l = bm.Variable(bm.zeros(self.num))
    self.Isrsr = bm.Variable(bm.zeros(self.num))
    self.Islsl = bm.Variable(bm.zeros(self.num))
    self.Ishift = bm.Variable(bm.zeros(self.num))
    self.Ishift_l = bm.Variable(bm.zeros(self.num))
    mask = bm.zeros_like(self.s0, dtype=bool)
    self.mask = mask.at[self.fixed_end_width+50:-self.fixed_end_width-50].set(True) # mask to cut off the two ends of the edge for display
    if optimize_offset:
      self.edge_offset = self.find_zero_velocity_asymmetric_edge_shift()
    else:
      self.edge_offset = edge_offset
    self.conn_mat_ss = self.make_edge_conn_mat(self.edge_offset, direction = 'right')
    self.Iss = bm.Variable(bm.zeros(self.num))
    self.Islsl = bm.Variable(bm.zeros(self.num))
    self.Ius = bm.Variable(bm.zeros(self.num))
    self.Ius_l = bm.Variable(bm.zeros(self.num))
    self.s_pos = bm.Variable(0.0)
    self.sl_pos = bm.Variable(0.0)
    self.I_pause = bm.Variable(bm.zeros(self.num))
    self.f_kernel = lambda a, x0, x: 1/(bm.sqrt(2*bm.pi) * a) *  bm.exp(-(x-x0)**2/(2*a**2)) 
    
  @property
  def derivative(self):
    du = lambda u, t, I_ext1: (-u + self.Iuu + self.Ishift + I_ext1) / self.tau_bump
    du_l = lambda u_l, t, I_ext2: (-u_l + self.Iulul + self.Ishift_l + I_ext2) / self.tau_bump
    dv = lambda v, t: (-v + self.m * self.u) / self.tau_v
    ds = lambda s, t: (-s + self.Iss + self.Ius) / self.tau_edge
    ds_l = lambda s_l, t: (-s_l + self.Islsl + self.Ius_l) / self.tau_edge
    return bp.JointEq([du, du_l, dv, ds, ds_l])
    
  def update(self, x=None):
    _t = bp.share['t']
    # Compute the new hit condition
    if_bump_hit= bm.where(bm.sign(self.u_pos) * self.u_pos < self.boundary-0.05, bm.bool(False), bm.bool(True))
    if_edge_hit= bm.where(bm.sign(self.s_pos) * self.s_pos < self.boundary, bm.bool(False), bm.bool(True))
    # Once hit is True, keep it True for subsequent updates
    self.first_hit = bm.logical_and(if_edge_hit, bm.logical_not(self.if_edge_hit))
    self.RT = bm.where(self.first_hit, _t-self.t_prep, self.RT)
    self.if_bump_hit = bm.logical_or(self.if_bump_hit, if_bump_hit)
    self.if_edge_hit = bm.logical_or(self.if_edge_hit, if_edge_hit)

    self.Iuu[:] = self.conn_mat @ self.get_u_fr(self.u)
    self.Iulul[:] = self.conn_mat @ self.get_u_fr(self.u_l)
    self.Iss[:] = self.conn_mat_ss @ self.get_s_fr(self.s)
    self.Islsl[:] = self.conn_mat_ss @ self.get_s_fr(self.s_l)
    
    Ius = self.clicks_right * self.u + self.clicks_left * (-self.u)
    Ius_l = self.clicks_right * (-self.u_l) + self.clicks_left * self.u_l

    #Ius = bm.where(_t<self.t_prep, 0, self.c1 * Ius)
    self.c1_dyn[:] = self.c1 * self.alpha ** (-3*(self.boundary-self.u_pos))
    self.c1_dyn_left[:] = self.c1 * self.alpha ** (-3*(self.boundary-self.ul_pos))
    self.c2_dyn[:] = self.c2
    self.Ius[:] = bm.where(bm.logical_or(self.if_edge_hit, _t < self.t_prep), 0, self.c1_dyn * Ius)
    self.Ius_l[:] = bm.where(bm.logical_or(self.if_edge_hit, _t < self.t_prep), 0, self.c1_dyn_left * Ius_l)

    #Ishift = self.c2*(self.clicks_left * self.conn_left @ self.r + self.clicks_right * self.conn_right @ self.r)
    Ishift = bm.zeros_like(self.s0)
    Ishift_l = bm.zeros_like(self.s0)
    Ishift[self.mask] = -self.c2_dyn * bm.diff(self.s[self.mask], prepend=self.s[self.mask][0])
    Ishift_l[self.mask] = -self.c2_dyn * bm.diff(self.s_l[self.mask], prepend=self.s_l[self.mask][0])
    #Ishift = bm.where(bm.sign(self.u_pos) * self.u_pos < self.boundary, Ishift, 0)
    #self.Ishift[:] = bm.where(_t<self.t_prep, 0, Ishift)
    
    self.Ishift[:] = bm.where(bm.logical_or(self.if_bump_hit, _t < self.t_prep), 0, Ishift)
    self.Ishift_l[:] = bm.where(bm.logical_or(self.if_bump_hit, _t < self.t_prep), 0, Ishift_l)

    #self.I_pause[:] = bm.where(bm.sign(self.u_pos) * self.u_pos < self.boundary, 0, -10)
    self.I_pause[:] = bm.where(self.if_edge_hit, -10, 0)
    u, u_l, v, s, s_l = self.integral(self.u, self.u_l, self.v, self.s, self.s_l, _t, self.input1, self.input2)
    self.u[:] = bm.where(u>0, u, 0)
    self.u_l[:] = bm.where(u_l>0, u_l, 0)
    self.v[:] = v 
    self.s[:] = bm.where(_t<self.t_prep, self.get_edge_clamped(s, direction='right'), s)
    self.s_l[:] = bm.where(_t<self.t_prep, self.get_edge_clamped(s_l, direction='right'), s_l)
    #self.s[:] = self.get_edge_clamped(s, direction='right')
    u_pos = self.find_bump_location(self.u)
    ul_pos = self.find_bump_location(self.u_l)
    self.u_dpos = u_pos - self.u_pos
    self.u_pos = u_pos
    self.ul_pos = ul_pos
    self.s_pos = self.find_edge_location(self.s)
    self.sl_pos = self.find_edge_location(self.s_l)
    
    self.input1[:] = 0.
    self.input2[:] = 0.


import numpy as np
import brainpy as bp
import brainpy.math as bm
from CANN_DDM_models import CANN_DDM_bump_edge_model
from tqdm import tqdm
import gc
import os
import subprocess
from scipy import stats
import jax
from jax import clear_caches
import psutil

class NetworkSimulator:
    def __init__(self, DDM_params=None, CANN_params=None, mon_vars=None, c1_range=None, save_runner=False, platform='cpu'):
        default_DDM_params = {
            'dt': 1.,
            'dt_DDM': 10.,
            'v': 0.5,
            'sig_W': .5,
            'boundary': .8,
        }

        default_CANN_params = {
            'dur1': 500,
            'dur2': 4000,
            'edge_type': 'tanh',
            'num': 1024,
            'tau_bump': 0.1,
            'tau_edge': 1,
            'beta': 2,
            'offset': 3.85,
            'delta_z': 1/40,
            'J0_bump': 4,
            'J0_edge': 1,
            'a': 0.25,
            'A': 10,
            'c2': 1
        }
        default_mon_vars = ['u_pos', 'RT']

        self.DDM_params = {**default_DDM_params, **(DDM_params or {})}
        self.CANN_params = {**default_CANN_params, **(CANN_params or {})}
        self.mon_vars = list(set(default_mon_vars) | set(mon_vars if mon_vars is not None else default_mon_vars))
        bm.set_platform(platform)
        self.platform = platform.lower()
        
        if 'c1' in self.CANN_params.keys():
            self.c1_opt = self.CANN_params['c1']
        else:
            print("c1 not found in CANN_params, finding optimal c1")
            print("-" * 50)
            self.c1_opt, self.err_per_c1 = self.find_optimal_c1(c1_range=c1_range)
            c1_opt_err = np.min(np.mean(self.err_per_c1, axis=1))
            print(f"Find the optimal c1: {self.c1_opt} with average error: {c1_opt_err}")
        self.save_runner = save_runner
        if save_runner:
            self.runners_log = dict()
        self.t_prep = int((self.CANN_params['dur1'] + 0.1 * self.CANN_params['dur2']) / self.DDM_params['dt_DDM'])

            
     


    def _force_memory_cleanup(self, aggressive=True):
        """
        Aggressive memory cleanup function
        """
        import gc
        import sys
        import ctypes
        
        # Multiple rounds of garbage collection
        for _ in range(5 if aggressive else 3):
            gc.collect()
            
        # Clear JAX
        clear_caches()
        #jax.clear_backends()
        
        # Clear BrainPy caches
        if hasattr(bm, 'clear_cache'):
            bm.clear_cache()
        if hasattr(bp, 'clear_cache'):
            bp.clear_cache()
            
        # Clear Python's internal caches
        if hasattr(sys, 'intern'):
            sys.intern.clear() if hasattr(sys.intern, 'clear') else None
            
        # Force memory compaction on supported systems
        if aggressive and hasattr(ctypes, 'windll'):
            # Windows specific memory cleanup
            try:
                ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
            except:
                pass
        elif aggressive:
            # Unix-like systems
            try:
                import mmap
                # Try to hint the OS to free memory
                pass
            except:
                pass
                
        # Final garbage collection
        gc.collect()

    def generate_evidence_variable(self, seed=None):
        v = self.DDM_params['v']
        sig_W = self.DDM_params['sig_W']
        dur1 = self.CANN_params['dur1']
        dur2 = self.CANN_params['dur2']
        dt_DDM = self.DDM_params['dt_DDM']
        dt = self.DDM_params['dt']

    
        l =  bm.sqrt(sig_W**2 * dt_DDM/1e3 + (v*dt_DDM/1e3)**2)
        num1 = int(dur1 / dt_DDM) 
        num2 = int(dur2 / dt_DDM)
        p_right = 1/2 * (1 + v * (dt_DDM/1e3)/l)
        p_left = 1 - p_right
        clicks_left = bm.random.binomial(1, p_left, num1+num2)
        clicks_left[:self.t_prep] = 0
        clicks_right = 1 - clicks_left
        clicks_right[:self.t_prep] = 0
        clicks_left = bm.repeat(clicks_left, int(dt_DDM / dt))
        clicks_right = bm.repeat(clicks_right, int(dt_DDM / dt))
        
        self.dx = l
        self.p_right = p_right
        return clicks_left, clicks_right


    def run_CANN_simulation(self, model, mon_vars, seed_evidence=None, progress_bar=True):
      clicks_left, clicks_right = self.generate_evidence_variable(seed=seed_evidence)
      dur1 = self.CANN_params['dur1']
      dur2 = self.CANN_params['dur2']
      dt = self.DDM_params['dt']
      assert np.shape(clicks_left) == np.shape(clicks_right), "Evidence vectors must have the same length."
      assert len(clicks_left) == int((dur1 + dur2) / dt), "Evidence length should be equal to the simulatin duration."
      I1 = 0.1 * model.get_stimulus_by_pos(0)
      Iext, duration = bp.inputs.section_input(values=[I1, 0], durations=[dur1/10, dur2/10], return_length=True)
      runner=bp.DSRunner(model, inputs=[('input', Iext, 'iter'), ('clicks_left', clicks_left,'iter', '='), ('clicks_right', clicks_right,'iter', '=')],
                      monitors=mon_vars, dyn_vars=model.vars(), progress_bar=progress_bar, dt=dt)
      runner.run(dur1+dur2)
      return runner

    def run_CANN_simulation_two_edges(self, model, mon_vars, seed_evidence=None, progress_bar=True):
      clicks_left, clicks_right = self.generate_evidence_variable(seed=seed_evidence)
      dur1 = self.CANN_params['dur1']
      dur2 = self.CANN_params['dur2']
      dt = self.DDM_params['dt']
      assert np.shape(clicks_left) == np.shape(clicks_right), "Evidence vectors must have the same length."
      assert len(clicks_left) == int((dur1 + dur2) / dt), "Evidence length should be equal to the simulatin duration."
      I1 = 0.1 * model.get_stimulus_by_pos(0)
      Iext1, duration = bp.inputs.section_input(values=[I1, 0], durations=[dur1/10, dur2/10], return_length=True)
      Iext2 = Iext1.copy()
      runner=bp.DSRunner(model, inputs=[('input1', Iext1, 'iter'), ('input2', Iext2, 'iter'), ('clicks_left', clicks_left,'iter', '='), ('clicks_right', clicks_right,'iter', '=')],
                      monitors=mon_vars, dyn_vars=model.vars(), progress_bar=progress_bar, dt=dt)
      runner.run(dur1+dur2)
      return runner

    def _check_memory_usage(self, threshold_mb=8000):
        """
        Check current memory usage based on platform type and return True if above threshold
        
        Args:
            threshold_mb: Memory threshold in MiB
            
        Returns:
            tuple: (is_above_threshold, current_memory_usage_mb, memory_type)
        """
        try:
            if self.platform == 'gpu':
                # 检查GPU内存使用情况
                return self._check_gpu_memory_usage(threshold_mb)
            else:
                # 检查CPU内存使用情况
                return self._check_cpu_memory_usage(threshold_mb)
        except Exception as e:
            print(f"Warning: Error checking {self.platform} memory usage: {e}")
            # 如果GPU检查失败，回退到CPU检查
            if self.platform == 'gpu':
                print("Falling back to CPU memory checking")
                return self._check_cpu_memory_usage(threshold_mb)
            else:
                # 如果CPU检查也失败，返回安全值
                return False, 0, 'unknown'
    
    def _check_cpu_memory_usage(self, threshold_mb):
        """
        Check CPU memory usage
        """
        process = psutil.Process(os.getpid())
        current_mem = process.memory_info().rss / 1024**2  # in MiB
        return current_mem > threshold_mb, current_mem, 'cpu'
    
    def _check_gpu_memory_usage(self, threshold_mb):
        """
        实时检测 GPU 内存占用情况。如果占用超过阈值，则返回 True。
        支持使用 pynvml（推荐）或 fallback 到 nvidia-smi / PyTorch。
        """
        try:
            # 优先使用 pynvml 获取 GPU 内存信息
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 默认检测第一个 GPU
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used_mb = meminfo.used / 1024**2
                pynvml.nvmlShutdown()
                return memory_used_mb > threshold_mb, memory_used_mb, 'gpu'
            except ImportError:
                pass  # 如果未安装 pynvml，继续尝试其他方法

            # fallback 使用 nvidia-smi 命令行
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=3)
                if result.returncode == 0:
                    memory_used_mb = float(result.stdout.strip().split('\n')[0])
                    return memory_used_mb > threshold_mb, memory_used_mb, 'gpu'
            except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
                pass

            # fallback 使用 PyTorch
            try:
                import torch
                if torch.cuda.is_available():
                    memory_used_bytes = torch.cuda.memory_allocated()
                    memory_used_mb = memory_used_bytes / 1024**2
                    return memory_used_mb > threshold_mb, memory_used_mb, 'gpu'
            except ImportError:
                pass

            print("Warning: Unable to get GPU memory info, falling back to CPU memory")
            return self._check_cpu_memory_usage(threshold_mb)

        except Exception as e:
            print(f"Unexpected error while checking GPU memory: {e}")
            return False, 0.0, 'unknown'


    def simulate_network(self, num_trials=500, batch_size=5, save_dir=None, memory_threshold=8000, seed_mode='fixed', seed_start=2025, seed_evidence=None):
        # Handle seed generator
        if seed_mode == 'fixed':
            seed_sequence = np.arange(num_trials) +seed_start
        elif seed_mode == 'random':
            seed_sequence = [bm.random.randint(0, int(1e9)) for _ in range(num_trials)]
        else:
            raise ValueError("seed_mode must be either 'random' or 'fixed'")


        dt = self.DDM_params['dt']
        v = self.DDM_params['v']
        boundary = self.DDM_params['boundary']
        
        dur1 = self.CANN_params['dur1']
        dur2 = self.CANN_params['dur2']
        edge_type = self.CANN_params['edge_type']
        num = self.CANN_params['num']
        tau_bump = self.CANN_params['tau_bump']
        tau_edge = self.CANN_params['tau_edge']
        beta = self.CANN_params['beta']
        sigma_edge = self.CANN_params['sigma_edge']
        offset = self.CANN_params['offset']
        delta_z = self.CANN_params['delta_z']
        J0_bump = self.CANN_params['J0_bump']
        J0_edge = self.CANN_params['J0_edge']
        a = self.CANN_params['a']
        A = self.CANN_params['A']
        c2 = self.CANN_params['c2']

        RT_corr = []
        RT_incorr = []
        runners = dict()
        total_batches = (num_trials + batch_size - 1) // batch_size
        
        print(f"\nStarting simulation with {num_trials} trials in {total_batches} batches of size {batch_size}")
        print(f"Memory threshold set to: {memory_threshold} MiB")
        print("=" * 50)
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, num_trials)
            batch_RT_corr = []
            batch_RT_incorr = []
            
            print(f"\nProcessing Batch {batch_idx + 1}/{total_batches} (Trials {batch_start + 1}-{batch_end})")
            print("-" * 50)
            
            for trial_idx in tqdm(range(batch_start, batch_end), 
                            desc=f"Trials in batch {batch_idx + 1}",
                            ncols=80):
                
                # Pre-trial memory check and cleanup
                high_mem, current_mem, mem_type = self._check_memory_usage(memory_threshold)
                if high_mem:
                    print(f"\nWarning: High {mem_type} memory usage detected ({current_mem:.1f} MiB). Forcing cleanup...")
                    self._force_memory_cleanup(aggressive=True)
                else:
                    self._force_memory_cleanup(aggressive=False)
                
                seed_network = int(seed_sequence[trial_idx])
                if seed_evidence is None:
                    seed_evidence = seed_network
                clicks_left, clicks_right = self.generate_evidence_variable(seed=seed_evidence)
                
                self.model = CANN_DDM_bump_edge_model(num=num, c1=self.c1_opt, c2=c2, J0_bump=J0_bump,
                                                tau_bump=tau_bump,
                                                tau_edge=tau_edge,
                                                J0_edge=J0_edge, edge_offset=offset,
                                                optimize_offset=False, a=a, A=A,
                                                beta=beta, sigma_edge=sigma_edge, 
                                                edge_type=edge_type, delta_z=delta_z, 
                                                seed=seed_network, boundary=boundary)
            
                runner = self.run_CANN_simulation(self.model, self.mon_vars, progress_bar=False)
                
                if runner.mon.RT[-1] != bm.inf:
                    rt_val = float(runner.mon.RT[-1])
                    u_pos_val = float(runner.mon.u_pos[-1])
                    if bm.sign(u_pos_val) == bm.sign(v):
                        batch_RT_corr.append(rt_val)
                    else:
                        batch_RT_incorr.append(rt_val)
                
                # Comprehensive memory cleanup
                # Clear monitor data
                if hasattr(runner, 'mon'):
                    for key in list(runner.mon.__dict__.keys()):
                        if hasattr(runner.mon.__dict__[key], '__del__'):
                            delattr(runner.mon, key)
                        
                # Clear model internals
                if hasattr(self.model, 'clear_cache'):
                    self.model.clear_cache()
                if hasattr(self.model, 'reset_state'):
                    self.model.reset_state()
                    
                # Explicit deletion of large objects
                del clicks_left, clicks_right
                if self.save_runner:
                    self.runners_log[seed_network] = runner
                else:
                    del runner
                # Force aggressive memory cleanup
                self._force_memory_cleanup(aggressive=True)
            
            print(f"\nBatch {batch_idx + 1} completed:")
            print(f"Correct responses: {len(batch_RT_corr)}")
            print(f"Incorrect responses: {len(batch_RT_incorr)}")
            
            # Memory status after batch
            _, mem, mem_type = self._check_memory_usage()
            print(f"{mem_type.upper()} Memory after batch {batch_idx + 1}: {mem:.2f} MiB")
            print("-" * 50)
            
            RT_corr.extend(batch_RT_corr)
            RT_incorr.extend(batch_RT_incorr)
            del batch_RT_corr, batch_RT_incorr
            
            # Inter-batch aggressive memory cleanup
            self._force_memory_cleanup(aggressive=True)
        
        print("\nSimulation completed!")
        print(f"Total correct responses: {len(RT_corr)}")
        print(f"Total incorrect responses: {len(RT_incorr)}")
        _, final_mem, mem_type = self._check_memory_usage()
        print(f"Final {mem_type.upper()} Memory Usage: {final_mem:.2f} MiB")
        print("=" * 50)
        
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_path = os.path.join(save_dir, 'RT_sim.npz')

            RT_corr = np.array(RT_corr)
            RT_incorr = np.array(RT_incorr)

            if os.path.exists(save_path):
                data_old = np.load(save_path)
                RT_corr = np.concatenate([data_old['RT_corr'], RT_corr])
                RT_incorr = np.concatenate([data_old['RT_incorr'], RT_incorr])

            np.savez_compressed(save_path, RT_corr=RT_corr, RT_incorr=RT_incorr)

        return RT_corr, RT_incorr

    def generate_dice_RT(self, n_trials=500, seed=2025, max_steps=1000):
        bm.random.seed(seed)
        
        v = self.DDM_params.get('v', 1)
        sig_W = self.DDM_params.get('sig_W', 1)
        B = self.DDM_params.get('boundary', 1)
        dt_DDM = self.DDM_params.get('dt_DDM', 1)
        
        dl = np.sqrt(sig_W**2 * dt_DDM*1e-3 + (v * dt_DDM*1e-3)**2)
        p = 0.5 * (1 + (v * dt_DDM*1e-3) / dl)
        print(f"Discrete simulation parameters: step length dl = {dl:.5f}, probability p = {p:.5f}")
        
        pos_RT_dice = []
        neg_RT_dice = []
        
        for _ in range(n_trials):
            x = 0.0
            t = 0.0
            hit_boundary = False
            for _ in range(max_steps):
                t += dt_DDM*1e-3
                if bm.random.rand() < p:
                    x += dl
                else:
                    x -= dl
                if x >= B:
                    pos_RT_dice.append(t)
                    hit_boundary = True
                    break
                elif x <= -B:
                    neg_RT_dice.append(t)
                    hit_boundary = True
                    break
            if not hit_boundary:
                pos_RT_dice.append(np.nan)
                neg_RT_dice.append(np.nan)
                
        return np.array(pos_RT_dice), np.array(neg_RT_dice)
    
    # def find_optimal_c1(self, c1_range=None, num_seeds=10):
    #     """
    #     Find the optimal c1 value for given DDM and CANN parameters by sampling different c1 values
    #     and using linear regression to find the best fit.
        
    #     Parameters:
    #     -----------
    #     DDM_params : dict, optional
    #         Parameters for the DDM model. If None, default values will be used.
    #     CANN_params : dict, optional
    #         Parameters for the CANN model. If None, default values will be used.
    #     c1_range : array-like, optional
    #         Range of c1 values to sample. If None, will use np.linspace(0.1, 2.5, 20)
    #     num_seeds : int, optional
    #         Number of random seeds to use per c1 value
            
    #     Returns:
    #     --------
    #     optimal_c1 : float
    #         The optimal c1 value found
    #     """

    #     # Update with provided parameters
    #     DDM_params = self.DDM_params
    #     CANN_params = self.CANN_params

    #     # Set default c1 range if not provided
    #     if c1_range is None:
    #         c1_range = np.linspace(0.1, 2.5, 20)

    #     # Extract parameters
    #     dt = DDM_params['dt']
    #     dt_DDM = DDM_params['dt_DDM']
    #     v = DDM_params['v']
    #     sig_W = DDM_params['sig_W']
    #     boundary = DDM_params['boundary']

    #     dur1 = CANN_params['dur1']
    #     dur2 = CANN_params['dur2']
    #     t_prep = dur1 + 0.1 * dur2

    #     # Calculate edge velocities for different c1 values
    #     all_moving_distances = []
    #     v_edge = []
        
    #     for c1 in tqdm(c1_range):
    #         moving_distances_per_c1 = []
    #         v_edge_per_c1 = []
    #         for i in range(num_seeds):
    #             seed = 2025 + i
    #             clicks_left, clicks_right = self.generate_evidence_variable(seed=i)
                
    #             my_model = CANN_DDM_bump_edge_model(
    #                 num=CANN_params['num'],
    #                 c1=c1,
    #                 c2=CANN_params['c2'],
    #                 J0_bump=CANN_params['J0_bump'],
    #                 tau_bump=CANN_params['tau_bump'],
    #                 J0_edge=CANN_params['J0_edge'],
    #                 edge_offset=CANN_params['offset'],
    #                 optimize_offset=False,
    #                 t_prep=t_prep,
    #                 a=CANN_params['a'],
    #                 A=CANN_params['A'],
    #                 beta=CANN_params['beta'],
    #                 edge_type=CANN_params['edge_type'],
    #                 delta_z=CANN_params['delta_z'],
    #                 seed=seed,
    #                 boundary=boundary
    #             )
    #             mon_vars = ['u_dpos', 'RT']
    #             runner = self.run_CANN_simulation(my_model, mon_vars, progress_bar=False)
                
    #             # Calculate edge velocity
    #             RT = runner.mon.RT[-1]
    #             if RT < dur1 + dur2 - t_prep:
    #                 edge_velocity = bm.mean(bm.abs(runner.mon.u_dpos[int(t_prep):int(t_prep+RT)])) / dt * 1e3
    #             else:
    #                 edge_velocity = bm.mean(bm.abs(runner.mon.u_dpos[int(t_prep):])) / dt * 1e3
    #             v_edge_per_c1.append(edge_velocity)
                
    #         v_edge.append(v_edge_per_c1)
        
    #     # Calculate mean edge velocity for each c1
    #     mean_v_edge = np.mean(v_edge, axis=1)
        
    #     # Perform linear regression
    #     slope, intercept, r_value, p_value, std_err = stats.linregress(c1_range, mean_v_edge)
        
    #     # Calculate theoretical optimal c1
    #     # Based on the theoretical prediction: v = c1 * bump_height / sigma
    #     # where sigma = 0.5/beta
    #     bump_height = 0.57  # Typical bump height
    #     sigma = 0.5/CANN_params['beta']
    #     du = np.sqrt(sig_W**2 * (dt_DDM*1e-3) + (v * dt_DDM*1e-3)**2)
    #     v_DDM = du / (dt_DDM*1e-3)
        
    #     # Find the c1 value that gives the closest velocity to the theoretical prediction
    #     optimal_c1 = (v_DDM - intercept) / slope
        
    #     return optimal_c1, {
    #         'slope': slope,
    #         'intercept': intercept,
    #         'r_value': r_value,
    #         'p_value': p_value,
    #         'std_err': std_err,
    #         'theoretical_c1': optimal_c1,
    #         'mean_v_edge': mean_v_edge,
    #         'c1_range': c1_range
    #     }



    def find_optimal_c1(self, c1_range=None, num_seeds=10):
        """
        Find the optimal c1 value for given DDM and CANN parameters by sampling different c1 values
        and using linear regression to find the best fit.

        Parameters:
        -----------
        DDM_params : dict, optional
            Parameters for the DDM model. If None, default values will be used.
        CANN_params : dict, optional
            Parameters for the CANN model. If None, default values will be used.
        c1_range : array-like, optional
            Range of c1 values to sample. If None, will use np.linspace(0.1, 2.5, 20)
        num_seeds : int, optional
            Number of random seeds to use per c1 value

        Returns:
        --------
        optimal_c1 : float
            The optimal c1 value found
        """

        # Update with provided parameters
        DDM_params = self.DDM_params
        CANN_params = self.CANN_params

        # Set default c1 range if not provided
        if c1_range is None:
            c1_range = np.linspace(0.1, 5, 20)


        # Extract parameters
        dt = DDM_params['dt']
        dt_DDM = DDM_params['dt_DDM']
        v = DDM_params['v']
        sig_W = DDM_params['sig_W']
        boundary = DDM_params['boundary']

        dur1 = CANN_params['dur1']
        dur2 = CANN_params['dur2']

        # Calculate edge velocities for different c1 values
        all_moving_distances = []
        v_edge = []
        err_per_c1 = []
        for c1 in tqdm(c1_range):
            err = []
            for i in range(num_seeds):
                clicks_left, clicks_right = self.generate_evidence_variable(seed=i)
                my_model = CANN_DDM_bump_edge_model(
                    num=CANN_params['num'],
                    c1=c1,
                    c2=CANN_params['c2'],
                    J0_bump=CANN_params['J0_bump'],
                    tau_bump=CANN_params['tau_bump'],
                    tau_edge=CANN_params['tau_edge'],
                    sigma_edge=CANN_params['sigma_edge'],
                    J0_edge=CANN_params['J0_edge'],
                    edge_offset=CANN_params['offset'],
                    optimize_offset=False,
                    a=CANN_params['a'],
                    A=CANN_params['A'],
                    beta=CANN_params['beta'],
                    edge_type=CANN_params['edge_type'],
                    delta_z=CANN_params['delta_z'],
                    seed=i,
                    boundary=boundary
                )
                mon_vars = ['u_dpos', 's_pos', 'RT']
                runner = self.run_CANN_simulation(my_model, mon_vars, progress_bar=False)
                # Calculate the expected decision variable trajectory 
                RT_sim = runner.mon.RT[-1] if runner.mon.RT[-1] != bm.inf else dur1 + dur2 - self.t_prep - 1
                RT_sim = int(RT_sim)
                x_pred = np.cumsum(clicks_left[self.t_prep:] * (-self.dx) / dt_DDM  + clicks_right[self.t_prep:] * (self.dx) / dt_DDM)
                x_sim = runner.mon.s_pos[self.t_prep:]
                cross_idx = np.where(bm.abs(x_pred) > boundary)[0]
                RT_pred = cross_idx[0] if cross_idx.size > 0 else dur1 + dur2 - self.t_prep
                RT = np.min([RT_sim, RT_pred])
                err.append(np.abs(x_sim[RT] - x_pred[RT]))
            err_per_c1.append(err)

        err_per_c1 = np.array(err_per_c1)
        c1_opt = c1_range[np.argmin(err_per_c1.mean(axis=1))]
        return c1_opt, err_per_c1


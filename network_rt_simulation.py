import numpy as np
import matplotlib.pyplot as plt
from CANN_DDM_models import get_DDM_simulation, get_RT, run_CANN_simulation, CANN_DDM_bump_edge_model
from tqdm import tqdm
from brainpy import math as bm
import gc
import os

def simulate_network(DDM_params=None, CANN_params=None, num_trials=500):
    """
    Simulate network behavior and collect response times for correct and incorrect responses.
    
    Parameters:
    -----------
    DDM_params : dict, optional
        Parameters for the DDM model
    CANN_params : dict, optional
        Parameters for the CANN model
    num_trials : int, optional
        Number of trials to simulate
        
    Returns:
    --------
    tuple
        Arrays of response times for correct and incorrect responses
    """
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

    DDM_params = {**default_DDM_params, **(DDM_params or {})}
    CANN_params = {**default_CANN_params, **(CANN_params or {})}

    # read DDM parameters
    dt = DDM_params['dt']
    dt_DDM = DDM_params['dt_DDM']
    v = DDM_params['v']
    sig_W = DDM_params['sig_W']
    boundary = DDM_params['boundary']
    du = np.sqrt(sig_W**2 * dt_DDM + (v * dt_DDM)**2)
    v_sim = du / dt

    # read CANN parameters
    dur1 = CANN_params['dur1']
    dur2 = CANN_params['dur2']
    t_prep = dur1 + 0.1 * dur2
    edge_type = CANN_params['edge_type']
    num = CANN_params['num']
    tau_bump = CANN_params['tau_bump']
    tau_edge = CANN_params['tau_edge']
    beta = CANN_params['beta']
    offset = CANN_params['offset']
    delta_z = CANN_params['delta_z']
    J0_bump = CANN_params['J0_bump']
    J0_edge = CANN_params['J0_edge']
    a = CANN_params['a']
    A = CANN_params['A']
    c2 = CANN_params['c2']
    
    # find the theoretical c1 given the drift rate v
    c1_pred = lambda sigma, v, bump_height: v * sigma / bump_height
    c1_fit = c1_pred(.5/beta, v_sim, bump_height=0.57)
    RT_corr = []
    RT_incorr = []

    for seed in tqdm(np.arange(num_trials)):
        seed = int(seed)
        clicks_left, clicks_right, l, p = get_DDM_simulation(v, sig_W, dur1, dur2, dt_DDM, dt=dt, seed=seed)
        my_model = CANN_DDM_bump_edge_model(num=num, c1=c1_fit, c2=c2, J0_bump=J0_bump,
                                        tau_bump = tau_bump,
                                        J0_edge=J0_edge, edge_offset=offset,
                                        optimize_offset=False,
                                        t_prep = t_prep, a=a, A=A,
                                        beta=beta, edge_type=edge_type, delta_z=delta_z, seed=seed, boundary=boundary)
        mon_vars=['u','v', 's' ,'Ishift', 'u_pos', 'u_dpos', 's_pos', 'Ius', 'RT', 'first_hit', 'if_bump_hit', 'if_edge_hit']
        runner = run_CANN_simulation(my_model, clicks_left, clicks_right, dur1, dur2, mon_vars, dt=dt, progress_bar=False)
        my_model.reset_state()
        if runner.mon.RT[-1] != bm.inf:
            if bm.sign(runner.mon.u_pos[-1]) == bm.sign(v):
                RT_corr.append(runner.mon.RT[-1])
            else:
                RT_incorr.append(runner.mon.RT[-1])
        del my_model, runner, clicks_left, clicks_right, l, p
        gc.collect()
    return np.array(RT_corr), np.array(RT_incorr)

def plot_RT_distribution(v=1, sig_W=1, B=1, dt=0.001, dt_dice=0.01, n_trials=500, seed=2025, max_steps=1000, pos_RT=None, neg_RT=None):
    """
    Plot reaction time (RT) distributions for positive and negative responses.
    
    Parameters:
    -----------
    v : float, optional
        Drift rate
    sig_W : float, optional
        Diffusion coefficient
    B : float, optional
        Boundary value
    dt : float, optional
        Time step for potential simulation
    dt_dice : float, optional
        Time step for dice-roll simulation
    n_trials : int, optional
        Number of trials to simulate
    seed : int, optional
        Random seed
    max_steps : int, optional
        Maximum number of steps per trial
    pos_RT : array_like, optional
        Reaction times for positive boundary hits
    neg_RT : array_like, optional
        Reaction times for negative boundary hits
    """
    np.random.seed(seed)
    bins = np.arange(0, max_steps * dt, dt_dice*10)
    
    # Discrete Dice-Roll Simulation
    dl = np.sqrt(sig_W**2 * dt_dice + (v * dt_dice)**2)
    p = 0.5 * (1 + (v * dt_dice) / dl)
    print(f"Discrete simulation parameters: step length dl = {dl:.5f}, probability p = {p:.5f}")
    
    pos_RT_dice = []
    neg_RT_dice = []
    
    for trial in range(n_trials):
        x = 0.0
        t = 0.0
        hit_boundary = False
        for step in range(max_steps):
            t += dt_dice
            if np.random.rand() < p:
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
            
    pos_RT_dice = np.array(pos_RT_dice)
    neg_RT_dice = np.array(neg_RT_dice)
    
    def compute_ecdf(data):
        data = data[~np.isnan(data)]
        sorted_data = np.sort(data)
        n = sorted_data.size
        ecdf = np.arange(1, n + 1) / n
        return sorted_data, ecdf
    
    pos_x, pos_ecdf = compute_ecdf(pos_RT_dice)
    neg_x, neg_ecdf = compute_ecdf(neg_RT_dice)
    
    if pos_RT is not None:
        pos_x_extra, pos_ecdf_extra = compute_ecdf(np.array(pos_RT))
    if neg_RT is not None:
        neg_x_extra, neg_ecdf_extra = compute_ecdf(np.array(neg_RT))
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Positive Boundary plots
    ax = axs[0, 0]
    ax.hist(pos_RT_dice, bins=bins, density=True, alpha=0.5, label="Positive RT (Dice Simulation)")
    if pos_RT is not None:
        ax.hist(pos_RT, bins=bins, density=True, alpha=0.5, label="Positive RT")
    ax.set_xlabel("Reaction Time (s)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Positive Boundary Reaction Time Histogram")
    ax.legend()
    
    ax = axs[0, 1]
    ax.step(pos_x, pos_ecdf, where='post', label="Positive RT (Dice Simulation)")
    if pos_RT is not None:
        ax.step(pos_x_extra, pos_ecdf_extra, where='post', label="Positive RT")
    ax.set_xlabel("Reaction Time (s)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Positive Boundary Reaction Time Empirical CDF")
    ax.legend()
    
    # Negative Boundary plots
    ax = axs[1, 0]
    ax.hist(neg_RT_dice, bins=bins, density=True, alpha=0.5, label="Negative RT (Dice Simulation)")
    if neg_RT is not None:
        ax.hist(neg_RT, bins=bins, density=True, alpha=0.5, label="Negative RT")
    ax.set_xlabel("Reaction Time (s)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Negative Boundary Reaction Time Histogram")
    ax.legend()
    
    ax = axs[1, 1]
    ax.step(neg_x, neg_ecdf, where='post', label="Negative RT (Dice Simulation)")
    if neg_RT is not None:
        ax.step(neg_x_extra, neg_ecdf_extra, where='post', label="Negative RT")
    ax.set_xlabel("Reaction Time (s)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Negative Boundary Reaction Time Empirical CDF")
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    return pos_RT_dice, neg_RT_dice

def run_and_save_simulation(num_trials=500, save_dir='results'):
    """
    Run network simulation and save results.
    
    Parameters:
    -----------
    num_trials : int, optional
        Number of trials to simulate
    save_dir : str, optional
        Directory to save results
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Run simulation
    RT_corr, RT_incorr = simulate_network(num_trials=num_trials)
    
    # Save results
    np.save(os.path.join(save_dir, 'RT_corr.npy'), RT_corr)
    np.save(os.path.join(save_dir, 'RT_incorr.npy'), RT_incorr)
    
    # Plot and save distributions
    plot_RT_distribution(pos_RT=RT_corr, neg_RT=RT_incorr)
    plt.savefig(os.path.join(save_dir, 'rt_distributions.png'))
    
    return RT_corr, RT_incorr

if __name__ == "__main__":
    # Example usage
    RT_corr, RT_incorr = run_and_save_simulation(num_trials=500) 
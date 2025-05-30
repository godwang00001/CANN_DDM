import sys
sys.path.append('..')
import argparse
from network_rt_simulation import NetworkSimulator
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run CANN-DDM network simulation with configurable parameters')
    parser.add_argument('--num_trials', type=int, default=500,
                      help='Number of trials to simulate (default: 500)')
    parser.add_argument('--batch_size', type=int, default=50,
                      help='Batch size for processing trials (default: 50)')
    parser.add_argument('--save_dir', type=str, default='../figs_code',
                      help='Directory to save results (default: results)')
    parser.add_argument('--no_plot', action='store_true',
                      help='Disable plotting of results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load saved RT data
    DDM_params = {
        'v': .5,
        'sig_W': .5,
        'boundary': .8,
        'dt':1.,
        'dt_DDM': 25.
    }

    CANN_params = {
        'dur1': 500,
        'dur2': 2000,
        'edge_type': 'tanh', 
        'num': 1024,
        'tau_bump': 0.08,
        'tau_edge': 4,
        'beta': 2,
        'offset': 3.85,
        'delta_z': 1/40,
        'J0_bump': 4,
        'J0_edge': 1,
        'a': 0.25,
        'A': 10,
        'c2': 1
    }
    CANN_params.update({'c1': 2.563}) # pre-calculated optimal c1
    Simulator = NetworkSimulator(DDM_params, CANN_params, save_runner=False)


    print(f"\nStarting simulation with:")
    print(f"Number of trials: {args.num_trials}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save directory: {args.save_dir}")
    print("=" * 50)
    
    # Run network simulation
    Simulator.simulate_network(num_trials=args.num_trials, batch_size=args.batch_size, save_dir=args.save_dir)
    
    # Generate dice experiment RT data
    # max_steps = CANN_params['dur1'] + CANN_params['dur2']
    # RT_dice_corr, RT_dice_incorr = Simulator.generate_dice_RT(n_trials=args.num_trials, max_steps=max_steps)

    # print("\nSimulation completed successfully!")
    # if args.save_dir:
        
    #     print(f"Results saved in: {args.save_dir}")
    # else:
    #     print("Results not saved")

if __name__ == "__main__":
    main() 
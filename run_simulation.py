import sys
sys.path.append('..')
import argparse
import numpy as np
from network_rt_simulation import run_and_save_simulation, generate_dice_RT, plot_rt_distribution

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run CANN-DDM network simulation with configurable parameters')
    parser.add_argument('--trials', type=int, default=500,
                      help='Number of trials to simulate (default: 500)')
    parser.add_argument('--batch_size', type=int, default=50,
                      help='Batch size for processing trials (default: 50)')
    parser.add_argument('--save_dir', type=str, default='../figs_code',
                      help='Directory to save results (default: results)')
    parser.add_argument('--no_plot', action='store_true',
                      help='Disable plotting of results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Default DDM parameters
    DDM_params = {
        'dt': 1.,
        'dt_DDM': 25.,
        'v': .5,
        'sig_W': .5,
        'boundary': .8,
    }
    
    # Default CANN parameters
    CANN_params = {
        'dur1': 500,
        'dur2': 4000,
        'edge_type': 'tanh',
        'num': 1024,
        'tau_bump': 0.1,
        'tau_edge': 1.5,
        'beta': 2,
        'offset': 3.85,
        'delta_z': 1/40,
        'J0_bump': 4,
        'J0_edge': 1,
        'a': 0.25,
        'A': 10,
        'c1': 3.025,
        'c2': .5
    }
    
    print(f"\nStarting simulation with:")
    print(f"Number of trials: {args.trials}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save directory: {args.save_dir}")
    print("=" * 50)
    
    # Run network simulation
    RT_sim_corr, RT_sim_incorr = run_and_save_simulation(
        DDM_params=DDM_params,
        CANN_params=CANN_params,
        num_trials=args.trials,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )
    
    # Generate dice experiment RT data
    RT_dice_corr, RT_dice_incorr = generate_dice_RT(DDM_params, n_trials=args.trials)
    
    # Plot results if not disabled
    if not args.no_plot:
        plot_rt_distribution(
            RT_sim_corr, RT_sim_incorr,
            RT_dice_corr, RT_dice_incorr,
            save_path=f'{args.save_dir}/rt_distribution.png'
        )
    
    print("\nSimulation completed successfully!")
    print(f"Results saved in: {args.save_dir}")

if __name__ == "__main__":
    main() 
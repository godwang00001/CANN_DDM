# CANN-DDM Network Simulation

This repository contains code for simulating and analyzing neural network behavior using the CANN-DDM (Continuous Attractor Neural Network - Drift Diffusion Model) approach.

## Features

- Network simulation with customizable parameters
- Response time analysis and visualization
- Support for both correct and incorrect response analysis
- Automatic result saving and plotting

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CANN_DDM_repo.git
cd CANN_DDM_repo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulation:
```python
from network_rt_simulation import run_and_save_simulation

# Run simulation with default parameters
RT_corr, RT_incorr = run_and_save_simulation(num_trials=500)
```

## Parameters

The simulation can be customized using various parameters:

- DDM parameters (drift rate, noise, boundary)
- CANN parameters (network size, time constants, etc.)
- Number of trials
- Save directory

## Results

Results are saved in the specified directory (default: 'results/') including:
- Response time data (.npy files)
- Distribution plots (.png files)

## License

MIT License 
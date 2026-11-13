# Q-Learning Market Simulation

This project simulates firms that repeatedly choose prices and investment levels in a dynamic market. Each firm learns which action to take in each market state using a Q-learning approach. The simulation can compare firms that optimize different objectives and records outcomes such as prices, profits, market shares, and consumer surplus.

## Code structure

| File or folder | Purpose |
| --- | --- |
| `Main.py` | Builds the state and action spaces, runs the repeated-market simulation, updates firm learning, tracks convergence, and saves results. Command-line options in this file can override simulation parameters. |
| `Firm.py` | Defines firm instance - storing the Q-matrix, action selection, visit counts, and learning updates. |
| `Settings.py` | Defines the `Config` object, including market and learning parameters, demand, and calculated price and investment options. |
| `Visualize.py` | Contains plotting and analysis routines for simulation histories, strategies, welfare, and cluster results. |
| `Helper.py` | Small utility functions. |
| `BestResponses.py` | Standalone script that calculates and plots a leader’s price best response to a follower’s price. |
| `TrainingResults/` | Output location used by the simulation for plots, summaries, and other run artifacts. |
| `requirements.txt` | Lists the Python packages used by the project. |

## Running

Install the listed dependencies with:

```bash
python -m pip install -r requirements.txt
```

Run the simulation with:

```bash
python Main.py
```

Simulation parameters can be provided as command-line options. For example:

```bash
python Main.py --firms 2 --lags 1 --K 100 --objective profit
```

See the arguments defined in `Main.py` for the full list of configurable parameters. Simulation outputs are written to the TrainingResults directories.

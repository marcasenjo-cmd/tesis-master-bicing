# Optimization Pipeline

Implements the optimization algorithms for bike-sharing system (BSS) station placement using network-based distance measures, and metaheuristic algorithms such as genetic algorithms (GA) and simulated annealing.

## 📊 Modules

### `helper_optimization.py` - Core Optimization Functions
- Elevation-aware distance calculations and adjustments
- All-pairs shortest path computation
- Graph-based distance metrics and centrality computations
- Optimization result management and validation
- Output: Distance matrices, graph metrics, and analysis results

### `GA/` - Genetic Algorithm Implementation
- **`GA.py`**: Main genetic algorithm with selection, elitism, crossover, and mutation
- **`helper_crossover.py`**: crossover strategies (greedy, top-first, weighted)
- **`helper_mutate.py`**: mutation operator
- **`graph_metric/`**: Network analysis and graph-based distance metrics for the optimization.

### `experiments/` - Optimization Experiments

- **`helper_experiment.py`**: Data loading and experiment setup utilities

- **`GA_hyperparameter_optimization/`**: 1st step. Genetic algorithm parameter tuning for all potential scenarios.

- **`GA_vs_MILP/`**: 2nd step. Comparison between the GA and MILP optimization to validate the GA.

- **`scenarios/`**: 3rd step. This folder contains the main experiments, which are performed across 3 scenarios (see `bss_weights.yaml`):
    - `no_graph_metrics.py`: Optimization without graph-based distance metrics based solely on the spatial factors.
    - `alpha_screening.py`: Optimization including the graph-based distance metrics too
    - `alpha_screening_with_altitude.py`: Optimization including the graph-based distance metrics and also the elevation-aware graph. 

- **`BSS ampliation`**: 4th step. Expansion of an existing BSS using l'Eixample neighborhood from Barcelona as playground.

- **`graph_metrics/`**: Preliminary experiments on the network-based metrics.


## 🚀 Usage

```bash
# Run generic GA optimization
python src/optimization/GA/GA.py

# Ensure data pipeline is completed
# The optimization pipeline requires processed node attributes and network graphs
# from the `assign_to_nodes` pipeline to be available in `data/processed/`
```

### Before the main experiments
```bash
# Hyperparameter optimization
python src/optimization/experiments/GA_hyperparameter_optimization/gridsearch.py

# GA vs MILP comparison to validate GA hyperparameters
python src/optimization/experiments/GA_vs_MILP/stations/GA_execution.py
python src/optimization/experiments/GA_vs_MILP/stations/MILP_execution.py

```
### Main Optimization Scenarios
```bash
# Run optimization without graph metrics
python src/optimization/experiments/scenarios/no_graph_metrics.py


# Run alpha screening experiments (recommended starting point)
python src/optimization/experiments/scenarios/alpha_screening.py


# Run optimization with elevation-aware graph metrics
python src/optimization/experiments/scenarios/alpha_screening_altitude.py

# BSS expansion experiments
python src/optimization/experiments/BSS_ampliation/BSS_ampliation.py
```


### ⚙️ Configuration
The main configuration is in `src/optimization/experiments/scenarios/bss_weights.yaml` which defines multiple scenarios for systematic comparison.

- **N_STATIONS**: Number of stations to place (configurable per experiment)
- **STATION_MIN_DISTANCE**: Minimum distance between stations (default: 300m)
- **Weights**: Configurable importance of different optimization criteria via `bss_weights.yaml`
- **Strategy**: Optimization algorithm parameters (GA, MILP)

### Algorithm Parameters
- **Population Size**: Configurable (default: 50)
- **Generations**: Configurable with early stopping (default: 10,000)
- **Selection Strategy**: Tournament or roulette-wheel selection
- **Crossover Strategy**: Greedy, top-first, or weighted random
- **Mutation Rate**: Configurable (default: 0.05)
- **Elite Fraction**: Percentage of best solutions to preserve (default: 0.05)

### File Paths
- Configured in `paths.py` at project root
- Input data from `data/processed/` directory obtained through `assign_to_nodes` pipeline
- Results saved to `data/processed/experiments/`

## 📁 Output

```
data/processed/
├── experiments/              # Experiment outputs
│   ├── scenarios/            # Main scenario results
│   ├── GA_vs_MILP/          # Comparison results
│   ├── GA_hyperparameter_optimization/  # Parameter tuning results
│   ├── graph_metrics/        # Network analysis results
│   └── BSS_ampliation/       # Expansion experiment results
└── nodes/                   # Input node attributes
    └── normalized_node_attributes.csv
```

### Performance Tips
- Use parallel processing for multiple experiments (adjust `processes` parameter)
- Start with smaller population sizes and fewer generations for testing
- Monitor memory usage with large distance matrices

---
**Author**: Jordi Grau Escolano


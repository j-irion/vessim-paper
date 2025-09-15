# Optimizing Microgrid Composition for Sustainable Data Centers

This repository provides simulation and optimization tools to analyze renewable energy and battery configurations for microgrids in data centers. It was developed for the [Sustainable Supercomputing Workshop](https://sites.google.com/view/sc25/home) at [SC25](https://sc25.supercomputing.org/).

Two alternative approaches are supported:

- **Exhaustive Search**: Brute-force evaluation over the parameter space
- **Hyperparameter Optimization**: Efficient parameter tuning using Optuna

These methods are not meant to be run in succession, but as alternative strategies.

---

## 1. Setup

Install dependencies via [Poetry](https://python-poetry.org/) and change into the working directory:

```bash
poetry install
cd examples
```

## 2. Exhaustive Search
### 2.1 Run the Simulation

**Berkeley:**
```bash
poetry run python renewable_battery_analysis.py --config-name config_battery_analysis_sweep --multirun
```

**Houston:**
```bash
poetry run python renewable_battery_analysis.py --config-name config_battery_analysis_sweep_houston --multirun
```

### 2.2 Convert Results to CSV
After the simulation completes, convert the multirun results into a single CSV file:
```bash
poetry run python convert_results_to_df.py -d ./multirun/<date>/<time> -l berkley  # or houston
```

### 2.3 Visualize Results
You can visualize the results using the provided Jupyter notebook [emissions_analysis.ipynb](./examples/emissions_analysis.ipynb). This notebook will help you analyze the performance of different configurations based on the simulation results.

## 3. Hyperparameter Optimization (Optuna)
Run the Optuna-based optimization for more efficient search:

**Berkeley:**
```bash
poetry run python renewable_battery_analysis.py --config-name config_battery_analysis_sweep_optuna.yaml --multirun
```

**Houston:**
```bash
poetry run python renewable_battery_analysis.py --config-name config_battery_analysis_sweep_optuna_houston.yaml --multirun
```

---

## Notes
- The exhaustive search and Optuna optimization are alternative methods and should not be run in succession.
- All simulation logic is managed via hydra and joblib for parallel execution.
- Configuration files are located in the `configs/` directory.
- CSV outputs contain performance metrics for each configuration, suitable for analysis or visualization.
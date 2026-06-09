# Interactive Map Package

This package provides tools for creating interactive maps to visualize bike-sharing station solutions in Barcelona.

## Package Structure

The package is organized into the following modules:

- `__init__.py` - Package initialization and exports
- `data_loader.py` - Functions to load solution data, weights, and datasets
- `map_generator.py` - Functions to create the base map and add basic layers
- `graph_analyzer.py` - Network analysis and path calculation functions
- `node_features.py` - Functions for visualizing nodes on the map
- `edge_features.py` - Functions for visualizing edges on the map
- `interactive_js.py` - JavaScript functionality for interactivity
- `main.py` - Main entry point for creating maps

## Usage

To create an interactive map, you can use the following code:

```python
from src.results_exploration.interactive_map import create_interactive_map

# Create map with default settings
m = create_interactive_map(output_file="barcelona_stations_map.html")

# Or customize parameters
m = create_interactive_map(
    solution=my_custom_solution,  # Custom list of node IDs
    output_file="custom_map.html",
    root="path/to/project/root",
    experiment_path="path/to/experiment/results"
)
```

You can also run the package from the command line:

```bash
python -m src.results_exploration.create_interactive_map
```

Or with custom parameters:

```bash
python -m src.results_exploration.interactive_map.main -o my_map.html -r ../../ -e path/to/experiment
```

## Features

The interactive map includes the following features:

- Color-coded nodes based on their score
- Paths between nodes with direction indicators
- Popups with detailed information for nodes and edges
- Toggle connections between nodes
- Multiple base map options
- Barcelona city boundary overlay 
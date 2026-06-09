# Node Attributes Assignment Pipeline

Assigns spatial and demographic attributes to graph nodes. The assigned data will be used later in the BSS station optimization.

## 📊 Modules

### `node_attributes_assignation.py` - Main Assignment Pipeline
- Assigns infrastructure, urbanism, and socioeconomic data to network nodes
- Uses building-area weighted assignments for accurate population distribution
- Processes data in parallel batches for efficiency
- Output: `node_attributes.csv`

### `normalize_and_plot_nodes_attributes.py` - Data Normalization
- Applies statistical normalization methods to prepare data for optimization algorithms
- Generates comprehensive visualizations of data distributions
- Creates correlation matrices for feature selection
- Output: `normalized_node_attributes.csv`

### `utils/` - Core Assignment Functions
- **`class_node_assigner.py`**: Main assignment class with spatial methods
- **`class_dataset_normalizer.py`**: Advanced normalization and visualization
- **`load_input_datasets.py`**: Data loading utilities
- **`assign_*.py`**: Specialized assignment functions for different data types

## 🚀 Usage

```bash
# Run complete pipeline
python src/assign_to_nodes/node_attributes_assignation.py
python src/assign_to_nodes/normalize_and_plot_nodes_attributes.py
```

## ⚙️ Configuration

- **Buffer size**: 300 meters (modifiable in each file)
- **Coordinate system**: EPSG:25831 (modifiable in each file)
- **Location**: Barcelona, Spain (modifiable in each file)
- **File paths**: Configured in `paths.py`

## 📁 Output

```
data/processed/nodes/
├── node_attributes.csv          # Raw assigned attributes
└── normalized_node_attributes.csv # Normalized for ML algorithms
```

## 🔧 Dependencies

- **Input**: Datasets from `create_input_dataset` pipeline

## 📊 Data Types Assigned

### Infrastructure
- Public transport lines (bus, metro, tram)
- Bike lane kilometers
- Transport accessibility metrics

### Urbanism
- Points of Interest (POIs) counts
- POI diversity and entropy
- Cycling infrastructure metrics

### Socioeconomic
- Population demographics (age, gender)
- Education levels (primary, secondary, college)
- Economic indicators (income, unemployment)
- Vehicle ownership (cars, motorcycles)

## 📈 Normalization Methods

- **minmax**: Scale to [0, 1] range
- **log**: Natural logarithm for skewed data
- **boxcox**: Box-Cox transformation for non-normal distributions
- **robust**: Median and IQR-based normalization
- **zscore**: Standard score with scaling
- **inverted_***: Inverted versions of all methods

---

**Author**: Jordi Grau Escolano

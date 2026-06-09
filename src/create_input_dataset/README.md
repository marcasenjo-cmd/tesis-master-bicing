# Input Dataset Creation Pipeline

Creates analysis-ready datasets from multiple sources for BSS optimization. The output files are saved in `data/processed/input_variables`

## 📊 Modules

### `socioeconomic.py` - Demographic & Economic Data
- Population by age/sex, education, unemployment, income, vehicle ownership
- Sources: INE (Spanish Census), Barcelona Open Data
- Output: `socioeconomic_census_section_*.csv` and `socioeconomic_neighborhood_*.csv` 

### `infrastructure.py` - Public Transport
- Metro, tram, and bus networks from OpenStreetMap
- Output: `infrastructure_*.csv`

### `urbanism.py` - Urban Infrastructure
- Bike lanes and Points of Interest (12 categories)
- Output: `urbanism_*.csv`

### `topology.py` & `topology_helper.py` - Terrain Analysis
- Elevation data and slope calculations from OpenTopoData
- Output: `topology_altitude_slope_{resolution}m.csv`

## 🚀 Usage

```bash
# Run all modules
python src/create_input_dataset/socioeconomic.py
python src/create_input_dataset/infrastructure.py
python src/create_input_dataset/urbanism.py
python src/create_input_dataset/topology.py
```

## ⚙️ Configuration

- File paths: `paths.py`
- Coordinate systems: WGS84 input → EPSG:25831 output (modifiable in each file).


# tesis-master-bicing

Repositorio asociado a la tesis de máster de **Marc Asenjo**, centrada en el análisis territorial y predictivo de la ocupación de estaciones del sistema **Bicing** de Barcelona.

El proyecto combina datos abiertos de movilidad, información territorial, variables socioeconómicas y análisis geoespacial para construir un **índice explicativo territorial por estación** y estudiar su relación con la **ocupación de las estaciones de Bicing**. Además, se desarrolla un modelo predictivo para analizar hasta qué punto las características del entorno urbano permiten explicar o anticipar patrones de disponibilidad y uso del sistema.

---

## Objetivo del proyecto

El objetivo principal es evaluar si existe una relación significativa entre el contexto territorial de cada estación de Bicing y su nivel de ocupación.

De forma más concreta, el proyecto busca:

- Construir un dataset maestro por estación de Bicing. (\data\processed\stationsbicing_station_master_2026_03)
- Integrar variables urbanas, territoriales y socioeconómicas.
- Calcular un índice explicativo territorial asociado a cada estación. (I_i = wA·A_i + wS·S_i + wC·C_i + wM·M_i + wP·P_i
A_i = accesibilidad / conectividad
S_i = factores socioeconómicos
C_i = cardinalidad / centralidad / densidad de nodos
M_i = modularidad o estructura espacial de red
P_i = componente predictiva relacionada con ocupación/demanda)
- Analizar la correlación entre variables explicativas y ocupación.
- Detectar problemas de multicolinealidad entre variables.
- Entrenar modelos predictivos para estimar variables relacionadas con la ocupación.
- Evaluar si el índice territorial aporta capacidad explicativa sobre la ocupación real del sistema.

---

## Preguntas de investigación

Algunas de las preguntas que guían el análisis son:

1. ¿Qué factores territoriales están más relacionados con la ocupación de las estaciones de Bicing?
2. ¿Existe una asociación entre accesibilidad, entorno urbano, características socioeconómicas y disponibilidad de bicicletas?
3. ¿Puede construirse un índice territorial que sintetice el potencial explicativo de cada estación?
4. ¿Hasta qué punto un modelo predictivo puede anticipar la ocupación de estaciones a partir de variables territoriales?
5. ¿Existen patrones espaciales o desigualdades territoriales en la distribución y uso del sistema?

---

## Estructura del repositorio

```text
tesis-master-bicing/
│
├── data/
│   ├── raw/
│   │   ├── bcn/
│   │   ├── bicing/
│   │   ├── ine/
│   │   └── osm/
│   │
│   └── processed/
│       ├── eda/
│       ├── final_analysis/
│       ├── input_variables/
│       ├── modeling/
│       ├── socioeconomic_validation/
│       ├── stations/
│       └── visual_analysis/
│
├── notebooks/
│
├── scripts/
│   ├── build_station_master_dataset.py
│   ├── build_station_master_dataset_repaired.py
│   ├── build_station_status_targets.py
│   ├── build_modeling_dataset.py
│   ├── build_explanatory_score.py
│   ├── eda_feature_selection.py
│   ├── validate_and_standardize_socioeconomic.py
│   ├── train_predictive_models.py
│   ├── compare_explanatory_vs_predictive.py
│   ├── compare_demographic_feature_sets.py
│   └── visualize_bicing_results.py
│
├── src/
│   ├── assign_to_nodes/
│   ├── create_input_dataset/
│   ├── input/
│   ├── optimization/
│   ├── results_exploration/
│   └── data_loader.py
│
├── visualizations/
│
├── docs/
│   └── thesis/
│
├── .devcontainer/
├── .vscode/
├── Dockerfile
├── requirements.txt
├── paths.py
├── README.md
├── LICENSE
└── .gitignore

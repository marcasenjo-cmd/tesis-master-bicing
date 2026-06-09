"""
This module handles the extraction, processing, and aggregation of socioeconomic data
from multiple Spanish statistical sources.

The module processes data from multiple sources:
1. **INE (Spanish National Statistics Institute)**: Population demographics by age and sex
2. **Barcelona Open Data**: Education levels, employment rates, vehicle ownership
3. **Municipal Statistics**: Income levels, household characteristics, nationality data

Key Features:
- Census section-level and neighborhood level aggregation for spatial analysis
- Automatic file caching to avoid redundant processing

Data Categories:
- **Demographics**: Age groups, sex distribution, population density
- **Education**: Primary, secondary, and tertiary education levels
- **Employment**: Unemployment rates by neighborhood
- **Economics**: Income levels, household sizes, vehicle ownership
- **Diversity**: Nationality and immigration statistics

Output Files:
- socioeconomic_census_section_population_sex_age.csv: Demographic breakdowns
- socioeconomic_census_section_education.csv: Education level distributions
- socioeconomic_census_section_employment.csv: Employment statistics
- socioeconomic_census_section_vehicle_ownership.csv: Vehicle ownership data
- socioeconomic_census_section_income.csv: Income and economic indicators
- socioeconomic_census_section_household_size.csv: Household characteristics
- socioeconomic_census_section_non_spanish_population.csv: Nationality data

Author: Jordi Grau Escolano
"""

import sys
import os
import random
from pathlib import Path
from matplotlib import pyplot as plt
import yaml
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore

# Set up project root and import project-specific paths
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from paths import *
import src.data_loader as dl
import src.create_input_dataset.socioeconomic_helper as sh

EPSG = 25831

################################################################################
############################ POPULATION DATA ###################################
################################################################################

def create_population_sex_age_census_section_dataset():
    """   
    This function processes INE population data to create demographic breakdowns
    by age groups and sex for each census section in Barcelona. 
    
    Data Source:
    - INE Population Census: https://ine.es/pcaxisdl/t20/e245/p07/a2022/l0/0001.px
        
    Returns:
        None: The function saves the processed dataset to a CSV file.
        
    Processing Steps:
    1. Load raw population data from INE source
    2. Extract census section boundaries and geometries
    3. Compute demographic breakdowns by sex and age groups
    4. Aggregate population totals by census section
    5. Merge demographic data with spatial boundaries
    6. Export to CSV for further analysis
        
    Output Columns:
    - census_section: Unique identifier for each census section
    - f, m: Female and male population counts
    - age groups: Population counts for 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+
    - total_population: Total population per census section

    """
    output_file = f"{PR_INPUT}/socioeconomic_census_section_population_sex_age.csv"
    if os.path.exists(output_file):
        print(f"Population_census_section dataset already exists in {output_file}")
        return None
              
    # Load data
    df_pop = dl.load_clean_bcn_population_data(root='')
    census_sections = dl.load_bcn_census_sections()

    # Compute demographic data
    df_sex = sh.compute_demographic_data(df_pop, 'sex', output_type='absolute')
    df_age = sh.compute_demographic_data(df_pop, 'age', output_type='absolute')
    df_pop_cs = df_pop.groupby('census_section')['population'].sum()

    # Merge demographic data with census sections
    census_sections_pop = (
        census_sections
        .join(df_sex, how='left')
        .join(df_age, how='left')
        .join(df_pop_cs.rename('total_population'), how='left')
    ).drop('geometry', axis=1)

    census_sections_pop.drop(['0-9'], axis=1, inplace=True)

    # Save the resulting DataFrame to a CSV file
    census_sections_pop.to_csv(output_file, index=True)
    print(f"Population_census_section dataset created successfully and saved to {output_file}.")

def create_education_census_section_dataset():
    """    
    This function processes Barcelona Open Data education statistics to create
    education level distributions across census sections. 
    
    Data Sources:
    - Barcelona Open Data: https://opendata-ajuntament.barcelona.cat/data/ca/dataset/pad_mdbas_niv-educa-esta_sexe
    - Education Categories: https://opendata-ajuntament.barcelona.cat/data/ca/dataset/pad-dimensions/resource/b00be3f8-9328-4175-8689-24a25bc0907c
        
    Returns:
        None: The function saves the processed dataset to a CSV file.
        
    Processing Steps:
    1. Load raw education data from Barcelona Open Data
    2. Handle statistical confidentiality (replace ".." with random values)
    3. Map detailed education levels to standardized categories
    4. Aggregate data by census section and education category
    5. Create education distribution columns
    6. Export standardized dataset
        
    Education Categories:
    - **Primary**: Less than primary education and primary education
    - **Secondary**: Lower and upper secondary education
    - **College**: Tertiary education (university and college)
        
    Data Handling:
    - Values below 5 are marked as ".." for confidentiality
    - Random values (0-5) are assigned to maintain statistical integrity
    - Education levels are mapped to three broad categories
    - Sex-specific data is aggregated to total counts
        
    Output Columns:
    - census_section: Unique identifier for each census section
    - education_primary: Count of people with primary education or less
    - education_secondary: Count of people with secondary education
    - education_college: Count of people with tertiary education
        
    Applications:
    - Education-based accessibility analysis
    - Socioeconomic status modeling
    - Station placement optimization
    - Demographic demand forecasting
    """
    output_file = f"{PR_INPUT}/socioeconomic_census_section_education.csv"
    if os.path.exists(output_file):
        print(f"Education_census_section dataset already exists in {output_file}")
        return None
    
    # Load the education level data
    input_file = f'{RAW_BCN}/2024_pad_mdbas_niv-educa-esta_sexe.csv'
    df_education = pd.read_csv(input_file, sep=',', encoding='ISO-8859-1')

    # Drop unnecessary columns
    df_education.drop(
        ['Data_Referencia', 'Codi_Districte', 'Nom_Districte', 'Codi_Barri', 'Nom_Barri', 'AEB'], 
        axis=1, inplace=True
    )

    # Rename columns
    df_education.rename(columns={
        'Seccio_Censal': 'census_section',
        'Valor': 'value', 
        'NIV_EDUCA_esta': 'education_level',
        'SEXE': 'sex'
    }, inplace=True)

    # Create census_section field
    df_education['census_section'] = '08019' + df_education['census_section'].apply(str).str.zfill(5)

    # Replace ".." values with random values
    random_values = [0, 1, 2, 3, 4, 5]
    df_education['value'] = df_education['value'].apply(
        lambda x: random.choice(random_values) if x == ".." else x
    )
    df_education['value'] = df_education['value'].astype(int)

    # Map existing education levels to new categories
    education_mapping = {
        1: 'primary',  # Sense estudis
        2: 'primary',  # Estudis primaris
        3: 'secondary',  # ESO
        4: 'secondary',  # Batchillerat/Supperior
        5: 'college',  # Estudis universitaris
        6: 'not available' 
    }
    df_education['education_category'] = df_education['education_level'].map(education_mapping)

    # Aggregate values that are split by sex
    df_education = df_education.groupby(['census_section', 'education_category'])['value'].sum().reset_index()

    # Pivot the table to have education categories as columns
    df_education = df_education.pivot(index='census_section', columns='education_category', values='value')
    df_education = df_education.fillna(0)
    df_education.columns = [f'education_{col}' for col in df_education.columns]

    # Convert all values to integers
    for col in df_education.columns:
        df_education[col] = df_education[col].apply(int)

    # Filter columns
    df_education.drop('education_not available', axis=1, inplace=True)
    cols = ['education_primary', 'education_secondary', 'education_college']
    df_education = df_education[cols]

    # # Transform columns from absolute numbers to fractions
    # df_education[cols] = df_education[cols].div(df_education[cols].sum(axis=1), axis=0)
    # df_education[cols] = round(df_education[cols], 2)

    # Save
    df_education.to_csv(output_file, index=True)
    print(f"Education dataset created successfully and saved to {output_file}.")

def create_non_spanish_population_census_section_dataset():
    """   
    This function processes Barcelona Open Data nationality statistics to quantify
    immigrant populations by census section.
    
    Data Source:
    - Barcelona Open Data: https://opendata-ajuntament.barcelona.cat/data/ca/dataset/pad_mdbas_nacionalitat-g_sexe
        
    Returns:
        None: The function saves the processed dataset to a CSV file.
        
    Processing Steps:
    1. Load raw nationality data from Barcelona Open Data
    2. Handle statistical confidentiality issues
    3. Categorize nationality groups
    4. Aggregate by census section
    5. Export non-Spanish population counts
        
    Nationality Categories:
    - **Spanish**: Spanish nationals
    - **European Union**: EU citizens outside Spain
    - **Rest of World**: Non-EU international residents
        
    Data Handling:
    - Values below 5 are marked as ".." for confidentiality
    - Random values (0-5) are assigned to maintain statistical integrity
    - Focus on non-Spanish population for diversity analysis
        
    Output Columns:
    - census_section: Unique identifier for each census section
    - non_spanish_population: Count of non-Spanish residents
    """
    output_file = f"{PR_INPUT}/socioeconomic_census_section_non_spanish_population.csv"
    if os.path.exists(output_file):
        print(f"Non-Spanish population dataset already exists in {output_file}")
        return None
    
    # Load the education level data
    input_file = f'{RAW_BCN}/2024_pad_mdbas_nacionalitat-g_sexe.csv'
    df = pd.read_csv(input_file)

    # Drop unnecessary columns
    drop_columns = ['Data_Referencia', 'Codi_Districte', 'Nom_Districte', 'Codi_Barri', 'Nom_Barri', 'AEB']
    df.drop(columns=drop_columns, inplace=True)

    # Rename columns for clarity
    rename_columns = {
        'Seccio_Censal': 'census_section',
        'Valor': 'value', 
        'NACIONALITAT_G': 'nationality',
        'SEXE': 'sex'
    }
    df.rename(columns=rename_columns, inplace=True)

    # Create a full census_section field by padding with zeros
    df['census_section'] = '08019' + df['census_section'].astype(str).str.zfill(5)

    # Remove rows where nationality equals 1 (Spanish population)
    df = df[df['nationality'] != 1]

    # Replace ".." in the 'value' column with random values and convert to integer
    random_values = [0, 1, 2, 3, 4, 5]
    df['value'] = df['value'].apply(lambda x: random.choice(random_values) if x == ".." else x)
    df['value'] = df['value'].astype(int)

    # Group by census_section and sum the 'value' column
    df_grouped = df.groupby('census_section', as_index=False)['value'].sum()
    df_grouped.rename(columns={'value': 'non_spanish_population'}, inplace=True)

    # Save the resulting DataFrame to a CSV file
    df_grouped.to_csv(output_file, index=False)
    print(f"Non-Spanish dataset created successfully and saved to {output_file}.")

def create_unemployment_neighborhood_dataset():
    """   
    This function processes 2024 unemployment data from Barcelona's open data portal and assigns
    unemployment statistics to neighborhoods by spatially joining census sections with neighborhood
    boundaries. It handles cases where census sections don't perfectly align with neighborhood
    boundaries by using iterative buffering.
    
    Dataset source: https://portaldades.ajuntament.barcelona.cat/ca/estad%C3%ADstiques/av935yrhx9?view=table
    
    Processing steps:
    1. Loads unemployment percentage data by neighborhood from January 31st, 2024
    2. Loads population data at census section level
    3. Spatially joins census sections with neighborhood boundaries
    4. Uses iterative buffering to assign unassigned census sections to neighborhoods
    5. Calculates absolute unemployment counts from percentages
    6. Creates visualizations of unemployment data
    7. Saves the processed dataset
    
    Returns:
        None: The function saves the processed dataset to the specified output file.
        
    Output file:
        socioeconomic_neighborhood_unemployment.csv containing:
        - neighborhood: neighborhood name
        - unemployment_percentage: unemployment rate as percentage
        - unemployment: absolute unemployment count
    """
    input_file = f'{RAW_BCN}/bcn_atur.csv'
    visualization_file = f'{VISUALIZATIONS}/raw_data/unemployment.png'
    output_file = f"{PR_INPUT}/socioeconomic_neighborhood_unemployment.csv"
    
    # if os.path.exists(output_file):
    #     print(f"Unemployment_neighborhood dataset already exists in {output_file}")
    #     return None

    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Extract neighborhood name for rows with '-' in 'Tipus de territori'
    mask = df['Tipus de territori'] == '-'
    if mask.any():
        df.loc[mask, 'Tipus de territori'] = df.loc[mask, 'Territori'].str.extract(r'\(([^)]+)\)', expand=False)

    df = df[df['Tipus de territori'] == 'Barri']

    # Select the relevant columns: 'Territori' and '31 ene 2026'
    df_filtered = df[['Territori', '31 ene 2026']]

    # Rename columns
    df_filtered.columns = ['neighborhood', 'unemployment_percentage']

    # Transform from percentage to absolute counts using population data at census-section level
    # Load population data at the census section geometries
    df_pop = dl.load_clean_bcn_population_data()
    df_pop = df_pop[~df_pop['age'].isin(['0-9', '10-19', '70+'])]
    df_pop = df_pop.groupby('census_section')['population'].sum().reset_index()
    census_sections = dl.load_bcn_census_sections()
    census_sections = census_sections.to_crs(EPSG)

    # Merge population data with census sections geometries 
    df_pop = df_pop.merge(census_sections, on='census_section', how='left')
    df_pop = gpd.GeoDataFrame(df_pop)

    # Find census sections inside of neighborhoods. There is no neighborhood column 
    neighborhoods = dl.load_bcn_neighborhoods()
    
    # Initial assignment with no buffer
    df_pop = gpd.sjoin(df_pop, neighborhoods, how='left', predicate='within')
    initial_unassigned = df_pop['neighborhood'].isna().sum()
    total_sections = len(df_pop)
    print(f"\nInitial assignment (no buffer):")
    print(f"- Total census sections: {total_sections}")
    print(f"- Unassigned sections: {initial_unassigned} ({(initial_unassigned/total_sections)*100:.1f}%)\n")

    # Iteratively increase buffer for unassigned census sections
    buffer_size = 50
    max_buffer = 500  # Maximum buffer size to try

    while buffer_size <= max_buffer:
        # Only proceed if there are unassigned census sections
        mask_na = df_pop['neighborhood'].isna()
        if not mask_na.any():
            break
        
        print(f"Buffer {buffer_size}m: {mask_na.sum()} unassigned ({(mask_na.sum()/total_sections)*100:.1f}%)")
        
        # Create buffered neighborhoods
        neighborhoods['geometry'] = neighborhoods.geometry.buffer(buffer_size)
        
        # Only perform spatial join for unassigned census sections
        df_pop_na = df_pop[mask_na].copy()
        df_pop_assigned = df_pop[~mask_na].copy()
        
        # Perform spatial join with current buffer
        df_pop_na_joined = gpd.sjoin(
            df_pop_na[['census_section', 'population', 'geometry']], 
            neighborhoods, how='left', predicate='within')
        
        # Combine assigned and newly assigned sections
        df_pop = pd.concat([df_pop_assigned, df_pop_na_joined], ignore_index=True)
        
        # Print results of this iteration
        newly_assigned = mask_na.sum() - df_pop['neighborhood'].isna().sum()
        print(f"  → Assigned {newly_assigned}, remaining {df_pop['neighborhood'].isna().sum()} ({(df_pop['neighborhood'].isna().sum()/total_sections)*100:.1f}%)")
        
        # Increase buffer for next iteration
        buffer_size += 10

    # Print final summary
    final_unassigned = df_pop['neighborhood'].isna().sum()
    print(f"\nFinal results: {total_sections - final_unassigned}/{total_sections} assigned ({((total_sections - final_unassigned)/total_sections)*100:.1f}%), {final_unassigned} unassigned ({(final_unassigned/total_sections)*100:.1f}%)")

    # Get population per neighborhood and merge to unemployment data
    df_pop = df_pop.groupby('neighborhood')['population'].sum().reset_index()
    df_unemployment = df_pop.merge(df_filtered, on='neighborhood', how='left')
    df_unemployment['unemployment'] = df_unemployment['unemployment_percentage'] * df_unemployment['population'] / 100
    
    # Merge neighborhood geometries without buffers
    neighborhoods = dl.load_bcn_neighborhoods()
    df_unemployment = df_unemployment.merge(neighborhoods, on='neighborhood', how='left')
    df_unemployment = gpd.GeoDataFrame(df_unemployment)
    
    # if not os.path.exists(visualization_file):
    fig, axs = plt.subplots(1,3, figsize=(12, 5))
    bcn_boundary = dl.load_bcn_boundary()
    
    # Plot boundary on all subplots
    for ax in axs:
        bcn_boundary.plot(ax=ax, color='black', linewidth=1)
    
    # Define columns and titles for plotting
    columns = ['unemployment_percentage', 'unemployment', 'population']
    titles = ['Unemployment percentage', 'Unemployment absolute', 'Population']
    
    # Plot data and set properties for each subplot
    for i, (column, title) in enumerate(zip(columns, titles)):
        df_unemployment.plot(ax=axs[i], column=column, legend=True, legend_kwds={'shrink': 0.8})
        axs[i].axis('off')
        axs[i].set_title(title)

    fig.savefig(visualization_file, dpi=300)
    plt.close(fig)


    df_unemployment.drop(['population', 'geometry'], axis=1, inplace=True)

    # Save the processed data to the output file
    df_unemployment.to_csv(output_file, index=False)

################################################################################
########################### ECONOMICAL DATA ####################################
################################################################################

def create_income_census_section_dataset():
    """   
    This function processes income data from Spain's National Statistics Institute (INE) to create
    a dataset of average income statistics at the census section level. It handles
    both personal and household income metrics for the year 2022.
    
    Dataset source: INE (Instituto Nacional de Estadística) - Average income statistics
    
    Processing steps:
    1. Loads raw income data from INE CSV files
    2. Cleans and filters the data to remove missing values
    3. Standardizes column names and income type categories
    4. Pivots the data to create separate columns for personal and household income
    5. Formats census section identifiers consistently
    6. Saves the processed dataset
    
    Returns:
        None: The function saves the processed dataset to the specified output file.
        
    Output file:
        socioeconomic_census_section_income.csv containing:
        - census_section: census section identifier
        - income_2022_pers: average personal income for 2022
        - income_2022_house: average household income for 2022
    """
    output_file = f"{PR_INPUT}/socioeconomic_census_section_income.csv"
    if os.path.exists(output_file):
        print(f"Unemployment_neighborhood dataset already exists in {output_file}")
        return None
    
    # Load the income data
    input_file = f'{RAW_INE}/ine_avg_income.csv'
    df_income = pd.read_csv(input_file, sep=';', encoding='utf-8-sig', thousands='.')

    # Drop unnecessary columns
    df_income.drop(['Municipios', 'Distritos'], axis=1, inplace=True)

    # Drop rows with missing values
    df_income.dropna(inplace=True)

    # Rename columns for clarity
    df_income.rename(columns={
        'Secciones': 'census_section',
        'Indicadores de renta media y mediana': 'income',
        'Periodo': 'year',
        'Total': 'value'
    }, inplace=True)

    # Replace income type values
    income_str = {
        'Renta neta media por persona': 'pers',
        'Renta neta media por hogar': 'house'
    }
    df_income['income'] = df_income['income'].map(income_str)

    # Split the census_section field if needed
    df_income[['cs_id', 'cs_name']] = df_income['census_section'].str.split(" ", n=1, expand=True)
    df_income.drop(['census_section', 'cs_name'], axis=1, inplace=True)
    df_income.rename(columns={'cs_id': 'census_section'}, inplace=True)

    # Pivot the table to have income types as columns
    df_income = df_income.pivot(index='census_section', columns='income', values='value')
    df_income.rename(columns={'pers': 'income_2022_pers', 'house': 'income_2022_house'}, inplace=True)
    df_income.columns.name = ''

    # Save the resulting DataFrame to a CSV file
    df_income.to_csv(output_file, index=True)
    print(f"Income dataset created successfully and saved to {output_file}.")

def create_household_size_census_section_dataset():
    """   
    This function processes cadastral data from Barcelona's open data portal to create a dataset
    of average household sizes (in square meters) at the census section level.
    
    Dataset source: 
    https://opendata-ajuntament.barcelona.cat/data/ca/dataset/est-cadastre-habitatges-superficie-mitjana

    Processing steps:
    1. Loads raw household size data from Barcelona's open data portal and removes unnecessary columns
    2. Renames and formats columns and spatial regions for clarity and consistency
    6. Sets census section as the index and saves the dataset
    
    Returns:
        None: The function saves the processed dataset to the specified output file.
        
    Output file:
        socioeconomic_census_section_household_size.csv containing:
        - census_section: census section identifier (index)
        - household_avg_m2: average household size in square meters
    """
    output_file = f"{PR_INPUT}/socioeconomic_census_section_household_size.csv"
    if os.path.exists(output_file):
        print(f"Unemployment_neighborhood dataset already exists in {output_file}")
        return None

    # Load the household size data
    input_file = f'{RAW_BCN}/2024_loc_hab_sup_mitjana.csv'
    df_household = pd.read_csv(input_file, sep=',', encoding='ISO-8859-1')

    # Drop unnecessary columns
    df_household.drop(['Any', 'Nom_districte', 'Nom_barri'], axis=1, inplace=True)

    # Rename columns for clarity
    df_household.rename(columns={
        'Codi_districte': 'district_code',
        'Codi_barri': 'neigh_code',
        'Seccio_censal': 'census_section_partial',
        'Sup_mitjana_m2': 'household_avg_m2'
    }, inplace=True)

    # Format the census_section field
    for col in ['district_code', 'neigh_code']:
        df_household[col] = df_household[col].apply(str).str.zfill(2)

    df_household['census_section_partial'] = df_household['census_section_partial'].apply(str).str.zfill(3)

    df_household['census_section'] = (
        '08019' + 
        df_household['district_code'] + 
        df_household['census_section_partial']
    )

    # Drop unnecessary columns
    df_household.drop(['district_code', 'neigh_code', 'census_section_partial'], axis=1, inplace=True)

    # Set the index to census_section
    df_household.set_index('census_section', inplace=True)

    # Save the resulting DataFrame to a CSV file
    df_household.to_csv(output_file)

    print(f"Household size dataset created successfully and saved to {output_file}.")

def car_ownership_census_section_dataset():
    """   
    This function processes vehicle registration data from Barcelona's open data portal to create
    a dataset of vehicle ownership at the census section level. It handles multiple
    vehicle types and converts motorization indices (vehicles per 1000 inhabitants) to absolute counts.
    
    Dataset sources:
    - Motorization index: https://portaldades.ajuntament.barcelona.cat/ca/estad%C3%ADstiques/om9l2845pv
    - Raw data: https://opendata-ajuntament.barcelona.cat/data/en/dataset/est_vehicles_index_motor
    
    The function:
    1. Loads vehicle motorization index data from Barcelona's open data portal
    2. Cleans and standardizes column names and vehicle type categories
    3. Formats census section identifiers consistently
    4. Translates vehicle type names to English
    5. Pivots data to create separate columns for each vehicle type and combines related vehicle types
    6. Converts motorization indices to absolute counts using population data
    7. Creates visualizations of the data
    9. Saves the processed dataset
    
    Returns:
        None: The function saves the processed dataset to the specified output file.
        
    Output file:
        socioeconomic_census_section_car_ownership.csv containing:
        - census_section: census section identifier (index)
        - cars: motorization index for cars (per 1000 inhabitants)
        - motos: motorization index for motorcycles and mopeds (per 1000 inhabitants)
        - others: motorization index for other vehicles (per 1000 inhabitants)
        - population: total population in the census section
        - cars_abs: absolute count of cars
        - motos_abs: absolute count of motorcycles and mopeds
        - others_abs: absolute count of other vehicles
    """

    output_file = f"{PR_INPUT}/socioeconomic_census_section_car_ownership.csv"
    if os.path.exists(output_file):
        print(f"Car ownership dataset already exists in {output_file}")
        return None
    
    # Load the dataset
    input_file = f'{RAW_BCN}/2023_parc_vehicles_index_motoritzacio.csv'
    df = pd.read_csv(input_file)

    # Drop unnecessary columns
    drop_columns = ['Any', 'Nom_Districte', 'Codi_Barri', 'Nom_Barri']
    df.drop(columns=drop_columns, inplace=True)

    # Rename columns
    rename_columns = {
        'Seccio_Censal': 'census_section',
        'Codi_Districte': 'district',
        'Tipus_Vehicle': 'vehicle_type', 
        'Index_Motoritzacio': 'motorization_index',
    }
    df.rename(columns=rename_columns, inplace=True)

    # Filter out invalid census sections
    df = df[df['census_section'] != 'NC']

    # Create a full census_section field by padding with zeros
    df['census_section'] = '08019' + df['district'].astype(str).str.zfill(2)  + \
        df['census_section'].astype(str).str.zfill(3)

    # Translate vehicle names
    vehicle_translation = {
        'Turismes': 'cars',
        'Motos': 'motorcycles',
        'Ciclomotors': 'mopeds',
        'Furgonetes': 'vans',
        'Camions': 'trucks',
        'Resta de vehicles': 'others'
    }
    df['vehicle_type'] = df['vehicle_type'].map(vehicle_translation)
    
    # Pivot the DataFrame to create columns for each vehicle type
    df = df.pivot(index='census_section', columns='vehicle_type', values='motorization_index')

    # Replace NaN values and convert to int
    df = df.fillna(0)
    for col in df.columns:
        df[col] = df[col].apply(int)
    
    # Select columns
    df['motos'] = df['motorcycles'] + df['mopeds']
    df['others'] = df['others'] + df['trucks'] + df['vans']
    df.drop(['motorcycles', 'mopeds', 'trucks', 'vans'], axis=1, inplace=True)   
    cols = ['cars', 'motos', 'others']

    # Transform from motorization index (tant per mil) to absolute counts
    # Load population data and merge with car ownership data
    df_pop = dl.load_clean_bcn_population_data()
    df_pop = df_pop.groupby('census_section')['population'].sum().reset_index()
    df = df.merge(df_pop, on='census_section', how='left')

    # Transformation: multiply by population to get absolute counts
    for col in cols:
        df[f"{col}_abs"] = df['population'] * (df[col] / 1000)
        df[col] = round(df[col], 2)
        df[f"{col}_abs"] = round(df[f"{col}_abs"], 2)
    # df.drop('population', axis=1, inplace=True)

    fig, ax = plt.subplots(1,3, figsize=(12, 5))
    print(df['cars_abs'].max())
    gdf = dl.load_bcn_census_sections()
    gdf = gdf.merge(df, on='census_section', how='left')
    gdf.plot(ax=ax[0], column='cars', legend=True, legend_kwds={'shrink': 0.3})
    gdf.plot(ax=ax[1], column='population', legend=True, legend_kwds={'shrink': 0.3})
    gdf.plot(ax=ax[2], column='cars_abs', legend=True, legend_kwds={'shrink': 0.3})
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    plt.tight_layout()
    fig.savefig(f'{VISUALIZATIONS}/raw_data/raw__TEST_car_ownership.png', dpi=300)
    plt.close(fig)

    # Save
    df.set_index('census_section', inplace=True)
    df.to_csv(output_file, index=True)
    print(f"Car ownership dataset created successfully and saved to {output_file}.")


if __name__ == "__main__":


    location = 'Barcelona, Barcelona, Catalunya, España'

    # Download raw data
    sh.download_and_clean_population_data('08019')
    sh.download_residential_buildings(location)

    # Socioeconomic data
    create_population_sex_age_census_section_dataset()  # Population, sex %, age %
    create_education_census_section_dataset()  # People's education in counts for 3 categories
    create_unemployment_neighborhood_dataset()  # Unemployment %
    create_non_spanish_population_census_section_dataset()  # Non-Spanish people count
    car_ownership_census_section_dataset() # Number of vehicles per 1000 people

    # Economic data
    create_income_census_section_dataset()  # Avg income person and household 
    create_household_size_census_section_dataset()  # Avg household size

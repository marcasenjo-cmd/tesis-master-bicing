import numpy as np  # type: ignore


def generate_weight_combinations_3_variables(step=0.33):
    """
    Generate systematic weight combinations for three features.
    
    Args:
        step (float): The step size for weight increments (0.33)
    
    Returns:
        list: List of dictionaries containing weight combinations
    """
    weight_combinations = []
    
    # Round to avoid floating point precision issues
    round_to = 2
    
    # Generate all possible combinations that sum to 1
    for w1 in np.arange(0, 1 + step, step):
        for w2 in np.arange(0, 1 - w1 + step, step):
            w3 = round(1 - w1 - w2, round_to)
            if w3 >= 0:
                # Create weight combination dictionary
                weights = {
                    'population': round(float(w1), round_to),
                    'education_primary': round(float(w2), round_to),
                    'unemployment_percentage': round(float(w3), round_to)
                }
                weight_combinations.append(weights)
    
    return weight_combinations

def generate_weight_vectors(variables, n_samples=50, seed=42):
    """
    Generate representative set of weight vectors using Monte Carlo sampling.
    Each vector is normalized so components sum to 1.
    
    Args:
        variables (list): List of variable names
        n_samples (int): Number of unique weight vectors to generate
        seed (int): Random seed for reproducibility
    
    Returns:
        list: List of dictionaries, each containing normalized weights
    """
    np.random.seed(seed)
    
    # Number of variables
    n_variables = len(variables)
    
    # Generate unique weight combinations
    weight_vectors = set()
    while len(weight_vectors) < n_samples:
        # Generate random weights between 0 and 1
        weights = np.random.random(n_variables)
        # Normalize so they sum to 1
        weights = weights / np.sum(weights)
        # Round to 3 decimal places to avoid floating point issues
        weights = np.round(weights, 3)
        # Convert to tuple for hashability (to use in set)
        weights = tuple(weights)
        
        # Only add if sum is still 1 after rounding
        if np.sum(weights) == 1.0:
            weight_vectors.add(weights)
    
    # Convert to list of dictionaries with variable names
    weight_dicts = []
    for weights in weight_vectors:
        weight_dict = {var: weight for var, weight in zip(variables, weights)}
        weight_dicts.append(weight_dict)
    
    return weight_dicts
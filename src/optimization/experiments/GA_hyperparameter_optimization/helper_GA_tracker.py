import os
import pandas as pd  # type: ignore
import numpy as np  # type: ignore

def track_exploration_metrics(generations_population, total_graph_nodes, output_file, weights=None):
    exploration_metrics = []
    prev_gen_nodes = set()
    cumulative_explored_nodes = set()

    if weights:
        weights_str = {k: v for k, v in weights.items()}
    
    for gen, population in enumerate(generations_population):
        current_generation_nodes = set(np.concatenate(population))
        
        # Count the number of unique nodes in the current generation
        unique_nodes_count = len(current_generation_nodes)
        
        # Calculate the number of new nodes introduced in this generation
        new_nodes_count = len(current_generation_nodes - prev_gen_nodes)
        
        # Calculate the fraction of new nodes relative to the number of nodes in the current generation
        fraction_new_nodes = new_nodes_count / unique_nodes_count
        
        # Calculate the coverage of the current generation in relation to total graph nodes
        generation_coverage = unique_nodes_count / total_graph_nodes
        
        # Update cumulative explored nodes with the current generation's nodes
        cumulative_explored_nodes.update(current_generation_nodes)
        
        # Store the calculated metrics in a dictionary
        metrics = {
            'gen': gen,
            'unique_nodes_count': unique_nodes_count,
            'fraction_new_nodes': fraction_new_nodes,
            'generation_coverage': generation_coverage,
            'cumulative_exploration_fraction': len(cumulative_explored_nodes) / total_graph_nodes
        }
        
        # Add weights to metrics if provided
        if weights:
            # Add a single column with the full weights dictionary as a string
            metrics['weights'] = str(weights)
        
        # Append the metrics for the current generation to the exploration metrics list
        exploration_metrics.append(metrics)

        # Update previous generation nodes for the next iteration
        prev_gen_nodes = current_generation_nodes
    
    # Convert the exploration metrics list to a DataFrame
    df_metrics = pd.DataFrame(exploration_metrics)
    
    # Save the metrics DataFrame to a CSV file, appending if the file already exists
    df_metrics.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
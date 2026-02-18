import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.cluster import AgglomerativeClustering 
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

import networkx as nx

def create_distance_matrix(
    similarity_pair: pd.DataFrame, 
    pair1_column_name:str = "category_x",
    pair2_column_name:str = "category_y",
    similarity_column_name:str = "similarity",
    max_similarity:int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Converts a similarity pair DataFrame into a distance matrix and its condensed form.
    This function calculates a distance matrix by subtracting the similarity values 
    from a maximum similarity value, ensuring that higher similarity corresponds to 
    lower distance. The resulting distance matrix is symmetric, with diagonal values 
    set to zero (distance to self). Additionally, the function generates a condensed 
    distance matrix suitable for clustering algorithms.
        similarity_pair (pd.DataFrame): A DataFrame containing similarity pairs with 
            columns for the two categories and their similarity score.
        pair1_column_name (str, optional): The column name for the first category in 
            the similarity pair. Defaults to "category_x".
        pair2_column_name (str, optional): The column name for the second category in 
            the similarity pair. Defaults to "category_y".
        similarity_column_name (str, optional): The column name for the similarity 
            score. Defaults to "similarity".
        max_similarity (int, optional): The maximum similarity value used to calculate 
            distances. Defaults to 4.
        Tuple[pd.DataFrame, np.ndarray]: 
            - A DataFrame representing the distance matrix, with rows and columns 
              corresponding to unique categories.
            - A 1D NumPy array representing the condensed distance matrix, suitable 
              for clustering algorithms like hierarchical clustering.
    similarity_pair: pd.DataFrame, 
    pair1_column_name:str = "category_x",
    pair2_column_name:str = "category_y",
    similarity_column_name:str = "similarity",
    max_similarity:int = 4) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    
    # Revert the similarity to distance for easier and meaningful representation
    similarity_pair['distance'] = max_similarity - similarity_pair[similarity_column_name]
    
    # Create matrix form
    terms = pd.unique(similarity_pair[[pair1_column_name, pair2_column_name]].values.ravel())
    n = len(terms)
    distance_matrix = pd.DataFrame(np.zeros((n, n)), index=terms, columns=terms)
    
    for _, row in similarity_pair.iterrows():
        distance_matrix.loc[row[pair1_column_name], row[pair2_column_name]] = row['distance']
        distance_matrix.loc[row[pair2_column_name], row[pair1_column_name]] = row['distance']

    # Diagonal is 0 (distance to self)
    np.fill_diagonal(distance_matrix.values, 0)

    condensed_dist = squareform(distance_matrix.values)
    
    return terms, distance_matrix, condensed_dist

def build_similarity_graph(terms_pair, threshold=5):
    """_summary_

    Args:
        terms_pair (_type_): _description_
        threshold (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    
    graph = nx.Graph()
    
    # Add all unique terms as nodes
    all_terms = set(terms_pair['term_x'].unique()) | set(terms_pair['term_y'].unique())
    print(f"All terms ({len(all_terms)}): {sorted(all_terms)}")
    
    for term in all_terms:
        graph.add_node(term)
    
    # Add edges based on similarity threshold
    for _, row in terms_pair.iterrows():
        if row['similarity'] >= threshold:
            graph.add_edge(
                row['term_x'],
                row['term_y'],
                weight=float(row['similarity']),
                distance=float(10 - row['similarity'])
            )
    
    # Print selected edges
    print(f"\nEdges added (threshold >= {threshold}):")
    for u, v, data in graph.edges(data=True):
        print(f"  {u} -- {v} (weight: {data['weight']:.2f})")
    
    print(f"\nSummary: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    return graph
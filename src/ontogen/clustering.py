"""
Graph building, similarity graphs, and distance matrix utilities.

This module provides functions for constructing similarity-based graphs
from pairwise term comparisons and converting them to distance matrices
for clustering algorithms.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Set

import networkx as nx
from scipy.spatial.distance import squareform


def create_distance_matrix(
    similarity_pair: pd.DataFrame,
    pair1_column_name: str = "category_x",
    pair2_column_name: str = "category_y",
    similarity_column_name: str = "similarity",
    max_similarity: int = 5,
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Convert a similarity-pair DataFrame into a distance matrix and its condensed form.

    Calculates a distance matrix by subtracting similarity values from a maximum
    similarity value, ensuring that higher similarity corresponds to lower distance.
    The resulting distance matrix is symmetric with diagonal values set to zero.

    Args:
        similarity_pair: DataFrame containing similarity pairs with columns for
            the two categories and their similarity score.
        pair1_column_name: Column name for the first category. Defaults to "category_x".
        pair2_column_name: Column name for the second category. Defaults to "category_y".
        similarity_column_name: Column name for the similarity score. Defaults to "similarity".
        max_similarity: Maximum similarity value used to calculate distances. Defaults to 5.

    Returns:
        A tuple of (terms, distance_matrix, condensed_distance):
            - terms: Array of unique category names.
            - distance_matrix: DataFrame with rows/columns as categories.
            - condensed_distance: 1D array for hierarchical clustering.
    """
    # Revert the similarity to distance for easier and meaningful representation
    similarity_pair = similarity_pair.copy()
    similarity_pair['distance'] = max_similarity - similarity_pair[similarity_column_name]

    # Create matrix form
    terms = pd.unique(
        similarity_pair[[pair1_column_name, pair2_column_name]].values.ravel()
    )
    n = len(terms)
    distance_matrix = pd.DataFrame(np.zeros((n, n)), index=terms, columns=terms)

    for _, row in similarity_pair.iterrows():
        distance_matrix.loc[row[pair1_column_name], row[pair2_column_name]] = row['distance']
        distance_matrix.loc[row[pair2_column_name], row[pair1_column_name]] = row['distance']

    # Diagonal is 0 (distance to self)
    np.fill_diagonal(distance_matrix.values, 0)

    condensed_dist = squareform(distance_matrix.values)

    return terms, distance_matrix, condensed_dist


def build_similarity_graph(
    terms_pair: pd.DataFrame,
    threshold: float = 5.0,
) -> nx.Graph:
    """
    Build an undirected similarity graph from pairwise similarity scores.

    Nodes represent terms, edges connect terms whose similarity meets or
    exceeds the threshold. Edge weights store the raw similarity score.

    Args:
        terms_pair: DataFrame with columns 'term_x', 'term_y', and 'similarity'.
        threshold: Minimum similarity score to create an edge. Defaults to 5.0.

    Returns:
        A networkx Graph with similarity-based edges.
    """
    graph = nx.Graph()

    # Add all unique terms as nodes
    all_terms: Set[str] = (
        set(terms_pair['term_x'].unique()) | set(terms_pair['term_y'].unique())
    )
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
                distance=float(10 - row['similarity']),
            )

    # Print selected edges
    print(f"\nEdges added (threshold >= {threshold}):")
    for u, v, data in graph.edges(data=True):
        print(f"  {u} -- {v} (weight: {data['weight']:.2f})")

    print(f"\nSummary: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    return graph

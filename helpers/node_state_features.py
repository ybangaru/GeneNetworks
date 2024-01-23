"""
Experimental feature for the node identification into "edge", "inner" and "free" states
"""
import numpy as np
import networkx as nx


def calculate_node_state(node, sub_data, threshold_edge):
    # Get the adjacency matrix of the subgraph
    adjacency_matrix = nx.to_numpy_matrix(sub_data)

    # Get the indices of the nodes that are adjacent to the current node
    adjacent_nodes = np.where(adjacency_matrix[node] > 0)[0]

    # Calculate the average feature vector of the adjacent nodes
    average_adjacent_feature_vector = np.mean(sub_data.x[adjacent_nodes], axis=0)

    # Calculate the difference between the feature vector of the current node and the average feature vector of the adjacent nodes
    difference = sub_data.x[node] - average_adjacent_feature_vector

    # If the difference is large, the node is in the "edge" state
    if np.linalg.norm(difference) > threshold_edge:
        return 0  # "edge" state

    # If the node has more than one adjacent node, it's in the "inner" state
    elif len(adjacent_nodes) > 1:
        return 1  # "inner" state

    # If the node has no adjacent nodes or only one, it's in the "free" state
    else:
        return 2  # "free" state

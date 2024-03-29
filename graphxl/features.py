"""
Helper functions for processing node/edge features
"""

import numpy as np
import networkx as nx
from scipy.stats import rankdata
import torch
import torch_geometric as tg

from .graph_build import extract_boundary_features
from .logging_setup import logger


def process_biomarker_expression(
    G,
    node_ind,
    biomarkers=None,
    biomarker_expression_process_method="raw",
    biomarker_expression_lower_bound=-3,
    biomarker_expression_upper_bound=3,
    **kwargs
):
    """Process biomarker expression

    Args:
        G (nx.Graph): full cellular graph of the region
        node_ind (int): target node index
        biomarkers (list): list of biomarkers
        biomarker_expression_process_method (str): process method, one of 'raw', 'linear', 'log', 'rank'
        biomarker_expression_lower_bound (float): lower bound for min-max normalization, used for 'linear' and 'log'
        biomarker_expression_upper_bound (float): upper bound for min-max normalization, used for 'linear' and 'log'

    Returns:
        list: processed biomarker expression values
    """

    bm_exp_dict = G.nodes[node_ind]["biomarker_expression"]
    bm_exp_vec = []
    for bm in biomarkers:
        if bm_exp_dict is None or bm not in bm_exp_dict:
            bm_exp_vec.append(0.0)
        else:
            bm_exp_vec.append(float(bm_exp_dict[bm]))

    bm_exp_vec = np.array(bm_exp_vec)
    lb = biomarker_expression_lower_bound
    ub = biomarker_expression_upper_bound

    if biomarker_expression_process_method == "raw":
        return list(bm_exp_vec)
    elif biomarker_expression_process_method == "linear":
        bm_exp_vec = np.clip(bm_exp_vec, lb, ub)
        bm_exp_vec = (bm_exp_vec - lb) / (ub - lb)
        return list(bm_exp_vec)
    elif biomarker_expression_process_method == "log":
        bm_exp_vec = np.clip(np.log(bm_exp_vec + 1e-9), lb, ub)
        bm_exp_vec = (bm_exp_vec - lb) / (ub - lb)
        return list(bm_exp_vec)
    elif biomarker_expression_process_method == "rank":
        bm_exp_vec = rankdata(bm_exp_vec, method="min")
        num_exp = len(bm_exp_vec)
        bm_exp_vec = (bm_exp_vec - 1) / (num_exp - 1)
        return list(bm_exp_vec)
    else:
        raise ValueError("expression process method %s not recognized" % biomarker_expression_process_method)


def process_neighbor_composition(G, node_ind, cell_type_mapping=None, neighborhood_size=10, **kwargs):
    """Calculate the composition vector of k-nearest neighboring cells

    Args:
        G (nx.Graph): full cellular graph of the region
        node_ind (int): target node index
        cell_type_mapping (dict): mapping of unique cell types to integer indices
        neighborhood_size (int): number of nearest neighbors to consider

    Returns:
        comp_vec (list): composition vector of k-nearest neighboring cells
    """
    center_coord = G.nodes[node_ind]["center_coord"]

    def node_dist(c1, c2):
        return np.linalg.norm(np.array(c1) - np.array(c2), ord=2)

    radius = 1
    neighbors = {}
    while len(neighbors) < 2 * neighborhood_size and radius < 5:
        radius += 1
        ego_g = nx.ego_graph(G, node_ind, radius=radius)
        neighbors = {n: feat_dict["center_coord"] for n, feat_dict in ego_g.nodes.data()}

    closest_neighbors = sorted(neighbors.keys(), key=lambda x: node_dist(center_coord, neighbors[x]))
    closest_neighbors = closest_neighbors[1 : (neighborhood_size + 1)]  # noqa: E203

    comp_vec = np.zeros((len(cell_type_mapping),))
    for n in closest_neighbors:
        cell_type = cell_type_mapping[G.nodes[node_ind]["cell_type"]]
        comp_vec[cell_type] += 1
    comp_vec = list(comp_vec / comp_vec.sum())
    return comp_vec


def process_edge_distance(G, edge_ind, log_distance_lower_bound=2.0, log_distance_upper_bound=5.0, **kwargs):
    """Process edge distance, distance will be log-transformed and min-max normalized

    Default parameters assume distances are usually within the range: 10-100 pixels / 3.7-37 um

    Args:
        G (nx.Graph): full cellular graph of the region
        edge_ind (int): target edge index
        log_distance_lower_bound (float): lower bound for log-transformed distance
        log_distance_upper_bound (float): upper bound for log-transformed distance

    Returns:
        list: list of normalized log-transformed distance
    """
    dist = G.edges[edge_ind]["distance"]
    log_dist = np.log(dist + 1e-5)
    _d = np.clip(
        (log_dist - log_distance_lower_bound) / (log_distance_upper_bound - log_distance_lower_bound),
        0,
        1,
    )
    return [_d]


def process_feature(G, feature_item, node_ind=None, edge_ind=None, **feature_kwargs):
    """Process a single node/edge feature item

    The following feature items are supported, note that some of them require
    keyword arguments in `feature_kwargs`:

    Node features:
        - feature_item: "cell_type"
            (required) "cell_type_mapping"
        - feature_item: "center_coord"
        - feature_item: "biomarker_expression"
            (required) "biomarkers",
            (optional) "biomarker_expression_process_method",
            (optional, if method is "linear" or "log") "biomarker_expression_lower_bound",
            (optional, if method is "linear" or "log") "biomarker_expression_upper_bound"
        - feature_item: "neighborhood_composition"
            (required) "cell_type_mapping",
            (optional) "neighborhood_size"
        - other additional feature items stored in the node attributes
            (see `graph_build.construct_graph_for_region`, argument `cell_features_file`)

    Edge features:
        - feature_item: "edge_type"
        - feature_item: "distance"
            (optional) "log_distance_lower_bound",
            (optional) "log_distance_upper_bound"

    Args:
        G (nx.Graph): full cellular graph of the region
        feature_item (str): feature item
        node_ind (int): target node index (if feature item is node feature)
        edge_ind (tuple): target edge index (if feature item is edge feature)
        feature_kwargs (dict): arguments for processing features

    Returns:
        v (list): feature vector
    """
    # Node features
    if node_ind is not None and edge_ind is None:
        if feature_item == "cell_type":
            # Integer index of the cell type
            assert (
                "cell_type_mapping" in feature_kwargs
            ), "'cell_type_mapping' is required in the kwargs for feature item 'cell_type'"
            v = [feature_kwargs["cell_type_mapping"][G.nodes[node_ind]["cell_type"]]]
            # [feature_kwargs["cell_type_mapping"][G.nodes[node_ind]["cell_type"]]]
            return v
        elif feature_item == "center_coord":
            # Coordinates of the cell centroid
            v = list(G.nodes[node_ind]["center_coord"])
            return v
        elif feature_item == "biomarker_expression":
            # Biomarker expression of the cell
            assert (
                "biomarkers" in feature_kwargs
            ), "'biomarkers' is required in the kwargs for feature item 'biomarker_expression'"
            v = process_biomarker_expression(G, node_ind, **feature_kwargs)
            return v
        elif feature_item == "neighborhood_composition":
            # Composition vector of the k-nearest neighboring cells
            assert (
                "cell_type_mapping" in feature_kwargs
            ), "'cell_type_mapping' is required in the kwargs for feature item 'neighborhood_composition'"
            v = process_neighbor_composition(G, node_ind, **feature_kwargs)
            return v
        elif feature_item == "voronoi_polygon" or feature_item == "boundary_polygon":
            v = extract_boundary_features(G.nodes[node_ind][feature_item])
            return v
        elif feature_item in G.nodes[node_ind]:
            # Additional features specified by users, e.g. "volume" in the example
            v = [G.nodes[node_ind][feature_item]]
            return v
        else:
            raise ValueError(
                "Feature %s not found in the node attributes of graph %s, node %s"
                % (feature_item, G.region_id, str(node_ind))
            )

    # Edge features
    elif edge_ind is not None and node_ind is None:
        if feature_item == "edge_type":
            v = [feature_kwargs["edge_types"][G.edges[edge_ind]["edge_type"]]]
            return v
        elif feature_item == "distance":
            v = process_edge_distance(G, edge_ind, **feature_kwargs)
            return v
        elif feature_item in G.edges[edge_ind]:
            v = [G.edges[edge_ind][feature_item]]
            return v
        else:
            raise ValueError(
                "Feature %s not found in the edge attributes of graph %s, edge %s"
                % (feature_item, G.region_id, str(edge_ind))
            )

    else:
        raise ValueError("One of node_ind or edge_ind should be specified")


def nx_to_tg_graph(
    G,
    node_features=[
        "cell_type",
        "biomarker_expression",
        "center_coord",
        "volume",
    ],
    edge_features=["edge_type", "distance"],
    **feature_kwargs
):
    """Build pyg data objects from nx graphs

    Args:
        G (nx.Graph): full cellular graph of the region
        node_features (list, optional): list of node feature items
        edge_features (list, optional): list of edge feature items
        feature_kwargs (dict): arguments for processing features

    Returns:
        data_list (list): list of pyg data objects
    """
    data_list = []

    # Each connected component of the cellular graph will be a separate pyg data object
    # Usually there should only be one connected component for each cellular graph
    for inds in nx.connected_components(G):
        # Skip small connected components
        if len(inds) < len(G) * 0.1:
            continue
        sub_G = G.subgraph(inds)

        # Relabel nodes to be consecutive integers, note that node indices are
        # not meaningful here, cells are identified by the key "cell_id" in each node
        mapping = {n: i for i, n in enumerate(sorted(sub_G.nodes))}
        sub_G = nx.relabel.relabel_nodes(sub_G, mapping)
        assert np.all(sub_G.nodes == np.arange(len(sub_G)))

        # Append node and edge features to the pyg data object
        data = {"x": [], "edge_attr": [], "edge_index": [], "is_padding": [], "cell_id": []}
        for node_ind in sub_G.nodes:
            feat_val = []
            for key in node_features:
                feat_val.extend(process_feature(sub_G, key, node_ind=node_ind, **feature_kwargs))
            data["x"].append(feat_val)
            data["is_padding"].append(sub_G.nodes[node_ind]["is_padding"])
            data["cell_id"].append(sub_G.nodes[node_ind]["cell_id"])

        for edge_ind in sub_G.edges:
            feat_val = []
            for key in edge_features:
                feat_val.extend(process_feature(sub_G, key, edge_ind=edge_ind, **feature_kwargs))
            data["edge_attr"].append(feat_val)
            data["edge_index"].append(edge_ind)
            data["edge_attr"].append(feat_val)
            data["edge_index"].append(tuple(reversed(edge_ind)))

        for key, item in data.items():
            if key != "cell_id":
                data[key] = torch.tensor(item)
        data["edge_index"] = data["edge_index"].t().long()
        data = tg.data.Data.from_dict(data)
        data.num_nodes = sub_G.number_of_nodes()
        data.region_id = G.region_id
        data_list.append(data)
    return data_list


def get_feature_names(features, cell_type_mapping=None, biomarkers=None):
    """Helper fn for getting a list of feature names from a list of feature items

    Args:
        features (list): list of feature items
        cell_type_mapping (dict): mapping of unique cell types to integer indices
        biomarkers (list): list of biomarkers

    Returns:
        feat_names(list): list of feature names
    """
    feat_names = []
    for feat in features:
        if feat in ["distance", "cell_type", "edge_type"]:
            # feature "cell_type", "edge_type" will be a single integer indice
            # feature "distance" will be a single float value
            feat_names.append(feat)
        elif feat == "center_coord":
            # feature "center_coord" will be a tuple of two float values
            feat_names.extend(["center_coord-x", "center_coord-y"])
        elif feat == "biomarker_expression":
            # feature "biomarker_expression" will contain a list of biomarker expression values
            feat_names.extend(["biomarker_expression-%s" % bm for bm in biomarkers])
        elif feat == "neighborhood_composition":
            # feature "neighborhood_composition" will contain a composition vector of the immediate neighbors
            # The vector will have the same length as the number of unique cell types
            feat_names.extend(
                [
                    "neighborhood_composition-%s" % ct
                    for ct in sorted(cell_type_mapping.keys(), key=lambda x: cell_type_mapping[x])
                ]
            )
        elif feat == "voronoi_polygon":
            # TODO: feature "voronoi_polygon" will contain a list of hu moments /
            # handle variations in feature length
            feat_names.extend(["voronoi_polygon-%s" % i for i in range(1, 8)])
        elif feat == "boundary_polygon":
            feat_names.extend(["boundary_polygon-%s" % i for i in range(1, 8)])
        else:
            logger.info("Using additional feature: %s" % feat)
            feat_names.append(feat)
    return feat_names

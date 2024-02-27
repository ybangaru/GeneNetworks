# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 14:53:12 2021

@author: zhenq
"""
import json
import os
import numpy as np
import pandas as pd
import pickle


def get_cell_type_metadata(nx_graph_files):
    """
    Read cell type metadata info from clustering/networkx segment config directory,
    defaults to mlflow run directory at init config in pipeline_build_nx_graphs.py

    Args:
        nx_graph_files (list/str): path/list of paths to cellular graph files (gpickle)

    Returns:
        cell_type_hashmap (dict): mapping of unique cell types to integer indices
        cell_type_proportion (dict): mapping of unique cell types to their frequency
    """
    if isinstance(nx_graph_files, str):
        nx_graph_files = [nx_graph_files]

    directory_path = os.path.dirname(os.path.dirname(nx_graph_files[0]))

    cell_type_mapping_path = os.path.join(directory_path, "cell_type_mapping.json")
    cell_type_freq_path = os.path.join(directory_path, "cell_type_freq.json")
    cell_type_color_path = os.path.join(directory_path, "color_dict.json")

    try:
        cell_type_hashmap = json.load(open(cell_type_mapping_path))
        cell_type_proportion = json.load(open(cell_type_freq_path))

    except FileNotFoundError:
        # cell_type_mapping = {}
        cell_metadata_loc = os.path.join(directory_path, "cell_type_to_id.json")
        if os.path.exists(cell_metadata_loc):
            with open(cell_metadata_loc, "r") as f:
                cell_metadata = json.load(f)
        else:
            raise FileNotFoundError(f"cell_type_to_id.json not found in {directory_path}")

        # count values of each cell type
        sequential_mapping = {k: len(v) for k, v in cell_metadata.items()}
        # sort mapping dictionary by frequency of cell types in descending order
        sequential_mapping = dict(sorted(sequential_mapping.items(), key=lambda item: item[1], reverse=True))
        # create a dictionary with the cell type and an index for each cell type starting from 0 for the most frequent
        cell_type_hashmap = {key: index for index, key in enumerate(sequential_mapping.keys())}
        # create a dictionary with the cell type and the proportion of cells of that type
        total_cells = sum(sequential_mapping.values())
        cell_type_proportion = {k: v / total_cells for k, v in sequential_mapping.items()}

        with open(cell_type_mapping_path, "w") as json_file:
            json.dump(cell_type_hashmap, json_file, indent=2)
        with open(cell_type_freq_path, "w") as json_file:
            json.dump(cell_type_proportion, json_file, indent=2)

    cell_type_color = json.load(open(cell_type_color_path))
    return cell_type_hashmap, cell_type_proportion, cell_type_color


def get_biomarker_metadata(nx_graph_files):
    """
    Read cell biomarker metadata info from clustering/networkx segment config directory,
    defaults to mlflow run directory at init config in pipeline_build_nx_graphs.py

    Args:
        nx_graph_files (list/str): path/list of paths to cellular graph files (gpickle)

    Returns:
        shared_bms (list): list of biomarkers shared by all cells (intersect)
        all_bms (list): list of all biomarkers (union)
    """

    if isinstance(nx_graph_files, str):
        nx_graph_files = [nx_graph_files]

    directory_path = os.path.dirname(os.path.dirname(os.path.dirname(nx_graph_files[0])))
    shared_bms_loc = os.path.join(directory_path, "biomarkers_list_shared.csv")
    all_bms_loc = os.path.join(directory_path, "biomarkers_list_all.csv")

    try:
        shared_bms = pd.read_csv(shared_bms_loc, header=None).values.flatten().tolist()
        all_bms = pd.read_csv(all_bms_loc, header=None).values.flatten().tolist()

    except FileNotFoundError:
        all_bms = set()
        shared_bms = None
        for g_f in nx_graph_files:
            G = pickle.load(open(g_f, "rb"))
            for n in G.nodes:
                bms = sorted(G.nodes[n]["biomarker_expression"].keys())
                for bm in bms:
                    all_bms.add(bm)
                valid_bms = [
                    bm for bm in bms if G.nodes[n]["biomarker_expression"][bm] == G.nodes[n]["biomarker_expression"][bm]
                ]
                shared_bms = set(valid_bms) if shared_bms is None else shared_bms & set(valid_bms)
        shared_bms = sorted(shared_bms)
        all_bms = sorted(all_bms)

        # save csv files for shared and all biomarkers
        pd.DataFrame(shared_bms).to_csv(shared_bms_loc, header=None, index=False)
        pd.DataFrame(all_bms).to_csv(all_bms_loc, header=None, index=False)

    return shared_bms, all_bms


def get_graph_splits(dataset, split="random", cv_k=5, seed=None, fold_mapping=None):
    """Define train/valid split

    Args:
        dataset (CellularGraphDataset): dataset to split
        split (str): split method, one of 'random', 'fold'
        cv_k (int): number of splits for random split
        seed (int): random seed
        fold_mapping (dict): mapping of region ids to folds,
            fold could be coverslip, patient, etc.

    Returns:
        split_inds (list): fold indices for each region in the dataset
    """
    splits = {}
    region_ids = set([dataset.get_full(i).region_id for i in range(dataset.N)])
    _region_ids = sorted(region_ids)
    if split == "random":
        if seed is not None:
            np.random.seed(seed)
        if fold_mapping is None:
            fold_mapping = {region_id: region_id for region_id in _region_ids}
        # `_ids` could be sample ids / patient ids / certain properties
        _folds = sorted(set(list(fold_mapping.values())))
        np.random.shuffle(_folds)
        cv_shard_size = len(_folds) / cv_k
        for i, region_id in enumerate(_region_ids):
            splits[region_id] = _folds.index(fold_mapping[region_id]) // cv_shard_size
    elif split == "fold":
        # Split into folds, one fold per group
        assert fold_mapping is not None
        _folds = sorted(set(list(fold_mapping.values())))
        for i, region_id in enumerate(_region_ids):
            splits[region_id] = _folds.index(fold_mapping[region_id])
    else:
        raise ValueError("split mode not recognized")

    split_inds = []
    for i in range(dataset.N):
        split_inds.append(splits[dataset.get_full(i).region_id])
    return split_inds

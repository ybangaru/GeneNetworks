# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 14:53:12 2021

@author: zhenq
"""
import json
import os
import numpy as np
import pickle
from helpers import ANNOTATION_DICT

EDGE_TYPES = {
    "neighbor": 0,
    "distant": 1,
    "self": 2,
}

# # Metadata for the example dataset
# BIOMARKERS_UPMC = [
#     "CD11b", "CD14", "CD15", "CD163", "CD20", "CD21", "CD31", "CD34", "CD3e",
#     "CD4", "CD45", "CD45RA", "CD45RO", "CD68", "CD8", "CollagenIV", "HLA-DR",
#     "Ki67", "PanCK", "Podoplanin", "Vimentin", "aSMA",
# ]

# CELL_TYPE_MAPPING_UPMC = {
#     'APC': 0,
#     'B cell': 1,
#     'CD4 T cell': 2,
#     'CD8 T cell': 3,
#     'Granulocyte': 4,
#     'Lymph vessel': 5,
#     'Macrophage': 6,
#     'Naive immune cell': 7,
#     'Stromal / Fibroblast': 8,
#     'Tumor': 9,
#     'Tumor (CD15+)': 10,
#     'Tumor (CD20+)': 11,
#     'Tumor (CD21+)': 12,
#     'Tumor (Ki67+)': 13,
#     'Tumor (Podo+)': 14,
#     'Vessel': 15,
#     'Unassigned': 16,
# }

# CELL_TYPE_FREQ_UPMC = {
#     'APC': 0.038220815854819415,
#     'B cell': 0.06635091324932002,
#     'CD4 T cell': 0.09489001514723677,
#     'CD8 T cell': 0.07824503590797544,
#     'Granulocyte': 0.026886102677111563,
#     'Lymph vessel': 0.006429085023448621,
#     'Macrophage': 0.10251942892685563,
#     'Naive immune cell': 0.033537398925429215,
#     'Stromal / Fibroblast': 0.07692583870182068,
#     'Tumor': 0.10921293560435145,
#     'Tumor (CD15+)': 0.06106975782857908,
#     'Tumor (CD20+)': 0.02098925720318548,
#     'Tumor (CD21+)': 0.053892044158901406,
#     'Tumor (Ki67+)': 0.13373768013421947,
#     'Tumor (Podo+)': 0.06276108605978743,
#     'Vessel': 0.034332604596958326,
#     'Unassigned': 0.001,
# }


def get_cell_type_metadata(nx_graph_files):
    """Find all unique cell types from a list of cellular graphs

    Args:
        nx_graph_files (list/str): path/list of paths to cellular graph files (gpickle)

    Returns:
        cell_type_mapping (dict): mapping of unique cell types to integer indices
        cell_type_freq (dict): mapping of unique cell types to their frequency
    """
    if isinstance(nx_graph_files, str):
        nx_graph_files = [nx_graph_files]

    directory_path = os.path.dirname(nx_graph_files[0])
    cell_type_mapping_path = os.path.join(directory_path, 'cell_type_mapping.json')
    cell_type_freq_path = os.path.join(directory_path, 'cell_type_freq.json')
    cell_annotation_frequencies_path = os.path.join(directory_path,'cell_annotation_freq.json')

    try:
        cell_type_mapping = json.load(open(cell_type_mapping_path))
        cell_type_freq = json.load(open(cell_type_freq_path))
        sorted_cell_annotation_freq = json.load(open(cell_annotation_frequencies_path))
    
    except FileNotFoundError:

        cell_type_mapping = {}
        cell_annotation_frequencies = {}
        for g_f in nx_graph_files:
            G = pickle.load(open(g_f, 'rb'))

            assert 'cell_type' in G.nodes[0]
            for n in G.nodes:
                ct = G.nodes[n]['cell_type']
                if ct not in cell_type_mapping:
                    cell_type_mapping[ct] = 0
                cell_type_mapping[ct] += 1

                ann_ct = ANNOTATION_DICT[ct]
                cur_ann_ct_count = cell_annotation_frequencies.get(ann_ct, 0)
                cell_annotation_frequencies[ann_ct] = cur_ann_ct_count + 1

        # TODO: Handle when keys are not int type
        unique_cell_types = sorted(cell_type_mapping.keys(), key=lambda x: int(x))
        unique_cell_types_ct = [cell_type_mapping[ct] for ct in unique_cell_types]
        unique_cell_type_freq = [count / sum(unique_cell_types_ct) for count in unique_cell_types_ct]
        cell_type_mapping = {ct: i for i, ct in enumerate(unique_cell_types)}
        cell_type_freq = dict(zip(unique_cell_types, unique_cell_type_freq))

        cell_annotation_frequencies = {item: cell_annotation_frequencies[item]/sum(cell_annotation_frequencies.values()) for item in cell_annotation_frequencies}
        sorted_cell_annotation_freq = dict(sorted(cell_annotation_frequencies.items(), key=lambda item: item[1], reverse=True))

        with open(cell_type_mapping_path, 'w') as json_file:
            json.dump(cell_type_mapping, json_file, indent=2)
        with open(cell_type_freq_path, 'w') as json_file:
            json.dump(cell_type_freq, json_file, indent=2)
        with open(cell_annotation_frequencies_path, 'w') as json_file:
            json.dump(sorted_cell_annotation_freq, json_file, indent=2)

    return cell_type_mapping, cell_type_freq, sorted_cell_annotation_freq


def get_biomarker_metadata(nx_graph_files, file_loc=None):
    """Load all biomarkers from a list of cellular graphs

    Args:
        nx_graph_files (list/str): path/list of paths to cellular graph files (gpickle)

    Returns:
        shared_bms (list): list of biomarkers shared by all cells (intersect)
        all_bms (list): list of all biomarkers (union)
    """
    if isinstance(nx_graph_files, str):
        nx_graph_files = [nx_graph_files]
    all_bms = set()
    shared_bms = None
    for g_f in nx_graph_files:
        G = pickle.load(open(g_f, 'rb'))
        for n in G.nodes:
            bms = sorted(G.nodes[n]["biomarker_expression"].keys())
            for bm in bms:
                all_bms.add(bm)
            valid_bms = [
                bm for bm in bms if G.nodes[n]["biomarker_expression"][bm] == G.nodes[n]["biomarker_expression"][bm]]
            shared_bms = set(valid_bms) if shared_bms is None else shared_bms & set(valid_bms)
    shared_bms = sorted(shared_bms)
    all_bms = sorted(all_bms)
    return shared_bms, all_bms


def get_graph_splits(dataset,
                     split='random',
                     cv_k=5,
                     seed=None,
                     fold_mapping=None):
    """ Define train/valid split

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
    if split == 'random':
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
    elif split == 'fold':
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

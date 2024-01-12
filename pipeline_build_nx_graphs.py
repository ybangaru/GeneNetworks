
"""
This script is typically used after leiden clustering to create networkx graphs for each slice segmented into multiple regions
and save them in the form of pickle files. These pickle files are then used for training the graph neural network.
"""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from helpers import logger, NO_JOBS
from joblib import Parallel, delayed
# import torch.nn as nn

from helpers import spatialPipeline, ANNOTATION_DICT, plotly_spatial_scatter_edges, plotly_spatial_scatter_categorical
from helpers import build_graph_from_cell_coords, assign_attributes, calcualte_voronoi_from_coords, plot_voronoi_polygons, plot_graph
from helpers import BoundaryDataLoader, SLICE_PIXELS_EDGE_CUTOFF, logger
import matplotlib.pyplot as plt


def main():
    # y_data_filter_name = "Liver12Slice12"
    # y_resolution = 0.7
    # ann_data = read_run_result_ann_data(y_data_filter_name, y_resolution)

    # ann_data.obs = ann_data.obs.rename(columns={"sample": "liver_slice"})
    # ann_data.obs['CELL_TYPE'] = ann_data.obs['leiden_res'].tolist() 

    all_liver_samples = ["Liver1Slice1", "Liver1Slice2", "Liver2Slice1", "Liver2Slice2"]
    data_filter_name = "Liver12Slice12"
    leiden_resolution = 0.7

    data_options = {
        "dir_loc" : f"/data/qd452774/spatial_transcriptomics/",
        "names_list" : all_liver_samples,
        "mlflow_config" : {
            "id"  : None,
            "data_filter_name" : data_filter_name,
            "leiden_resolution" : leiden_resolution,
        },
    }
    pipeline_instance = spatialPipeline(options=data_options)
    pipeline_instance.data.obs['cell_type'] = pipeline_instance.data.obs['leiden_res'].map(ANNOTATION_DICT)


    NODE_FEATURES = ["cell_type", "center_coord", "biomarker_expression", "volume", "boundary_polygon"]
    EDGE_FEATURES = ["distance", "edge_type"]

    segment_config = {
        "segments_per_dimension" : 20,
        "sample_name" : "Liver1Slice1",
        "region_id" : (14, 17),
        "padding" : {
            "value" : SLICE_PIXELS_EDGE_CUTOFF["Liver1Slice1"]*40,
            "units" : "pixels",
        },
    }

    pretransform_networkx_configs = [
        {
            "boundary_type" : "voronoi",
            "edge_config" : {
                "type" : "Delaunay",
            },
            "neighbor_edge_cutoff" : SLICE_PIXELS_EDGE_CUTOFF["Liver1Slice1"],
        },
        {
            "boundary_type" : "given",
            "edge_config" : {
                "type" : "Delaunay",
            },
            "neighbor_edge_cutoff" : SLICE_PIXELS_EDGE_CUTOFF["Liver1Slice1"],
        },        
        {
            "boundary_type" : "given",
            "edge_config" : {
                "type" : "R3Index",
                "bound_type" : "rectangle",
                "threshold_distance" : 0,
            },    
            "neighbor_edge_cutoff" : SLICE_PIXELS_EDGE_CUTOFF["Liver1Slice1"],
        }, 
        {
            "boundary_type" : "given",
            "edge_config" : {
                "type" : "R3Index",
                "bound_type" : "rotated_rectangle",
                "threshold_distance" : 0,
            },    
            "neighbor_edge_cutoff" : SLICE_PIXELS_EDGE_CUTOFF["Liver1Slice1"],
        },                

    ]

    network_features_config = {
        "node_features" : NODE_FEATURES,
        "edge_features" : EDGE_FEATURES,
    }

    for pretransform_networkx_config in pretransform_networkx_configs:
        curr_nx_graph = pipeline_instance.build_networkx_for_region(segment_config, pretransform_networkx_config, network_features_config)
        pipeline_instance.build_networkx_plots(curr_nx_graph, pretransform_networkx_config, save_=True)


    dataset_root = f"/data/qd452774/spatial_transcriptomics/data/{y_data_filter_name}_leiden_res_{y_resolution}"
    nx_graph_root = os.path.join(dataset_root, f"graph_{y_data_filter_name}_{y_resolution}")
    all_graph_names = sorted([f for f in os.listdir(nx_graph_root) if f.endswith('_voronoi.gpkl')])
    nx_graph_files = [os.path.join(nx_graph_root, f) for f in all_graph_names]

    from helpers import get_biomarker_metadata
    biomarkers_list, _ = get_biomarker_metadata(nx_graph_files)
    save_biomarkers_list = os.path.join(dataset_root, "biomarkers_list.csv")
    biomarkers_df = pd.DataFrame(biomarkers_list)
    biomarkers_df.to_csv(save_biomarkers_list, index=False, header=False)
    logger.info(f"Saved biomarkers list to {save_biomarkers_list}")

if __name__=="__main__":
    main()
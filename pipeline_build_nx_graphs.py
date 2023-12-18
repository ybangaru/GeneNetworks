
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

# import torch.nn as nn
import spacegm
import pickle

from helpers import logger, NO_JOBS
from joblib import Parallel, delayed

# from spacegm.graph_build import load_cell_coords, load_cell_types, load_cell_biomarker_expression, load_cell_features

# space_gm_data_loc = "/data/qd452774/space-gm"
# region_id = f"{space_gm_data_loc}/UPMC_c001_v001_r001_reg001"
# cell_coords_file = f'{space_gm_data_loc}/data/voronoi/UPMC_c001_v001_r001_reg001.cell_data.csv'
# cell_types_file = f'{space_gm_data_loc}/data/voronoi/UPMC_c001_v001_r001_reg001.cell_types.csv'
# cell_biomarker_expression_file = f'{space_gm_data_loc}/data/voronoi/UPMC_c001_v001_r001_reg001.expression.csv'
# cell_features_file = f'{space_gm_data_loc}/data/voronoi/UPMC_c001_v001_r001_reg001.cell_features.csv'

# graph_label_file = f'{space_gm_data_loc}/data/metadata/full_graph_labels.csv'

# print("\nInputs for region %s:" % region_id)
# print("\nCell coordinates")
# # display(load_cell_coords(cell_coords_file))
# print("\nCell types")
# # display(load_cell_types(cell_types_file))
# print("\nCell biomarker expression")
# # display(load_cell_biomarker_expression(cell_biomarker_expression_file))
# print("\nAdditional cell features")
# # display(load_cell_features(cell_features_file))

# print("\nGraph-level tasks")
# # display(pd.read_csv(graph_label_file, index_col=0))


# # <br/><br/><br/><br/>
# # 
# # A networkx graph can be constructed using the inputs above

# # In[3]:

from cluster_comparison import read_run_result_ann_data
from spacegm import build_graph_from_cell_coords, assign_attributes, calcualte_voronoi_from_coords, plot_voronoi_polygons, plot_graph
from helpers import BoundaryDataLoader, SLICE_PIXELS_EDGE_CUTOFF, logger
import matplotlib.pyplot as plt


def build_graph_for_region(slice_info_temp):

    slice_region_id = f"{slice_info_temp[0]}-{slice_info_temp[1]}-{slice_info_temp[2]}"
    liver_slice_name = slice_info_temp[3]
    leiden_resolution = slice_info_temp[4]
    logger.info(f"Starting to build graph for slice - {slice_region_id}")
    try:
        boundary_data_instance = BoundaryDataLoader(slice_info_temp[0], slice_info_temp[1], slice_info_temp[2])
        # slice_ann_data = 
        slice_boundaries = boundary_data_instance.filter_by_centroid_coordinates(ann_data)
        slice_mask = np.zeros(ann_data.obs.shape[0], dtype=bool)
        slice_mask[
            np.where(ann_data.obs.index.isin(slice_boundaries.keys()))
        ] = True

        slice_spatial_data = ann_data.obsm["spatial"][slice_mask]

        slice_cell_data = pd.DataFrame(
            slice_spatial_data,
            columns=["X", "Y"],
            index=ann_data.obs.index[slice_mask],
        ).reset_index()
        slice_cell_data = slice_cell_data.rename(columns={"index": "CELL_ID"})
        slice_cell_data["CELL_TYPE"] = ann_data.obs['CELL_TYPE'][slice_mask].to_list()
        slice_cell_data['SIZE'] = ann_data.obs['volume'][slice_mask].to_list()
        bm_columns = [f"BM-{bm_temp}" for bm_temp in ann_data.var.index.to_list()]
        slice_cell_data[bm_columns] = pd.DataFrame(ann_data[slice_cell_data['CELL_ID'], :].X.toarray())

        slice_cell_boundaries_given = [
            slice_boundaries[item][0] for item in slice_cell_data["CELL_ID"].values
        ]

        neighbor_edge_cutoff = SLICE_PIXELS_EDGE_CUTOFF[slice_info_temp[0]]


        voronoi_polygons_calculated = calcualte_voronoi_from_coords(
            slice_spatial_data[:, 0], slice_spatial_data[:, 1]
        )
        G_voronoi_boundary_delaunay_edges, node_to_cell_mapping_voronoi_boundary_delaunay_edges = build_graph_from_cell_coords(
            slice_cell_data, voronoi_polygons_calculated, edge_logic='Delaunay'
        )
        G_voronoi_boundary_delaunay_edges = assign_attributes(
            G_voronoi_boundary_delaunay_edges, slice_cell_data, node_to_cell_mapping_voronoi_boundary_delaunay_edges, neighbor_edge_cutoff
        )
        G_voronoi_boundary_delaunay_edges.region_id = slice_region_id

        G_given_boundary_delaunay_edges, node_to_cell_mapping_given_boundary_delaunay_edges = build_graph_from_cell_coords(
            slice_cell_data, slice_cell_boundaries_given, edge_logic='Delaunay'
        )
        G_given_boundary_delaunay_edges = assign_attributes(G_given_boundary_delaunay_edges, slice_cell_data, node_to_cell_mapping_given_boundary_delaunay_edges, neighbor_edge_cutoff)
        G_given_boundary_delaunay_edges.region_id = slice_region_id

        G_given_boundary_r3index_edges, node_to_cell_mapping_given_boundary_r3index_edges = build_graph_from_cell_coords(
            slice_cell_data, slice_cell_boundaries_given, edge_logic='R3Index'
        )
        G_given_boundary_r3index_edges = assign_attributes(G_given_boundary_r3index_edges, slice_cell_data, node_to_cell_mapping_given_boundary_r3index_edges, neighbor_edge_cutoff)
        G_given_boundary_r3index_edges.region_id = slice_region_id

        dataset_root = f"/data/qd452774/spatial_transcriptomics/data/{liver_slice_name}_leiden_res_{leiden_resolution}"

        nx_graph_root = os.path.join(dataset_root, f"graph_{liver_slice_name}_{leiden_resolution}")
        fig_save_root = os.path.join(dataset_root, f"fig_{liver_slice_name}_{leiden_resolution}")
        model_save_root = os.path.join(dataset_root, f'model_{liver_slice_name}_{leiden_resolution}')

        os.makedirs(nx_graph_root, exist_ok=True)
        os.makedirs(fig_save_root, exist_ok=True)
        os.makedirs(model_save_root, exist_ok=True)

        voronoi_polygon_img_output = os.path.join(fig_save_root, "%s_voronoi.png" % slice_region_id)
        given_boundary_delayney_edges_img_output = os.path.join(fig_save_root, "%s_actual_bound_del_edge.png" % slice_region_id)
        given_boundary_r3index_edges_img_output = os.path.join(fig_save_root, "%s_actual_bound_r3index_edge.png" % slice_region_id)

        graph_img_output = os.path.join(fig_save_root, "%s_voronoi.png" % slice_region_id)
        gb_de_graph_img_output = os.path.join(fig_save_root, "%s_actual_bound_del_edge.png" % slice_region_id)
        gb_re_graph_img_output = os.path.join(fig_save_root, "%s_actual_bound_r3index_edge.png" % slice_region_id)


        graph_output = os.path.join(nx_graph_root, "%s_voronoi.gpkl" % slice_region_id)
        gb_de_graph_output = os.path.join(nx_graph_root, "%s_actual_bound_del_edge.gpkl" % slice_region_id)
        gb_re_graph_output = os.path.join(nx_graph_root, "%s_actual_bound_r3index_edge.gpkl" % slice_region_id)

        figsize = 10

        # Visualization of cellular graph
        if voronoi_polygon_img_output is not None:
            plt.clf()
            plt.figure(figsize=(figsize, figsize))
            plot_voronoi_polygons(G_voronoi_boundary_delaunay_edges)
            plt.axis('scaled')
            plt.savefig(voronoi_polygon_img_output, dpi=300, bbox_inches='tight')
            plt.close()
        if graph_img_output is not None:
            plt.clf()
            plt.figure(figsize=(figsize, figsize))
            plot_graph(G_voronoi_boundary_delaunay_edges)
            plt.axis('scaled')
            plt.savefig(graph_img_output, dpi=300, bbox_inches='tight')
            plt.close()
        if graph_output is not None:
            with open(graph_output, 'wb') as f:
                pickle.dump(G_voronoi_boundary_delaunay_edges, f)

        if given_boundary_delayney_edges_img_output is not None:
            plt.clf()
            plt.figure(figsize=(figsize, figsize))
            plot_graph(G_given_boundary_delaunay_edges)
            plt.axis('scaled')
            plt.savefig(given_boundary_delayney_edges_img_output, dpi=300, bbox_inches='tight')
            plt.close()
        if gb_de_graph_img_output is not None:
            plt.clf()
            plt.figure(figsize=(figsize, figsize))
            plot_graph(G_given_boundary_delaunay_edges)
            plt.axis('scaled')
            plt.savefig(gb_de_graph_img_output, dpi=300, bbox_inches='tight')
            plt.close()
        if gb_de_graph_output is not None:
            with open(gb_de_graph_output, 'wb') as f:
                pickle.dump(G_given_boundary_delaunay_edges, f)

        if given_boundary_r3index_edges_img_output is not None:
            plt.clf()
            plt.figure(figsize=(figsize, figsize))
            plot_graph(G_given_boundary_r3index_edges)
            plt.axis('scaled')
            plt.savefig(given_boundary_r3index_edges_img_output, dpi=300, bbox_inches='tight')
            plt.close()
        if gb_re_graph_img_output is not None:
            plt.clf()
            plt.figure(figsize=(figsize, figsize))
            plot_graph(G_given_boundary_r3index_edges)
            plt.axis('scaled')
            plt.savefig(gb_re_graph_img_output, dpi=300, bbox_inches='tight')
            plt.close()
        if gb_re_graph_output is not None:
            with open(gb_re_graph_output, 'wb') as f:
                pickle.dump(G_given_boundary_r3index_edges, f)

        logger.info(f"Finished building graph for slice - {slice_region_id}")
    except Exception as e:
        logger.error(f"Error for slice - {slice_region_id}")
        logger.error(e)


def main():
    y_data_filter_name = "Liver12Slice12"
    y_resolution = 0.7
    ann_data = read_run_result_ann_data(y_data_filter_name, y_resolution)

    ann_data.obs = ann_data.obs.rename(columns={"sample": "liver_slice"})
    ann_data.obs['CELL_TYPE'] = ann_data.obs['leiden_res'].tolist() 

    all_slices_names = ann_data.obs['liver_slice'].unique().tolist()

    all_slices_names_grid = []
    for item_slice in all_slices_names:
        for x_index in list(range(20)):
            for y_index in list(range(20)):
                all_slices_names_grid.append((item_slice, x_index, y_index, y_data_filter_name, y_resolution))
    # print(all_slices_names_grid[-1])

    logger.info("Starting to build graphs")

    # slice_info_temp = all_slices_names_grid[368]

    def run_parallel(all_combinations):
        Parallel(n_jobs=NO_JOBS)(delayed(build_graph_for_region)(combination) for combination in all_combinations)
 
    # '/data/qd452774/spatial_transcriptomics/data/example_dataset/graph/Liver1Slice1-14-17.gpkl'
    # for slice_info_temp in all_slices_names_grid:
    # test_case = all_slices_names_grid[368]
    # build_graph_for_region(test_case)


    run_parallel(all_slices_names_grid)

    # nx_graph_root = os.path.join(dataset_root, f"graph_{liver_slice_name}_{leiden_resolution}")
    # fig_save_root = os.path.join(dataset_root, f"fig_{liver_slice_name}_{leiden_resolution}")
    # model_save_root = os.path.join(dataset_root, f'model_{liver_slice_name}_{leiden_resolution}')

    dataset_root = f"/data/qd452774/spatial_transcriptomics/data/{y_data_filter_name}_leiden_res_{y_resolution}"
    nx_graph_root = os.path.join(dataset_root, f"graph_{y_data_filter_name}_{y_resolution}")
    all_graph_names = sorted([f for f in os.listdir(nx_graph_root) if f.endswith('_voronoi.gpkl')])
    nx_graph_files = [os.path.join(nx_graph_root, f) for f in all_graph_names]

    from spacegm import get_biomarker_metadata
    biomarkers_list, _ = get_biomarker_metadata(nx_graph_files)
    save_biomarkers_list = os.path.join(dataset_root, "biomarkers_list.csv")
    biomarkers_df = pd.DataFrame(biomarkers_list)
    biomarkers_df.to_csv(save_biomarkers_list, index=False, header=False)
    logger.info(f"Saved biomarkers list to {save_biomarkers_list}")

if __name__=="__main__":
    main()
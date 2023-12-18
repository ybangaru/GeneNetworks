import os
import ast
import numpy as np
import pandas as pd
import mlflow
from helpers import logger
from joblib import Parallel, delayed
import geopandas as gpd
from shapely.geometry import Polygon
from rtree import index
from spacegm import (
    calcualte_voronoi_from_coords,
    build_graph_from_voronoi_polygons,
    build_voronoi_polygon_to_cell_mapping,
    assign_attributes,
    build_graph_from_cell_coords,
)
from helpers import (
    plotly_spatial_scatter_categorical,
    plotly_spatial_scatter_numerical,
    plotly_spatial_scatter_pca,
    plotly_spatial_scatter_umap,
    spatialPipeline,
    spatialPreProcessor,
    BoundaryDataLoader,
    VisualizePipeline,
    plotly_spatial_scatter_edges,
    logger
)
import plotly.subplots as sp
import plotly.io as pio


# filters_test = {
#     "data_filter_name": "Liver1Slice1",
#     "names_list": ["Liver1Slice1"],
#     "state": "steady-state",
#     "filter_nans": False,
# }

# filters_test = {
#     "data_filter_name": "Liver1Slice12",
#     "names_list": ["Liver1Slice1", "Liver1Slice2"],
#     "state": "steady-state",
#     "filter_nans": False,
# }

# data_options = {
#     "dir_loc": os.path.realpath(os.path.join(os.getcwd())),
#     "names_list": filters_test["names_list"],
# }

# pp_options = {
#     "filter_cells": dict(min_genes=25),
#     "filter_genes": dict(min_cells=10),
#     "find_mt": False,
#     "find_rp": False,
#     "var_names_make_unique": True,
#     "qc_metrics": dict(
#         percent_top=None, log1p=False, inplace=True  # qc_vars=["mt", "rp"],
#     ),
#     "normalize_totals": dict(target_sum=5_000),
#     "log1p_transformation": dict(),
#     "scale": dict(zero_center=False),
# }

# pca_options = dict(random_state=0, n_comps=220)
# ng_options = dict(n_neighbors=50, random_state=0)
# clustering_options = {
#     "umap": dict(n_components=2),
#     "louvain": dict(flavor="igraph"),
#     "leiden": dict(resolution=0.5),
# }


# pipeline_instance = spatialPipeline(options=data_options)
# preprocessor_instance = spatialPreProcessor(options=pp_options)
# pipeline_instance.preprocess_data(preprocessor_instance)
# pipeline_instance.perform_pca(pca_options)
# pipeline_instance.compute_neighborhood_graph(ng_options)
# pipeline_instance.perform_clustering(clustering_options)

def run_pipeline_visualizations(combination, pipeline_instance):    
    try:
        boundary_data_instance = BoundaryDataLoader(combination[0], combination[1], combination[2])
        test_boundaries = pipeline_instance.filter_by_centroid_coordinates(boundary_data_instance)
        mask = np.zeros(pipeline_instance.data.obs.shape[0], dtype=bool)
        mask[
            np.where(pipeline_instance.data.obs.index.isin(test_boundaries.keys()))
        ] = True

        filtered_spatial_data = pipeline_instance.data.obsm["spatial"][mask]

        cell_data = pd.DataFrame(
            filtered_spatial_data,
            columns=["X", "Y"],
            index=pipeline_instance.data.obs.index[mask],
        ).reset_index()
        cell_data = cell_data.rename(columns={"index": "CELL_ID"})


        cell_boundaries_given = [
            test_boundaries[item][0] for item in cell_data["CELL_ID"].values
        ]

        G_given_boundary_delaunay_edges, node_to_cell_mapping_given_boundary_delaunay_edges = build_graph_from_cell_coords(
            cell_data, cell_boundaries_given, edge_logic='Delaunay'
        )
        G_given_boundary_delaunay_edges = assign_attributes(G_given_boundary_delaunay_edges, cell_data, node_to_cell_mapping_given_boundary_delaunay_edges)


        voronoi_polygons_calculated = calcualte_voronoi_from_coords(
            filtered_spatial_data[:, 0], filtered_spatial_data[:, 1]
        )
        G_voronoi_boundary_delaunay_edges, node_to_cell_mapping_voronoi_boundary_delaunay_edges = build_graph_from_cell_coords(
            cell_data, voronoi_polygons_calculated, edge_logic='Delaunay'
        )
        G_voronoi_boundary_delaunay_edges = assign_attributes(
            G_voronoi_boundary_delaunay_edges, cell_data, node_to_cell_mapping_voronoi_boundary_delaunay_edges
        )

        G_given_boundary_r3index_edges, node_to_cell_mapping_given_boundary_r3index_edges = build_graph_from_cell_coords(
            cell_data, cell_boundaries_given, edge_logic='R3Index'
        )
        G_given_boundary_r3index_edges = assign_attributes(G_given_boundary_r3index_edges, cell_data, node_to_cell_mapping_given_boundary_r3index_edges)

        given_boundaries = {G_given_boundary_delaunay_edges.nodes[n]['cell_id'] : G_given_boundary_delaunay_edges.nodes[n]["voronoi_polygon"] for n in G_given_boundary_delaunay_edges.nodes}
        voronoi_boundaries = {G_voronoi_boundary_delaunay_edges.nodes[n]['cell_id'] : G_voronoi_boundary_delaunay_edges.nodes[n]["voronoi_polygon"] for n in G_voronoi_boundary_delaunay_edges.nodes}
        rectangle_boundaries = {G_given_boundary_r3index_edges.nodes[n]['cell_id'] : G_given_boundary_r3index_edges.nodes[n]["voronoi_polygon"] for n in G_given_boundary_r3index_edges.nodes}


        # categorical_columns = ["louvain", "leiden"]
        categorical_columns = ["leiden_res"]

        for cat_column in categorical_columns:

            given_bound_fig = plotly_spatial_scatter_categorical(given_boundaries, pipeline_instance.data.obs[cat_column])
            given_bound_fig.write_html(file=f'assets/{combination[0]}/{combination[1]}x{combination[2]}_{cat_column}_given_bound.html')

            vor_bound_fig = plotly_spatial_scatter_categorical(voronoi_boundaries, pipeline_instance.data.obs[cat_column])
            vor_bound_fig.write_html(file=f'assets/{combination[0]}/{combination[1]}x{combination[2]}_{cat_column}_vor_bound.html')

            rectangle_bound_fig = plotly_spatial_scatter_categorical(rectangle_boundaries, pipeline_instance.data.obs[cat_column])
            rectangle_bound_fig.write_html(file=f'assets/{combination[0]}/{combination[1]}x{combination[2]}_{cat_column}_rectangle_bound.html')

            delayney_edge_fig = plotly_spatial_scatter_edges(G_given_boundary_delaunay_edges, pipeline_instance.data.obs[cat_column], edge_info='Delaynay')
            delayney_edge_fig.write_html(file=f'assets/{combination[0]}/{combination[1]}x{combination[2]}_{cat_column}_delayney_edge.html')

            r3index_edge_fig = plotly_spatial_scatter_edges(G_given_boundary_r3index_edges, pipeline_instance.data.obs[cat_column], edge_info='R3Index')
            r3index_edge_fig.write_html(file=f'assets/{combination[0]}/{combination[1]}x{combination[2]}_{cat_column}_r3index_edge.html')

        pca_given_boundaries_fig = plotly_spatial_scatter_pca(pipeline_instance, given_boundaries, component=1)
        pca_given_boundaries_fig.write_html(file=f"assets/{combination[0]}/{combination[1]}x{combination[2]}_pca_1_given_bound.html")
        pca_voronoi_boundaries_fig = plotly_spatial_scatter_pca(pipeline_instance, voronoi_boundaries, component=1)
        pca_voronoi_boundaries_fig.write_html(file=f"assets/{combination[0]}/{combination[1]}x{combination[2]}_pca_1_voronoi_bound.html")
        pca_rectangle_boundaries_fig = plotly_spatial_scatter_pca(pipeline_instance, rectangle_boundaries, component=1)
        pca_rectangle_boundaries_fig.write_html(file=f"assets/{combination[0]}/{combination[1]}x{combination[2]}_pca_1_rectangle_bound.html")

        umap_given_boundaries_fig = plotly_spatial_scatter_umap(pipeline_instance, given_boundaries, component=1)
        umap_given_boundaries_fig.write_html(file=f"assets/{combination[0]}/{combination[1]}x{combination[2]}_umap_1_given_bound.html")
        umap_voronoi_boundaries_fig = plotly_spatial_scatter_umap(pipeline_instance, voronoi_boundaries, component=1)
        umap_voronoi_boundaries_fig.write_html(file=f"assets/{combination[0]}/{combination[1]}x{combination[2]}_umap_1_voronoi_bound.html")
        umap_rectangle_boundaries_fig = plotly_spatial_scatter_umap(pipeline_instance, rectangle_boundaries, component=1)
        umap_rectangle_boundaries_fig.write_html(file=f"assets/{combination[0]}/{combination[1]}x{combination[2]}_umap_1_rectangle_bound.html")

    except Exception as e:
        logger.error(f"The following error occurred while testing - {combination[0]} - {combination[1]} - {combination[2]}")
        logger.error(str(e))

# def run_parallel(all_combinations):
#     Parallel(n_jobs=4)(delayed(run_pipeline_visualizations)(combination) for combination in all_combinations)


# run_parallel(all_combinations)

if __name__ == "__main__":    


    mlflow.set_tracking_uri("/data/qd452774/spatial_transcriptomics/mlruns/")
    client = mlflow.tracking.MlflowClient()

    experiments_config = []

    filters_list = [
        # {
        #     "data_filter_name" : "Liver1Slice1",
        #     "names_list" : ["Liver1Slice1"],
        #     "state": "steady-state",
        #     "filter_nans" : False,
        # },
        # {
        #     "data_filter_name" : "Liver1Slice2",
        #     "names_list" : ["Liver1Slice2"],
        #     "state": "steady-state",
        #     "filter_nans" : False,
        # }, 
        # {
        #     "data_filter_name" : "Liver2Slice1",
        #     "names_list" : ["Liver2Slice1"],
        #     "state": "steady-state",
        #     "filter_nans" : False,
        # }, 
        # {
        #     "data_filter_name" : "Liver2Slice2",
        #     "names_list" : ["Liver2Slice2"],
        #     "state": "steady-state",
        #     "filter_nans" : False,
        # },
        # {
        #     "data_filter_name" : "Liver1Slice12",
        #     "names_list" : ["Liver1Slice1", "Liver1Slice2"],
        #     "state": "steady-state",
        #     "filter_nans" : False,
        # }, 
        # {
        #     "data_filter_name" : "Liver2Slice12",
        #     "names_list" : ["Liver2Slice1", "Liver2Slice2"],
        #     "state": "steady-state",
        #     "filter_nans" : False,
        # },
        {
            "data_filter_name" : "Liver12Slice12",
            "names_list" : ["Liver1Slice1", "Liver1Slice2", "Liver2Slice1", "Liver2Slice2"],
            "state": "steady-state",
            "filter_nans" : False,
        },
    ]

    for experiment_config in filters_list:
        
        my_experiment = f"spatial_clustering_{experiment_config['data_filter_name']}"
        my_experiment_id = client.get_experiment_by_name(my_experiment).experiment_id

        my_runs = client.search_runs(experiment_ids=[my_experiment_id])

        for each_run in my_runs:
            
            if 'leiden-0.7' in each_run.info.run_name:# or 'leiden-1.0' in each_run.info.run_name or 'leiden-1.1' in each_run.info.run_name:
                try:
                    artifact_location_base = f"/data/qd452774/spatial_transcriptomics/mlruns/{my_experiment_id}/{each_run.info.run_uuid}/artifacts"

                    if each_run.data.tags['state'] == 'disease-state':
                        my_filename = f"{artifact_location_base}/data/spatial_clustering_ds_{each_run.data.tags['data_filter']}.h5ad"
                    else:
                        my_filename = f"{artifact_location_base}/data/spatial_clustering_ss_{each_run.data.tags['data_filter']}.h5ad"
                    
                    experiments_config.append(
                        {
                            "data_filter_name": each_run.data.tags['data_filter'],
                            "state": each_run.data.tags["state"],
                            "experiment": my_experiment,
                            "experiment_id": my_experiment_id,
                            "run_name": each_run.data.tags["run_name"],
                            "run_id": each_run.info.run_uuid,
                            "data_file_name": my_filename,
                            "names_list": ast.literal_eval(each_run.data.params['names_list']),
                            
                        }
                    )
                except:
                    pass    

    # pipeline_instance = VisualizePipeline()

    filters = experiments_config[0]
    experiment_id = filters["experiment_id"]
    run_id = filters["run_id"]

    pipeline_instance = VisualizePipeline(filters)

    sample_slices = [pipeline_instance.names_list[0]]
    x_indices = [18] #list(range(20))
    y_indices = [7] #list(range(20))

    all_combinations = []
    for item_slice in sample_slices:
        for x_index in x_indices:
            for y_index in y_indices:
                all_combinations.append((item_slice, x_index, y_index))

    test_combination = all_combinations[0]

    from helpers import ANNOTATION_DICT

    pipeline_instance.data.obs['leiden_res'] = pipeline_instance.data.obs['leiden_res'].map(ANNOTATION_DICT)
    pipeline_instance.data.obs['liver_slice'] = pipeline_instance.data.obs['sample']


    mask_combined = (pipeline_instance.data.obs['sample'] == 'Liver1Slice1')
    pipeline_instance.data = pipeline_instance.data[mask_combined]

    run_pipeline_visualizations(test_combination, pipeline_instance)


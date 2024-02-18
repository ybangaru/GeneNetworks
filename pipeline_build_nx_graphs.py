"""
This script is typically used after leiden clustering to create networkx graphs for each slice segmented into multiple regions
and save them in the form of pickle files. These pickle files are then used for training the graph neural network.
"""
import warnings
import multiprocessing
from helpers import spatialPipeline, logger, NO_JOBS, DATA_DIR, COLORS_LIST

warnings.filterwarnings("ignore")


CLUSTERING_RUN_ID = "c750f81070fa4ccbbd731b92746bebc6"
UNIQUE_IDENTIFIER = "subcluster-finetune"
GRAPH_FOLDER_NAME = "graph"

data_options = {
    "data_dir": DATA_DIR,
    "mlflow_config": {
        "run_id": CLUSTERING_RUN_ID,
    },
    "unique_identifier": UNIQUE_IDENTIFIER,
    "cell_type_column": "Annotation_2",
    "base_microns_for_edge_cutoff": 5,
}
pipeline_instance = spatialPipeline(options=data_options)
pipeline_instance.data.obs["cell_type"] = pipeline_instance.data.obs[data_options["cell_type_column"]]
pipeline_instance.build_network_edge_cutoffs(data_options["base_microns_for_edge_cutoff"])

unique_colors = sorted(pipeline_instance.data.obs["cell_type"].unique())
unique_colors.sort()
COLOR_DICT = dict(zip(unique_colors, COLORS_LIST[: len(unique_colors)]))
COLOR_DICT["Unknown"] = COLORS_LIST[len(unique_colors)]

NODE_FEATURES = [
    "cell_type",
    "center_coord",
    "biomarker_expression",
    "volume",
    "boundary_polygon",
    "cell_id",
    "is_padding",
    "perimeter",
    "area",
    "mean_radius",
    "compactness_circularity",
    "fractality",
    "concavity",
    "elongation",
    "overlap_index",
    "sbro_orientation",
    "wswo_orientation",
]
EDGE_FEATURES = ["distance", "edge_type"]


pretransform_networkx_configs = [
    {
        "boundary_type": "voronoi",
        "edge_config": {
            "type": "Delaunay",
            "bound_type": "",
        },
    },
    {
        "boundary_type": "given",
        "edge_config": {
            "type": "Delaunay",
            "bound_type": "",
        },
    },
    {
        "boundary_type": "given",
        "edge_config": {
            "type": "R3Index",
            "bound_type": "rectangle",
            "threshold_distance": 0,
        },
    },
    {
        "boundary_type": "given",
        "edge_config": {
            "type": "R3Index",
            "bound_type": "rotated_rectangle",
            "threshold_distance": 0,
        },
    },
    {
        "boundary_type": "given",
        "edge_config": {
            "type": "MST",
            "bound_type": "",
        },
    },
]

all_slices_names_grid = []
for item_slice in pipeline_instance.slides_list:
    for x_index in list(range(20)):
        for y_index in list(range(20)):
            temp_segment_config = {
                "segments_per_dimension": 20,
                "sample_name": item_slice,
                "region_id": (x_index, y_index),
                "padding": {
                    "value": pipeline_instance.slice_pixels_edge_cutoff[item_slice] * 40,
                    "units": "pixels",
                },
                "neighbor_edge_cutoff": pipeline_instance.slice_pixels_edge_cutoff[item_slice],
            }
            all_slices_names_grid.append(temp_segment_config)

network_features_config = {
    "node_features": NODE_FEATURES,
    "edge_features": EDGE_FEATURES,
}


def parallel_function(segment_config, pretransform_networkx_config, network_features_config):
    try:
        pipeline_instance.save_networkx_config_pretransform_and_data(
            pretransform_networkx_config,
            segment_config,
            network_features_config,
            save_to="mlflow_run",
            graph_folder_name=GRAPH_FOLDER_NAME,
        )
        curr_nx_graph = pipeline_instance.build_networkx_for_region(
            segment_config, pretransform_networkx_config, network_features_config
        )
        if curr_nx_graph:
            pipeline_instance.build_networkx_plots(
                curr_nx_graph,
                segment_config,
                pretransform_networkx_config,
                COLOR_DICT,
                save_=True,
                save_to="mlflow_run",
                graph_folder_name=GRAPH_FOLDER_NAME,
            )
        else:
            logger.info("No cells in segment")
            logger.info(
                f"{segment_config['sample_name']} - x={segment_config['region_id'][0]} - y={segment_config['region_id'][1]}"
            )
    except Exception as e:
        logger.info("------------------")
        logger.info(
            f"{segment_config['sample_name']} - x={segment_config['region_id'][0]} - y={segment_config['region_id'][1]}"
        )
        logger.info(pretransform_networkx_config)
        logger.error(f"{str(e)}")


def main():
    func_args = [
        (segment_config, pretransform_networkx_config, network_features_config)
        for segment_config in all_slices_names_grid
        for pretransform_networkx_config in pretransform_networkx_configs
    ]

    # Create a multiprocessing Pool and execute the function in parallel
    with multiprocessing.Pool(processes=NO_JOBS) as pool:
        pool.starmap(parallel_function, func_args)
    # test_index = 250 # 1357
    # parallel_function(*func_args[test_index])


if __name__ == "__main__":
    main()

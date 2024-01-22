"""
This script is typically used after leiden clustering to create networkx graphs for each slice segmented into multiple regions
and save them in the form of pickle files. These pickle files are then used for training the graph neural network.
"""
import warnings
import multiprocessing

from helpers import spatialPipeline, logger, NO_JOBS, ANNOTATION_DICT, SLICE_PIXELS_EDGE_CUTOFF, PROJECT_DIR

warnings.filterwarnings("ignore")


all_liver_samples = ["Liver1Slice1", "Liver1Slice2", "Liver2Slice1", "Liver2Slice2"]
data_filter_name = "Liver12Slice12"
leiden_resolution = 0.7

data_options = {
    "dir_loc": f"{PROJECT_DIR}/",
    "names_list": all_liver_samples,
    "mlflow_config": {
        "id": None,
        "data_filter_name": data_filter_name,
        "leiden_resolution": leiden_resolution,
    },
}
pipeline_instance = spatialPipeline(options=data_options)
pipeline_instance.data.obs["cell_type"] = pipeline_instance.data.obs["leiden_res"].map(ANNOTATION_DICT)


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
        "neighbor_edge_cutoff": SLICE_PIXELS_EDGE_CUTOFF["Liver1Slice1"],
    },
    {
        "boundary_type": "given",
        "edge_config": {
            "type": "Delaunay",
            "bound_type": "",
        },
        "neighbor_edge_cutoff": SLICE_PIXELS_EDGE_CUTOFF["Liver1Slice1"],
    },
    {
        "boundary_type": "given",
        "edge_config": {
            "type": "R3Index",
            "bound_type": "rectangle",
            "threshold_distance": 0,
        },
        "neighbor_edge_cutoff": SLICE_PIXELS_EDGE_CUTOFF["Liver1Slice1"],
    },
    {
        "boundary_type": "given",
        "edge_config": {
            "type": "R3Index",
            "bound_type": "rotated_rectangle",
            "threshold_distance": 0,
        },
        "neighbor_edge_cutoff": SLICE_PIXELS_EDGE_CUTOFF["Liver1Slice1"],
    },
    {
        "boundary_type": "given",
        "edge_config": {
            "type": "MST",
            "bound_type": "",
        },
        "neighbor_edge_cutoff": SLICE_PIXELS_EDGE_CUTOFF["Liver1Slice1"],
    },
]

all_slices_names_grid = []
for item_slice in all_liver_samples:
    for x_index in list(range(20)):
        for y_index in list(range(20)):
            temp_segment_config = {
                "segments_per_dimension": 20,
                "sample_name": item_slice,
                "region_id": (x_index, y_index),
                "padding": {
                    "value": SLICE_PIXELS_EDGE_CUTOFF[item_slice] * 40,
                    "units": "pixels",
                },
            }
            all_slices_names_grid.append(temp_segment_config)

network_features_config = {
    "node_features": NODE_FEATURES,
    "edge_features": EDGE_FEATURES,
}


def parallel_function(segment_config, pretransform_networkx_config, network_features_config):
    try:
        curr_nx_graph = pipeline_instance.build_networkx_for_region(
            segment_config, pretransform_networkx_config, network_features_config
        )
        if curr_nx_graph:
            pipeline_instance.build_networkx_plots(
                curr_nx_graph, segment_config, pretransform_networkx_config, save_=True
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

    # parallel_function(*func_args[250])


if __name__ == "__main__":
    main()

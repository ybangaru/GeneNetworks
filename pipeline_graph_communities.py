import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch_geometric.transforms as T
from graphxl import (
    spatialPipeline,
    DATA_DIR,
    COLORS_LIST,
    MLFLOW_CLIENT,
    GraphCommunityDataset,
    MLFLOW_TRACKING_URI,
    train_graph_community_ensemble,
    logger,
)


def build_base_data_config():
    CLUSTERING_EXP_ID = "834659805990264659"
    CLUSTERING_RUN_ID = "c750f81070fa4ccbbd731b92746bebc6"
    UNIQUE_IDENTIFIER = "subcluster-finetune"
    GRAPH_FOLDER_NAME = "graph_communities"

    SEGMENTS_PER_DIMENSION = 20
    chosen_network = "knn"  # "voronoi_delaunay", "given_r3index", "given_mst"
    network_type = f"{chosen_network}_{SEGMENTS_PER_DIMENSION}"

    all_liver_samples = ["Liver1Slice1", "Liver1Slice2", "Liver2Slice1", "Liver2Slice2"]
    data_filter_name = "Liver12Slice12"

    data_options = {
        "data_dir": DATA_DIR,
        "mlflow_config": {
            "run_id": CLUSTERING_RUN_ID,
            "networkx_gpkl_root": f"{MLFLOW_TRACKING_URI}{CLUSTERING_EXP_ID}/{CLUSTERING_RUN_ID}/artifacts/{UNIQUE_IDENTIFIER}/{GRAPH_FOLDER_NAME}/{network_type}",
            "data_filter_name": data_filter_name,
        },
        "slides_list": all_liver_samples,
        "unique_identifier": UNIQUE_IDENTIFIER,
        "cell_type_column": "Annotation_2",
        # "base_microns_for_edge_cutoff": 5,
        "network_type": network_type,
    }

    return {
        "data_options": data_options,
        "SEGMENTS_PER_DIMENSION": SEGMENTS_PER_DIMENSION,
    }


def build_base_pipeline(data_options, SEGMENTS_PER_DIMENSION):
    pipeline_instance = spatialPipeline(options=data_options)
    pipeline_instance.data.obs["cell_type"] = pipeline_instance.data.obs[data_options["cell_type_column"]]
    # pipeline_instance.build_network_edge_cutoffs(data_options["base_microns_for_edge_cutoff"])

    unique_colors = sorted(pipeline_instance.data.obs["cell_type"].unique())
    unique_colors.sort()
    COLOR_DICT = dict(zip(unique_colors, COLORS_LIST[: len(unique_colors)]))
    COLOR_DICT["Unknown"] = COLORS_LIST[len(unique_colors)]

    return {
        "pipeline_instance": pipeline_instance,
        "color_dict": COLOR_DICT,
        "segments_per_dim": SEGMENTS_PER_DIMENSION,
    }


def build_graph_segment_config(pipeline_instance, color_dict, segments_per_dim):
    pretransform_networkx_config = {
        # only required for building the segments grid, not being used as feature
        "boundary_type": "given",
        "edge_config": {
            "type": "knn",
            "bound_type": "",
            "n_neighbors": 69,
        },
    }

    all_slices_names_grid = []
    for item_slice in pipeline_instance.slides_list:
        for x_index in list(range(segments_per_dim)):
            for y_index in list(range(segments_per_dim)):
                temp_segment_config = {
                    "segments_per_dimension": segments_per_dim,
                    "sample_name": item_slice,
                    "region_id": (x_index, y_index),
                    "padding": None,
                    "neighbor_edge_cutoff": None,
                }
                all_slices_names_grid.append(temp_segment_config)

    network_features_config = {
        "node_features": ["one-hot-encoding"],
        "edge_features": None,
    }

    return {
        "pretransform_networkx_config": pretransform_networkx_config,
        "network_features_config": network_features_config,
        "all_slides_names_grid": all_slices_names_grid,
    }


def build_torch_dataset(
    data_options,
    all_slides_names_grid,
    pipeline_instance,
    pretransform_networkx_config,
    network_features_config,
    save_files=True,
    max_nodes_per_segment=None,
    **kwargs,
):
    clustering_run_id = data_options["mlflow_config"]["run_id"]
    clustering_run_info = MLFLOW_CLIENT.get_run(clustering_run_id)
    clustering_run_data = clustering_run_info.data

    ROOT_DIR = data_options["mlflow_config"]["networkx_gpkl_root"]
    RAW_FILENAME = "torch_files"
    RAW_TORCH_FOLDER_DIR = f"{ROOT_DIR}/{RAW_FILENAME}"
    os.makedirs(RAW_TORCH_FOLDER_DIR, exist_ok=True)

    dataset_kwargs = {}
    dataset_kwargs.update(
        {
            "experiment_name": "graph_communities",
            "run_name": "baseline community model testing",
            "clustering_run_id": data_options["mlflow_config"]["run_id"],
            "dataset_filter": data_options["mlflow_config"]["data_filter_name"],
            "leiden_annotation_column": data_options["cell_type_column"],
            "leiden_cluster_res": eval(clustering_run_data.params["leiden"])["resolution"],
            "transform": [],
            "pre_transform": None,
            "dataset_root": ROOT_DIR,
            "raw_folder_name": RAW_FILENAME,
            "processed_folder_name": f"{RAW_FILENAME}_processed",
            "network_type": data_options["network_type"],
            "node_features": network_features_config["node_features"],
            "edge_features": network_features_config["edge_features"],
        }
    )

    if save_files:
        MAX_NODES_SEGMENT = -1
        for segment_config in all_slides_names_grid:
            torch_graph_temp = pipeline_instance.build_networkx_for_region(
                segment_config, pretransform_networkx_config, network_features_config, convert_to_torch=True
            )
            if torch_graph_temp is not None:
                MAX_NODES_SEGMENT = max(MAX_NODES_SEGMENT, torch_graph_temp.x.shape[0])
                filename_tg = f"{RAW_TORCH_FOLDER_DIR}/{torch_graph_temp.region_id}.gpt"
                torch.save(torch_graph_temp, filename_tg)
        logger.info(f"Max nodes per segment: {MAX_NODES_SEGMENT}")
    else:
        if not max_nodes_per_segment:
            raise ValueError("max_nodes_per_segment is required when save_files is False")
        MAX_NODES_SEGMENT = max_nodes_per_segment

    dataset_kwargs.update(
        {
            "max_nodes_segment": MAX_NODES_SEGMENT,
        }
    )

    dataset = GraphCommunityDataset(**dataset_kwargs)
    transformers = [T.ToDense(MAX_NODES_SEGMENT)]
    dataset.set_transforms(transformers)

    # Set a random seed for reproducibility
    np.random.seed(42)
    num_regions = len(dataset.region_ids)
    all_inds = list(range(num_regions))
    train_inds, valid_inds = train_test_split(all_inds, test_size=0.20)

    dataset_kwargs.update(
        {
            "transform": transformers,
            "train_inds": train_inds,
            "valid_inds": valid_inds,
        }
    )
    dataset.set_train_valid_inds(train_inds, valid_inds)
    logger.info(f"Number of graphs: {len(dataset)}")

    # Print the indices for the training and validation sets
    train_inds_str = ",".join(map(str, train_inds))
    valid_inds_str = ",".join(map(str, valid_inds))
    logger.info(f"Training indices: {train_inds_str}")
    logger.info(f"Validation indices: {valid_inds_str}")

    return dataset, dataset_kwargs


def build_graph_community_model_kwargs(dataset, color_dict, **kwargs):
    num_tcn = len(color_dict) - 1  # -1 for unknown
    num_ensemble_models = 10  # for ensemble
    num_training_iterations = 100
    embedding_dimension = 128
    lr_train = 0.0001
    loss_cutoff = -0.06

    return {
        "num_tcn": num_tcn,
        "num_ensembles": num_ensemble_models,
        "num_train_iterations": num_training_iterations,
        "emb_dim": embedding_dimension,
        "lr": lr_train,
        "loss_cutoff": loss_cutoff,
    }


def main():
    save_files = False
    max_nodes_per_segment = 1778  # required when save files is False

    base_data_config = build_base_data_config()
    base_pipeline_info = build_base_pipeline(**base_data_config)
    graph_segment_config = build_graph_segment_config(**base_pipeline_info)
    torch_dataset, dataset_config = build_torch_dataset(
        **base_data_config,
        **graph_segment_config,
        **base_pipeline_info,
        save_files=save_files,
        max_nodes_per_segment=max_nodes_per_segment,
    )
    model_config = build_graph_community_model_kwargs(dataset=torch_dataset, **dataset_config, **base_pipeline_info)
    train_graph_community_ensemble(torch_dataset, dataset_config, **model_config)


def combine_ensemble_results(exp_id, run_id):
    import glob
    import pandas as pd

    run_data = MLFLOW_CLIENT.get_run(run_id)
    run_data = run_data.data

    artifact_location = f"/data/qd452774/spatial_transcriptomics/mlruns/{exp_id}/{run_id}/artifacts"
    ensemble_results = {}

    all_segments = {
        "train": [],
        "valid": [],
    }
    num_ensemble_models = int(run_data.params["num_ensembles"])
    for dtype in ["train", "valid"]:
        for i in range(1, num_ensemble_models + 1):
            try:
                all_files_ = glob.glob(f"{artifact_location}/{dtype}_results/model_{i}/*")
                for filename_ in all_files_:
                    all_segments[dtype].append(filename_[filename_.rfind("/") + 1 :].split("_")[0])
            except Exception as e:
                logger.info(f"Model {i} - data type {dtype} files missing")

    all_segments["train"] = list(set(all_segments["train"]))
    all_segments["valid"] = list(set(all_segments["valid"]))

    for dtype in ["train", "valid"]:
        for segment in all_segments[dtype]:
            all_labels_list = []
            for i in range(1, num_ensemble_models + 1):
                try:
                    assign_matrix = pd.read_csv(
                        f"{artifact_location}/{dtype}_results/model_{i}/{segment}_assignment_matrix.csv", header=None
                    )
                    node_mask = pd.read_csv(
                        f"{artifact_location}/{dtype}_results/model_{i}/{segment}_node_mask.csv", header=None
                    )
                    nonzero_ind = node_mask[node_mask[0] == 1].index
                    assign_matrix = assign_matrix.iloc[nonzero_ind, :]
                    segment_labels = np.array(assign_matrix).argmax(axis=1) + 1
                    all_labels_list.append(segment_labels)
                except Exception as e:
                    logger.info(f"Model {i} - segment {segment} assignment matrix missing")

            ensemble_results[segment] = pd.DataFrame(all_labels_list).T

    for dtype in ["train", "valid"]:
        os.makedirs(f"{artifact_location}/{dtype}_results/models_combined", exist_ok=True)
        for segment in all_segments[dtype]:
            ensemble_results[segment].to_csv(
                f"{artifact_location}/{dtype}_results/models_combined/{segment}.csv", index=False
            )


def build_visualizations(exp_id, run_id):
    pass


if __name__ == "__main__":
    # stage 1
    # main()
    # stage 2
    run_id = "c98854892d514a12a1ef24fab7b0f893"
    exp_id = "548471598705222754"
    combine_ensemble_results(exp_id, run_id)
    # stage 3
    # use R diceR to generate the final results
    # stage 4
    # build visualizations for the segments
    # build_visualizations(exp_id, run_id)

"""
This module instanciates an MLflow client and provides functions to read the results of a run from MLflow.
"""
import glob
import random
import re
import pandas as pd
import numpy as np
import mlflow
import scanpy as sc
from ast import literal_eval
from .logging_setup import logger
from .local_config import MLFLOW_TRACKING_URI
from .models import GNN_pred
from .transform import AddCenterCellType, FeatureMask
from .data import CellularGraphDataset, SubgraphSampler

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MLFLOW_CLIENT = mlflow.tracking.MlflowClient()


def get_run_info(x_experiment, x_resolution):
    xexp_name = f"spatial_clustering_{x_experiment}"
    xrun_name = f"steady-state-using-220-PCs-50-neighbors-leiden-{x_resolution}"

    # Get the experiment ID by name
    experiment = MLFLOW_CLIENT.get_experiment_by_name(xexp_name)
    experiment_id = experiment.experiment_id

    # Search for the run by run name within the specified experiment
    runs = MLFLOW_CLIENT.search_runs(experiment_ids=[experiment_id], filter_string=f"tags.run_name='{xrun_name}'")

    # Check if any runs match the criteria
    if len(runs) > 1:
        logger.error("more runs are there with same name than expected")
        return None
    elif len(runs) == 1:
        return runs[0]
    else:
        logger.debug(f"No matching run found for run name: {xrun_name} in experiment: {xexp_name}")
        return None


def read_run_annotations(run_id=None, slides_name=None, resolution=None):
    if (run_id and slides_name) or (run_id and resolution):
        logger.info("Provided run_id and slides_name or resolution, using run_id directly")

    if run_id:
        xrun_info = MLFLOW_CLIENT.get_run(run_id)
    else:
        assert (
            slides_name and resolution
        ), "Please provide slides custom name and resolution to filter runs for the experiment"
        xrun_info = get_run_info(slides_name, resolution)

    exp_id = xrun_info.info.experiment_id
    run_id = xrun_info.info.run_id
    annotations_json_file = f"{MLFLOW_TRACKING_URI}{exp_id}/{run_id}/artifacts/metadata/annotations_dict.json"
    with open(annotations_json_file, "r") as json_file:
        annotations_dict = literal_eval(json_file.read())

    return annotations_dict


def read_run_annotation_colors(run_id=None, slides_name=None, resolution=None):
    if (run_id and slides_name) or (run_id and resolution):
        logger.info("Provided run_id and slides_name or resolution, using run_id directly")

    if run_id:
        xrun_info = MLFLOW_CLIENT.get_run(run_id)
    else:
        assert (
            slides_name and resolution
        ), "Please provide slides custom name and resolution to filter runs for the experiment"
        xrun_info = get_run_info(slides_name, resolution)

    exp_id = xrun_info.info.experiment_id
    run_id = xrun_info.info.run_id

    try:
        annotation_colors_json_file = f"{MLFLOW_TRACKING_URI}{exp_id}/{run_id}/artifacts/metadata/cell_type_color.json"
        with open(annotation_colors_json_file, "r") as json_file:
            annotation_colors_dict = literal_eval(json_file.read())
    except FileNotFoundError:
        dataset_root_ = xrun_info.data.params["dataset_root"]
        annotation_colors_json_file = f"{dataset_root_}/color_dict.json"
        with open(annotation_colors_json_file, "r") as json_file:
            annotation_colors_dict = literal_eval(json_file.read())

    return annotation_colors_dict


def read_run_annotation_mapping(run_id=None, slides_name=None, resolution=None):
    if (run_id and slides_name) or (run_id and resolution):
        logger.info("Provided run_id and slides_name or resolution, using run_id directly")

    if run_id:
        xrun_info = MLFLOW_CLIENT.get_run(run_id)
    else:
        assert (
            slides_name and resolution
        ), "Please provide slides custom name and resolution to filter runs for the experiment"
        xrun_info = get_run_info(slides_name, resolution)

    exp_id = xrun_info.info.experiment_id
    run_id = xrun_info.info.run_id

    try:
        annotation_mapping_json_file = f"{MLFLOW_TRACKING_URI}{exp_id}/{run_id}/artifacts/metadata/cell_mapping.json"
        with open(annotation_mapping_json_file, "r") as json_file:
            annotation_mapping_dict = literal_eval(json_file.read())
    except FileNotFoundError:
        dataset_root_ = xrun_info.data.params["dataset_root"]
        annotation_mapping_json_file = f"{dataset_root_}/cell_mapping.json"
        with open(annotation_mapping_json_file, "r") as json_file:
            annotation_mapping_dict = literal_eval(json_file.read())

    return annotation_mapping_dict


def read_run_attribute_classification(run_id=None, attribute_name=None):
    if not run_id:
        raise ValueError("Please provide run_id")

    xrun_info = MLFLOW_CLIENT.get_run(run_id)
    return literal_eval(xrun_info.data.params[attribute_name])


def read_run_attribute_clustering(run_id=None, slides_name=None, resolution=None, attribute_name=None):
    if (run_id and slides_name) or (run_id and resolution):
        logger.info("Provided run_id and slides_name or resolution, using run_id directly")

    if run_id:
        xrun_info = MLFLOW_CLIENT.get_run(run_id)
    else:
        assert (
            slides_name and resolution
        ), "Please provide slides custom name and resolution to filter runs for the experiment"
        xrun_info = get_run_info(slides_name, resolution)

    return literal_eval(xrun_info.data.params[attribute_name])


def read_run_result_ann_data(run_id=None, slides_name=None, resolution=None):
    if (run_id and slides_name) or (run_id and resolution):
        logger.info("Provided run_id and slides_name or resolution, using run_id directly")

    if run_id:
        xrun_info = MLFLOW_CLIENT.get_run(run_id)
    else:
        assert (
            slides_name and resolution
        ), "Please provide slides custom name and resolution to filter runs for the experiment"
        xrun_info = get_run_info(slides_name, resolution)

    exp_id = xrun_info.info.experiment_id
    run_id = xrun_info.info.run_id

    # read filename automatically ending with h5ad
    file_names = glob.glob(f"{MLFLOW_TRACKING_URI}{exp_id}/{run_id}/artifacts/data/*.h5ad")
    if len(file_names) > 1:
        logger.error("more than one h5ad files found")
        raise ValueError("more than one h5ad files found")
    x_data = sc.read_h5ad(filename=file_names[0])

    return x_data


def read_run_node_true_pred_labels(experiment_id, run_id, pred_as_dist=None, train_or_test="train"):
    if train_or_test == "train":
        true_labels_folder = f"{MLFLOW_TRACKING_URI}{experiment_id}/{run_id}/artifacts/node_class"
        pred_labels_folder = f"{MLFLOW_TRACKING_URI}{experiment_id}/{run_id}/artifacts/node_probs"
    elif train_or_test == "test":
        true_labels_folder = f"{MLFLOW_TRACKING_URI}{experiment_id}/{run_id}/artifacts/eval/node_class"
        pred_labels_folder = f"{MLFLOW_TRACKING_URI}{experiment_id}/{run_id}/artifacts/eval/node_probs"
    true_labels_file_names = sorted(
        glob.glob(f"{true_labels_folder}/*.npy"),
        key=lambda x: int(re.search(r"\d+", x).group()),
    )
    pred_labels_file_names = sorted(
        glob.glob(f"{pred_labels_folder}/*.npy"),
        key=lambda x: int(re.search(r"\d+", x).group()),
    )

    true_numbers = []
    pred_numbers = []
    true_labels = []
    pred_labels = []

    for file_name in true_labels_file_names:
        number = int(file_name.split("/")[-1].split(".")[0])
        true_numbers.append(number)

        true_array = np.load(file_name)
        true_labels.append(true_array)

    for file_name in pred_labels_file_names:
        number = int(file_name.split("/")[-1].split(".")[0])
        pred_numbers.append(number)

        pred_array = np.load(file_name)
        pred_labels.append(pred_array)

    df = pd.DataFrame(
        {
            "true_num": true_numbers,
            "pred_num": pred_numbers,
            "true_labels": true_labels,
            "pred_labels": pred_labels,
        }
    )

    if not pred_as_dist:
        df["pred_labels"] = df["pred_labels"].apply(lambda x: np.argmax(x, axis=1))

    assert (df["true_num"] == df["pred_num"]).all()
    df["Number"] = df["true_num"]
    df = df.drop(["true_num", "pred_num"], axis=1)
    return df


def read_run_embeddings_df(experiment_id, run_id, type_="nodes"):
    embedding_folder = f"{MLFLOW_TRACKING_URI}{experiment_id}/{run_id}/artifacts/embeddings"

    all_file_names = glob.glob(f"{embedding_folder}/*.npy")
    if len(all_file_names) == 0:
        logger.error(f"No files found in {embedding_folder}")
        raise ValueError(f"No files found in {embedding_folder}")

    edge_embb_file_names = [x for x in all_file_names if "edge" in x]
    node_embb_file_names = set(all_file_names) - set(edge_embb_file_names)
    node_embb_file_names = list(node_embb_file_names)

    if type_ == "nodes":
        file_names = node_embb_file_names
    elif type_ == "edges":
        file_names = edge_embb_file_names
    else:
        logger.error(f"Invalid type_ to read_run_embeddings_df={type_}")
        raise ValueError(f"Invalid type_={type_} @ read_run_embeddings_df")

    numbers = []
    embeddings = []
    keys = []

    if type_ == "nodes":
        for file_name in file_names:
            number = int(file_name.split("/")[-1].split(".")[0])
            numbers.append(number)

            embedding_array = np.load(file_name)
            embeddings.append(embedding_array)

    elif type_ == "edges":
        for file_name in file_names:
            gnn_edge_number = file_name.split("/")[-1].split(".")[0]
            number = int(gnn_edge_number.split("_")[2])
            numbers.append(number)

            embedding_array = np.load(file_name)
            embeddings.append(embedding_array)
            keys.append(int(gnn_edge_number.split("_")[1][4:]))

    # Create a DataFrame
    df = pd.DataFrame({"Number": numbers, "Embedding": embeddings})
    df["Dim0"] = df["Embedding"].apply(lambda x: x[:, 0])
    df["Dim1"] = df["Embedding"].apply(lambda x: x[:, 1])
    df["Dim2"] = df["Embedding"].apply(lambda x: x[:, 2])

    if type_ == "edges":
        df["hop_num"] = keys

    return df


def load_node_classification_model(experiment_id, run_id, model_type, return_items=["dataset", "data_iter"]):
    # model_type: "best" or "latest"

    run_info = MLFLOW_CLIENT.get_run(run_id)
    run_data = run_info.data

    clustering_run_id = run_data.params["clustering_run_id"]
    clustering_run_info = MLFLOW_CLIENT.get_run(clustering_run_id)
    clustering_run_data = clustering_run_info.data

    network_type = run_data.params["network_type"]

    data_options = {
        "names_list": list(clustering_run_data.params["names_list"]),
        "mlflow_config": {
            "id": clustering_run_id,
            "networkx_gpkl_root": run_data.params["dataset_root"],
            "data_filter_name": run_data.params["dataset_filter"],
        },
    }

    dataset_kwargs = {
        "clustering_run_id": data_options["mlflow_config"]["id"],
        "dataset_root": run_data.params["dataset_root"],
        "raw_folder_name": run_data.params["raw_folder_name"],
        "processed_folder_name": run_data.params["processed_folder_name"],
        "network_type": network_type,
        "node_features": run_data.params["node_features"].split(","),
        "node_features_mask": run_data.params["node_features_mask"].split(","),
        "center_node_features_mask": run_data.params["center_node_features_mask"].split(","),
        "edge_features": run_data.params["edge_features"].split(","),
        "edge_features_mask": run_data.params["edge_features_mask"].split(","),
        "edge_types": eval(run_data.params["edge_types"]),
        "graph_type": run_data.params["graph_type"],
        "node_embedding_size": int(run_data.params["node_embedding_size"]),
        "edge_embedding_size": int(run_data.params["edge_embedding_size"]),
        "subgraph_size": int(run_data.params["subgraph_size"]),
        "subgraph_source": run_data.params["subgraph_source"],
        "subgraph_allow_distant_edge": bool(run_data.params["subgraph_allow_distant_edge"]),
        "subgraph_radius_limit": int(run_data.params["subgraph_radius_limit"]),
        "sampling_avoid_unassigned": bool(run_data.params["sampling_avoid_unassigned"]),
        "unassigned_cell_type": run_data.params["unassigned_cell_type"]
        if "unassigned_cell_type" in run_data.params
        else "Unassigned",
    }

    transform_kwargs = {
        "transform": [],
        "pre_transform": None,
    }

    dataset_kwargs.update(transform_kwargs)

    dataset = CellularGraphDataset(root=dataset_kwargs["dataset_root"], **dataset_kwargs)
    transformers = [
        AddCenterCellType(dataset),
    ]

    if dataset_kwargs["node_features_mask"]:
        use_node_features = [
            x for x in dataset_kwargs["node_features"] if x not in dataset_kwargs["node_features_mask"]
        ]
        if dataset_kwargs["center_node_features_mask"]:
            use_center_node_features = [
                x for x in use_node_features if x not in dataset_kwargs["center_node_features_mask"]
            ]
        transformers.append(
            FeatureMask(
                dataset,
                use_center_node_features=use_center_node_features
                if dataset_kwargs["center_node_features_mask"]
                else use_node_features,
                use_neighbor_node_features=use_node_features,
                use_edge_features=dataset.edge_feature_names,
            )
        )

    dataset.set_transforms(transformers)

    model_kwargs = {
        "num_layer": dataset.subgraph_size,  # same number of layers as number of hops in the subgraphs
        # +1 needed to assign the Unknown type during cell type masking transformation
        "num_node_type": len(dataset.cell_type_mapping) + 1,
        # number of embeddings for different cell types (plus one placeholder cell type)
        "num_feat": dataset[0].x.shape[1] - 1,  # exclude the cell type column
        "emb_dim": dataset_kwargs["node_embedding_size"],
        "emb_dim_edge": dataset_kwargs["edge_embedding_size"],
        "num_node_tasks": len(
            dataset.cell_type_mapping
        ),  # A multi-class classification task: predicting center cell type
        "num_graph_tasks": 0,  # a binary classification task
        "node_embedding_output": "last",
        "drop_ratio": 0.15,
        "graph_pooling": "max",
        "gnn_type": dataset_kwargs["graph_type"],
        "return_node_embedding": False,
        "edge_types": dataset_kwargs["edge_types"],
    }

    model = GNN_pred(**model_kwargs)

    model_file = f'{run_data.params["model_folder"]}/{model_type}_model.pt'
    model.from_pretrained(model_file, strict_bool=True)

    data_iter = SubgraphSampler(
        dataset,
        selected_inds=dataset._indices if dataset._indices else random.sample(range(dataset.N), 10),
        batch_size=int(run_data.params["batch_size"]),
        num_regions_per_segment=int(run_data.params["num_regions_per_segment"]),
        steps_per_segment=int(run_data.params["num_iterations_per_segment"]),
        num_workers=int(run_data.params["num_workers"]),
    )

    if "dataset" in return_items and "data_iter" in return_items:
        return model, dataset, data_iter
    elif "dataset" in return_items:
        return model, dataset
    elif "data_iter" in return_items:
        return model, data_iter

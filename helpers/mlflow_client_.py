"""
This module instanciates an MLflow client and provides functions to read the results of a run from MLflow.
"""
import glob
import re
import pandas as pd
import numpy as np
import mlflow
import scanpy as sc
from ast import literal_eval
from .logging_setup import logger
from .local_config import MLFLOW_TRACKING_URI

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

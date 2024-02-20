# updates to backfill mlflow runs with config data
import os
import pandas as pd
import mlflow
from helpers import MLFLOW_CLIENT, logger


def update_19_02_2024():
    """
    Reading runs info from csv file downloaded from mlflow dashboard
    as filtering for None values is not possible, for details, https://github.com/mlflow/mlflow/issues/2922
    """

    runs_df = pd.read_csv("runs.csv")

    fill_center_cell_feature_mask_ids = runs_df[runs_df["center_node_features_mask"].isna()]["Run ID"].tolist()
    for run_id in fill_center_cell_feature_mask_ids:
        run_info = MLFLOW_CLIENT.get_run(run_id)
        run_info = run_info.data
        MLFLOW_CLIENT.log_param(run_id, "center_node_features_mask", run_info.params["node_features_mask"])

    fill_clustering_run_ids = runs_df[runs_df["clustering_run_id"].isna()]["Run ID"].tolist()
    for run_id in fill_clustering_run_ids:
        run_info = MLFLOW_CLIENT.get_run(run_id)
        run_info = run_info.data
        MLFLOW_CLIENT.log_param(run_id, "clustering_run_id", "c750f81070fa4ccbbd731b92746bebc6")

    fill_edge_embedding_size_ids = runs_df[runs_df["edge_embedding_size"].isna()]["Run ID"].tolist()
    for run_id in fill_edge_embedding_size_ids:
        run_info = MLFLOW_CLIENT.get_run(run_id)
        run_info = run_info.data
        MLFLOW_CLIENT.log_param(run_id, "edge_embedding_size", run_info.params["node_embedding_size"])

    fill_edge_types_ids = runs_df[runs_df["edge_types"].isna()]["Run ID"].tolist()
    for run_id in fill_edge_types_ids:
        run_info = MLFLOW_CLIENT.get_run(run_id)
        run_info = run_info.data
        MLFLOW_CLIENT.log_param(
            run_id,
            "edge_types",
            {
                "neighbor": 0,
                "distant": 1,
                "self": 2,
            },
        )

    fill_edge_embedding_bool_ids = runs_df[runs_df["log_edge_embeddings"].isna()]["Run ID"].tolist()
    for run_id in fill_edge_embedding_bool_ids:
        run_info = MLFLOW_CLIENT.get_run(run_id)
        run_info = run_info.data
        MLFLOW_CLIENT.log_param(run_id, "log_edge_embeddings", False)

    data_filter_name_ids = runs_df[runs_df["dataset_filter"].isna()]["Run ID"].tolist()
    for run_id in data_filter_name_ids:
        run_info = MLFLOW_CLIENT.get_run(run_id)
        run_info = run_info.data
        MLFLOW_CLIENT.log_param(run_id, "dataset_filter", "Liver12Slice12")

    for run_id in data_filter_name_ids:
        run_info = MLFLOW_CLIENT.get_run(run_id)
        run_info = run_info.data
        MLFLOW_CLIENT.log_param(run_id, "leiden_cluster_res", 0.7)
        MLFLOW_CLIENT.log_param(run_id, "leiden_annotation_column", "Annotation_2")

    fill_leiden_annotation_column_ids = runs_df[runs_df["leiden_annotation_column"].isna()]["Run ID"].tolist()
    for run_id in fill_leiden_annotation_column_ids:
        run_info = MLFLOW_CLIENT.get_run(run_id)
        run_info = run_info.data
        MLFLOW_CLIENT.log_param(run_id, "leiden_annotation_column", "Annotation_1")


if "__name__ == __main__":
    update_19_02_2024()

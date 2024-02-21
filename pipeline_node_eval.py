import os
import warnings
from torchviz import make_dot
from helpers import (
    MLFLOW_TRACKING_URI,
    plotly_confusion_matrix_anime,
    plotly_embeddings_anime,
    plotly_precision_recall_curve_anime,
    load_node_classification_model,
    logger,
)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")


def build_eval_anime(exp_id, run_id):
    dir_location = f"{MLFLOW_TRACKING_URI}{exp_id}/{run_id}/artifacts"

    if not os.path.exists(f"{dir_location}/figs"):
        os.makedirs(f"{dir_location}/figs")

    # TODO: build classification report and auc plots
    # classification_report_anime = plotly_classification_report_anime(exp_id, run_id, "test")
    # classification_report_anime.write_html(f"{dir_location}/figs/class_report.html")

    plots_ = {}

    plots_.update(**plotly_embeddings_anime(exp_id, run_id))

    for data_type in ["train", "test"]:
        plots_.update(**plotly_precision_recall_curve_anime(exp_id, run_id, data_type))
        plots_.update(**plotly_confusion_matrix_anime(exp_id, run_id, data_type))

    for key, value in plots_.items():
        value.write_html(f"{dir_location}/figs/{key}.html")


def save_model_architecture_fig(exp_id, run_id, model_type="best"):
    test_model, dataset, data_iter = load_node_classification_model(
        exp_id, run_id, model_type=model_type, return_items=["dataset", "data_iter"]
    )

    batch = next(data_iter)
    batch = batch.to("cpu")
    output_tensor = test_model(batch)  # Forward pass through the model
    dot_graph = make_dot(output_tensor[0], params=dict(test_model.named_parameters()), show_attrs=True, show_saved=True)

    save_folder = f"{MLFLOW_TRACKING_URI}{exp_id}/{run_id}/artifacts/figs"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    dot_graph.render(f"{save_folder}/model_visualization_best", format="png")


if __name__ == "__main__":
    exp_id = "303267778283029741"
    run_ids = [
        "ee8781107115495b87142ea068eda61a",
        "d78e239af13c41f5a4bd67369354746c",
        "b0730f640077408faf145fcc4e6bc354",
        "47893dda56a84854a911d29a9dbdf154",
        "cd7fa3415cc042f1b8f55e054d722bbe",
        "b1241973be1943fc9e8a88ab96b0c8fd",
    ]

    for run_id in run_ids:
        try:
            save_model_architecture_fig(exp_id, run_id, model_type="best")
            build_eval_anime(exp_id, run_id)
        except Exception as e:
            logger.error(f"Error in run_id: {run_id} with error: {e}")

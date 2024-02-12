import os
from helpers import (
    plotly_confusion_matrix_anime,
    plotly_embeddings_anime_2d,
    plotly_embeddings_anime_3d,
    plotly_precision_recall_curve_anime,
)


def build_eval_plots(exp_id, run_id):
    dir_location = f"/data/qd452774/spatial_transcriptomics/mlruns/{exp_id}/{run_id}/artifacts"

    if not os.path.exists(f"{dir_location}/figs"):
        os.makedirs(f"{dir_location}/figs")

    # TODO: build classification report and auc plots
    # classification_report_anime = plotly_classification_report_anime(exp_id, run_id, "test")
    # classification_report_anime.write_html(f"{dir_location}/figs/class_report.html")

    emb_fig_2d = plotly_embeddings_anime_2d(exp_id, run_id)
    emb_fig_2d.write_html(f"{dir_location}/figs/embeddings_2d_plot.html")

    emb_fig_3d = plotly_embeddings_anime_3d(exp_id, run_id)
    emb_fig_3d.write_html(f"{dir_location}/figs/embeddings_3d_plot.html")

    pr_anime_fig_train = plotly_precision_recall_curve_anime(exp_id, run_id, "train")
    pr_anime_fig_train.write_html(f"{dir_location}/figs/pr_train_anime.html")

    confusion_matrix_fig_train = plotly_confusion_matrix_anime(exp_id, run_id, "train")
    confusion_matrix_fig_train.write_html(f"{dir_location}/figs/cm_train_anime.html")

    pr_anime_fig_test = plotly_precision_recall_curve_anime(exp_id, run_id, "test")
    pr_anime_fig_test.write_html(f"{dir_location}/figs/pr_test_anime.html")

    confusion_matrix_fig_test = plotly_confusion_matrix_anime(exp_id, run_id, "test")
    confusion_matrix_fig_test.write_html(f"{dir_location}/figs/cm_test_anime.html")


if __name__ == "__main__":
    exp_id = "303267778283029741"
    run_id = "cd7fa3415cc042f1b8f55e054d722bbe"
    build_eval_plots(exp_id, run_id)

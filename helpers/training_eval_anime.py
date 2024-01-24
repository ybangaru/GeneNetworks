"""
This module contains functions to build animation plots for continuous evaluation of training
results.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, roc_curve, classification_report
from sklearn.preprocessing import label_binarize


# TODO: Build sklearn.metrics classification report animation as well

from .mlflow_client_ import (
    MLFLOW_CLIENT,
    read_run_embeddings_df,
    read_run_node_true_pred_labels,
)
from .plotly_helpers import COLORS_LIST
from .experiment_config import ANNOTATION_DICT


def plotly_embeddings_anime_2d(experiment_id, run_id):
    df = read_run_embeddings_df(experiment_id, run_id)

    num_iterations = int(df["Number"].max() / 10)
    cell_types_list = list(ANNOTATION_DICT.values()) + ["Unknown_Z"]
    unique_colors = sorted(cell_types_list)
    color_dict = dict(zip(unique_colors, COLORS_LIST[: len(unique_colors)]))

    df_exploded = pd.DataFrame(
        np.column_stack(
            (
                np.repeat(df["Number"], df["Embedding"].apply(len)),
                np.concatenate(df["Embedding"].apply(lambda x: x[:, 0])),
                np.concatenate(df["Embedding"].apply(lambda x: x[:, 1])),
            )
        ),
        columns=["Number", "Dim0", "Dim1"],
    )
    df_exploded["CellTypes"] = cell_types_list * num_iterations
    df_exploded["ColorCol"] = df_exploded["CellTypes"].map(color_dict)

    fig = px.scatter(
        df_exploded,
        x="Dim0",
        y="Dim1",
        animation_frame="Number",
        color="CellTypes",
        color_discrete_map=dict(zip(df_exploded["CellTypes"], df_exploded["ColorCol"])),
        text="CellTypes",
        size=[5] * df_exploded.shape[0],
        title="Embeddings Visualization Over Training Iterations",
    )

    fig.update_xaxes(range=[df_exploded["Dim0"].min(), df_exploded["Dim0"].max()])
    fig.update_yaxes(range=[df_exploded["Dim1"].min(), df_exploded["Dim1"].max()])

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(frame=dict(duration=50, redraw=True), fromcurrent=True),
                        ],
                    )
                ],
            )
        ]
    )

    return fig


def plotly_embeddings_anime_3d(experiment_id, run_id):
    df = read_run_embeddings_df(experiment_id, run_id)

    num_iterations = int(df["Number"].max() / 10)
    cell_types_list = list(ANNOTATION_DICT.values()) + ["Unknown_Z"]
    unique_colors = sorted(cell_types_list)
    color_dict = dict(zip(unique_colors, COLORS_LIST[: len(unique_colors)]))

    df_exploded = pd.DataFrame(
        np.column_stack(
            (
                np.repeat(df["Number"], df["Embedding"].apply(len)),
                np.concatenate(df["Embedding"].apply(lambda x: x[:, 0])),
                np.concatenate(df["Embedding"].apply(lambda x: x[:, 1])),
                np.concatenate(df["Embedding"].apply(lambda x: x[:, 2])),
            )
        ),
        columns=["Number", "Dim0", "Dim1", "Dim2"],
    )
    df_exploded["CellTypes"] = cell_types_list * num_iterations
    df_exploded["ColorCol"] = df_exploded["CellTypes"].map(color_dict)

    fig = px.scatter_3d(
        df_exploded,
        x="Dim0",
        y="Dim1",
        z="Dim2",
        animation_frame="Number",
        color="CellTypes",
        color_discrete_map=dict(zip(df_exploded["CellTypes"], df_exploded["ColorCol"])),
        text="CellTypes",
        size=[5] * df_exploded.shape[0],
        title="Embeddings Visualization Over Training Iterations",
    )

    fig.update_xaxes(range=[df_exploded["Dim0"].min(), df_exploded["Dim0"].max()])
    fig.update_yaxes(range=[df_exploded["Dim1"].min(), df_exploded["Dim1"].max()])

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(frame=dict(duration=50, redraw=True), fromcurrent=True),
                        ],
                    )
                ],
            )
        ]
    )

    return fig


def create_confusion_matrix(true_labels, pred_labels, labels):
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    return cm_normalized


def plotly_confusion_matrix_anime(experiment_id, run_id):
    df = read_run_node_true_pred_labels(experiment_id, run_id)
    act_labels = [int(item) for item in ANNOTATION_DICT.keys()]
    df["normalized_confusion_matrix"] = df.apply(
        lambda row: create_confusion_matrix(row["true_labels"], row["pred_labels"], labels=act_labels),
        axis=1,
    )

    fig = make_subplots(rows=1, cols=1)  # , subplot_titles=['Confusion Matrix Animation'])

    # Initialize the heatmap with the first confusion matrix

    heatmap = go.Heatmap(
        z=np.nan_to_num(df.loc[0, "normalized_confusion_matrix"], nan=0.0),
        colorscale="Viridis",
        x=list(ANNOTATION_DICT.values()),
        y=list(ANNOTATION_DICT.values()),
    )
    frames = [
        go.Frame(
            data=[
                go.Heatmap(
                    z=np.nan_to_num(df.loc[i, "normalized_confusion_matrix"], nan=0.0),
                    x=list(ANNOTATION_DICT.values()),
                    y=list(ANNOTATION_DICT.values()),
                )
            ],
            name=str(df.loc[i, "Number"]),
        )
        for i in range(len(df))
    ]

    # Add the first frame and initialize the layout
    fig.add_trace(heatmap)
    fig.frames = frames

    # Add a slider for controlling the training iteration
    slider_steps = [
        dict(
            label=str(iteration),
            method="animate",
            args=[
                [str(iteration)],
                {
                    "frame": {"duration": 500, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 300},
                },
            ],
            value=iteration,
        )
        for iteration in df["Number"]
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 300},
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ]
    )

    fig.update_layout(
        sliders=[
            dict(
                steps=slider_steps,
                active=0,
                x=0.1,
                y=0,
                xanchor="left",
                yanchor="top",
                len=0.9,
                pad={"t": 50},
            )
        ]
    )

    # Update layout settings
    fig.update_layout(title_text="Confusion Matrix Animation", showlegend=False)
    return fig


def plotly_precision_recall_curve_anime(experiment_id, run_id):
    class_names = [int(item) for item in list(ANNOTATION_DICT.keys())]
    df = read_run_node_true_pred_labels(experiment_id, run_id, pred_as_dist=True)

    # TODO: add attribute for filtering iterations in read_run_node_true_pred_labels
    df = df[df["Number"] % 20 == 0]

    df["probability_labels"] = df.apply(
        lambda row: row["pred_labels"][np.arange(len(row["true_labels"])), row["true_labels"]], axis=1
    )
    df["true_labels_bin"] = df.apply(lambda row: label_binarize(row["true_labels"], classes=class_names), axis=1)

    plot_data = []

    for item in class_names:
        df[f"precision_recall_bs_{item}"] = df.apply(
            lambda row: precision_recall_curve(row["true_labels_bin"][:, item], row["pred_labels"][:, item]), axis=1
        )
        df[f"roc_curve_bs_{item}"] = df.apply(
            lambda row: roc_curve(row["true_labels_bin"][:, item], row["pred_labels"][:, item]), axis=1
        )
        df[f"precision_{item}"] = df.apply(lambda row: row[f"precision_recall_bs_{item}"][0], axis=1)
        df[f"recall_{item}"] = df.apply(lambda row: row[f"precision_recall_bs_{item}"][1], axis=1)
        df[f"tpr_{item}"] = df.apply(lambda row: row[f"roc_curve_bs_{item}"][0], axis=1)
        df[f"fpr_{item}"] = df.apply(lambda row: row[f"roc_curve_bs_{item}"][1], axis=1)
        # df[f"auc_{item}"] = df.apply(lambda row: auc(row[f"precision_{item}"], row[f"recall_{item}"]), axis=1)

        temp_df = (
            df[[f"precision_{item}", f"recall_{item}", "Number"]]
            .explode([f"precision_{item}", f"recall_{item}"])
            .reset_index(drop=True)
        )
        temp_df = temp_df.rename(
            columns={
                f"precision_{item}": "precision",
                f"recall_{item}": "recall",
            }
        )
        temp_df["precision"] = temp_df["precision"].astype(float)
        temp_df["recall"] = temp_df["recall"].astype(float)
        temp_df["cell_type"] = [ANNOTATION_DICT[str(item)]] * len(temp_df)
        plot_data.append(temp_df)

    unique_colors = sorted(list(ANNOTATION_DICT.values()))
    unique_colors.sort()
    color_dict = dict(zip(unique_colors, COLORS_LIST[: len(unique_colors)]))

    plot_data = pd.concat(plot_data)

    fig = px.line(
        plot_data,
        x="precision",
        y="recall",
        animation_frame="Number",
        color="cell_type",
        color_discrete_map=color_dict,
        labels={"precision": "Precision", "recall": "Recall"},
        title="Precision and Recall for each Cell Type",
    )

    return fig


def main():
    from .local_config import PROJECT_DIR

    exp_id = "303267778283029741"
    my_runs = MLFLOW_CLIENT.search_runs(experiment_ids=[exp_id])
    run_ids = [run.info.run_uuid for run in my_runs]
    run_id = run_ids[0]

    confusion_matrix_fig = plotly_confusion_matrix_anime(exp_id, run_id)
    confusion_matrix_fig.write_html(f"{PROJECT_DIR}/test_confusion_matrix.html")

    emb_fig = plotly_embeddings_anime_2d(exp_id, run_id)
    emb_fig.write_html(f"{PROJECT_DIR}/test_embeddings_2d.html")

    emb_fig_3d = plotly_embeddings_anime_3d(exp_id, run_id)
    emb_fig_3d.write_html(f"{PROJECT_DIR}/test_embeddings_3d.html")


if __name__ == "__main__":
    main()

"""
This module contains functions to build animation plots for continuous evaluation of training
results.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from .mlflow_client_ import (
    MLFLOW_CLIENT,
    read_run_embeddings_df,
    read_run_node_true_pred_labels,
    read_run_annotations,
    read_run_annotation_colors,
    read_run_annotation_mapping,
    read_run_attribute_classification,
)
from .plotly_helpers import COLORS_LIST


def plotly_classification_report_anime(exp_id, run_id, dataset_type):
    df = read_run_node_true_pred_labels(experiment_id=exp_id, run_id=run_id, train_or_test=dataset_type)
    annotation_dict = read_run_annotations(exp_id, run_id)
    df["report"] = df.apply(
        lambda row: classification_report(row["true_labels"], row["pred_labels"], output_dict=True), axis=1
    )
    df["report"] = df.apply(lambda row: pd.DataFrame(row["report"]).reset_index().melt(id_vars=["index"]), axis=1)
    df = df[["Number", "report"]]

    plot_data = []
    for row in df.itertuples():
        temp_df = row.report
        temp_df["Number"] = row.Number
        plot_data.append(temp_df)

    plot_data = pd.concat(plot_data)
    plot_data = plot_data.rename(
        columns={"index": "metric", "value": "score", "Number": "Epoch", "variable": "cell_type"}
    )
    plot_data["cell_type"] = plot_data["cell_type"].replace(annotation_dict)

    # TODO: add attributes for which score, which plot type, also auc plots from sklearn.metrics
    print(plot_data)


def plotly_scatter_anime(
    df, x, y, color_label, title, animation_label, color_discrete_map, labels, z=None, facet_col=None
):
    coords_ = {
        "x": x,
        "y": y,
    }

    if z:
        function_ = px.scatter_3d
        coords_["z"] = z
    else:
        function_ = px.scatter
        # coords_["facet_col"] = facet_col

    fig = function_(
        df,
        **coords_,
        animation_frame=animation_label,
        animation_group=color_label,
        facet_col=facet_col,
        color=color_label,
        color_discrete_map=dict(zip(df[color_label], df[color_discrete_map])),
        text=color_label,
        size=[5] * len(df),
        title=title,
    )

    fig.update_xaxes(autorange=True)
    fig.update_yaxes(autorange=True)

    fig.update_layout(
        transition={"duration": 1},
    )

    return fig


def plotly_embeddings_anime(experiment_id, run_id, spaces=["2d", "3d"], types=["nodes", "edges"]):
    figs_ = {}
    dfs = {}
    # build dataframes for nodes and edge embeddings along with their train iteration number
    for item_ in types:
        dfs[item_] = read_run_embeddings_df(experiment_id, run_id, item_)

    color_dict = read_run_annotation_colors(run_id)
    annotation_emb_reference = read_run_annotation_mapping(run_id)
    edge_emb_reference = read_run_attribute_classification(run_id, attribute_name="edge_types")
    if len(annotation_emb_reference) != len(color_dict):
        annotation_emb_reference["Unknown"] = len(color_dict)
    # sort embeddings dictionaries by values to align with embedding vector order
    annotation_emb_reference = dict(sorted(annotation_emb_reference.items(), key=lambda item: item[1]))
    edge_emb_reference = dict(sorted(edge_emb_reference.items(), key=lambda item: item[1]))

    color_dict_edge = dict(zip(edge_emb_reference.keys(), COLORS_LIST[: len(edge_emb_reference)]))

    dfs_exploded = {}
    for item_ in types:
        for space in spaces:
            curr_cols = ["Number", "Dim0", "Dim1"]
            temp_arrays = (
                np.repeat(dfs[item_]["Number"], dfs[item_]["Embedding"].apply(len)),
                np.concatenate(dfs[item_]["Dim0"]),
                np.concatenate(dfs[item_]["Dim1"]),
            )
            if space == "3d":
                temp_arrays += (np.concatenate(dfs[item_]["Dim2"]),)
                curr_cols.append("Dim2")
            if item_ == "edges":
                temp_arrays += (np.repeat(dfs[item_]["hop_num"], dfs[item_]["Embedding"].apply(len)),)
                curr_cols.append("hop_num")

            dfs_exploded[item_] = pd.DataFrame(np.column_stack(temp_arrays), columns=curr_cols)

            if item_ == "nodes":
                dfs_exploded[item_]["CellTypes"] = (
                    list(annotation_emb_reference.keys()) * dfs_exploded[item_]["Number"].nunique()
                )
                dfs_exploded[item_]["ColorCol"] = dfs_exploded[item_]["CellTypes"].map(color_dict)
                dfs_exploded[item_] = dfs_exploded[item_].sort_values(by=["Number", "CellTypes"])
            else:
                for hop_num in dfs_exploded[item_]["hop_num"].unique():
                    dfs_exploded[item_].loc[dfs_exploded[item_]["hop_num"] == hop_num, "EdgeTypes"] = (
                        list(edge_emb_reference.keys()) * dfs_exploded[item_]["Number"].nunique()
                    )

                dfs_exploded[item_]["ColorCol"] = dfs_exploded[item_]["EdgeTypes"].map(color_dict_edge)
                dfs_exploded[item_] = dfs_exploded[item_].sort_values(by=["Number", "EdgeTypes"])

            file_name = f"embeddings_anime_{item_}_{space}"
            label_col = "CellTypes" if item_ == "nodes" else "EdgeTypes"
            figs_[file_name] = plotly_scatter_anime(
                df=dfs_exploded[item_],
                x="Dim0",
                y="Dim1",
                z="Dim2" if space == "3d" else None,
                color_label=label_col,
                title=f"Training Iterations - {item_} embeddings",
                animation_label="Number",
                color_discrete_map="ColorCol",
                labels=label_col,
                facet_col="hop_num" if item_ == "edges" else None,
            )

    return figs_


def create_confusion_matrix(true_labels, pred_labels, labels):
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    return cm_normalized


def plotly_confusion_matrix_anime(experiment_id, run_id, train_or_test="train"):
    df = read_run_node_true_pred_labels(experiment_id, run_id, train_or_test=train_or_test)
    annotation_dict = read_run_annotation_mapping(run_id)
    # sort annotation dictionary by values to align with confusion matrix order
    annotation_dict = dict(sorted(annotation_dict.items(), key=lambda item: item[1]))
    act_labels = list(annotation_dict.values())
    df["normalized_confusion_matrix"] = df.apply(
        lambda row: create_confusion_matrix(row["true_labels"], row["pred_labels"], labels=act_labels),
        axis=1,
    )

    fig = make_subplots(rows=1, cols=1)  # , subplot_titles=['Confusion Matrix Animation'])

    # Initialize the heatmap with the first confusion matrix
    x_labels = y_labels = list(annotation_dict.keys())

    heatmap = go.Heatmap(
        z=np.nan_to_num(df.loc[0, "normalized_confusion_matrix"], nan=0.0),
        colorscale="Viridis",
        x=x_labels,
        y=y_labels,
    )
    frames = [
        go.Frame(
            data=[
                go.Heatmap(
                    z=np.nan_to_num(df.loc[i, "normalized_confusion_matrix"], nan=0.0),
                    x=x_labels,
                    y=y_labels,
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
    fig_name = f"confusion_matrix_anime_{train_or_test}"
    return {fig_name: fig}


def plotly_precision_recall_curve_anime(experiment_id, run_id, train_or_test="train"):
    annotation_dict = read_run_annotation_mapping(run_id)
    color_dict = read_run_annotation_colors(run_id)

    df = read_run_node_true_pred_labels(experiment_id, run_id, pred_as_dist=True, train_or_test=train_or_test)

    # TODO: add attribute for filtering iterations in read_run_node_true_pred_labels
    df = df[df["Number"] % 20 == 0]

    df["probability_labels"] = df.apply(
        lambda row: row["pred_labels"][np.arange(len(row["true_labels"])), row["true_labels"]], axis=1
    )
    df["true_labels_bin"] = df.apply(
        lambda row: label_binarize(row["true_labels"], classes=list(annotation_dict.values())), axis=1
    )

    plot_data = []

    for key, item in annotation_dict.items():
        df[f"precision_recall_bs_{item}"] = df.apply(
            lambda row: precision_recall_curve(row["true_labels_bin"][:, item], row["pred_labels"][:, item]), axis=1
        )
        # df[f"roc_curve_bs_{item}"] = df.apply(
        #     lambda row: roc_curve(row["true_labels_bin"][:, item], row["pred_labels"][:, item]), axis=1
        # )
        df[f"precision_{item}"] = df.apply(lambda row: row[f"precision_recall_bs_{item}"][0], axis=1)
        df[f"recall_{item}"] = df.apply(lambda row: row[f"precision_recall_bs_{item}"][1], axis=1)
        # df[f"tpr_{item}"] = df.apply(lambda row: row[f"roc_curve_bs_{item}"][0], axis=1)
        # df[f"fpr_{item}"] = df.apply(lambda row: row[f"roc_curve_bs_{item}"][1], axis=1)
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
        temp_df["cell_type"] = [key] * len(temp_df)
        plot_data.append(temp_df)

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

    file_name = f"precision_recall_curve_anime_{train_or_test}"

    return {file_name: fig}


def main():
    from .local_config import PROJECT_DIR

    exp_id = "303267778283029741"
    my_runs = MLFLOW_CLIENT.search_runs(experiment_ids=[exp_id])
    run_ids = [run.info.run_uuid for run in my_runs]
    run_id = run_ids[0]

    confusion_matrix_fig = plotly_confusion_matrix_anime(exp_id, run_id)
    confusion_matrix_fig.write_html(f"{PROJECT_DIR}/test_confusion_matrix.html")

    # emb_fig = plotly_embeddings_anime_2d(exp_id, run_id)
    # emb_fig.write_html(f"{PROJECT_DIR}/test_embeddings_2d.html")

    # emb_fig_3d = plotly_embeddings_anime_3d(exp_id, run_id)
    # emb_fig_3d.write_html(f"{PROJECT_DIR}/test_embeddings_3d.html")


if __name__ == "__main__":
    main()

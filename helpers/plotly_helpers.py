"""
This file contains helper functions for building plots using Plotly.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio
import scanpy as sc
from typing import (
    Union,
    Optional,
)
import numpy as np
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
from sklearn.preprocessing import LabelBinarizer

from .logging_setup import logger

pio.templates.default = "plotly_dark"
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, facecolor="white")

COLORS_LIST = (
    px.colors.qualitative.Bold
    + px.colors.qualitative.Light24
    # + px.colors.qualitative.Vivid
    # + px.colors.qualitative.D3
    # + px.colors.qualitative.Set1
    + px.colors.qualitative.Prism
    # + px.colors.qualitative.Set3
    + px.colors.qualitative.Pastel
    + px.colors.qualitative.Plotly
)


def plotly_pca_categorical(
    adata,
    filters,
    color_key: str,
    show: Optional[bool] = None,
    return_fig: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    x_dim: int = 0,
    y_dim: int = 1,
):
    if "pca" not in adata.obsm.keys() and "X_pca" not in adata.obsm.keys():
        raise KeyError(f"Could not find entry in `obsm` for 'pca'.\n" f"Available keys are: {list(adata.obsm.keys())}.")

    # Get the PCA coordinates and variance explained
    pca_coords = adata.obsm["pca"] if "pca" in adata.obsm.keys() else adata.obsm["X_pca"]
    var_exp = adata.uns["pca"]["variance_ratio"] if "pca" in adata.obsm.keys() or "X_pca" in adata.obsm.keys() else None

    # Create a dataframe of the PCA coordinates with sample names as the index
    pca_df = pd.DataFrame(
        data=pca_coords[:, [x_dim, y_dim]],
        columns=["PC{}".format(x_dim + 1), "PC{}".format(y_dim + 1)],
        index=adata.obs_names,
    )

    # return pca_df
    # Create a list of colors for each data point
    # nan_mask = np.isnan(pca_df).any(axis=1)
    color_list = adata.obs[color_key].astype(str).replace("nan", "Unknown")
    distinct_values = color_list.unique()
    distinct_values.sort()
    color_mapping = dict(zip(distinct_values, COLORS_LIST[: len(distinct_values)]))
    pca_df["category"] = color_list

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="category",
        color_discrete_map=color_mapping
        # color=color_list.map(color_mapping),
    )

    # Set the axis labels
    x_label = (
        "PC{} ({}%)".format(x_dim + 1, round(var_exp[x_dim] * 100, 2))
        if var_exp is not None
        else "PC{}".format(x_dim + 1)
    )
    y_label = (
        "PC{} ({}%)".format(y_dim + 1, round(var_exp[y_dim] * 100, 2))
        if var_exp is not None
        else "PC{}".format(y_dim + 1)
    )
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=800,
        # width = 1200,
        legend={
            "title": f"{color_key} categories",
            "itemsizing": "constant",
            "itemwidth": 30,
        },
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        title=f"scRNA PCA({color_key} representation) using {filters} observations",
    )

    fig.update_traces(marker_size=2.5)

    # Show or save the plot
    if save is not None:
        fig.write_html(save)
    if show is not False:
        fig.show()

    # Return the plotly figure object if requested
    if return_fig is True:
        return fig


def plotly_pca_numerical(
    adata,
    filters,
    color_key: str,
    show: Optional[bool] = None,
    return_fig: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    x_dim: int = 0,
    y_dim: int = 1,
):
    if "pca" not in adata.obsm.keys() and "X_pca" not in adata.obsm.keys():
        raise KeyError(f"Could not find entry in `obsm` for 'pca'.\n" f"Available keys are: {list(adata.obsm.keys())}.")

    # Get the PCA coordinates and variance explained
    pca_coords = adata.obsm["pca"] if "pca" in adata.obsm.keys() else adata.obsm["X_pca"]
    var_exp = adata.uns["pca"]["variance_ratio"] if "pca" in adata.obsm.keys() or "X_pca" in adata.obsm.keys() else None

    # Create a dataframe of the PCA coordinates with sample names as the index
    pca_df = pd.DataFrame(
        data=pca_coords[:, [x_dim, y_dim]],
        columns=["PC{}".format(x_dim + 1), "PC{}".format(y_dim + 1)],
        index=adata.obs_names,
    )

    # Create a list of colors for each data point
    color_list = adata.obs[color_key]

    fig = px.scatter(
        x=pca_df.iloc[:, 0],
        y=pca_df.iloc[:, 1],
        color=color_list,
    )

    # Set the axis labels
    x_label = (
        "PC{} ({}%)".format(x_dim + 1, round(var_exp[x_dim] * 100, 2))
        if var_exp is not None
        else "PC{}".format(x_dim + 1)
    )
    y_label = (
        "PC{} ({}%)".format(y_dim + 1, round(var_exp[y_dim] * 100, 2))
        if var_exp is not None
        else "PC{}".format(y_dim + 1)
    )
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=800,
        # width = 1200,
        legend={
            "title": f"{color_key} categories",
            "itemsizing": "constant",
            "itemwidth": 30,
        },
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        title=f"scRNA PCA({color_key} representation) using {filters} observations",
    )

    fig.update_traces(marker_size=2.5)

    # Show or save the plot
    if save is not None:
        fig.write_html(save)
    if show is not False:
        fig.show()

    # Return the plotly figure object if requested
    if return_fig is True:
        return fig


def plotly_umap_categorical(adata, chosen_key_value, color_key, from_annotations):
    fig = None

    if from_annotations is True:
        umap_coords = adata.obs[["UMAP_1", "UMAP_2"]].values
        color_list = adata.obs[color_key]
        if "all" in chosen_key_value:
            plot_title = "UMAP source data, using known observations"
        else:
            plot_title = f"UMAP source data, using {chosen_key_value} observations"

    else:
        umap_coords = adata.obsm["X_umap"]
        color_list = adata.obs[color_key].astype(str).replace("nan", "Unknown")
        if "all" in chosen_key_value:
            plot_title = "UMAPs constructed from known observations"
        else:
            plot_title = f"UMAP constructed using {chosen_key_value} observations"

    umap_df = pd.DataFrame(
        data=umap_coords[:, [0, 1]],
        columns=["UMAP_{}".format(0 + 1), "UMAP_{}".format(1 + 1)],
        index=adata.obs_names,
    )

    distinct_values = color_list.unique()
    distinct_values.sort()
    color_mapping = dict(zip(distinct_values, COLORS_LIST[: len(distinct_values)]))
    umap_df["category"] = color_list

    fig = px.scatter(
        umap_df,
        x="UMAP_1",
        y="UMAP_2",
        color="category",
        color_discrete_map=color_mapping,
        category_orders={"category": distinct_values},
    )

    x_label = "UMAP1"
    y_label = "UMAP2"
    fig.update_layout(
        title=f"{plot_title}",
        height=800,  # width=800,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend={
            "title": f"{color_key} categories",
            "itemsizing": "constant",
            "itemwidth": 30,
        },
        xaxis_showgrid=False,
        yaxis_showgrid=False,
    )
    fig.update_traces(marker=dict(size=2.0))
    return fig


def plotly_umap_numerical(adata, chosen_key_value, color_key, from_annotations):
    fig = None

    if from_annotations is True:
        umap_coords = adata.obs[["UMAP_1", "UMAP_2"]].values
        if "all" in chosen_key_value:
            plot_title = "UMAPs source data, known observations"
        else:
            plot_title = f"UMAP source data, {chosen_key_value}"

    else:
        umap_coords = adata.obsm["X_umap"]
        if "all" in chosen_key_value:
            plot_title = f"UMAPs constructed, {chosen_key_value} observations"
        else:
            plot_title = f"UMAP constructed, {chosen_key_value} observations"

    color_list = adata.obs[color_key]

    fig = px.scatter(
        x=umap_coords[:, 0],
        y=umap_coords[:, 1],
        color=color_list,
        labels={"color": f"{color_key}"},
    )

    x_label = "UMAP1"
    y_label = "UMAP2"
    fig.update_traces(marker=dict(size=2.5))
    fig.update_layout(
        title=f"{plot_title}",
        height=800,  # width=800,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend={
            "title": f"{color_key}",
            "itemsizing": "constant",
            "itemwidth": 30,
        },
        xaxis_showgrid=False,
        yaxis_showgrid=False,
    )
    return fig


def plotly_spatial_scatter_categorical(test_boundaries, color_column, color_dict=None):
    fig = go.Figure()

    x_label = "X-position"
    y_label = "Y-position"

    if color_dict is None:
        # Get unique values of color_key
        unique_colors = sorted(color_column.unique())
        unique_colors.sort()
        color_dict = dict(zip(unique_colors, COLORS_LIST[: len(unique_colors)]))
    else:
        unique_colors = list(color_dict.keys())
        unique_colors.sort()

    legend_boolen_set = set()

    for ordered_cluster in unique_colors:
        temp_boundaries = color_column[color_column == ordered_cluster]
        # for key, value in temp_boundaries.items():
        for key in temp_boundaries.index:
            try:
                x = test_boundaries[key][:, 0]
                y = test_boundaries[key][:, 1]
                if color_column.loc[key] in legend_boolen_set:
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            fill="toself",
                            mode="lines",
                            line=dict(color=color_dict[color_column.loc[key]]),
                            legendgroup=color_column.loc[key],
                            showlegend=False,
                            name=f"{color_column.loc[key]}",
                            text=f"{key}",
                            hovertemplate="<b>%{text}</b>",
                        ),
                    )
                else:
                    legend_boolen_set.add(color_column.loc[key])
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            fill="toself",
                            mode="lines",
                            line=dict(color=color_dict[color_column.loc[key]]),
                            legendgroup=color_column.loc[key],
                            name=f"{color_column.loc[key]}",
                            text=f"{key}",
                            hovertemplate="<b>%{text}</b>",
                        ),
                    )
            except KeyError as e:
                logger.info(f"KeyError: {e}")
                continue

    # Set layout properties
    fig.update_layout(
        title=f"{color_column.name} representation",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=800,
        legend=dict(
            itemsizing="constant",
            title_font_family="Courier New",
        ),
    )

    return fig


def plotly_spatial_scatter_numerical(test_boundaries, color_column):
    fig = go.Figure()
    min_color_value = color_column.min()
    max_color_value = color_column.max()

    x_label = "X-position"
    y_label = "Y-position"

    normalized_color_values = {}

    for key, value in test_boundaries.items():
        try:
            color_value = color_column[key]
            normalized_color = (color_value - min_color_value) / (max_color_value - min_color_value)
            color_hex = mcolors.rgb2hex((normalized_color, 0.5, 0.5))
            normalized_color_values[normalized_color] = color_hex  # Collect color values in the list
            x = value[:, 0]
            y = value[:, 1]
            hover_text = f"{color_value:.2f}"

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    fill="toself",
                    mode="lines",
                    line=dict(color=color_hex),
                    showlegend=False,
                    name=hover_text,
                ),
            )
        except KeyError as e:
            print(f"KeyError: {e}")
            continue
    colorscale_min_value = min(normalized_color_values, key=normalized_color_values.get)
    colorscale_max_value = max(normalized_color_values, key=normalized_color_values.get)

    colorbar_trace = go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            colorscale=[
                [0, normalized_color_values[colorscale_min_value]],
                [1, normalized_color_values[colorscale_max_value]],
            ],  # Custom color scale using color values
            showscale=True,
            cmin=min_color_value,
            cmax=max_color_value,
            color=[min_color_value, max_color_value],
            colorbar=dict(thickness=20, tickmode="auto", ticks="outside", outlinewidth=0),
        ),
        hoverinfo="none",
        showlegend=False,
    )

    fig.add_trace(colorbar_trace)  # Add colorbar trace to the figure

    # Set the layout
    fig.update_layout(
        title=f"{color_column.name} representation",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=800,
    )

    return fig


def plotly_spatial_scatter_pca(pipeline_instance, test_boundaries, component):
    color_column = pd.Series(
        pipeline_instance.data.obsm["X_pca"][:, component],
        index=pipeline_instance.data.obs.index,
        name=f"PCA_component_{component}",
    )
    fig = plotly_spatial_scatter_numerical(test_boundaries, color_column)
    return fig


def plotly_spatial_scatter_umap(pipeline_instance, test_boundaries, component):
    color_column = pd.Series(
        pipeline_instance.data.obsm["X_umap"][:, component],
        index=pipeline_instance.data.obs.index,
        name=f"UMAP_component_{component}",
    )
    fig = plotly_spatial_scatter_numerical(test_boundaries, color_column)
    return fig


def plotly_spatial_scatter_edges(G, node_color_col, edge_info, color_dict=None):
    """Plot dot-line graph for the cellular graph using Plotly

    Args:
        G (nx.Graph): full cellular graph of the region
        node_colors (list): list of node colors. Defaults to None.
    """
    x_label = "X-position"
    y_label = "Y-position"

    # Extract basic node attributes
    node_coords = np.array([G.nodes[n]["center_coord"] for n in G.nodes])
    # Extract cell IDs of nodes for hover text
    node_cell_ids = [G.nodes[n]["cell_id"] for n in G.nodes]

    if not color_dict:
        # Get unique values of color_key
        unique_colors = sorted(node_color_col.unique())
        unique_colors.sort()
        color_dict = dict(zip(unique_colors, COLORS_LIST[: len(unique_colors)]))
        color_dict["Unknown"] = COLORS_LIST[len(unique_colors)]
    else:
        unique_colors = list(color_dict.keys())
        unique_colors.sort()

    node_colors = []
    node_names = []
    for n in G.nodes:
        try:
            node_names.append(node_color_col.loc[G.nodes[n]["cell_id"]])
            temp_node_color = color_dict[node_color_col.loc[G.nodes[n]["cell_id"]]]
            node_colors.append(temp_node_color)
        except KeyError:
            node_names.append("Unknown")
            node_colors.append(color_dict["Unknown"])

    # Create edges for Plotly Scatter plot
    edge_x_solid = []
    edge_y_solid = []
    edge_x_dash = []
    edge_y_dash = []
    for i, j, edge_type in G.edges.data():
        xi, yi = G.nodes[i]["center_coord"]
        xj, yj = G.nodes[j]["center_coord"]

        if edge_type["edge_type"] == "neighbor":
            edge_x_solid.extend([xi, xj, None])  # Add None to create a break in the line
            edge_y_solid.extend([yi, yj, None])
        else:
            edge_x_dash.extend([xi, xj, None])  # Add None to create a break in the line
            edge_y_dash.extend([yi, yj, None])

    edge_solid_trace = go.Scatter(
        x=edge_x_solid,
        y=edge_y_solid,
        mode="lines",
        line=dict(width=0.8, color="white", dash="solid"),
        name="Neighbor Edges",
        showlegend=True,
    )

    edge_dash_trace = go.Scatter(
        x=edge_x_dash,
        y=edge_y_dash,
        mode="lines",
        line=dict(width=0.8, color="gold", dash="dash"),
        name="Distant Edges",
        showlegend=True,
    )

    # Create a dictionary to store node traces for each cell type
    node_traces = {}

    for cell_type in unique_colors:
        mask = node_color_col.loc[node_cell_ids] == cell_type
        cell_node_trace = go.Scatter(
            x=node_coords[mask, 0],
            y=node_coords[mask, 1],
            mode="markers",
            marker=dict(
                size=8,
                color=color_dict[cell_type],
                line=dict(width=0.5, color=color_dict[cell_type]),
            ),
            name=str(cell_type),  # Set the name for the legend entry
        )
        node_traces[cell_type] = cell_node_trace

    # Create the figure and add both traces
    fig = go.Figure(data=list(node_traces.values()))
    fig.add_traces([edge_solid_trace, edge_dash_trace])

    # Set axis limits based on node coordinates
    x_min, x_max = node_coords[:, 0].min(), node_coords[:, 0].max()
    y_min, y_max = node_coords[:, 1].min(), node_coords[:, 1].max()

    # Add some padding to the axis limits
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05

    fig.update_xaxes(range=[x_min - x_padding, x_max + x_padding])
    fig.update_yaxes(range=[y_min - y_padding, y_max + y_padding])

    # Remove axis ticks and labels for cleaner visualization
    fig.update_xaxes(showticklabels=True, showgrid=False)
    fig.update_yaxes(showticklabels=True, showgrid=False)

    # Set the title and layout properties
    fig.update_layout(
        title=f"{node_color_col.name} node - {edge_info} edges representation",
        title_x=0.5,
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode="closest",
        showlegend=True,
        height=800,
    )

    return fig


def plotly_spatial_scatter_subgraph(df, subgraph_edges=None):
    fig = go.Figure()

    x_label = "X-position"
    y_label = "Y-position"

    legend_boolen_set = set()
    for key, value in df.iterrows():
        try:
            x = value["boundary"][:, 0]
            y = value["boundary"][:, 1]
            if value["cell_type"] in legend_boolen_set:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        fill="toself",
                        mode="lines",
                        line=dict(color=value["cell_color"]),
                        legendgroup=f"{value['cell_type']}",
                        showlegend=False,
                        name=f"{value['cell_type']}",
                        text=f"{value['cell_id']} - {value['cell_type']}",
                        hovertemplate="<b>%{text}</b>",
                    ),
                )
            else:
                legend_boolen_set.add(value["cell_type"])
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        fill="toself",
                        mode="lines",
                        line=dict(color=value["cell_color"]),
                        legendgroup=f"{value['cell_type']}",
                        name=f"{value['cell_type']}",
                        text=f"{value['cell_id']} - {value['cell_type']}",
                        hovertemplate="<b>%{text}</b>",
                    ),
                )
        except KeyError as e:
            logger.info(f"KeyError: {e}")
            continue

    # Add edges to the plot if subgraph_edges is provided
    if subgraph_edges is not None:
        edge_types_design = {
            "neighbor": "solid",
            "distant": "dash",
            "self": "dot",
        }
        edge_types_color = {
            "neighbor": "white",
            "distant": "gold",
            "self": "gray",
        }
        for design_type in edge_types_design:
            filtered_edges = subgraph_edges[subgraph_edges["edge_type"] == design_type]
            # Extract coordinates from the DataFrame using vectorized operations
            if not filtered_edges.empty:
                for other_edge in list(filtered_edges["edge_type"].unique()):
                    filtered_x_edges = []
                    filtered_y_edges = []
                    other_edge_df = filtered_edges[filtered_edges["edge_type"] == other_edge]

                    for indx, row in other_edge_df.iterrows():
                        xi, yi = row["from_loc"]
                        xj, yj = row["to_loc"]
                        filtered_x_edges.extend([xi, xj, None])
                        filtered_y_edges.extend([yi, yj, None])

                    edge_trace = go.Scatter(
                        x=filtered_x_edges,
                        y=filtered_y_edges,
                        mode="lines",
                        line=dict(width=0.6, color=edge_types_color[design_type], dash=edge_types_design[design_type]),
                        name=f"{design_type} edges",
                        showlegend=True,
                        opacity=0.8,
                    )
                    fig.add_trace(edge_trace)

    # Set layout properties
    fig.update_layout(
        title="cell type representation",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=600,
        # width=500,
        legend=dict(
            itemsizing="constant",
            title_font_family="Courier New",
        ),
    )

    return fig


def plotly_precision_recall_curve(y_true, y_scores, class_names, title="Precision-Recall Curve"):
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)

    fig = go.Figure()

    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        pr_auc = auc(recall, precision)

        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"{class_names[i]} (AUC = {pr_auc:.2f})",
            )
        )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Recall"),
        yaxis=dict(title="Precision"),
        #   legend=dict(x=0, y=1, traceorder='normal')
    )
    return fig


def plotly_confusion_matrix(y_true, y_pred, labels, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # Normalize confusion matrix

    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="True"),
        x=class_names,
        y=class_names,
        color_continuous_scale="Viridis",
        title=title,
        origin="upper",
    )

    fig.update_layout(xaxis=dict(side="top"))
    fig.update_layout(coloraxis_colorbar=dict(title="Normalized Count"))

    return fig


def plotly_node_embeddings_2d(embeddings, labels, class_names, title="Node Embeddings in 2D"):
    # Map numeric labels to class names
    label_names = [class_names[label] for label in labels]

    df = pd.DataFrame(
        {
            "X": embeddings[:, 0],
            "Y": embeddings[:, 1],
            "Label": label_names,
        }  # Use the mapped class names as labels
    )

    fig = px.scatter(
        df,
        x="X",
        y="Y",
        color="Label",
        hover_name="Label",
        title=title,
        category_orders={"Label": class_names},
    )

    return fig


def plotly_node_embeddings_3d(embeddings, labels, class_names, title="Node Embeddings in 3D"):
    # Map numeric labels to class names
    label_names = [class_names[label] for label in labels]

    df = pd.DataFrame(
        {
            "X": embeddings[:, 0],
            "Y": embeddings[:, 1],
            "Z": embeddings[:, 2],  # Add Z coordinate for the 3D plot
            "Label": label_names,
        }
    )

    fig = px.scatter_3d(
        df,
        x="X",
        y="Y",
        z="Z",
        color="Label",
        hover_name="Label",
        title=title,
        category_orders={"Label": class_names},
    )

    return fig


if __name__ == "__main__":
    from .data import CellularGraphDataset
    from .local_config import PROJECT_DIR

    dataset_root = f"{PROJECT_DIR}/data/example_dataset"
    dataset_kwargs = {
        "transform": [],
        "pre_transform": None,
        "raw_folder_name": "graph",  # os.path.join(dataset_root, "graph") is the folder where we saved nx graphs
        "processed_folder_name": "tg_graph",  # processed dataset files will be stored here
        "node_features": [
            "cell_type",
            "volume",
            "biomarker_expression",
            "neighborhood_composition",
            "center_coord",
        ],  # There are all the cellular features that we want the dataset to compute
        "edge_features": ["edge_type", "distance"],  # edge (cell pair) features
        "subgraph_size": 3,  # indicating we want to sample 3-hop subgraphs from these regions (for training/inference), this is a core parameter for SPACE-GM.
        "subgraph_source": "on-the-fly",
        "subgraph_allow_distant_edge": True,
        "subgraph_radius_limit": 400.0,
    }

    feature_kwargs = {
        "biomarker_expression_process_method": "linear",
        "biomarker_expression_lower_bound": 0,
        "biomarker_expression_upper_bound": 18,
        "neighborhood_size": 10,
    }
    dataset_kwargs.update(feature_kwargs)

    logger.info("Loading CellularGraphDataset for the following comfig...")
    logger.info(dataset_kwargs)
    dataset = CellularGraphDataset(dataset_root, **dataset_kwargs)
    logger.info(dataset)

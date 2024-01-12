import plotly.express as px
from plotly.subplots import make_subplots
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

pio.templates.default = "plotly_dark"
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, facecolor="white")

COLORS_FROM_PLOTLY = (
    px.colors.qualitative.Bold
    + px.colors.qualitative.Dark24
    # + px.colors.qualitative.Vivid
    # + px.colors.qualitative.D3
    + px.colors.qualitative.Set1
    + px.colors.qualitative.Set2
    + px.colors.qualitative.Set3    
)


def plotly_spatial_scatter_subgraph(test_boundaries, color_column, subgraph_edges=None):

    fig = go.Figure()

    x_label = "X-position"
    y_label = "Y-position"

    # Get unique values of color_key
    unique_colors = sorted(color_column.unique())

    unique_colors.sort()
    color_dict = dict(zip(unique_colors, COLORS_FROM_PLOTLY[:len(unique_colors)]))
    color_dict['Unknown'] = COLORS_FROM_PLOTLY[len(unique_colors)]

    legend_boolen_set = set()

    for key, value in test_boundaries.items():
        try:
            x = value[:, 0]
            y = value[:, 1]
            
            if color_column.loc[key] in legend_boolen_set:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        fill="toself",
                        mode="lines",
                        line=dict(color=color_dict[color_column.loc[key]]),
                        legendgroup=f"{color_column.loc[key]}",
                        showlegend=False,
                        name=f"{color_column.loc[key]}",
                        text=f"{key}",
                        hovertemplate='<b>%{text}</b>',
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
                        legendgroup=f"{color_column.loc[key]}",
                        name=f"{color_column.loc[key]}",
                        text=f"{key}",
                        hovertemplate='<b>%{text}</b>',
                    ),
                )
                
            # Add edges to the plot if subgraph_edges is provided
            if subgraph_edges:
                for edge in subgraph_edges:
                    x_edge = [value[edge[0], 0], value[edge[1], 0]]
                    y_edge = [value[edge[0], 1], value[edge[1], 1]]
                    fig.add_trace(
                        go.Scatter(
                            x=x_edge,
                            y=y_edge,
                            mode="lines",
                            line=dict(color="gray"),  # Adjust the edge color as needed
                            showlegend=False,
                            hoverinfo="none",
                        ),
                    )
        except KeyError as e:
            print(f"KeyError: {e}")
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
        raise KeyError(
            f"Could not find entry in `obsm` for 'pca'.\n"
            f"Available keys are: {list(adata.obsm.keys())}."
        )

    # Get the PCA coordinates and variance explained
    pca_coords = (
        adata.obsm["pca"] if "pca" in adata.obsm.keys() else adata.obsm["X_pca"]
    )
    var_exp = (
        adata.uns["pca"]["variance_ratio"]
        if "pca" in adata.obsm.keys() or "X_pca" in adata.obsm.keys()
        else None
    )

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
    color_mapping = dict(zip(distinct_values, COLORS_FROM_PLOTLY[:len(distinct_values)]))
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
        raise KeyError(
            f"Could not find entry in `obsm` for 'pca'.\n"
            f"Available keys are: {list(adata.obsm.keys())}."
        )

    # Get the PCA coordinates and variance explained
    pca_coords = (
        adata.obsm["pca"] if "pca" in adata.obsm.keys() else adata.obsm["X_pca"]
    )
    var_exp = (
        adata.uns["pca"]["variance_ratio"]
        if "pca" in adata.obsm.keys() or "X_pca" in adata.obsm.keys()
        else None
    )

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


def plotly_umap_categorical(
    adata, chosen_key_value, color_key, from_annotations
):
    fig = None

    if from_annotations is True:
        umap_coords = adata.obs[["UMAP_1", "UMAP_2"]].values
        color_list = adata.obs[color_key]
        if "all" in chosen_key_value:
            plot_title = "UMAP source data, using known observations"
        else:
            plot_title = (
                f"UMAP source data, using {chosen_key_value} observations"
            )

    else:
        umap_coords = adata.obsm["X_umap"]
        color_list = adata.obs[color_key].astype(str).replace("nan", "Unknown")
        if "all" in chosen_key_value:
            plot_title = "UMAPs constructed from known observations"
        else:
            plot_title = (
                f"UMAP constructed using {chosen_key_value} observations"
            )

    umap_df = pd.DataFrame(
        data=umap_coords[:, [0, 1]],
        columns=["UMAP_{}".format(0 + 1), "UMAP_{}".format(1 + 1)],
        index=adata.obs_names,
    )

    distinct_values = color_list.unique()
    distinct_values.sort()
    color_mapping = dict(zip(distinct_values, COLORS_FROM_PLOTLY[:len(distinct_values)]))
    umap_df["category"] = color_list

    fig = px.scatter(
        umap_df,
        x="UMAP_1",
        y="UMAP_2",
        color="category",
        color_discrete_map=color_mapping,  
        category_orders={"category": distinct_values}
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


def plotly_umap_categorical_old(
    adata, chosen_key_value, color_key, from_annotations
):
    fig = None

    if from_annotations is True:
        umap_coords = adata.obs[["UMAP_1", "UMAP_2"]].values
        color_list = adata.obs[color_key]
        if "all" in chosen_key_value:
            plot_title = "UMAP source data, using known observations"
        else:
            plot_title = (
                f"UMAP source data, using {chosen_key_value} observations"
            )

    else:
        umap_coords = adata.obsm["X_umap"]
        color_list = adata.obs[color_key].astype(str).replace("nan", "Unknown")
        if "all" in chosen_key_value:
            plot_title = "UMAPs constructed from known observations"
        else:
            plot_title = (
                f"UMAP constructed using {chosen_key_value} observations"
            )

    # Get unique values of color_key
    unique_colors = color_list.unique()

    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[
            [{"secondary_y": False}],
        ],
    )

    # Get a list of discrete colors from Plotly
    colors_from_plotly = (
        px.colors.qualitative.Bold
        + px.colors.qualitative.Pastel
        + px.colors.qualitative.Vivid
    )
    # print(colors_from_plotly)
    # colors_from_plotly = pc.DEFAULT_PLOTLY_COLORS
    # Create a dictionary that maps each unique string to a color
    color_dict = {}
    for i, string in enumerate(unique_colors):
        color_dict[string] = colors_from_plotly[i % len(colors_from_plotly)]

    # Iterate over unique colors to create trace for each color
    for color in unique_colors:
        color_mask = color_list == color
        fig.add_trace(
            go.Scatter(
                x=umap_coords[color_mask, 0],
                y=umap_coords[color_mask, 1],
                mode="markers",
                marker_color=[color_dict[c] for c in color_list[color_mask]],
                name=str(color),
            )
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
    fig.update_traces(marker=dict(size=2.5))
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


def plotly_spatial_scatter_categorical(test_boundaries, color_column):
    fig = go.Figure()

    x_label = "X-position"
    y_label = "Y-position"

    # Get unique values of color_key
    unique_colors = sorted(color_column.unique())

    unique_colors.sort()
    color_dict = dict(zip(unique_colors, COLORS_FROM_PLOTLY[:len(unique_colors)]))

    legend_boolen_set = set()

    for ordered_cluster in unique_colors:
        temp_boundaries = color_column[color_column == ordered_cluster]
        # for key, value in temp_boundaries.items():
        for key in temp_boundaries.index:
            try:
                # x = value[0, :, 0]
                # y = value[0, :, 1]
                x = test_boundaries[key][:, 0]
                y = test_boundaries[key][:, 1]
                # x = value[:, 0]
                # y = value[:, 1]
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
                            hovertemplate='<b>%{text}</b>',
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
                            hovertemplate='<b>%{text}</b>',
                        ),
                    )
            except KeyError as e:
                # print(f"KeyError: {e}")
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
            normalized_color = (color_value - min_color_value) / (
                max_color_value - min_color_value
            )
            color_hex = mcolors.rgb2hex((normalized_color, 0.5, 0.5))
            normalized_color_values[
                normalized_color
            ] = color_hex  # Collect color values in the list
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
    colorscale_min_value = min(
        normalized_color_values, key=normalized_color_values.get
    )
    colorscale_max_value = max(
        normalized_color_values, key=normalized_color_values.get
    )

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
            colorbar=dict(
                thickness=20, tickmode="auto", ticks="outside", outlinewidth=0
            ),
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


def plotly_spatial_scatter_edges(G, node_color_col, edge_info):
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

    # Get unique values of color_key
    unique_colors = sorted(node_color_col.unique())

    unique_colors.sort()
    color_dict = dict(zip(unique_colors, COLORS_FROM_PLOTLY[:len(unique_colors)]))
    color_dict['Unknown'] = COLORS_FROM_PLOTLY[len(unique_colors)]

    node_colors = []
    node_names = []
    for n in G.nodes:
        try:
            node_names.append(node_color_col.loc[G.nodes[n]["cell_id"]])
            temp_node_color = color_dict[node_color_col.loc[G.nodes[n]["cell_id"]]]
            node_colors.append(temp_node_color)
        except KeyError:
            node_names.append('Unknown')
            node_colors.append(color_dict['Unknown'])

    # Create edges for Plotly Scatter plot
    edge_x = []
    edge_y = []
    edge_colors = []
    edge_dashes = []    
    for i, j, edge_type in G.edges.data():
        xi, yi = G.nodes[i]["center_coord"]
        xj, yj = G.nodes[j]["center_coord"]

        if edge_type["edge_type"] == "neighbor":
            edge_x.extend(
                [xi, xj, None]
            )  # Add None to create a break in the line
            edge_y.extend([yi, yj, None])
            edge_colors.append("brown")
            edge_dashes.append("solid")
        else:
            edge_x.extend(
                [xi, xj, None]
            )  # Add None to create a break in the line
            edge_y.extend([yi, yj, None])
            edge_colors.append("gold")
            edge_dashes.append("dash")

    # Create a dictionary to store node traces for each cell type
    node_traces = {}
    
    for cell_type in unique_colors:

        mask = node_color_col.loc[node_cell_ids] == cell_type
        cell_node_trace = go.Scatter(
            x=node_coords[mask, 0],
            y=node_coords[mask, 1],
            mode="markers",
            marker=dict(
                size=8, color=color_dict[cell_type], line=dict(width=0.5, color=color_dict[cell_type])
            ),
            #text=[name for i, name in enumerate(node_names) if mask[i]],
            #hoverinfo="text",
            name=str(cell_type),  # Set the name for the legend entry
        )
        node_traces[cell_type] = cell_node_trace


    # ind_neighbors = np.where(np.array(edge_colors) == "brown")[0]
    # ind_distinct = np.where(np.array(edge_colors) == "gold")[0]

    # Create scatter plot for edges (lines)
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="wheat"), #edge_colors, dash=edge_dashes
        hoverinfo="none",
        name="Edges",
        showlegend=True,
    )

    # Create the figure and add both traces
    fig = go.Figure(data=list(node_traces.values()) + [edge_trace])

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


class VisualizePipeline:

    def __init__(self, info_dict):
        self.info_cluster = info_dict
        self.file_name = self.info_cluster["data_file_name"]
        self.state = self.info_cluster["state"]
        self.data_filter_name = self.info_cluster["data_filter_name"]
        self.names_list = self.info_cluster["names_list"]
        # self.filters = (self.info_cluster[0], self.info_cluster[1])

        self.all_generated_pcas = {}
        self.all_generated_umaps = {}
        # self.all_generated_spatial = {}

        self.data = self.read_data()
        self.categorical_columns = self.data.obs.select_dtypes(
            include=["category", "object", "bool"]
        ).columns
        self.numerical_columns = self.data.obs.select_dtypes(
            include=["float32", "int32", "float64", "int64"]
        ).columns

    def read_data(self):
        # if not self.data:
        data = sc.read_h5ad(self.file_name)
        return data

    def generate_pca_plots(self):
        for cate_item in self.categorical_columns:
            fig = plotly_pca_categorical(
                self.data,
                self.data_filter_name,
                color_key=cate_item,
                return_fig=True,
                x_dim=0,
                y_dim=1,
                show=False,
            )
            self.all_generated_pcas[f"{cate_item}"] = fig
        for num_item in self.numerical_columns:
            fig = plotly_pca_numerical(
                self.data,
                self.data_filter_name,
                color_key=num_item,
                return_fig=True,
                x_dim=0,
                y_dim=1,
                show=False,
            )
            self.all_generated_pcas[f"{num_item}"] = fig

    def generate_umap_plots(self):

        for cate_item in self.categorical_columns:
            # fig_true = plotly_umap_categorical(
            #     self.data,
            #     self.data_filter_name,
            #     color_key=cate_item,
            #     from_annotations=True,
            # )
            # self.all_generated_umaps[f"UMAP_{cate_item}_from_annot"] = fig_true
            fig_false = plotly_umap_categorical(
                self.data,
                self.data_filter_name,
                color_key=cate_item,
                from_annotations=False,
            )
            self.all_generated_umaps[f"{cate_item}"] = fig_false
        for num_item in self.numerical_columns:
            # fig_true = plotly_umap_numerical(
            #     self.data,
            #     self.data_filter_name,
            #     color_key=num_item,
            #     from_annotations=True,
            # )
            # self.all_generated_umaps[f"UMAP_{num_item}_from_annot"] = fig_true
            fig_false = plotly_umap_numerical(
                self.data,
                self.data_filter_name,
                color_key=num_item,
                from_annotations=False,
            )
            self.all_generated_umaps[f"{num_item}"] = fig_false

    def filter_by_centroid_coordinates(self, spatial_loader):
        return spatial_loader.filter_by_centroid_coordinates(self.data)

    def get_boundaries_for_indices(self, spatial_loader):
        return spatial_loader.get_boundaries_for_indices(self.data)

    def get_boundaries_of_one_fov(self, spatial_loader, fov_value):
        return spatial_loader.get_boundaries_of_one_fov(self.data, fov_value)

    def get_boundaries_of_multiple_fov(self, spatial_loader, fov_values):
        return spatial_loader.get_boundaries_of_multiple_fov(self.data, fov_values)            
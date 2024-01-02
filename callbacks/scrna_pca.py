import numpy as np
import pandas as pd
from dash import Input, Output, State, callback
from .plots_parallel import generate_plots
from helpers import BoundaryDataLoader, plotly_spatial_scatter_numerical, SLICE_PIXELS_EDGE_CUTOFF, plotly_pca_categorical, plotly_pca_numerical
from spacegm import build_graph_from_cell_coords, assign_attributes
from .ann_data_store import ann_data, categorical_columns, numerical_columns

@callback(
    Output("pca_liverslice-dropdown", "options"),
    Output("pca_color-dropdown", "options"),
    # Output("pca_x-segment-dropdown", "options"),
    # Output("pca_y-segment-dropdown", "options"),
    Input("pca_plot-container", "children"),
    prevent_initial_call=False
)
def update_dropdown_options(_):
    
    liverslice_options = ["Liver1Slice1", "Liver1Slice2", "Liver2Slice1", "Liver2Slice2"]
    temp = categorical_columns.copy()
    temp.extend(numerical_columns)
    color_options = temp
    # x_segment_options = [{"label": file, "value": file} for file in list(range(20))]
    # y_segment_options = [{"label": file, "value": file} for file in list(range(20))]
    
    return liverslice_options, color_options#, x_segment_options, y_segment_options



@callback(
    Output("pca_plot-container", "figure"),
    Input("pca_load-button", "n_clicks"),
    # Input("ann-data-store", "data"),
    State("pca_liverslice-dropdown", "value"),
    State("pca_color-dropdown", "value"),
    # State("pca_x-segment-dropdown", "value"),
    # State("pca_y-segment-dropdown", "value"),
    prevent_initial_call=True
)
def load_plots(n_clicks, liverslice, color_choice):
    if not liverslice or not color_choice:
        return []
    
    # slice_region_id = f"{liverslice}-{x_segment}-{y_segment}"

    # print(slice_region_id)
    

    # boundary_data_instance = BoundaryDataLoader(liverslice, x_segment, y_segment)
    # slice_boundaries = boundary_data_instance.filter_by_centroid_coordinates(ann_data)
    # slice_mask = np.zeros(ann_data.obs.shape[0], dtype=bool)
    # slice_mask[
    #     np.where(ann_data.obs.index.isin(slice_boundaries.keys()))
    # ] = True

    # slice_spatial_data = ann_data.obsm["spatial"][slice_mask]

    # slice_cell_data = pd.DataFrame(
    #     slice_spatial_data,
    #     columns=["X", "Y"],
    #     index=ann_data.obs.index[slice_mask],
    # ).reset_index()
    # slice_cell_data = slice_cell_data.rename(columns={"index": "CELL_ID"})
    # slice_cell_data["CELL_TYPE"] = ann_data.obs['CELL_TYPE'][slice_mask].to_list()
    # slice_cell_data['SIZE'] = ann_data.obs['volume'][slice_mask].to_list()
    # bm_columns = [f"BM-{bm_temp}" for bm_temp in ann_data.var.index.to_list()]
    # slice_cell_data[bm_columns] = pd.DataFrame(ann_data[slice_cell_data['CELL_ID'], :].X.toarray())

    # slice_cell_boundaries_given = [
    #     slice_boundaries[item][0] for item in slice_cell_data["CELL_ID"].values
    # ]

    # G_given_boundary_delaunay_edges, node_to_cell_mapping_given_boundary_delaunay_edges = build_graph_from_cell_coords(
    #     slice_cell_data, slice_cell_boundaries_given, edge_logic='Delaunay'
    # )
    # neighbor_edge_cutoff = SLICE_PIXELS_EDGE_CUTOFF[liverslice]
    # G_given_boundary_delaunay_edges = assign_attributes(G_given_boundary_delaunay_edges, slice_cell_data, node_to_cell_mapping_given_boundary_delaunay_edges, neighbor_edge_cutoff)
    # G_given_boundary_delaunay_edges.region_id = slice_region_id    

    # given_boundaries = {G_given_boundary_delaunay_edges.nodes[n]['cell_id'] : G_given_boundary_delaunay_edges.nodes[n]["voronoi_polygon"] for n in G_given_boundary_delaunay_edges.nodes}

    if color_choice in categorical_columns:
        fig = plotly_pca_categorical(
            ann_data,
            liverslice,
            color_key=color_choice,
            return_fig=True,
            x_dim=0,
            y_dim=1,
            show=False,
        )
    elif color_choice in numerical_columns:
        fig = plotly_pca_numerical(
            ann_data,
            liverslice,
            color_key=color_choice,
            return_fig=True,
            x_dim=0,
            y_dim=1,
            show=False,
        )        

    # pca_given_boundaries_fig = plotly_spatial_scatter_pca(pipeline_instance, given_boundaries, component=1)
    # component=1
    # color_column = pd.Series(
    #     ann_data.obsm["X_pca"][:, component],
    #     index=ann_data.obs.index,
    #     name=f"PCA_component_{component}",
    # )
    # fig = plotly_spatial_scatter_numerical(given_boundaries, color_column)
    return fig
    
    # plot_paths = [
    #     f"assets/subgraphs/{liverslice}/{x_segment}x{y_segment}_{color_choice}_given_bound.html",
    #     f"assets/subgraphs/{liverslice}/{x_segment}x{y_segment}_{color_choice}_vor_bound.html",
    #     f"assets/subgraphs/{liverslice}/{x_segment}x{y_segment}_{color_choice}_delayney_edge.html",
    #     f"assets/subgraphs/{liverslice}/{x_segment}x{y_segment}_{color_choice}_rectangle_bound.html",
    #     f"assets/subgraphs/{liverslice}/{x_segment}x{y_segment}_{color_choice}_r3index_edge.html"
    # ]
    
    # return generate_plots(plot_paths)

from helpers import CellularGraphDataset
from helpers.plotly_helpers import *
from helpers import logger

def plotly_spatial_scatter_subgraph(test_boundaries, color_column, subgraph_edges=None):

    fig = go.Figure()

    x_label = "X-position"
    y_label = "Y-position"

    # Get unique values of color_key
    unique_colors = color_column.unique()
    # Get a list of discrete colors from Plotly
    color_dict = {}
    for i, string in enumerate(unique_colors):
        color_dict[string] = COLORS_LIST[i % len(COLORS_LIST)]

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


def build_subgraph_for_plotly(dataset, idx, center_ind):

    # def plot_subgraph(self, idx, center_ind):
    #     """Plot the n-hop subgraph around cell `center_ind` from region `idx`"""
    xcoord_ind = dataset.node_feature_names.index('center_coord-x')
    ycoord_ind = dataset.node_feature_names.index('center_coord-y')

    _subg = dataset.calculate_subgraph(idx, center_ind)
    coords = _subg.x.data.numpy()[:, [xcoord_ind, ycoord_ind]].astype(float)
    x_c, y_c = coords[_subg.center_node_index]

    G = dataset.get_full_nx(idx)
    sub_node_inds = []
    for n in G.nodes:
        c = np.array(G.nodes[n]['center_coord']).astype(float).reshape((1, -1))
        if np.linalg.norm(coords - c, ord=2, axis=1).min() < 1e-2:
            sub_node_inds.append(n)
    assert len(sub_node_inds) == len(coords)
    _G = G.subgraph(sub_node_inds)

    node_colors = {f"{_G.nodes[n]['cell_id']}" : dataset.cell_type_mapping[_G.nodes[n]['cell_type']] for n in _G.nodes}
    test_boundaries = {f"{_G.nodes[n]['cell_id']}": _G.nodes[n]['voronoi_polygon'] for n in _G.nodes}
    # color_column = pd.Series(node_colors, index=_G.nodes)
    color_column = pd.DataFrame.from_dict(node_colors, orient='index', columns=['leiden_res'])
    color_column = color_column['leiden_res']

    return plotly_spatial_scatter_subgraph(test_boundaries, color_column)


if __name__ == "__main__":
    
    dataset_root = "/data/qd452774/spatial_transcriptomics/data/example_dataset"
    dataset_kwargs = {
        'transform': [],
        'pre_transform': None,
        'raw_folder_name': 'graph',  # os.path.join(dataset_root, "graph") is the folder where we saved nx graphs
        'processed_folder_name': 'tg_graph',  # processed dataset files will be stored here
        'node_features': ["cell_type", "volume", "biomarker_expression", "neighborhood_composition", "center_coord"],  # There are all the cellular features that we want the dataset to compute
        'edge_features': ["edge_type", "distance"],  # edge (cell pair) features
        'subgraph_size': 3,  # indicating we want to sample 3-hop subgraphs from these regions (for training/inference), this is a core parameter for SPACE-GM.
        'subgraph_source': 'on-the-fly',
        'subgraph_allow_distant_edge': True,
        'subgraph_radius_limit': 400.,
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

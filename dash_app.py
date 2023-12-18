import dash
from dash import html, dcc, Input, Output, State
from concurrent.futures import ThreadPoolExecutor
from helpers import logger
import os
import sys
import base64

# Extract the port argument from command-line arguments
if "--port" in sys.argv:
    port_index = sys.argv.index("--port") + 1
    port = int(sys.argv[port_index])
else:
    port = 8050  # Default port if not specified

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# Define the layout
app.layout = html.Div([
    html.H1("Visualizations"),
    
    dcc.Tabs([
        dcc.Tab(label='Various graph representations', children=[
            # Dropdowns for selecting liverslice, x_segment, and y_segment
            dcc.Dropdown(id="liverslice-dropdown", placeholder="Select liverslice"),
            dcc.Dropdown(id="color-dropdown", placeholder="Select color scheme"),
            dcc.Dropdown(id="x-segment-dropdown", placeholder="Select x_segment"),
            dcc.Dropdown(id="y-segment-dropdown", placeholder="Select y_segment"),
            
            # Button to trigger loading of plots
            html.Button("Load Plots", id="load-button"),
            
            # Container for the plots
            dcc.Loading(
                id="loading-component",
                type="default",  # Options: "default", "circle", "dot", "default", "bars", "custom"
                children=html.Div(id="plot-container")
            )
        ]),
        
        dcc.Tab(label='Cluster comparison', children=[
            # Dropdowns for selecting liverslice-x, resolution-x, liverslice-y, and resolution-y
            dcc.Dropdown(id="cc-liverslice-x-dropdown", placeholder="Select liverslice-x"),
            dcc.Dropdown(id="cc-resolution-x-dropdown", placeholder="Select resolution-x"),
            dcc.Dropdown(id="cc-liverslice-y-dropdown", placeholder="Select liverslice-y"),
            dcc.Dropdown(id="cc-cell-counts-abs-or-pct", placeholder="Select abs or pct type"),

            # Button to trigger loading of plots
            html.Button("Load Plots", id="load-cc-button"),
            
            # Container for the plots
            dcc.Loading(
                id="cc-loading-component",
                type="default",  # Options: "default", "circle", "dot", "default", "bars", "custom"
                children=html.Div(id="cc-plot-container")
            )            
            
        ]),

        dcc.Tab(label='Subgraph Sampler', children=[
            # Dropdowns for selecting liverslice-x, resolution-x, liverslice-y, and resolution-y
            dcc.Dropdown(id="ss-liverslice-dropdown", placeholder="Select liverslice"),
            dcc.Dropdown(id="ss-x-segment-dropdown", placeholder="Select x_segment"),
            dcc.Dropdown(id="ss-y-segment-dropdown", placeholder="Select y_segment"),            
            dcc.Dropdown(id="ss-center-cell-idx-to-sample-from", placeholder="Select a cell to show neighborhood"),

            # Button to trigger loading of plots
            html.Button("Load Plots", id="load-ss-button"),
            
            # Container for the plots
            dcc.Loading(
                id="ss-loading-component",
                type="default",  # Options: "default", "circle", "dot", "default", "bars", "custom"
                children=html.Div(id="ss-plot-container")
            )            
            
        ]),        
    ])
])

from spacegm import CellularGraphDataset
from helpers.build_subsampler import build_subgraph_for_plotly
from Example1 import build_train_kwargs

data_filter_name = "Liver1Slice12"
resolution = 0.6
network_type = "voronoi_delaunay"
graph_type = "gin"
dataset_kwargs = build_train_kwargs(data_filter_name, resolution, network_type, graph_type)

dataset_root = dataset_kwargs["dataset_root"]

logger.info("Loading CellularGraphDataset for the following comfig...")
logger.info(dataset_kwargs)

dataset = CellularGraphDataset(dataset_root, **dataset_kwargs)
logger.info(dataset)

# Function to generate a single plot cell
def generate_plot_cell(plot_path):
    # return html.Div([
    #     html.Iframe(srcDoc=open(plot_path, "r").read(),
    #                 className="plot-iframe", style={"height": "800px", "width": "100%"},)
    # ], className="plot-cell")
    return html.Div([
        dcc.Loading(
            id={"type": "loading-plot", "index": plot_path},
            type="circle",
            children=html.Iframe(srcDoc=open(plot_path, "r").read(), className="plot-iframe", style={"height": "800px", "width": "100%"}),
        )
    ], className="plot-cell")

# Lazy loading using ThreadPoolExecutor
def generate_plots(plot_paths):
    with ThreadPoolExecutor() as executor:
        plot_cells = list(executor.map(generate_plot_cell, plot_paths))
    
    return plot_cells

@app.callback(
    Output("liverslice-dropdown", "options"),
    Output("color-dropdown", "options"),
    Output("x-segment-dropdown", "options"),
    Output("y-segment-dropdown", "options"),
    Input("plot-container", "children"),
    prevent_initial_call=False
)
def update_dropdown_options(_):
    
    liverslice_options = ["Liver1Slice1", "Liver1Slice2", "Liver2Slice1", "Liver2Slice2"]
    color_options = ['louvain', 'leiden']
    x_segment_options = [{"label": file, "value": file} for file in list(range(20))]
    y_segment_options = [{"label": file, "value": file} for file in list(range(20))]
    
    return liverslice_options, color_options, x_segment_options, y_segment_options

@app.callback(
    Output("cc-liverslice-x-dropdown", "options"),
    Output("cc-resolution-x-dropdown", "options"),
    Output("cc-liverslice-y-dropdown", "options"),
    Output("cc-cell-counts-abs-or-pct", "options"),
    Input("plot-container", "children"),
    prevent_initial_call=False
)
def update_dropdown_options_cluster_comparison(_):
    
    liverslice_options = ["Liver12Slice12", "Liver1Slice1", "Liver1Slice2", "Liver2Slice1", "Liver2Slice2", "Liver1Slice12", "Liver2Slice12"]
    x_resolutions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    liverslice_y_options = ["Liver12Slice12", "Liver1Slice1", "Liver1Slice2", "Liver2Slice1", "Liver2Slice2", "Liver1Slice12", "Liver2Slice12"]
    abs_or_pct_options = ["absolute", "percentages"]
    
    return liverslice_options, x_resolutions, liverslice_y_options, abs_or_pct_options


@app.callback(
    Output("ss-liverslice-dropdown", "options"),
    Output("ss-x-segment-dropdown", "options"),
    Output("ss-y-segment-dropdown", "options"),
    Output("ss-center-cell-idx-to-sample-from", "options"),
    Input("ss-plot-container", "children"),
    prevent_initial_call=False
)
def update_dropdown_options_cluster_comparison(_):
    
    liverslice_options = ["Liver1Slice1", "Liver1Slice2"]
    x_segments_options = [{"label": file, "value": file} for file in list(range(20))]
    y_segments_options = [{"label": file, "value": file} for file in list(range(20))]
    center_cell_idx_to_sample_from = [{"label": file, "value": file} for file in list(range(5000))]
    
    return liverslice_options, x_segments_options, y_segments_options, center_cell_idx_to_sample_from



@app.callback(
    Output("ss-plot-container", "children"),
    Input("load-ss-button", "n_clicks"),
    State("ss-liverslice-dropdown","value"),
    State("ss-x-segment-dropdown","value"),
    State("ss-y-segment-dropdown","value"),
    State("ss-center-cell-idx-to-sample-from", "value"),
    prevent_initial_call=True
)
def load_plots_ss(n_clicks, liverslice_name, segment_x, segment_y, center_cell_id):

    graph_index = dataset.region_ids.index(f"{liverslice_name}-{segment_x}-{segment_y}")

    # Generate the Matplotlib plot
    mpl_fig_buffer = None
    mpl_fig_buffer = dataset.plot_subgraph(graph_index, center_cell_id)

    # Convert the Matplotlib figure buffer to a base64-encoded image
    matplotly_image = base64.b64encode(mpl_fig_buffer.getvalue()).decode()

    # Create an HTML image element to display the plot
    image_element = html.Img(src=f"data:image/png;base64,{matplotly_image}")

    cur_subgraph_fig = build_subgraph_for_plotly(dataset, graph_index, center_cell_id)
    plot_fig = dcc.Graph(id='my-graph', figure=cur_subgraph_fig)

    return html.Div([
        plot_fig,
        image_element
    ])



@app.callback(
    Output("cc-plot-container", "children"),
    Input("load-cc-button", "n_clicks"),
    State("cc-liverslice-x-dropdown","value"),
    State("cc-resolution-x-dropdown","value"),
    State("cc-liverslice-y-dropdown","value"),
    State("cc-cell-counts-abs-or-pct", "value"),
    prevent_initial_call=True
)
def load_plots_cc(n_clicks, liverslice_x, resolution_x, liverslice_y, abs_or_pct):
    if not liverslice_x or not resolution_x or not liverslice_y or not abs_or_pct:
        return []
    
    # logger.info(abs_or_pct)
    if abs_or_pct == "absolute":
        tail_str = ""
    else:
        tail_str = "_vertical_percent"

    plot_paths = [
        f"assets/cluster_comparison/{liverslice_y}={0.3}-{liverslice_x}={resolution_x}{tail_str}.html",
        f"assets/cluster_comparison/{liverslice_y}={0.4}-{liverslice_x}={resolution_x}{tail_str}.html",
        f"assets/cluster_comparison/{liverslice_y}={0.5}-{liverslice_x}={resolution_x}{tail_str}.html",
        f"assets/cluster_comparison/{liverslice_y}={0.6}-{liverslice_x}={resolution_x}{tail_str}.html",
        f"assets/cluster_comparison/{liverslice_y}={0.7}-{liverslice_x}={resolution_x}{tail_str}.html",
        f"assets/cluster_comparison/{liverslice_y}={0.8}-{liverslice_x}={resolution_x}{tail_str}.html",
        f"assets/cluster_comparison/{liverslice_y}={0.9}-{liverslice_x}={resolution_x}{tail_str}.html",
        f"assets/cluster_comparison/{liverslice_y}={1.0}-{liverslice_x}={resolution_x}{tail_str}.html",
        f"assets/cluster_comparison/{liverslice_y}={1.1}-{liverslice_x}={resolution_x}{tail_str}.html",
    ]
    
    return generate_plots(plot_paths)

@app.callback(
    Output("plot-container", "children"),
    Input("load-button", "n_clicks"),
    State("liverslice-dropdown", "value"),
    State("color-dropdown", "value"),
    State("x-segment-dropdown", "value"),
    State("y-segment-dropdown", "value"),
    prevent_initial_call=True
)
def load_plots(n_clicks, liverslice, color_choice, x_segment, y_segment):
    if not liverslice or not x_segment or not y_segment:
        return []
    
    plot_paths = [
        f"assets/subgraphs/{liverslice}/{x_segment}x{y_segment}_{color_choice}_given_bound.html",
        f"assets/subgraphs/{liverslice}/{x_segment}x{y_segment}_{color_choice}_vor_bound.html",
        f"assets/subgraphs/{liverslice}/{x_segment}x{y_segment}_{color_choice}_delayney_edge.html",
        f"assets/subgraphs/{liverslice}/{x_segment}x{y_segment}_{color_choice}_rectangle_bound.html",
        f"assets/subgraphs/{liverslice}/{x_segment}x{y_segment}_{color_choice}_r3index_edge.html"
    ]
    
    return generate_plots(plot_paths)

if __name__ == "__main__":
    app.run_server(debug=True, port=port, host='0.0.0.0')
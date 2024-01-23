from dash import Input, Output, State, callback
from .plots_parallel import generate_plots


@callback(
    Output("cc-liverslice-x-dropdown", "options"),
    Output("cc-resolution-x-dropdown", "options"),
    Output("cc-liverslice-y-dropdown", "options"),
    Output("cc-cell-counts-abs-or-pct", "options"),
    Input("cc-plot-container", "children"),
    prevent_initial_call=False
)
def update_dropdown_options_cluster_comparison(_):
    
    liverslice_options = ["Liver12Slice12", "Liver1Slice1", "Liver1Slice2", "Liver2Slice1", "Liver2Slice2", "Liver1Slice12", "Liver2Slice12"]
    x_resolutions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    liverslice_y_options = ["Liver12Slice12", "Liver1Slice1", "Liver1Slice2", "Liver2Slice1", "Liver2Slice2", "Liver1Slice12", "Liver2Slice12"]
    abs_or_pct_options = ["absolute", "percentages"]
    
    return liverslice_options, x_resolutions, liverslice_y_options, abs_or_pct_options


@callback(
    Output("cc-plot-container", "children"),
    Input("cc-load-button", "n_clicks"),
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
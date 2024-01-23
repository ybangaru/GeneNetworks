from concurrent.futures import ThreadPoolExecutor
from dash import html, dcc

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
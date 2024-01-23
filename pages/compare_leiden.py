from dash import html, dcc, callback, Input, Output


import dash
dash.register_page(__name__, order=2, image='assets/compare_leiden.png')


def layout():

    return html.Div(children=[
            dcc.Dropdown(id="cc-liverslice-x-dropdown", placeholder="Select liverslice-x"),
            dcc.Dropdown(id="cc-resolution-x-dropdown", placeholder="Select resolution-x"),
            dcc.Dropdown(id="cc-liverslice-y-dropdown", placeholder="Select liverslice-y"),
            dcc.Dropdown(id="cc-cell-counts-abs-or-pct", placeholder="Select abs or pct type"),

            # Button to trigger loading of plots
            html.Button("Load Plots", id="cc-load-button"),
            
            # Container for the plots
            dcc.Loading(
                id="cc-loading-component",
                type="default",  # Options: "default", "circle", "dot", "default", "bars", "custom"
                children=html.Div(id="cc-plot-container")
            )		
        ])

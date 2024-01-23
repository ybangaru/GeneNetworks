import dash
from dash import html, dcc #, Input, Output, State


dash.register_page(
	__name__,
	path='/scrna-pca',
	name='scrna-pca',
	# description='SOTA Research of Spatial Transcriptomics',
	order=2,
	# redirect_from=['/old-home-page', '/v2'],
	# extra_template_stuff='yup'
    image='assets/scrna_pca.png'
)

layout = html.Div(children=[
	dcc.Dropdown(id="pca_liverslice-dropdown", placeholder="Select liverslice"),
	dcc.Dropdown(id="pca_color-dropdown", placeholder="Select color scheme"),
	# dcc.Dropdown(id="pca_x-segment-dropdown", placeholder="Select x_segment"),
	# dcc.Dropdown(id="pca_y-segment-dropdown", placeholder="Select y_segment"),
	
	# Button to trigger loading of plots
	html.Button("Load Plots", id="pca_load-button"),
	
	# Container for the plots
	dcc.Loading(
		id="loading-component",
		type="default",  # Options: "default", "circle", "dot", "default", "bars", "custom"
		children=dcc.Graph(id="pca_plot-container")
	),
    ])
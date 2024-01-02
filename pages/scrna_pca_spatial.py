import dash
from dash import html, dcc #, Input, Output, State


dash.register_page(
	__name__,
	path='/scrna-pca-spatial',
	name='scrna-pca-spatial',
	# description='SOTA Research of Spatial Transcriptomics',
	order=2,
    image='assets/scrna_pca_spatial.png'
	# redirect_from=['/old-home-page', '/v2'],
	# extra_template_stuff='yup'
)

layout = html.Div(children=[
	dcc.Dropdown(id="spca_liverslice-dropdown", placeholder="Select liverslice"),
	dcc.Dropdown(id="spca_color-dropdown", placeholder="Select color scheme"),
	dcc.Dropdown(id="spca_x-segment-dropdown", placeholder="Select x_segment"),
	dcc.Dropdown(id="spca_y-segment-dropdown", placeholder="Select y_segment"),
	
	# Button to trigger loading of plots
	html.Button("Load Plots", id="spca_load-button"),
	
	# Container for the plots
	dcc.Loading(
		id="spca_loading-component",
		type="default",  # Options: "default", "circle", "dot", "default", "bars", "custom"
		children=dcc.Graph(id="spca_plot-container")
	),
    ])
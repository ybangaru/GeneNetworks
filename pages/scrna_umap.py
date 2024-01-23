import dash
from dash import html, dcc #, Input, Output, State

dash.register_page(
	__name__,
	path='/scrna-umap',
	name='scrna-umap',
	# description='SOTA Research of Spatial Transcriptomics',
	order=3,
	# redirect_from=['/old-home-page', '/v2'],
	# extra_template_stuff='yup'
)

layout = html.Div('SCRNA UMAP results')
import dash
from dash import html, dcc #, Input, Output, State


dash.register_page(
	__name__,
	path='/',
	name='Home',
	description='SOTA Research of Spatial Transcriptomics',
	order=0,
	# redirect_from=['/old-home-page', '/v2'],
	# extra_template_stuff='yup'
)

layout = html.Div('Home Page')
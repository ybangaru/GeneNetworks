import dash
from dash import html, dcc #, Input, Output, State


dash.register_page(
	__name__,
	path='/scrna-seq-data',
	name='scrna-seq-data',
	# description='SOTA Research of Spatial Transcriptomics',
	order=1,
	# redirect_from=['/old-home-page', '/v2'],
	# extra_template_stuff='yup'
)

layout = html.Div('SCRNA seq data')
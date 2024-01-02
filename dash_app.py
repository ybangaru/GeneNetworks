import sys
import os

from dash import Dash, html, dcc, Input, Output
import dash
import pages_plugin

import dash_cytoscape as cyto

# Extract the port argument from command-line arguments
if "--port" in sys.argv:
    port_index = sys.argv.index("--port") + 1
    port = int(sys.argv[port_index])
else:
    port = 8050  # Default port if not specified

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, plugins=[pages_plugin], external_stylesheets=external_stylesheets)

server = app.server
server.secret_key = os.environ.get("SECRET_KEY", "my-secret-key-test")

pages_paths = {
	"home" : dash.page_registry["pages.home"]["path"],
	"scrna_data" : dash.page_registry["pages.scrna_data"]["path"],
	"scrna_pos_boundary" : dash.page_registry["pages.scrna_pos_boundary"]["path"],
	"compare_leiden" : dash.page_registry["pages.compare_leiden"]["path"],
	"scrna_pca" : dash.page_registry["pages.scrna_pca"]["path"],
	"scrna_pca_spatial" : dash.page_registry["pages.scrna_pca_spatial"]["path"],
	"scrna_umap" : dash.page_registry["pages.scrna_umap"]["path"],
	"scrna_umap_spatial" : dash.page_registry["pages.scrna_umap_spatial"]["path"],
	"cell_subgraphs" : dash.page_registry["pages.cell_subgraphs"]["path"],
	"center_cell_classification" : dash.page_registry["pages.center_cell_classification"]["path"],
	"forward_outlook" : dash.page_registry["pages.forward_outlook"]["path"],
    
}

pages_images = {
	"home" : dash.page_registry["pages.home"]["image"],
	"scrna_data" : dash.page_registry["pages.scrna_data"]["image"],
	"scrna_pos_boundary" : dash.page_registry["pages.scrna_pos_boundary"]["image"],
	"compare_leiden" : dash.page_registry["pages.compare_leiden"]["image"],
	"scrna_pca" : dash.page_registry["pages.scrna_pca"]["image"],
	"scrna_pca_spatial" : dash.page_registry["pages.scrna_pca_spatial"]["image"],
	"scrna_umap" : dash.page_registry["pages.scrna_umap"]["image"],
	"scrna_umap_spatial" : dash.page_registry["pages.scrna_umap_spatial"]["image"],
	"cell_subgraphs" : dash.page_registry["pages.cell_subgraphs"]["image"],
	"center_cell_classification" : dash.page_registry["pages.center_cell_classification"]["image"],
	"forward_outlook" : dash.page_registry["pages.forward_outlook"]["image"],
    
}


app.layout = html.Div([
    dcc.Store(id="ann-data-obs-store", storage_type="session"),
    dcc.Store(id="ann-data-var-store", storage_type="session"),
	html.Div([
        html.A([
            html.Img(src=pages_images['home'], style={'height': '100px'}),
        ], href=pages_paths['home'], title="Home"),
        html.Span(' → ', style={'font-size': '24px'}),
        html.A([
            html.Img(src=pages_images['scrna_data'], style={'height': '100px'}),
        ], href=pages_paths['scrna_data']),
        html.Span(' → ', style={'font-size': '24px'}), 
        html.A([
            html.Img(src=pages_images['scrna_pca'], style={'height': '100px'}),
        ], href=pages_paths['scrna_pca'], title="PCA"),
        html.Span(' → ', style={'font-size': '24px'}), 
        html.A([
            html.Img(src=pages_images['scrna_umap'], style={'height': '100px'}),
        ], href=pages_paths['scrna_umap']),        
		
	]),
	html.Div([
        html.A([
            html.Img(src=pages_images['compare_leiden'], style={'height': '100px'}),
        ], href=pages_paths['compare_leiden']),
        html.Span(' → ', style={'font-size': '24px'}),         
        html.A([
            html.Img(src=pages_images['scrna_pos_boundary'], style={'height': '100px'}),
        ], href=pages_paths['scrna_pos_boundary']),
        html.Span(' → ', style={'font-size': '24px'}), 
        html.A([
            html.Img(src=pages_images['scrna_pca_spatial'], style={'height': '100px'}),
        ], href=pages_paths['scrna_pca_spatial'], title="PCA Spatial"),
        html.Span(' → ', style={'font-size': '24px'}), 
        html.A([
            html.Img(src=pages_images['scrna_umap_spatial'], style={'height': '100px'}),
        ], href=pages_paths['scrna_umap_spatial']),
        html.Span(' → ', style={'font-size': '24px'}), 
        html.A([
            html.Img(src=pages_images['cell_subgraphs'], style={'height': '100px'}),
        ], href=pages_paths['cell_subgraphs'], title="Subgraph Sampler"),
        html.Span(' → ', style={'font-size': '24px'}), 
        html.A([
            html.Img(src=pages_images['center_cell_classification'], style={'height': '100px'}),
        ], href=pages_paths['center_cell_classification']),
	]),

	pages_plugin.page_container

])

from callbacks import *


if __name__ == "__main__":
    app.run_server(debug=False, port=port, host='0.0.0.0')

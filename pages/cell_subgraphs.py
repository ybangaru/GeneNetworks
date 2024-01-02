import dash

dash.register_page(__name__, path='/cell-subgraphs', order=4, image='assets/cell_subgraphs.png')


def layout(velocity=None, **other_unknown_query_strings):
    return dash.html.Div([
        dash.dcc.Input(id='velocity', value=velocity)
    ])


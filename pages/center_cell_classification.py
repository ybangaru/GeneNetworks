import dash

dash.register_page(__name__, path='/center-cell-classification', order=5)


def layout(velocity=None, **other_unknown_query_strings):
    return dash.html.Div([
        dash.dcc.Input(id='velocity', value=velocity)
    ])


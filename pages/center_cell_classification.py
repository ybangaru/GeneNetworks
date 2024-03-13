import dash
import plotly.express as px
from helpers import MLFLOW_CLIENT
dash.register_page(__name__, path='/center-cell-classification', order=5)


def layout():

    # TODO: add a dropdown for run id
    return dash.html.Div(
        [
            dash.dcc.Dropdown(
                id="run-id-choice",
                placeholder="Select run id"
            ),
        ]
    )
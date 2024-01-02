from dash import Output, Input, State, callback_context
import dash
import flask

from .ann_data_store import ann_data

# Callback to update the session storage with obs_data and var_data
@dash.callback(
    Output("ann-data-obs-store", "data"),
    Output("ann-data-var-store", "data"),
    Input("ann-data-obs-store", "data"),
    Input("ann-data-var-store", "data"),
    prevent_initial_call=True,
)
def update_session_storage(obs_store_data, var_store_data):
    # Check if the dictionaries are already present in session storage
    ctx = callback_context
    if not ctx.triggered_id:
        raise dash.exceptions.PreventUpdate

    if "ann-data-obs-store" in ctx.triggered_id and not obs_store_data:
        # Store obs_data in session if not already present
        flask.session["obs_data"] = ann_data.obs.to_dict()
    elif "ann-data-var-store" in ctx.triggered_id and not var_store_data:
        # Store var_data in session if not already present
        flask.session["var_data"] = ann_data.var.to_dict()

    return flask.session.get("obs_data", ann_data.obs.to_dict()), flask.session.get("var_data", ann_data.var.to_dict())



# Callback to use the dictionaries in another callback
@dash.callback(
    Output("output-div", "children"),
    Input("ann-data-obs-store", "data"),
    Input("ann-data-var-store", "data"),
    prevent_initial_call=True,
)
def use_dictionaries(obs_data, var_data):
    # Access the dictionaries and perform desired operations
    # For example, concatenate some values from both dictionaries
    # result = f"Concatenated values: {obs_data['key_obs']} - {var_data['key_var']}"
    print(obs_data)
    print(var_data)
    

# @dash.callback(
#     dash.dependencies.Output('ann-data-store', 'data'),
#     [dash.dependencies.Input({'type': 'dynamic-input', 'index': dash.dependencies.ALL}, 'value')]
# )
# def load_data(_):
#     # Convert your AnnData object to a format that can be stored
#     # For example, you might convert it to a dictionary
#     data = ann_data.to_dict()

#     return data
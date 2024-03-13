# from plotly import Input, Output, callback
# import plotly.express as px


# @callback(Output('cell-type-embeddings', 'figure'),
#               [Input('cell-type-embeddings', 'relayoutData')])
# def update_scatter_plot(relayoutData):
#     # Extract the current frame number from the relayoutData
#     current_frame = int(relayoutData['title']['text'].split()[-1])
#     embeddings = embeddings_list[current_frame]

#     # Create a new scatter plot figure
#     new_fig = px.scatter(x=embeddings[:, 0], y=embeddings[:, 1], color=[class_names[label] for label in labels],
#                          title=f'Node Embeddings in 2D - Iteration {current_frame}')

#     return new_fig
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder, LabelBinarizer


def plot_precision_recall_curve(y_true, y_scores, class_names, title='Precision-Recall Curve'):
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    
    fig = go.Figure()

    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        pr_auc = auc(recall, precision)

        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines',
                                 name=f'{class_names[i]} (AUC = {pr_auc:.2f})'))

    fig.update_layout(title=title,
                      xaxis=dict(title='Recall'),
                      yaxis=dict(title='Precision'),
                    #   legend=dict(x=0, y=1, traceorder='normal')
                      )
    return fig


# Add the following function to your code
def plot_confusion_matrix(y_true, y_pred, labels, class_names, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize confusion matrix

    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="True"),
                    x=class_names,
                    y=class_names,
                    color_continuous_scale="Viridis",
                    title=title,
                    origin='upper')

    fig.update_layout(xaxis=dict(side='top'))
    fig.update_layout(coloraxis_colorbar=dict(title='Normalized Count'))

    return fig

def plot_node_embeddings_2d(embeddings, labels, class_names, title='Node Embeddings in 2D'):
    # Map numeric labels to class names
    label_names = [class_names[label] for label in labels]

    df = pd.DataFrame({
        'X': embeddings[:, 0],
        'Y': embeddings[:, 1],
        'Label': label_names  # Use the mapped class names as labels
    })

    fig = px.scatter(df, x='X', y='Y', color='Label', hover_name='Label', title=title,
                     category_orders={'Label': class_names})

    return fig


def plot_node_embeddings_3d(embeddings, labels, class_names, title='Node Embeddings in 3D'):
    # Map numeric labels to class names
    label_names = [class_names[label] for label in labels]

    df = pd.DataFrame({
        'X': embeddings[:, 0],
        'Y': embeddings[:, 1],
        'Z': embeddings[:, 2],  # Add Z coordinate for the 3D plot
        'Label': label_names
    })

    fig = px.scatter_3d(df, x='X', y='Y', z='Z', color='Label', hover_name='Label', title=title,
                        category_orders={'Label': class_names})

    return fig
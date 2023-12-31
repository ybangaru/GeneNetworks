#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 11:28:18 2021

@author: zqwu

"""

import os
import numpy as np
import torch
import torch.optim
import mlflow
from spacegm.data import SubgraphSampler
from spacegm.inference import collect_predict_for_all_nodes, collect_predict_by_random_sample

from helpers import mlflow_client, logger


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
                      legend=dict(x=0, y=1, traceorder='normal'))
    return fig


# Add the following function to your code
def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
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

# # Modify your training loop to include predictions
# def train_subgraph(model, dataset, device, node_task_loss_fn=None, ...):
#     # Existing code...

#     for i_iter in range(int(num_iterations)):
#         batch = next(data_iter)
#         batch = batch.to(device)

#         # Forward pass
#         res = model(batch)

#         # Extract predictions
#         _, node_pred = res[0].max(dim=1)

#         # Extract true labels
#         node_y = batch.node_y

#         # Your existing code for loss calculation and backpropagation...

#         if i_iter > 0 and i_iter % evaluate_freq == 0:
#             # Your existing code...

#             # Visualize confusion matrix
#             plot_confusion_matrix(node_y.cpu().numpy(), node_pred.cpu().numpy(), class_names=dataset.cell_type_mapping.keys())



def train_subgraph(model,
                   dataset,
                   device,
                   node_task_loss_fn=None,
                   graph_task_loss_fn=None,
                   train_inds=None,
                   valid_inds=None,
                   num_iterations=1e5,
                   num_regions_per_segment=0,
                   num_iterations_per_segment=1e4,
                   num_workers=0,
                   evaluate_freq=1e4,
                   evaluate_fn=[],
                   evaluate_on_train=True,
                   batch_size=64,
                   lr=0.001,
                   graph_loss_weight=1.,
                   dataset_kwargs={},
                   **kwargs):
    """ Train a GNN model through sampling subgraphs

    Args:
        model (nn.Module): pytorch model
        dataset (CellularGraphDataset): dataset object
        device (str/torch.device): device to run the model on
        node_task_criterion (nn.Module): node task loss function
        train_inds (list): list of indices of training samples
        valid_inds (list): list of indices of validation samples
        num_iterations (int): number of iterations to train
        num_regions_per_segment (int): see `data.SubgraphSampler` for details
        num_iterations_per_segment (int): see `data.SubgraphSampler` for details
        num_workers (int): see `data.SubgraphSampler` for details
        evaluate_freq (int): evaluate the model by calling `evalute_fn`
            every `evaluate_freq` iterations
        evaluate_fn (list): list of callback functions to evaluate the model
            during training
        evaluate_on_train (bool): if to evaluate the model on training set,
            will take longer time
        batch_size (int): batch size
        lr (float): learning rate
        graph_loss_weight (float): weight of graph task loss relative to node task loss
        **kwargs: additional arguments for callback functions
    """

    mlflow.set_experiment(kwargs['experiment_name'])

    with mlflow.start_run(run_name=kwargs['run_name']) as run:

        train_inds_str = ",".join(map(str, train_inds))
        valid_inds_str = ",".join(map(str, valid_inds))

        for arg, value in dataset_kwargs.items():
            try:
                if type(value) is np.ndarray or type(value) is list:
                    value = ",".join(map(str, value))
                mlflow.log_param(arg, value)
            except Exception as e:
                logger.error(f"Failed to log {arg} with value {value} due to {e}")

        # Create a dictionary of parameters to log
        params_to_log = {
            "node_task_loss_fn": node_task_loss_fn,
            "graph_task_loss_fn": graph_task_loss_fn,
            # "train_inds": train_inds_str,
            # "valid_inds": valid_inds_str,
            "num_iterations": num_iterations,
            "num_regions_per_segment": num_regions_per_segment,
            "num_iterations_per_segment": num_iterations_per_segment,
            "num_workers": num_workers,
            "evaluate_freq": evaluate_freq,
            "evaluate_fn": evaluate_fn,
            "evaluate_on_train": evaluate_on_train,
            "batch_size": batch_size,
            "lr": lr,
            "graph_loss_weight": graph_loss_weight
        }

        # Log the parameters
        mlflow.log_params(params_to_log)

        # Log all other keyword arguments dynamically
        for arg, value in kwargs.items():
            mlflow.log_param(arg, value)

    
        model.zero_grad()
        model = model.to(device)
        model.train()
        if train_inds is None:
            train_inds = np.arange(len(dataset))

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        node_losses = []
        graph_losses = []

        data_iter = SubgraphSampler(dataset,
                                    selected_inds=train_inds,
                                    batch_size=batch_size,
                                    num_regions_per_segment=int(num_regions_per_segment),
                                    steps_per_segment=int(num_iterations_per_segment),
                                    num_workers=num_workers)
        
        for i_iter in range(int(num_iterations)):
            batch = next(data_iter)
            batch = batch.to(device)

            res = model(batch)
            loss = 0.
            if model.num_node_tasks > 0:
                assert node_task_loss_fn is not None, \
                    "Please specify `node_task_loss_fn` in the training kwargs"
                node_y = batch.node_y
                node_pred = res[0]
                
                node_loss = node_task_loss_fn(node_pred, node_y)
                loss += node_loss
                node_losses.append(node_loss.to('cpu').data.item())

            if model.num_graph_tasks > 0:
                assert graph_task_loss_fn is not None, \
                    "Please specify `graph_task_loss_fn` in the training kwargs"
                graph_y, graph_w = batch.graph_y.float(), batch.graph_w.float()
                graph_pred = res[-1]
                graph_loss = graph_task_loss_fn(graph_pred, graph_y, graph_w)
                loss += graph_loss * graph_loss_weight
                graph_losses.append(graph_loss.to('cpu').data.item())

            loss.backward()
            optimizer.step()
            model.zero_grad()

            if i_iter > 0 and i_iter % evaluate_freq == 0:
                summary_str = "Finished iterations %d" % i_iter
                if len(node_losses) > 100:
                    summary_str += ", node loss %.2f" % np.mean(node_losses[-100:])
                    mlflow.log_metric("train_node_loss", np.mean(node_losses[-100:]), step=i_iter)
                else:
                    summary_str += ", node loss %.2f" % np.mean(node_losses[:])
                    mlflow.log_metric("train_node_loss", np.mean(node_losses[:]), step=i_iter)

                # _, node_pred = res[0].max(dim=1)
                # confusion_matrix_fig = plot_confusion_matrix(node_y.cpu().numpy(), node_pred.cpu().numpy(), class_names=list(dataset.cell_annotation_mapping.keys()))
                # mlflow.log_figure(confusion_matrix_fig, f"confusion_matrix_{i_ter}.png")

                # node_probs = torch.nn.functional.softmax(res[0], dim=1).detach().cpu().numpy()
                # precision_recall_fig = plot_precision_recall_curve(node_y.cpu().numpy(), node_probs, class_names=list(dataset.cell_annotation_mapping.keys()))

                # node_embeddings  = res[-1].detach().cpu().numpy()
                # node_embb_2d_fig = plot_node_embeddings_2d(node_embeddings, node_y.cpu().numpy(), class_names=list(dataset.cell_annotation_mapping.keys()))
                # node_embb_2d_fig.show()

                # node_embb_3d_fig = plot_node_embeddings_3d(node_embeddings, node_y.cpu().numpy(), class_names= list(dataset.cell_annotation_mapping.keys()))
                # node_embb_3d_fig.show()
                # if len(graph_losses) > 100:
                #     summary_str += ", graph loss %.2f" % np.mean(graph_losses[-100:])
                #     mlflow.log_metric("train_graph_loss", np.mean(node_losses[-100:]))
                # else:
                #     summary_str += ", graph loss %.2f" % np.mean(graph_losses[:])
                #     mlflow.log_metric("train_graph_loss", np.mean(graph_losses[:]))

                # mlflow.log_metric("train_graph_loss", np.mean(graph_losses[-100:]))

                logger.info("Evaluation at iteration %d" % i_iter)
                logger.info(summary_str)
                for fn in evaluate_fn:
                    result_list = fn(
                        model,
                        dataset,
                        device,
                        train_inds=train_inds if evaluate_on_train else None,
                        valid_inds=valid_inds,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        **kwargs
                    )
                    graph_type, *_rest_test_valid = result_list
                    dataset_type, _, avg_pred, top1_acc, top3_acc, top5_acc, *_rest_valid = _rest_test_valid

                    mlflow.log_metric(f"{graph_type}-{dataset_type}-avg_pred", avg_pred, step=i_iter)
                    mlflow.log_metric(f"{graph_type}-{dataset_type}-top1_acc", top1_acc, step=i_iter)
                    mlflow.log_metric(f"{graph_type}-{dataset_type}-top3_acc", top3_acc, step=i_iter)
                    mlflow.log_metric(f"{graph_type}-{dataset_type}-top5_acc", top5_acc, step=i_iter)

                    if _rest_valid:

                        dataset_type, _, avg_pred, top1_acc, top3_acc, top5_acc = _rest_valid
                        mlflow.log_metric(f"{graph_type}-{dataset_type}-avg_pred", avg_pred, step=i_iter)
                        mlflow.log_metric(f"{graph_type}-{dataset_type}-top1_acc", top1_acc, step=i_iter)
                        mlflow.log_metric(f"{graph_type}-{dataset_type}-top3_acc", top3_acc, step=i_iter)
                        mlflow.log_metric(f"{graph_type}-{dataset_type}-top5_acc", top5_acc, step=i_iter)        

                for fn in kwargs['model_save_fn']:
                    fn(model,
                        dataset,
                        device,
                        train_inds=train_inds if evaluate_on_train else None,
                        valid_inds=valid_inds,
                        batch_size=batch_size,
                        **kwargs
                        )

                model.train()
        dataset.set_indices(np.arange(dataset.N))
    return model


# %% Callback functions for evaluation
def evaluate_by_sampling_subgraphs(model,
                                   dataset,
                                   device,
                                   train_inds=None,
                                   valid_inds=None,
                                   batch_size=64,
                                   node_task_evaluate_fn=None,
                                   graph_task_evaluate_fn=None,
                                   num_eval_iterations=300,
                                   num_workers=0,
                                   score_file=None,
                                   **kwargs):
    """ Callback function for evaluating GNN predictions on randomly sampled subgraphs

    This evaluation callback fn will randomly sample some batches of subgraphs from the dataset,
    and evaluate metrics based on these subgraphs. For graph-level tasks, the evaluation is
    performed treating each subgraph as an independent data point.

    Args:
        model (nn.Module): pytorch model
        dataset (CellularGraphDataset): dataset object
        device (str/torch.device): device to run the model on
        train_inds (list): list of indices of training samples, evaluation skipped if None
        valid_inds (list): list of indices of validation samples, evaluation skipped if None
        batch_size (int): batch size
        node_task_evaluate_fn (function): function to evaluate node-level task predictions,
            see `inference.cell_type_prediction_evaluate_fn` for example
        graph_task_evaluate_fn (function): function to evaluate graph-level task predictions,
            see `inference.graph_classification_evaluate_fn` for example
        num_eval_iterations (int): number of iterations for sampling subgraphs,
            results will be concatenated (`num_eval_iterations` * `batch_size`) for evaluation
        num_workers (int): number of workers for sampling subgraphs
        score_file (str): file to save the evaluation results

    Returns:
        list: list of evaluation results
    """

    dataset.set_indices(np.arange(dataset.N))
    score_row = ["Eval-Subgraph"]
    if train_inds is not None:
        # Evaluate on subgraphs sampled from training samples
        score_row.append("Train")
        # Collect predictions by randomly sampling subgraphs
        node_preds, node_labels, graph_preds, graph_ys, graph_ws = \
            collect_predict_by_random_sample(model, dataset, device,
                                             inds=train_inds,
                                             batch_size=batch_size,
                                             num_eval_iterations=num_eval_iterations,
                                             num_workers=num_workers,
                                             **kwargs)
        if len(node_preds) > 0:
            # Evalaute node-level predictions
            assert node_task_evaluate_fn is not None, \
                "Please specify `node_task_evaluate_fn` in the training kwargs"
            score_row.append("node-score")
            score_row.extend(node_task_evaluate_fn(node_preds, node_labels, print_res=False))

        if len(graph_preds) > 0:
            # Evaluate graph-level predictions
            assert graph_task_evaluate_fn is not None, \
                "Please specify `graph_task_evaluate_fn` in the training kwargs"
            score_row.append("graph-score")
            score_row.extend(graph_task_evaluate_fn(graph_preds, graph_ys, graph_ws, print_res=False))
    if valid_inds is not None:
        # Same for validation samples
        score_row.append("Valid")
        node_preds, node_labels, graph_preds, graph_ys, graph_ws = \
            collect_predict_by_random_sample(model, dataset, device,
                                             inds=valid_inds,
                                             batch_size=batch_size,
                                             num_eval_iterations=num_eval_iterations,
                                             num_workers=num_workers,
                                             **kwargs)
        if len(node_preds) > 0:
            assert node_task_evaluate_fn is not None, \
                "Please specify `node_task_evaluate_fn` in the training kwargs"
            score_row.append("node-score")
            score_row.extend(node_task_evaluate_fn(node_preds, node_labels, print_res=False))

        if len(graph_preds) > 0:
            assert graph_task_evaluate_fn is not None, \
                "Please specify `graph_task_evaluate_fn` in the training kwargs"
            score_row.append("graph-score")
            score_row.extend(graph_task_evaluate_fn(graph_preds, graph_ys, graph_ws, print_res=False))

    if score_file is not None:
        with open(score_file, 'a') as f:
            f.write(",".join([s if isinstance(s, str) else ("%.3f" % s) for s in score_row]) + '\n')
    return score_row


def evaluate_by_full_graph(model,
                           dataset,
                           device,
                           train_inds=None,
                           valid_inds=None,
                           batch_size=64,
                           shuffle=True,
                           subsample_ratio=0.2,
                           full_graph_node_task_evaluate_fn=None,
                           full_graph_graph_task_evaluate_fn=None,
                           score_file=None,
                           **kwargs):
    """ Callback function for evaluating GNN predictions on regions (full cellular graphs)

    Args:
        model (nn.Module): pytorch model
        dataset (CellularGraphDataset): dataset object
        device (str/torch.device): device to run the model on
        train_inds (list): list of indices of training samples, evaluation skipped if None
        valid_inds (list): list of indices of validation samples, evaluation skipped if None
        batch_size (int): batch size
        shuffle (bool): whether to shuffle the order of subgraphs for each region
        subsample_ratio (float): ratio of subgraphs to use for each region
        full_graph_node_task_evaluate_fn (function): function to evaluate node-level task predictions,
            see `inference.full_graph_cell_type_prediction_evaluate_fn` for example
        full_graph_graph_task_evaluate_fn (function): function to evaluate graph-level task predictions,
            see `inference.full_graph_graph_classification_evaluate_fn` for example
        score_file (str): file to save the evaluation results

    Returns:
        list: list of evaluation results
    """

    dataset.set_indices(np.arange(dataset.N))
    score_row = ["Eval-Full-Graph"]
    if train_inds is not None:
        # Evaluate on full graphs of training samples
        score_row.append("Train")
        # Collect all predictions with optional subsampling
        node_preds, graph_preds = collect_predict_for_all_nodes(
            model,
            dataset,
            device,
            inds=train_inds,
            batch_size=batch_size,
            shuffle=shuffle,
            subsample_ratio=subsample_ratio,
            **kwargs)
        if model.num_node_tasks > 0:
            # Evaluate node-level predictions
            assert full_graph_node_task_evaluate_fn is not None, \
                "Please specify `full_graph_node_task_evaluate_fn` in the training kwargs"
            score_row.append("node-score")
            score_row.extend(full_graph_node_task_evaluate_fn(dataset, node_preds, print_res=False))
        if model.num_graph_tasks > 0:
            # Evaluate graph-level predictions
            assert full_graph_graph_task_evaluate_fn is not None, \
                "Please specify `full_graph_graph_task_evaluate_fn` in the training kwargs"
            score_row.append("graph-score")
            score_row.extend(full_graph_graph_task_evaluate_fn(dataset, graph_preds, print_res=False))

    if valid_inds is not None:
        # Same for validation samples
        score_row.append("Valid")
        node_preds, graph_preds = collect_predict_for_all_nodes(
            model,
            dataset,
            device,
            inds=valid_inds,
            batch_size=batch_size,
            shuffle=shuffle,
            subsample_ratio=subsample_ratio,
            **kwargs)
        if model.num_node_tasks > 0:
            assert full_graph_node_task_evaluate_fn is not None, \
                "Please specify `full_graph_node_task_evaluate_fn` in the training kwargs"
            score_row.append("node-score")
            score_row.extend(full_graph_node_task_evaluate_fn(dataset, node_preds, print_res=False))
        if model.num_graph_tasks > 0:
            assert full_graph_graph_task_evaluate_fn is not None, \
                "Please specify `full_graph_graph_task_evaluate_fn` in the training kwargs"
            score_row.append("graph-score")
            score_row.extend(full_graph_graph_task_evaluate_fn(dataset, graph_preds, print_res=False))

    if score_file is not None:
        with open(score_file, 'a') as f:
            f.write(",".join([s if isinstance(s, str) else ("%.3f" % s) for s in score_row]) + '\n')
    return score_row


def save_model_weight(model,
                      dataset,
                      device,
                      model_folder=None,
                      **kwargs):
    if model_folder is not None:
        os.makedirs(model_folder, exist_ok=True)
        fs = [f for f in os.listdir(model_folder) if f.startswith('model_save')]
        torch.save(model.state_dict(), os.path.join(model_folder, 'model_save_%d.pt' % len(fs)))
    return

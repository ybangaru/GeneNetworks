#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 20:50:26 2021

@author: zqwu
"""
import torch
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, r2_score
from torch_geometric.data import Batch
from concurrent.futures import ThreadPoolExecutor

from .data import SubgraphSampler
from .local_config import NO_JOBS


def get_num_nodes(dataset, i):
    return dataset.get_full(i).x.shape[0]


def predict_on_batch(batch, model, device, dataset):
    """Wrapper function for generating predictions on a batch of subgraphs

    Args:
        batch (list): list of pyg data objects, subgraphs
        model (nn.Module): pytorch model
        device (str/torch.device): device to run the model on

    Returns:
        node_preds (None/np.ndarray): node-level predictions if there are node tasks in the model
        graph_preds (None/np.ndarray): graph-level predictions if there are graph tasks in the model
    """
    _batch = Batch.from_data_list(batch)
    _batch = _batch.to(device)

    node_preds = None
    graph_preds = None

    res = model(_batch)

    if model.num_node_tasks > 0:
        node_preds = res[0].cpu().data.numpy()
    if model.num_graph_tasks > 0:
        graph_preds = res[-1].cpu().data.numpy()

    _, node_pred = res[0].max(dim=1)
    node_y = _batch.node_y
    confusion_matrix_info = {
        "node_y": node_y.cpu().numpy(),
        "node_pred": node_pred.cpu().numpy(),
    }
    # confusion_matrix_fig = plotly_confusion_matrix(node_y.cpu().numpy(), node_pred.cpu().numpy(), labels=list(dataset.cell_type_mapping.values()), class_names=list(dataset.cell_annotation_mapping.keys()))
    node_probs = torch.nn.functional.softmax(res[0], dim=1).detach().cpu().numpy()
    precision_recall_info = {
        "node_y": node_y.cpu().numpy(),
        "node_probs": node_probs,
    }
    # precision_recall_fig = plotly_precision_recall_curve(node_y.cpu().numpy(), node_probs, class_names=list(dataset.cell_annotation_mapping.keys()))
    node_embeddings = res[-1].detach().cpu().numpy()
    return node_preds, graph_preds, confusion_matrix_info, precision_recall_info, node_embeddings


def save_pred(
    dataset,
    batch,
    node_preds,
    graph_preds,
    confusion_matrix_info,
    precision_recall_info,
    node_embeddings,
    node_results,
    graph_results,
    region_ids,
    confusion_matrix_results,
    precision_recall_results,
    node_embeddings_results,
):
    """Assign batch prediction results to dictionary of prediction

    Args:
        batch (list): list of pyg data objects, subgraphs
        node_preds (None/np.ndarray): node-level predictions if there are node tasks in the model
        graph_preds (None/np.ndarray): graph-level predictions if there are graph tasks in the model
        node_results (dict): dictionary of predictions on node-level tasks
        graph_results (dict): dictionary of predictions on graph-level tasks
    """
    for ind, sample in enumerate(batch):
        i = region_ids.index(sample.region_id)
        j = sample.original_center_node
        if node_preds is not None:
            if i in node_results and node_results[i][j] is None:
                node_results[i][j] = node_preds[ind]
        if graph_preds is not None:
            if i in graph_results and graph_results[i][j] is None:
                graph_results[i][j] = graph_preds[ind]

    confusion_matrix_results.append(confusion_matrix_info)
    precision_recall_results.append(precision_recall_info)
    node_embeddings_results.append(node_embeddings)

    # node_y_val = np.concatenate([item["node_y"] for item in confusion_matrix_results])
    # node_pred_val = np.concatenate([item["node_pred"] for item in confusion_matrix_results])
    # confusion_matrix_fig = plotly_confusion_matrix(node_y_val, node_pred_val, labels=list(dataset.cell_type_mapping.values()), class_names=list(dataset.cell_annotation_mapping.keys()))
    # confusion_matrix_fig.write_html(f"results/confusion_matrix_{len(node_y_val)}.html")

    # node_probs = np.concatenate([item["node_probs"] for item in precision_recall_results])
    # precision_recall_fig = plotly_precision_recall_curve(node_y_val, node_probs, class_names=list(dataset.cell_annotation_mapping.keys()))
    # precision_recall_fig.write_html(f"results/precision_recall_{len(node_y_val)}.html")

    return


def collect_predict_for_all_nodes(
    model, dataset, device, inds=None, batch_size=64, shuffle=False, subsample_ratio=1.0, print_progress=False, **kwargs
):
    """Collect predictions on all cells/nodes from the dataset

    Args:
        model (nn.Module): pytorch model
        dataset (CODEXGraphDataset): dataset of cellular graphs
        device (str/torch.device): device to run the model on
        inds (array-like): list of integer indices of graphs to predict on,
            used for training/validation/test splitting
        batch_size (int): batch size for model prediction
        shuffle (bool): if to shuffle the order of nodes during prediction
        subsample_ratio (float): subsample ratio of nodes to predict on,
            can be set to below 1.0 to speed up prediction
        print_progress (bool): if to print progress

    Returns:
        node_results (dict): dictionary of predictions on node-level tasks
        graph_results (dict): dictionary of predictions on graph-level tasks
            These two dictionaries are formulated as {region index: list of predictions},
            in which each entry represents a region (full cellular graph), and the value is
            a list of predictions from each subgraph (node) in the region.
    """
    model = model.to(device)
    model.eval()
    if inds is None:
        # Predict on all regions
        inds = np.arange(dataset.N)

    node_results = {}
    graph_results = {}
    confusion_matrix_results = []
    precision_recall_results = []
    node_embeddings_results = []

    if dataset.subgraph_size > 0:
        # Predicting on subgraphs
        # region_ids = [dataset.get_full(i).region_id for i in range(dataset.N)]
        region_ids = dataset.region_ids

        # `node_results` and `graph_results` are dictionaries, in which each
        # entry represents a region (full cellular graph). The value is a list of predictions
        # from each subgraph (node) in the region.
        # node_results = {i: [None] * get_num_nodes(dataset, i) for i in inds}
        # graph_results = {i: [None] * get_num_nodes(dataset, i) for i in inds}

        # for i in inds:

        def process_index(i):
            nonlocal node_results, graph_results, confusion_matrix_results, precision_recall_results, node_embeddings_results
            if print_progress:
                print("predict on %d" % i)
            if dataset.subgraph_source == "chunk_save":
                dataset.clear_cache()
                dataset.load_to_cache(i, subgraphs=True)
            batch = []

            all_inds = np.arange(get_num_nodes(dataset, i))
            if shuffle:
                np.random.shuffle(all_inds)
            for j in all_inds[: int(subsample_ratio * len(all_inds))]:
                data = dataset.get_subgraph(i, j)
                for transform_fn in dataset.transform:
                    data = transform_fn(data)
                batch.append(data)
                if len(batch) == batch_size:
                    (
                        node_preds,
                        graph_preds,
                        confusion_matrix_info,
                        precision_recall_info,
                        node_embeddings,
                    ) = predict_on_batch(batch, model, device, dataset)
                    # confusion_matrix_results.append(confusion_matrix_info)
                    # precision_recall_results.append(precision_recall_info)
                    # node_embeddings_results.append(node_embeddings)
                    save_pred(
                        dataset,
                        batch,
                        node_preds,
                        graph_preds,
                        confusion_matrix_info,
                        precision_recall_info,
                        node_embeddings,
                        node_results,
                        graph_results,
                        region_ids,
                        confusion_matrix_results,
                        precision_recall_results,
                        node_embeddings_results,
                    )
                    batch = []
            if len(batch) > 0:
                (
                    node_preds,
                    graph_preds,
                    confusion_matrix_info,
                    precision_recall_info,
                    node_embeddings,
                ) = predict_on_batch(batch, model, device, dataset)
                save_pred(
                    dataset,
                    batch,
                    node_preds,
                    graph_preds,
                    confusion_matrix_info,
                    precision_recall_info,
                    node_embeddings,
                    node_results,
                    graph_results,
                    region_ids,
                    confusion_matrix_results,
                    precision_recall_results,
                    node_embeddings_results,
                )

        with ThreadPoolExecutor(max_workers=NO_JOBS) as executor:
            executor.map(process_index, inds)

    elif dataset.subsample_neighbor_size == 0:
        # Predicting on full graphs
        # node_results = {}
        # graph_results = {}
        def process_index(i):
            nonlocal node_results, graph_results
            full_g = dataset.get_full(i)
            for transform_fn in dataset.transform:
                full_g = transform_fn(full_g)
            full_g = full_g.to(device)
            res = model(full_g)
            node_results[i] = res[0].cpu().data.numpy()
            graph_results[i] = res[1].cpu().data.numpy()

        with ThreadPoolExecutor(max_workers=NO_JOBS) as executor:
            executor.map(process_index, inds)

    return node_results, graph_results, confusion_matrix_results, precision_recall_results, node_embeddings_results


def collect_predict_by_random_sample(
    model, dataset, device, inds=None, batch_size=64, num_eval_iterations=300, num_workers=0, **kwargs
):
    """Collect predictions on randomly sampled subgraphs from the dataset

    Args:
        model (nn.Module): pytorch model
        dataset (CODEXGraphDataset): dataset of cellular graphs
        device (str/torch.device): device to run the model on
        inds (array-like): list of integer indices of graphs to predict on,
            used for training/validation/test splitting
        batch_size (int): batch size for model prediction
        num_eval_iterations (int): number of evaluation iterations
        num_workers (int): number of workers for data loading

    Returns:
        node_preds (list): list of batch predictions on node-level tasks,
            empty if there is no node-level task
        node_labels (list): list of batch labels of node-level tasks,
            empty if there is no node-level task
        graph_preds (list): list of batch predictions on graph-level tasks,
            empty if there is no graph-level task
        graph_ys (list): list of batch labels of graph-level tasks,
            empty if there is no graph-level task
        graph_ws (list): list of batch weights of graph-level tasks,
            empty if there is no graph-level task
    """

    model = model.to(device)
    model.eval()
    if inds is None:
        inds = np.arange(dataset.N)

    original_subgraph_source = dataset.subgraph_source
    dataset.subgraph_source = "on-the-fly"

    data_iter = SubgraphSampler(
        dataset,
        selected_inds=inds,
        batch_size=batch_size,
        num_regions_per_segment=0,
        steps_per_segment=int(num_eval_iterations),
        num_workers=num_workers,
    )

    node_preds = []
    node_labels = []
    graph_preds = []
    graph_ys = []
    graph_ws = []
    for i_iter in range(int(num_eval_iterations)):
        batch = next(data_iter)
        batch = batch.to(device)

        res = model(batch)
        if model.num_node_tasks > 0:
            node_y = batch.node_y
            node_pred = res[0]
            node_labels.append(node_y.detach().cpu().data.numpy())
            node_preds.append(node_pred.detach().cpu().data.numpy())

        if model.num_graph_tasks > 0:
            graph_y, graph_w = batch.graph_y.float(), batch.graph_w.float()
            graph_pred = res[-1]
            graph_preds.append(graph_pred.detach().cpu().data.numpy())
            graph_ys.append(graph_y.detach().cpu().data.numpy())
            graph_ws.append(graph_w.detach().cpu().data.numpy())

    dataset.set_subgraph_source(original_subgraph_source)
    dataset.set_indices(np.arange(dataset.N))
    if model.num_node_tasks > 0:
        node_preds = np.concatenate(node_preds, 0)
        node_labels = np.concatenate(node_labels, 0)
    if model.num_graph_tasks > 0:
        graph_preds = np.concatenate(graph_preds, 0)
        graph_ys = np.concatenate(graph_ys, 0)
        graph_ws = np.concatenate(graph_ws, 0)
    return node_preds, node_labels, graph_preds, graph_ys, graph_ws


# Evaluate fns (for random subgraph samples)
def cell_type_prediction_evaluate_fn(node_preds, node_labels, print_res=True):
    """Evaluate center cell type prediction accuracy

    Args:
        node_preds (array-like): logits of cell type prediction, (num_subgraphs, num_cell_types)
        node_labels (array-like): ground truth cell type labels, (num_subgraphs,)
        print_res (bool): if to print the accuracy results

    Returns:
        list: list of metrics
    """
    node_preds = softmax(node_preds, -1)

    # Average predicted probability of the true cell type
    node_labels = np.array(node_labels).astype(int).reshape((-1,))
    avg_pred = node_preds[(np.arange(node_preds.shape[0]), node_labels)].mean()

    # Top-1, 3, 5 accuracy
    pred_order = np.argsort(node_preds, -1)
    _labels = node_labels.reshape((-1, 1))
    top1_acc = (_labels == pred_order[:, -1:]).sum() / node_labels.size
    top3_acc = (_labels == pred_order[:, -3:]).sum() / node_labels.size
    top5_acc = (_labels == pred_order[:, -5:]).sum() / node_labels.size
    if print_res:
        print("NODE Avg-pred: %.2f, Acc top-1 %.2f; top-3 %.2f; top-5 %.2f" % (avg_pred, top1_acc, top3_acc, top5_acc))
    return [avg_pred, top1_acc, top3_acc, top5_acc]


def cell_bm_exp_prediction_evaluate_fn(node_preds, node_labels, print_res=True):
    """Evaluate cell-level biomarker expression prediction

    Args:
        node_preds (array-like): predicted biomarker expression, (num_subgraphs, num_biomarkers)
        node_labels (array-like): ground truth biomarker expression, (num_subgraphs, num_biomarkers)
        print_res (bool): if to print the accuracy results

    Returns:
        list: list of metrics (r2 scores)
    """
    r2s = [r2_score(node_labels[:, i], node_preds[:, i]) for i in range(node_labels.shape[1])]
    r2s.append(np.mean(r2s))
    if print_res:
        print("Avg-R2: %.3f" % np.mean(r2s))
    return r2s


def graph_classification_evaluate_fn(graph_preds, graph_ys, graph_ws=None, print_res=True):
    """Evaluate graph classification accuracy

    Args:
        graph_preds (array-like): binary classification logits for graph-level tasks, (num_subgraphs, num_tasks)
        graph_ys (array-like): binary labels for graph-level tasks, (num_subgraphs, num_tasks)
        graph_ws (array-like): weights for graph-level tasks, (num_subgraphs, num_tasks)
        print_res (bool): if to print the accuracy results

    Returns:
        list: list of metrics on all graph-level tasks
    """
    if graph_ws is None:
        graph_ws = np.ones_like(graph_ys)
    scores = []
    for task_i in range(graph_ys.shape[1]):
        _label = graph_ys[:, task_i]
        _pred = graph_preds[:, task_i]
        _w = graph_ws[:, task_i]
        s = roc_auc_score(_label[np.where(_w > 0)], _pred[np.where(_w > 0)])
        scores.append(s)
    if print_res:
        print("GRAPH %s" % str(scores))
    return scores


# Evaluate fns (for whole graph evaluation)
def full_graph_cell_type_prediction_evaluate_fn(dataset, node_results, print_res=True):
    node_preds = []
    node_labels = []
    for i in node_results:
        cell_type_ar = dataset.get_full(i).x[:, 0].cpu().data.numpy()
        for j, p in enumerate(node_results[i]):
            if p is not None:
                node_preds.append(p)
                node_labels.append(cell_type_ar[j])

    node_labels = np.array(node_labels).reshape((-1,))
    node_preds = np.stack(node_preds, 0)
    return cell_type_prediction_evaluate_fn(node_preds, node_labels, print_res=print_res)


def full_graph_cell_bm_exp_prediction_evaluate_fn(dataset, node_results, print_res=True):
    node_preds = []
    node_labels = []
    for i in node_results:
        for j, p in enumerate(node_results[i]):
            if p is not None:
                node_preds.append(p)
                # Get bm exp labels
                d = dataset.get_subgraph(i, j)
                for t in dataset.transform:
                    d = t(d)
                node_labels.append(d.node_y.cpu().data.numpy())

    node_labels = np.concatenate(node_labels, 0)
    node_preds = np.stack(node_preds, 0)
    return cell_bm_exp_prediction_evaluate_fn(node_preds, node_labels, print_res=print_res)


def full_graph_graph_classification_evaluate_fn(dataset, graph_results, aggr="mean", print_res=True):
    n_tasks = dataset[0].graph_y.data.numpy().size
    graph_preds = []
    graph_ys = []
    graph_ws = []
    for i in graph_results:
        graph_pred = [p for p in graph_results[i] if ((p is not None) and np.all(p == p))]
        graph_pred = np.stack(graph_pred, 0)

        if aggr == "mean":
            graph_pred = np.nanmean(graph_pred, 0)
        else:
            raise NotImplementedError("Only mean-aggregation is supported now")

        graph_y = dataset[i].graph_y.data.numpy().flatten()
        graph_w = dataset[i].graph_w.data.numpy().flatten()
        graph_preds.append(graph_pred)
        graph_ys.append(graph_y)
        graph_ws.append(graph_w)

    graph_preds = np.concatenate(graph_preds, 0).reshape((-1, n_tasks))
    graph_ys = np.concatenate(graph_ys, 0).reshape((-1, n_tasks))
    graph_ws = np.concatenate(graph_ws, 0).reshape((-1, n_tasks))
    return graph_classification_evaluate_fn(graph_preds, graph_ys, graph_ws, print_res=print_res)

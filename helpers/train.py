"""
Helper functions for training node and graph classification models, including
evaluation and model saving functions
"""

import os
import numpy as np
import torch
import torch.optim
import mlflow
from .data import SubgraphSampler
from .inference import collect_predict_for_all_nodes, collect_predict_by_random_sample
from .mlflow_client_ import MLFLOW_TRACKING_URI
from .logging_setup import logger


def train_subgraph(
    model,
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
    graph_loss_weight=1.0,
    dataset_kwargs={},
    embedding_log_freq=100_000,
    **kwargs,
):
    """Train a GNN model through sampling subgraphs

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
        embedding_log_freq (int) : log the cell type embeddings as numpy array
        evaluate_fn (list): list of callback functions to evaluate the model
            during training
        evaluate_on_train (bool): if to evaluate the model on training set,
            will take longer time
        batch_size (int): batch size
        lr (float): learning rate
        graph_loss_weight (float): weight of graph task loss relative to node task loss
        **kwargs: additional arguments for callback functions
    """

    mlflow.set_experiment(kwargs["experiment_name"])

    with mlflow.start_run(run_name=kwargs["run_name"]) as run:
        directory_run_artifacts = f"{MLFLOW_TRACKING_URI}{run.info.experiment_id}/{run.info.run_id}/artifacts"
        for item in [
            "embeddings",
            "node_probs",
            "node_class",
            "node_class_pred",
            "model",
            "scorefile",
        ]:
            if not os.path.exists(f"{directory_run_artifacts}/{item}"):
                os.makedirs(f"{directory_run_artifacts}/{item}")
        kwargs["model_folder"] = f"{directory_run_artifacts}/model"
        kwargs["score_file"] = f"{directory_run_artifacts}/scorefile/results.txt"

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
            "num_iterations": num_iterations,
            "num_regions_per_segment": num_regions_per_segment,
            "num_iterations_per_segment": num_iterations_per_segment,
            "num_workers": num_workers,
            "evaluate_freq": evaluate_freq,
            "embedding_log_freq": embedding_log_freq,
            "evaluate_fn": evaluate_fn,
            "evaluate_on_train": evaluate_on_train,
            "batch_size": batch_size,
            "lr": lr,
            "graph_loss_weight": graph_loss_weight,
        }

        # Log the parameters
        mlflow.log_params(params_to_log)

        # Log all other keyword arguments dynamically
        for arg, value in kwargs.items():
            mlflow.log_param(arg, value)

        model.zero_grad()
        best_node_loss_metric_value = float("inf")

        model = model.to(device)
        model.train()
        if train_inds is None:
            train_inds = np.arange(len(dataset))

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        node_losses = []
        graph_losses = []

        data_iter = SubgraphSampler(
            dataset,
            selected_inds=train_inds,
            batch_size=batch_size,
            num_regions_per_segment=int(num_regions_per_segment),
            steps_per_segment=int(num_iterations_per_segment),
            num_workers=num_workers,
        )

        for i_iter in range(int(num_iterations)):
            batch = next(data_iter)
            batch = batch.to(device)

            res = model(batch)
            loss = 0.0
            if model.num_node_tasks > 0:
                assert node_task_loss_fn is not None, "Please specify `node_task_loss_fn` in the training kwargs"
                node_y = batch.node_y
                node_pred = res[0]

                node_loss = node_task_loss_fn(node_pred, node_y)
                loss += node_loss
                node_losses.append(node_loss.to("cpu").data.item())

            if model.num_graph_tasks > 0:
                assert graph_task_loss_fn is not None, "Please specify `graph_task_loss_fn` in the training kwargs"
                graph_y, graph_w = batch.graph_y.float(), batch.graph_w.float()
                graph_pred = res[-1]
                graph_loss = graph_task_loss_fn(graph_pred, graph_y, graph_w)
                loss += graph_loss * graph_loss_weight
                graph_losses.append(graph_loss.to("cpu").data.item())

            loss.backward()
            optimizer.step()
            model.zero_grad()

            # Log embeddings
            if i_iter > 0 and i_iter % embedding_log_freq == 0:
                embeddings = model.gnn.x_embedding.weight.detach().cpu().numpy()
                node_classes = node_y.cpu().numpy()
                node_classes_pred = res[0].detach().cpu().numpy()
                node_probs = torch.nn.functional.softmax(res[0], dim=1).detach().cpu().numpy()

                embedding_filename = f"{directory_run_artifacts}/embeddings/{i_iter}.npy"
                node_probs_filename = f"{directory_run_artifacts}/node_probs/{i_iter}.npy"
                node_classes_filename = f"{directory_run_artifacts}/node_class/{i_iter}.npy"
                node_classes_pred_filename = f"{directory_run_artifacts}/node_class_pred/{i_iter}.npy"

                np.save(embedding_filename, embeddings)
                np.save(node_probs_filename, node_probs)
                np.save(node_classes_filename, node_classes)
                np.save(node_classes_pred_filename, node_classes_pred)

            if i_iter > 0 and i_iter % evaluate_freq == 0:
                summary_str = "Finished iterations %d" % i_iter
                if len(node_losses) > 100:
                    summary_str += ", node loss %.2f" % np.mean(node_losses[-100:])
                    mlflow.log_metric("train_node_loss", np.mean(node_losses[-100:]), step=i_iter)
                else:
                    summary_str += ", node loss %.2f" % np.mean(node_losses[:])
                    mlflow.log_metric("train_node_loss", np.mean(node_losses[:]), step=i_iter)

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
                        **kwargs,
                    )
                    graph_type, *_rest_test_valid = result_list
                    (
                        dataset_type,
                        _,
                        avg_pred,
                        top1_acc,
                        top3_acc,
                        top5_acc,
                        *_rest_valid,
                    ) = _rest_test_valid

                    mlflow.log_metric(f"{graph_type}-{dataset_type}-avg_pred", avg_pred, step=i_iter)
                    mlflow.log_metric(f"{graph_type}-{dataset_type}-top1_acc", top1_acc, step=i_iter)
                    mlflow.log_metric(f"{graph_type}-{dataset_type}-top3_acc", top3_acc, step=i_iter)
                    mlflow.log_metric(f"{graph_type}-{dataset_type}-top5_acc", top5_acc, step=i_iter)

                    if _rest_valid:
                        (
                            dataset_type,
                            _,
                            avg_pred,
                            top1_acc,
                            top3_acc,
                            top5_acc,
                        ) = _rest_valid
                        mlflow.log_metric(
                            f"{graph_type}-{dataset_type}-avg_pred",
                            avg_pred,
                            step=i_iter,
                        )
                        mlflow.log_metric(
                            f"{graph_type}-{dataset_type}-top1_acc",
                            top1_acc,
                            step=i_iter,
                        )
                        mlflow.log_metric(
                            f"{graph_type}-{dataset_type}-top3_acc",
                            top3_acc,
                            step=i_iter,
                        )
                        mlflow.log_metric(
                            f"{graph_type}-{dataset_type}-top5_acc",
                            top5_acc,
                            step=i_iter,
                        )

                for fn in kwargs["model_save_fn"]:
                    current_metric_value = (
                        np.mean(node_losses[-100:]) if len(node_losses) > 100 else np.mean(node_losses[:])
                    )
                    fn(
                        model,
                        dataset,
                        device,
                        best_model_metric="train_node_loss",
                        best_model_metric_value=best_node_loss_metric_value,
                        current_metric_value=current_metric_value,
                        **kwargs,
                    )
                    if current_metric_value < best_node_loss_metric_value:
                        best_node_loss_metric_value = current_metric_value

                model.train()
        dataset.set_indices(np.arange(dataset.N))
    return model


# %% Callback functions for evaluation
def evaluate_by_sampling_subgraphs(
    model,
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
    **kwargs,
):
    """Callback function for evaluating GNN predictions on randomly sampled subgraphs

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
        (
            node_preds,
            node_labels,
            graph_preds,
            graph_ys,
            graph_ws,
        ) = collect_predict_by_random_sample(
            model,
            dataset,
            device,
            inds=train_inds,
            batch_size=batch_size,
            num_eval_iterations=num_eval_iterations,
            num_workers=num_workers,
            **kwargs,
        )
        if len(node_preds) > 0:
            # Evalaute node-level predictions
            assert node_task_evaluate_fn is not None, "Please specify `node_task_evaluate_fn` in the training kwargs"
            score_row.append("node-score")
            score_row.extend(node_task_evaluate_fn(node_preds, node_labels, print_res=False))

        if len(graph_preds) > 0:
            # Evaluate graph-level predictions
            assert graph_task_evaluate_fn is not None, "Please specify `graph_task_evaluate_fn` in the training kwargs"
            score_row.append("graph-score")
            score_row.extend(graph_task_evaluate_fn(graph_preds, graph_ys, graph_ws, print_res=False))
    if valid_inds is not None:
        # Same for validation samples
        score_row.append("Valid")
        (
            node_preds,
            node_labels,
            graph_preds,
            graph_ys,
            graph_ws,
        ) = collect_predict_by_random_sample(
            model,
            dataset,
            device,
            inds=valid_inds,
            batch_size=batch_size,
            num_eval_iterations=num_eval_iterations,
            num_workers=num_workers,
            **kwargs,
        )
        if len(node_preds) > 0:
            assert node_task_evaluate_fn is not None, "Please specify `node_task_evaluate_fn` in the training kwargs"
            score_row.append("node-score")
            score_row.extend(node_task_evaluate_fn(node_preds, node_labels, print_res=False))

        if len(graph_preds) > 0:
            assert graph_task_evaluate_fn is not None, "Please specify `graph_task_evaluate_fn` in the training kwargs"
            score_row.append("graph-score")
            score_row.extend(graph_task_evaluate_fn(graph_preds, graph_ys, graph_ws, print_res=False))

    if score_file is not None:
        with open(score_file, "a") as f:
            f.write(",".join([s if isinstance(s, str) else ("%.3f" % s) for s in score_row]) + "\n")
    return score_row


def evaluate_by_full_graph(
    model,
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
    **kwargs,
):
    """Callback function for evaluating GNN predictions on regions (full cellular graphs)

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
            **kwargs,
        )
        if model.num_node_tasks > 0:
            # Evaluate node-level predictions
            assert (
                full_graph_node_task_evaluate_fn is not None
            ), "Please specify `full_graph_node_task_evaluate_fn` in the training kwargs"
            score_row.append("node-score")
            score_row.extend(full_graph_node_task_evaluate_fn(dataset, node_preds, print_res=False))
        if model.num_graph_tasks > 0:
            # Evaluate graph-level predictions
            assert (
                full_graph_graph_task_evaluate_fn is not None
            ), "Please specify `full_graph_graph_task_evaluate_fn` in the training kwargs"
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
            **kwargs,
        )
        if model.num_node_tasks > 0:
            assert (
                full_graph_node_task_evaluate_fn is not None
            ), "Please specify `full_graph_node_task_evaluate_fn` in the training kwargs"
            score_row.append("node-score")
            score_row.extend(full_graph_node_task_evaluate_fn(dataset, node_preds, print_res=False))
        if model.num_graph_tasks > 0:
            assert (
                full_graph_graph_task_evaluate_fn is not None
            ), "Please specify `full_graph_graph_task_evaluate_fn` in the training kwargs"
            score_row.append("graph-score")
            score_row.extend(full_graph_graph_task_evaluate_fn(dataset, graph_preds, print_res=False))

    if score_file is not None:
        with open(score_file, "a") as f:
            f.write(",".join([s if isinstance(s, str) else ("%.3f" % s) for s in score_row]) + "\n")
    return score_row


def save_model_weight(model, dataset, device, model_folder=None, **kwargs):
    if model_folder is not None:
        os.makedirs(model_folder, exist_ok=True)
        fs = [f for f in os.listdir(model_folder) if f.startswith("model_save")]
        torch.save(model.state_dict(), os.path.join(model_folder, "model_save_%d.pt" % len(fs)))
    return


def save_models_best_latest(
    model,
    dataset,
    device,
    model_folder=None,
    best_model_metric=None,
    best_model_metric_value=None,
    current_metric_value=None,
    **kwargs,
):
    """
    Save the latest model and update the best model based on a specified metric.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        dataset: Your dataset (if needed for metrics calculation).
        device: The device on which the model is stored.
        model_folder (str): The folder to save the models.
        best_model_metric (str): The metric to use for determining the best model.
        best_model_metric_value: The best metric value achieved so far.
        current_metric_value: The metric value for the current iteration.
        **kwargs: Additional arguments.

    Returns:
        None
    """
    if model_folder is not None:
        os.makedirs(model_folder, exist_ok=True)

        # Save the latest model
        torch.save(model.state_dict(), os.path.join(model_folder, "latest_model.pt"))

        # Update the best model if a metric is specified
        if best_model_metric is not None:
            if best_model_metric_value is None or current_metric_value < best_model_metric_value:
                torch.save(model.state_dict(), os.path.join(model_folder, "best_model.pt"))

    return

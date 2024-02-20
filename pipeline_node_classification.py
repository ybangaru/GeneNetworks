import warnings
import json
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
import helpers
from helpers import CellularGraphDataset, NO_JOBS, train_subgraph, logger, MLFLOW_TRACKING_URI, MLFLOW_CLIENT

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")


def train_node_classification(dataset_kwargs):
    dataset = CellularGraphDataset(root=dataset_kwargs["dataset_root"], **dataset_kwargs)
    # test = dataset.plot_subgraph(100, 800)
    # logger.info(dataset)
    # logger.info(dataset[0])

    transformers = [
        # `AddCenterCellType` will add `node_y` attribute to the subgraph for node-level prediction task
        # In this task we will mask the cell type of the center cell (replace it by a placeholder cell type)
        # and use its neighborhood to predict the true cell type
        helpers.AddCenterCellType(dataset),
        # `AddGraphLabel` will add `graph_y` and `graph_w` attributes to the subgraph for graph-level prediction task
        # helpers.AddGraphLabel(graph_label_file, tasks=['survival_status']),
    ]

    if dataset_kwargs["node_features_mask"]:
        use_node_features = [
            x for x in dataset_kwargs["node_features"] if x not in dataset_kwargs["node_features_mask"]
        ]
        if dataset_kwargs["center_node_features_mask"]:
            use_center_node_features = [
                x for x in use_node_features if x not in dataset_kwargs["center_node_features_mask"]
            ]
        transformers.append(
            helpers.FeatureMask(
                dataset,
                use_center_node_features=use_center_node_features
                if dataset_kwargs["center_node_features_mask"]
                else use_node_features,
                use_neighbor_node_features=use_node_features,
                use_edge_features=dataset.edge_feature_names,
            )
        )

    # TODO: handle edge features mask

    dataset.set_transforms(transformers)

    model_kwargs = {
        "num_layer": dataset.subgraph_size,  # same number of layers as number of hops in the subgraphs
        # +1 needed to assign the Unknown type during cell type masking transformation
        "num_node_type": len(dataset.cell_type_mapping) + 1,
        # number of embeddings for different cell types (plus one placeholder cell type)
        "num_feat": dataset[0].x.shape[1] - 1,  # exclude the cell type column
        "emb_dim": dataset_kwargs["node_embedding_size"],
        "emb_dim_edge": dataset_kwargs["edge_embedding_size"],
        "num_node_tasks": len(
            dataset.cell_type_mapping
        ),  # A multi-class classification task: predicting center cell type
        "num_graph_tasks": 0,  # a binary classification task
        "node_embedding_output": "last",
        "drop_ratio": 0.15,
        "graph_pooling": "max",
        "gnn_type": dataset_kwargs["graph_type"],
        "return_node_embedding": False,
        "edge_types": dataset_kwargs["edge_types"],
    }

    model = helpers.GNN_pred(**model_kwargs)
    # model_file = f"{PROJECT_DIR}/data/Liver12Slice12_leiden_res_0.7/model_Liver12Slice12_0.7/voronoi_delaunay-gin-emb_size_128-batch_size=128-lr=0.01/model_save_3188.pt"
    # model.from_pretrained(model_file, strict_bool=False)
    device = "cpu"

    learning_rate = 0.01
    BATCH_SIZE = 512

    train_kwargs = {
        "experiment_name": f'{dataset_kwargs["network_type"]}',
        "run_name": f'{dataset_kwargs["graph_type"]}-emb_dim={model_kwargs["emb_dim"]}-subgraph_size={dataset_kwargs["subgraph_size"]}-subgraph_radius={dataset_kwargs["subgraph_radius_limit"]}',
        "batch_size": BATCH_SIZE,  # No of subgraphs per batch
        "lr": learning_rate,
        "graph_loss_weight": 1.0,  # Weight of graph task loss relative to node task loss
        "num_iterations": 30_000,
        # Loss functions
        "node_task_loss_fn": nn.CrossEntropyLoss(),
        # 'graph_task_loss_fn': helpers.models.BinaryCrossEntropy(),
        # Evaluation during training
        "subsample_ratio": 0.1,  # Subsample 10% of the data for evaluation
        "evaluate_fn": [helpers.train.evaluate_by_sampling_subgraphs],  # , helpers.train.evaluate_by_full_graph
        "evaluate_on_train": True,
        "model_save_fn": [helpers.train.save_models_best_latest],
        "evaluate_freq": 20,  # Evaluate the model every 10 iterations
        "embedding_log_freq": 20,
        "log_edge_embeddings": True,
    }

    evaluate_kwargs = {
        "node_task_evaluate_fn": helpers.inference.cell_type_prediction_evaluate_fn,
        # 'graph_task_evaluate_fn': helpers.inference.graph_classification_evaluate_fn,
        # 'full_graph_node_task_evaluate_fn': helpers.inference.full_graph_cell_type_prediction_evaluate_fn,
        # 'full_graph_graph_task_evaluate_fn': helpers.inference.full_graph_graph_classification_evaluate_fn,
        "num_eval_iterations": 5,
    }
    train_kwargs.update(evaluate_kwargs)

    # Set a random seed for reproducibility
    np.random.seed(42)
    num_regions = len(dataset.region_ids)
    all_inds = list(range(num_regions))
    train_inds, valid_inds = train_test_split(all_inds, test_size=0.20)

    # Print the indices for the training and validation sets
    train_inds_str = ",".join(map(str, train_inds))
    valid_inds_str = ",".join(map(str, valid_inds))
    logger.info(f"Training indices: {train_inds_str}")
    logger.info(f"Validation indices: {valid_inds_str}")

    train_kwargs["train_inds"] = train_inds
    train_kwargs["valid_inds"] = valid_inds
    # train_kwargs['num_iterations'] = 10_000 -- being set above
    train_kwargs["num_regions_per_segment"] = 0
    train_kwargs["num_iterations_per_segment"] = 10
    train_kwargs["num_workers"] = NO_JOBS

    logger.info("Training model...")
    logger.info(train_kwargs["experiment_name"])
    logger.info(train_kwargs["run_name"])

    model = train_subgraph(model, dataset, device, dataset_kwargs=dataset_kwargs, **train_kwargs)


def build_train_kwargs(network_type, graph_type, kwargs):
    # TODO: subgraphs not using SUBGRAPH_RADIUS_LIMIT while plotting

    dataset_root = kwargs["mlflow_config"]["networkx_gpkl_root"]

    clustering_run_id = kwargs["mlflow_config"]["id"]
    clustering_run_info = MLFLOW_CLIENT.get_run(clustering_run_id)
    clustering_run_data = clustering_run_info.data

    networkx_build_config_path = f"{dataset_root}/networkx_build_config.json"
    with open(networkx_build_config_path, "r") as f:
        json_data = json.load(f)
        cell_annotation_key = json_data["cell_type_column"]

    SUBGRAPH_SIZE = 3
    SUBGRAPH_RADIUS_LIMIT = 184
    # NEIGHBORHOOD_SIZE = 15  # 15 * SUBGRAPH_SIZE
    NODE_EMBEDDING_SIZE = 130
    EDGE_EMBEDDING_SIZE = 130

    # TODO: test feature/layer normalization instead of batch normalization for training as shape features
    # features are resulting in overfitting and poor generalization

    NODE_FEATURES = [
        "cell_type",
        "center_coord",
        "biomarker_expression",
        "volume",
        # "neighborhood_composition",
        # "perimeter",
        # "area",
        # "mean_radius",
        # "compactness_circularity",
        # "fractality",
        # "concavity",
        # "elongation",
        # "overlap_index",
        # "sbro_orientation",
        # "wswo_orientation",
        "voronoi_polygon" if "voronoi" in network_type else "boundary_polygon",
    ]
    NODE_FEATURES_MASK = ["center_coord"]
    CENTER_NODE_FEATURES_MASK = ["center_coord", "biomarker_expression"]
    EDGE_FEATURES = ["edge_type", "distance"]  # edge_type must be first variable (cell pair) features "edge_type",
    EDGE_FEATURES_MASK = None
    EDGE_TYPES = {
        "neighbor": 0,
        "distant": 1,
        "self": 2,
    }

    dataset_kwargs = {
        "clustering_run_id": kwargs["mlflow_config"]["id"],
        "dataset_filter": kwargs["mlflow_config"]["data_filter_name"],
        "leiden_annotation_column": cell_annotation_key,
        "leiden_cluster_res": eval(clustering_run_data.params["leiden"])["resolution"],
        "transform": [],
        "pre_transform": None,
        "dataset_root": dataset_root,
        "raw_folder_name": "nx_files",
        "processed_folder_name": "nx_files_processed",
        "network_type": network_type,
        "node_features": NODE_FEATURES,
        "node_features_mask": NODE_FEATURES_MASK,
        "center_node_features_mask": CENTER_NODE_FEATURES_MASK,
        "edge_features": EDGE_FEATURES,
        "edge_features_mask": EDGE_FEATURES_MASK,
        "edge_types": EDGE_TYPES,
        "graph_type": graph_type,
        "node_embedding_size": NODE_EMBEDDING_SIZE,
        "edge_embedding_size": EDGE_EMBEDDING_SIZE,
        "subgraph_size": SUBGRAPH_SIZE,  # indicating we want to sample 3-hop subgraphs from these regions (for training/inference), this is a core parameter for SPACE-GM.
        "subgraph_source": "on-the-fly",
        "subgraph_allow_distant_edge": True,
        "subgraph_radius_limit": SUBGRAPH_RADIUS_LIMIT,
        "sampling_avoid_unassigned": True,
        # "unassigned_cell_type": "Unknown",
    }

    feature_kwargs = {
        # "biomarker_expression_process_method": "linear",
        # "biomarker_expression_lower_bound": 0,
        # "biomarker_expression_upper_bound": 18,
        # "neighborhood_size": NEIGHBORHOOD_SIZE,
    }
    dataset_kwargs.update(feature_kwargs)

    return dataset_kwargs


def main():
    all_liver_samples = ["Liver1Slice1", "Liver1Slice2", "Liver2Slice1", "Liver2Slice2"]
    data_filter_name = "Liver12Slice12"

    CLUSTERING_EXP_ID = "834659805990264659"
    CLUSTERING_RUN_ID = "c750f81070fa4ccbbd731b92746bebc6"
    UNIQUE_IDENTIFIER = "subcluster-finetune"
    GRAPH_FOLDER_NAME = "graph"

    SEGMENTS_PER_DIMENSION = 20
    chosen_network = "voronoi_delaunay"  # "given_delaunay", "given_r3index", "given_mst"
    graph_type = "gin"  # "gcn", "graphsage", "gat"
    network_type = f"{chosen_network}_{SEGMENTS_PER_DIMENSION}"

    data_options = {
        "names_list": all_liver_samples,
        "mlflow_config": {
            "id": CLUSTERING_RUN_ID,
            "networkx_gpkl_root": f"{MLFLOW_TRACKING_URI}{CLUSTERING_EXP_ID}/{CLUSTERING_RUN_ID}/artifacts/{UNIQUE_IDENTIFIER}/{GRAPH_FOLDER_NAME}/{network_type}",
            "data_filter_name": data_filter_name,
        },
    }

    try:
        dataset_kwargs = build_train_kwargs(network_type, graph_type, data_options)
        train_node_classification(dataset_kwargs)
    except Exception as e:
        logger.info(f"{network_type}-{graph_type} failed")
        logger.error(f"{str(e)}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 13:45:39 2021

@author: zhenq
"""

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from torch_geometric.data import Batch

from .data import SubgraphSampler


def get_random_sampled_subgraphs(dataset,
                                 inds=None,
                                 n_samples=32768,
                                 batch_size=64,
                                 num_workers=0,
                                 seed=123):
    """Randomly sample subgraphs from the dataset

    Args:
        dataset (CellularGraphDataset): target dataset
        inds (list): list of indices (of regions) to sample from,
            helpful for sampling only training/validation regions
        n_samples (int): number of subgraphs to sample
        batch_size (int): batch size for sampling
        num_workers (int): number of workers for sampling
        seed (int): random seed

    Returns:
        list: list of subgraphs (as pyg data objects)
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    original_indices = dataset.indices()
    # Reset dataset indices
    dataset.set_indices()
    if inds is None:
        inds = np.arange(dataset.N)
    n_iterations = int(np.ceil(n_samples / batch_size))
    data_iter = SubgraphSampler(dataset,
                                selected_inds=inds,
                                batch_size=batch_size,
                                num_regions_per_segment=0,
                                steps_per_segment=n_samples + 1,
                                num_workers=num_workers,
                                seed=seed)
    # Sample subgraphs
    data_list = []
    for _ in range(n_iterations):
        batch = next(data_iter)
        data_list.extend(batch.to_data_list())
    dataset.set_indices(original_indices)
    return data_list[:n_samples]


def get_embedding(model, data_list, device, batch_size=64):
    """Calculate center node embeddings, node predictions, graph embeddings,
    graph predictions for a list of subgraphs

    Args:
        model (nn.Module): model to calculate embeddings
        data_list (list): list of subgraphs (pyg data objects)
        device (str/torch.device): device to run the model on
        batch_size (int): batch size

    Returns:
        node_reps (np.ndarray): center node embeddings
        graph_reps (np.ndarray): graph embeddings
        preds (list): node and graph predictions
    """
    model = model.to(device)
    with torch.no_grad():
        num_batches = np.ceil(len(data_list) / batch_size)
        node_reps = []
        graph_reps = []
        preds = [[], []]
        for i in range(int(num_batches)):
            batch_data = data_list[i * batch_size:(i + 1) * batch_size]
            data = Batch.from_data_list(batch_data)
            data = data.to(device)

            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            batch = data.batch if 'batch' in data else torch.zeros((len(x),)).long().to(x.device)

            node_representation = model.gnn(x, edge_index, edge_attr)
            # Center node embedding
            center_node_rep = node_representation[data.center_node_index]
            node_reps.append(center_node_rep)

            # Graph embedding
            graph_rep = model.pool(node_representation, batch)
            graph_reps.append(graph_rep)

            # Node and graph predictions
            if model.num_node_tasks > 0:
                preds[0].append(model.node_pred_module(center_node_rep))
            if model.num_graph_tasks > 0:
                preds[1].append(model.graph_pred_module(graph_rep))

        node_reps = torch.cat(node_reps, 0).cpu().data.numpy()
        graph_reps = torch.cat(graph_reps, 0).cpu().data.numpy()
        preds = [torch.cat(p, 0).cpu().data.numpy() if len(p) > 0 else p for p in preds]
    return node_reps, graph_reps, preds


def get_composition_vector(data, n_cell_types=20):
    """Calculate the composition vector of a subgraph"""
    cts = data.x.cpu().data.numpy()[:, 0]
    composition_vec = np.zeros((n_cell_types,))
    for ct in cts:
        composition_vec[int(ct)] += 1
    return composition_vec / composition_vec.sum()


def get_adj_mat(data, n_cell_types=20):
    """Calculate the count matrix of cell type adjacency of a subgraph"""
    adj = np.zeros((n_cell_types, n_cell_types))
    for i_edge, edge in enumerate(np.transpose(data.edge_index.cpu().data.numpy())):
        if data.edge_attr[i_edge, 0].item() == 0:
            node1 = data.x[edge[0], 0].item()
            node2 = data.x[edge[1], 0].item()
            adj[int(node1), int(node2)] += 1
    return adj


def get_base_adj_mat(data_list):
    """Calculate the average frequency matrix of cell type adjacency
    of a list of subgraphs, column normalized
    """
    adj_i = np.stack([get_adj_mat(data) for data in data_list], 0).sum(0)
    adj_i = adj_i / adj_i.sum(1, keepdims=True)
    return adj_i


def dimensionality_reduction_combo(embs,
                                   n_pca_components=20,
                                   cluster_method='kmeans',
                                   n_clusters=10,
                                   seed=42,
                                   tool_saves=None):
    """Run a combination of dimensionality reduction and clustering

    Args:
        embs (np.ndarray): array of embeddings/composition vectors/other features
        n_pca_components (int): number of PCA components to use
        cluster_method (str): clustering method to use, one of "kmeans", "agg"
        n_clusters (int): number of clusters to use
        seed (int): random seed
        tool_saves (tuple): a tuple of dimensionality reduction and clustering objects, in the order of:
                (sklearn.decomposition.PCA, umap.UMAP, sklearn.cluster.KMeans/sklearn.cluster.AgglomerativeClustering)
            If provided, will use these objects to process the provided embeddings.
            If None, will create new objects and fit them to the provided embeddings.
            Note that UMAP will be skipped if not installed.

    Returns:
        np.ndarray: top PCs of the embeddings
        np.ndarray/None: UMAP-reduced embeddings if available
        np.ndarray: assigned cluster labels for the corresponding subgraphs
        tuple: tuple of dimensionality reduction and clustering objects, see docs for `tool_saves`
    """
    if seed is not None:
        np.random.seed(seed)
    if tool_saves is None:
        pca = PCA(n_components=n_pca_components, random_state=seed)
        pca_embs = pca.fit_transform(embs)

        try:
            import umap
            reducer = umap.UMAP(random_state=seed)
            umap_emb = reducer.fit_transform(pca_embs)
        except Exception as e:
            print("Error running umap, skipping: %s" % e)
            reducer = None
            umap_emb = None

        if cluster_method == 'agg':
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
        elif cluster_method == 'kmeans':
            clustering = KMeans(n_clusters=n_clusters, random_state=seed)
        else:
            raise NotImplementedError("Clustering method %s not supported" % cluster_method)

        clustering.fit(pca_embs)
        cluster_labels = clustering.labels_
    else:
        pca, reducer, clustering = tool_saves
        pca_embs = pca.transform(embs)
        umap_emb = None if reducer is None else reducer.transform(pca_embs)
        cluster_labels = clustering.predict(pca_embs)

    return pca_embs, umap_emb, cluster_labels, (pca, reducer, clustering)


def collect_cluster_label_for_all_nodes(model,
                                        dataset,
                                        device,
                                        tool_saves,
                                        inds=None,
                                        embedding_from='graph',
                                        print_progress=False):
    """Extract all subgraphs from the dataset, calculate their embeddings and
    cluster labels

    Args:
        model (nn.Module): model to calculate embeddings
        dataset (CellularGraphDataset): target dataset
        device (str/torch.device): device to run the model on
        tool_saves (tuple): tuple of dimensionality reduction and clustering objects,
            see docs for `tool_saves` in method `dimensionality_reduction_combo`
        inds (list): list of indices (of regions) to predict on,
            helpful for analyzing only training/validation regions
        embedding_from (str): source of embeddings, one of "node", "graph"
        print_progress (bool): if to print progress

    Returns:
        node_cluster_labels (dict): a dictionary of cluster labels formatted as:
            {region_index: [cluster_label_for_node_1, cluster_label_for_node_2, ...]}
    """

    original_indices = dataset.indices()
    dataset.set_indices()
    if inds is None:
        inds = np.arange(dataset.N)
    assert tool_saves is not None, "Please provide dimensionality reduction " + \
        "and clustering objects, see `dimensionality_reduction_combo` for details"

    def get_num_nodes(dataset, ind):
        return dataset.get_full(ind).num_nodes

    node_cluster_labels = {i: [None] * get_num_nodes(dataset, i) for i in inds}
    for i in inds:
        if print_progress:
            print("Predict on %d" % i)
        dataset.clear_cache()
        if dataset.subgraph_source == 'chunk_save':
            dataset.load_to_cache(i, subgraphs=True)

        # Collect all subgraphs from the region
        all_subgraphs = np.arange(get_num_nodes(dataset, i))
        data_dict = {}
        for j in all_subgraphs:
            data = dataset.get_subgraph(i, j)
            for transform_fn in dataset.transform:
                data = transform_fn(data)
            data = data.to(device)
            data_dict[(i, j)] = data

            if len(data_dict) >= 256 or j == all_subgraphs[-1]:
                keys = list(data_dict.keys())
                _data_list = [data_dict[k] for k in keys]
                node_embs, graph_embs, _ = get_embedding(model, _data_list, device)
                if embedding_from == 'node':
                    embs = node_embs
                elif embedding_from == 'graph':
                    embs = graph_embs
                else:
                    raise ValueError("Embedding from %s not supported" % embedding_from)
                _, _, cluster_labels, _ = dimensionality_reduction_combo(embs, tool_saves=tool_saves)
                for k, c in zip(keys, cluster_labels):
                    node_cluster_labels[k[0]][k[1]] = c
                data_dict = {}
        assert all([c is not None for c in node_cluster_labels[i]])

    dataset.set_indices(original_indices)
    return node_cluster_labels


def plot_umap(umap_emb, cluster_labels):
    """Plot UMAP embedding with cluster labels

    Args:
        umap_emb (np.ndarray): UMAP embedding
        cluster_labels (np.ndarray): cluster labels for the corresponding subgraphs
    """
    assert len(umap_emb) == len(cluster_labels)
    cm = matplotlib.cm.get_cmap("tab20")
    colors = [cm(cl % 20) for cl in cluster_labels]
    plt.scatter(umap_emb[:, 0], umap_emb[:, 1], s=1, c=colors)
    return


def get_biomarker_heatmap_for_cluster(data_list, cluster_labels, data_preds=None):
    """ Calculate biomarker heatmap for all subgraph clusters

    Args:
        data_list (list): list of subgraphs (as pyg data objects)
        cluster_labels (np.ndarray): cluster labels for the corresponding subgraphs
        data_preds (np.ndarray): predicted labels for the subgraphs, should be a 1-d array (optional)

    Returns:
        heatmap (np.ndarray): biomarker heatmap
        cluster_preds (np.ndarray): average predicted labels for the clusters,
            returned only when `data_preds` is provided`
        cluster_counts (np.ndarray): count of subgraphs in each cluster
    """
    n_cell_types = int(max(np.concatenate([d.x[:, 0].data.numpy() for d in data_list])) + 1)
    composition_vectors = [get_composition_vector(data, n_cell_types=n_cell_types) for data in data_list]
    assert len(cluster_labels) == len(data_list)

    unique_cluster_labels = sorted(set(cluster_labels))
    heatmap = np.zeros((len(unique_cluster_labels), n_cell_types))
    
    cluster_counts = np.zeros(len(unique_cluster_labels))
    if data_preds is not None:
        assert len(data_preds) == len(data_list)
        cluster_preds = np.zeros(len(unique_cluster_labels))

    for i in range(len(data_list)):
        comp_vec = composition_vectors[i]
        cluster_label = cluster_labels[i]
        assert np.allclose(comp_vec.sum(), 1)
        heatmap[unique_cluster_labels.index(cluster_label)] += comp_vec
        cluster_counts[unique_cluster_labels.index(cluster_label)] += 1
        if data_preds is not None:
            pred = data_preds[i]
            cluster_preds[unique_cluster_labels.index(cluster_label)] += pred

    heatmap /= cluster_counts.reshape(-1, 1)
    if data_preds is not None:
        cluster_preds /= cluster_counts
        return heatmap, cluster_preds, cluster_counts
    else:
        return heatmap, cluster_counts

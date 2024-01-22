# import os
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import torch
# import tempfile

# import spacegm
# from spacegm.graph_build import (
#     construct_graph_for_region,
#     calcualte_voronoi_from_coords,
#     build_graph_from_voronoi_polygons,
# )
# from spacegm.transform import (
#     FeatureMask,
#     AddCenterCellBiomarkerExpression,
#     AddCenterCellType,
#     AddCenterCellIdentifier,
#     AddGraphLabel,
# )
# from spacegm.embeddings_analysis import (
#     get_random_sampled_subgraphs,
#     get_embedding,
#     dimensionality_reduction_combo,
#     collect_cluster_label_for_all_nodes,
# )

# from spacegm.data import CellularGraphDataset, SubgraphSampler


# def simulate_data(n_cells=1000):
#     cell_ids = np.arange(n_cells)
#     cell_coords = {'CELL_ID': cell_ids,
#                    'X': np.random.uniform(0, 3000, (n_cells,)),
#                    'Y': np.random.uniform(0, 3000, (n_cells,))}
#     cell_coords = pd.DataFrame(cell_coords)

#     cell_types = {'CELL_ID': cell_ids,
#                   'CELL_TYPE': ['TYPE%d' % np.random.randint(0, 10) for _ in range(n_cells)]}
#     cell_types = pd.DataFrame(cell_types)

#     cell_bm_exp = {'CELL_ID': cell_ids}
#     for i in range(40):
#         cell_bm_exp['BIOMARKER%d' % i] = np.random.uniform(0., 100., (n_cells,))
#     cell_bm_exp = pd.DataFrame(cell_bm_exp)

#     cell_features = {'CELL_ID': cell_ids,
#                      'FEAT1': np.random.uniform(0, 1, (n_cells,)),
#                      'FEAT2': np.random.uniform(0, 1, (n_cells,))}
#     cell_features = pd.DataFrame(cell_features)

#     return cell_coords, cell_types, cell_bm_exp, cell_features


# def simulate_graph_labels(n_regions=5, region_prefix='REGION'):
#     df = {}
#     df['REGION_ID'] = ['%s%d' % (region_prefix, i) for i in range(n_regions)]
#     df['GRAPH_TASK1'] = [np.random.randint(0, 2) for _ in range(n_regions)]
#     df['GRAPH_TASK2'] = [np.random.uniform(0, 1) for _ in range(n_regions)]
#     df = pd.DataFrame(df)
#     return df


# def save_to_csv(cell_coords, cell_types, cell_bm_exp, cell_features=None, root=tempfile.mkdtemp()):
#     cell_coords.to_csv(os.path.join(root, 'cell_coords.csv'), index=False)
#     cell_types.to_csv(os.path.join(root, 'cell_types.csv'), index=False)
#     cell_bm_exp.to_csv(os.path.join(root, 'cell_bm_exp.csv'), index=False)
#     if cell_features is not None:
#         cell_features.to_csv(os.path.join(root, 'cell_features.csv'), index=False)
#     return root


# def build_dataset(n_regions=5, region_prefix='REGION'):
#     dataset_root = tempfile.mkdtemp()
#     os.makedirs(os.path.join(dataset_root, 'graph'), exist_ok=True)
#     for i_region in range(n_regions):
#         region_id = '%s%d' % (region_prefix, i_region)
#         n_cells = np.random.randint(200, 2000)
#         cell_coords, cell_types, cell_bm_exp, cell_features = simulate_data(n_cells=n_cells)
#         raw_root = save_to_csv(cell_coords, cell_types, cell_bm_exp, cell_features)
#         construct_graph_for_region(
#             region_id,
#             os.path.join(raw_root, 'cell_coords.csv'),
#             os.path.join(raw_root, 'cell_types.csv'),
#             os.path.join(raw_root, 'cell_bm_exp.csv'),
#             os.path.join(raw_root, 'cell_features.csv'),
#             graph_output=os.path.join(dataset_root, 'graph', '%s.gpkl' % region_id))

#     dataset_kwargs = {
#         'transform': [],
#         'pre_transform': None,
#         'raw_folder_name': 'graph',
#         'processed_folder_name': 'tg_graph',
#         'node_features': ["cell_type", "FEAT1", "FEAT2",
#                           "biomarker_expression", "neighborhood_composition", "center_coord"],
#         'edge_features': ["edge_type", "distance"],
#         'subgraph_size': 3,
#         'subgraph_source': 'on-the-fly',
#         'subgraph_allow_distant_edge': True,
#         'subgraph_radius_limit': 200.,
#     }

#     feature_kwargs = {
#         "biomarker_expression_process_method": "linear",
#         "biomarker_expression_lower_bound": 0.,
#         "biomarker_expression_upper_bound": 100.,
#         "neighborhood_size": 10,
#     }
#     dataset_kwargs.update(feature_kwargs)

#     dataset = CellularGraphDataset(dataset_root, **dataset_kwargs)
#     return dataset


# def test_calculate_voronoi_polygons():
#     n_cells = np.random.randint(200, 2000)
#     cell_coords, _, _, _ = simulate_data(n_cells=n_cells)
#     voronoi_polygons = calcualte_voronoi_from_coords(cell_coords['X'], cell_coords['Y'])
#     assert len(voronoi_polygons) == n_cells


# def test_initialize_graph_with_voronoi_polygons():
#     n_cells = np.random.randint(200, 2000)
#     cell_coords, _, _, _ = simulate_data(n_cells=n_cells)
#     voronoi_polygons = calcualte_voronoi_from_coords(cell_coords['X'], cell_coords['Y'])
#     G = build_graph_from_voronoi_polygons(voronoi_polygons)
#     assert len(G) == n_cells


# def test_construct_graph_for_region():
#     n_cells = np.random.randint(200, 2000)
#     cell_coords, cell_types, cell_bm_exp, cell_features = simulate_data(n_cells=n_cells)
#     root = save_to_csv(cell_coords, cell_types, cell_bm_exp, cell_features)
#     G = construct_graph_for_region(
#         'region',
#         os.path.join(root, 'cell_coords.csv'),
#         os.path.join(root, 'cell_types.csv'),
#         os.path.join(root, 'cell_bm_exp.csv'),
#         os.path.join(root, 'cell_features.csv'))
#     assert len(G) == n_cells
#     for n in G.nodes:
#         assert G.nodes[n].keys() == set(['voronoi_polygon', 'cell_id',
#                                          'center_coord', 'cell_type', 'biomarker_expression',
#                                          'FEAT1', 'FEAT2'])
#         assert G.nodes[n]['biomarker_expression'].keys() == set(['BIOMARKER%d' % i for i in range(40)])


# def test_construct_graph_with_incomplete_features():
#     n_cells = np.random.randint(200, 2000)
#     cell_coords, cell_types, cell_bm_exp, _ = simulate_data(n_cells=n_cells)
#     root = save_to_csv(cell_coords, cell_types, cell_bm_exp)
#     G = construct_graph_for_region(
#         'region',
#         os.path.join(root, 'cell_coords.csv'),
#         cell_types_file=None,
#         cell_biomarker_expression_file=os.path.join(root, 'cell_bm_exp.csv'))
#     assert len(G) == n_cells
#     for n in G.nodes:
#         assert G.nodes[n]['cell_type'] == 'Unassigned'
#     G = construct_graph_for_region(
#         'region',
#         os.path.join(root, 'cell_coords.csv'),
#         cell_types_file=None,
#         cell_biomarker_expression_file=None)
#     assert len(G) == n_cells
#     for n in G.nodes:
#         assert G.nodes[n]['biomarker_expression'] == {}


# def test_construct_graph_for_region_with_uneven_cell_counts():
#     n_cells = np.random.randint(200, 2000)
#     cell_coords, cell_types, cell_bm_exp, _ = simulate_data(n_cells=n_cells)

#     n_subsampled = np.random.randint(int(0.5 * n_cells), int(0.8 * n_cells))
#     subsampled_cell_ids = np.random.choice(cell_types['CELL_ID'], (n_subsampled), replace=False)

#     _cell_types = cell_types[cell_types['CELL_ID'].isin(subsampled_cell_ids)]
#     _cell_bm_exp = cell_bm_exp[cell_bm_exp['CELL_ID'].isin(subsampled_cell_ids)]
#     root = save_to_csv(cell_coords, _cell_types, _cell_bm_exp)
#     G = construct_graph_for_region(
#         'region',
#         os.path.join(root, 'cell_coords.csv'),
#         os.path.join(root, 'cell_types.csv'),
#         os.path.join(root, 'cell_bm_exp.csv'))
#     assert len(G) == n_subsampled


# def test_construct_graph_for_region_with_less_polygons():
#     n_cells = np.random.randint(200, 2000)
#     cell_coords, cell_types, cell_bm_exp, _ = simulate_data(n_cells=n_cells)

#     n_subsampled = np.random.randint(int(0.5 * n_cells), int(0.8 * n_cells))
#     subsampled_cell_ids = np.random.choice(cell_types['CELL_ID'], (n_subsampled), replace=False)

#     _cell_coords = cell_coords[cell_coords['CELL_ID'].isin(subsampled_cell_ids)]
#     voronoi_polygons = calcualte_voronoi_from_coords(_cell_coords['X'], _cell_coords['Y'])
#     root = save_to_csv(cell_coords, cell_types, cell_bm_exp)
#     with open(os.path.join(root, 'polygons.pkl'), 'wb') as f:
#         pickle.dump(voronoi_polygons, f)

#     G = construct_graph_for_region(
#         'region',
#         os.path.join(root, 'cell_coords.csv'),
#         os.path.join(root, 'cell_types.csv'),
#         os.path.join(root, 'cell_bm_exp.csv'),
#         voronoi_file=os.path.join(root, 'polygons.pkl'))
#     G2 = construct_graph_for_region(
#         'region',
#         os.path.join(root, 'cell_coords.csv'),
#         os.path.join(root, 'cell_types.csv'),
#         os.path.join(root, 'cell_bm_exp.csv'))
#     assert len(G) == n_subsampled
#     assert len(G2) == n_cells


# def test_generate_dataset():
#     n_regions = 10
#     prefix = 'REG'
#     dataset = build_dataset(n_regions=n_regions, region_prefix=prefix)
#     assert dataset.N == n_regions
#     assert dataset.processed_file_names == ['%s%d.0.gpt' % (prefix, i) for i in range(n_regions)]
#     assert len(dataset.cell_type_mapping) == 10
#     assert len(dataset.biomarkers) == 40
#     assert len(dataset.node_feature_names) == 55  # 1 (cell type) + 40 + 10 + 2 (coordinates) + 2(FEAT1 and FEAT2)
#     assert len(dataset.edge_feature_names) == 2
#     assert dataset.sampling_freq.shape[0] == 10
#     for i in range(n_regions):
#         assert dataset.get_full(i).x.shape == (dataset.get_full(i).num_nodes, 55)
#         assert dataset[i].region_id == '%s%d' % (prefix, i)
#         assert dataset['%s%d' % (prefix, i)].region_id == '%s%d' % (prefix, i)
#         assert dataset[i].x.shape[1] == 55
#         dataset.plot_subgraph(0, 1)
#         plt.clf()

#     dataset.set_indices(np.arange(5))
#     assert dataset[4].region_id == '%s%d' % (prefix, 4)
#     dataset.set_indices(np.arange(5, 10))
#     assert dataset[2].region_id == '%s%d' % (prefix, 7)

#     subset_inds = np.random.choice(np.arange(10), 5, replace=False)
#     dataset.set_indices(['%s%d' % (prefix, i) for i in subset_inds])
#     for i, ind in enumerate(subset_inds):
#         assert dataset[i].region_id == '%s%d' % (prefix, ind)
#         assert dataset['%s%d' % (prefix, ind)].region_id == '%s%d' % (prefix, ind)

#     dataset.set_indices(None)
#     dataset.save_all_subgraphs_to_chunk()
#     dataset.set_subgraph_source('chunk_save')
#     for i in range(n_regions):
#         assert dataset[i].region_id == '%s%d' % (prefix, i)
#         assert dataset[i].x.shape[1] == 55

#     for i in range(5):
#         dataset.clear_cache()
#         assert len(dataset.cached_data) == 0
#         dataset.load_to_cache(i, subgraphs=True)
#         assert i in dataset.cached_data
#         assert (i, 0) in dataset.cached_data


# def test_transformers():
#     n_regions = 7
#     prefix = ''
#     dataset = build_dataset(n_regions=n_regions, region_prefix=prefix)
#     d_n = dataset[0]

#     transformers = [
#         AddCenterCellType(dataset),
#         AddCenterCellIdentifier(),
#     ]
#     dataset.set_transforms(transformers)
#     dataset.clear_cache()
#     d_t = dataset[0]

#     assert d_t.x.shape[1] == d_n.x.shape[1] + 1
#     assert d_t.x[d_t.center_node_index, -1] == 1
#     assert d_t.x[:, -1].sum() == 1

#     assert d_t.node_y.shape == torch.Size([1])
#     assert d_t.node_y.dtype == torch.long
#     assert d_t.x[d_t.center_node_index, 0].long() == max(dataset.cell_type_mapping.values()) + 1

#     transformers = []
#     dataset.set_transforms(transformers)
#     dataset.clear_cache()
#     d_n = dataset[0]

#     transformers = [
#         FeatureMask(dataset,
#                     use_center_node_features=['cell_type', 'FEAT1'],
#                     use_neighbor_node_features=['cell_type', 'biomarker_expression']),
#         AddCenterCellBiomarkerExpression(dataset),
#     ]
#     dataset.set_transforms(transformers)
#     dataset.clear_cache()
#     d_t = dataset[0]

#     assert torch.abs(d_t.x[:, -12:]).sum() == 0
#     assert torch.abs(d_t.x[d_t.center_node_index, 2:]).sum() == 0
#     assert d_t.x[:, 3:43].sum() != 0
#     assert d_t.node_y.shape == torch.Size([1, 40])
#     assert d_t.node_y.dtype == torch.float

#     graph_label_file = tempfile.mkstemp()[1]
#     graph_label_df = simulate_graph_labels(n_regions=n_regions, region_prefix=prefix)
#     graph_label_df.to_csv(graph_label_file, index=False)
#     transformers = [AddGraphLabel(graph_label_file)]
#     dataset.set_transforms(transformers)
#     dataset.clear_cache()
#     for i in range(n_regions):
#         d_t = dataset[i]
#         assert d_t.graph_y.shape == torch.Size([1, 2])
#         assert d_t.graph_w.shape == torch.Size([1, 2])
#         assert torch.all(d_t.graph_w > 0)


# def test_subgraph_sampler():
#     n_regions = 10
#     prefix = 'REG'
#     dataset = build_dataset(n_regions=n_regions, region_prefix=prefix)
#     selected_inds = np.random.choice(np.arange(n_regions), 5, replace=False)
#     sampler = SubgraphSampler(
#         dataset,
#         selected_inds=selected_inds,
#         num_regions_per_segment=0)
#     for _ in range(5):
#         batch = next(sampler)
#         assert len(set(batch.region_id) - set('%s%d' % (prefix, i) for i in selected_inds)) == 0

#     selected_inds = np.random.choice(np.arange(n_regions), 5, replace=False)
#     subset_regions = ['%s%d' % (prefix, i) for i in selected_inds]
#     sampler = SubgraphSampler(
#         dataset,
#         selected_inds=subset_regions,
#         num_regions_per_segment=0)
#     for _ in range(5):
#         batch = next(sampler)
#         assert len(set(batch.region_id) - set('%s%d' % (prefix, i) for i in selected_inds)) == 0

#     dataset.save_all_subgraphs_to_chunk()
#     sampler = SubgraphSampler(
#         dataset,
#         selected_inds=subset_regions,
#         num_regions_per_segment=2,
#         steps_per_segment=3)
#     for _ in range(5):
#         batch = next(sampler)
#         assert len(set(batch.region_id) - set('%s%d' % (prefix, i) for i in selected_inds)) == 0
#         assert len(set(batch.region_id)) <= 2


# def test_get_random_sampled_subgraphs():
#     n_regions = 10
#     prefix = 'REG'
#     dataset = build_dataset(n_regions=n_regions, region_prefix=prefix)
#     selected_inds = np.random.choice(np.arange(n_regions), 5, replace=False)
#     data_list = get_random_sampled_subgraphs(
#         dataset, inds=selected_inds, n_samples=129, batch_size=32)
#     assert all([data.region_id in ['%s%d' % (prefix, i) for i in selected_inds] for data in data_list])

#     selected_inds = np.random.choice(np.arange(n_regions), 5, replace=False)
#     subset_regions = ['%s%d' % (prefix, i) for i in selected_inds]
#     data_list = get_random_sampled_subgraphs(
#         dataset, inds=subset_regions, n_samples=129, batch_size=32)
#     assert all([data.region_id in ['%s%d' % (prefix, i) for i in selected_inds] for data in data_list])


# def test_model_overfit():
#     n_regions = 4
#     prefix = 'TEST_REG'
#     dataset = build_dataset(n_regions=n_regions, region_prefix=prefix)
#     dataset.save_all_subgraphs_to_chunk()

#     graph_label_file = tempfile.mkstemp()[1]
#     graph_label_df = simulate_graph_labels(n_regions=n_regions, region_prefix=prefix)
#     graph_label_df['GRAPH_TASK1'] = [0, 0, 1, 1]
#     graph_label_df.to_csv(graph_label_file, index=False)

#     transformers = [AddGraphLabel(graph_label_file, tasks=['GRAPH_TASK1'])]
#     dataset.set_transforms(transformers)

#     model_kwargs = {
#         'num_layer': 3,
#         'num_node_type': 10,
#         'num_feat': dataset[0].x.shape[1] - 1,
#         'emb_dim': 32,
#         'num_node_tasks': 0,
#         'num_graph_tasks': 1,
#     }

#     model = spacegm.GNN_pred(**model_kwargs)
#     device = 'cpu'

#     model_save_root = tempfile.mkdtemp()
#     train_kwargs = {
#         'batch_size': 64,
#         'lr': 0.001,
#         'graph_loss_weight': 1.0,
#         'num_iterations': 50,
#         'graph_task_loss_fn': spacegm.models.BinaryCrossEntropy(),
#         'evaluate_fn': [spacegm.train.evaluate_by_sampling_subgraphs,
#                         spacegm.train.evaluate_by_full_graph,
#                         spacegm.train.save_model_weight],
#         'evaluate_freq': 10,
#         'graph_task_evaluate_fn': spacegm.inference.graph_classification_evaluate_fn,
#         'full_graph_graph_task_evaluate_fn': spacegm.inference.full_graph_graph_classification_evaluate_fn,
#         'num_eval_iterations': 10,
#         'score_file': os.path.join(model_save_root, 'GIN-example.txt'),
#         'model_folder': os.path.join(model_save_root, 'GIN-example'),
#     }

#     model = spacegm.train.train_subgraph(
#         model, dataset, device,
#         train_inds=[0, 1, 2, 3], valid_inds=None, **train_kwargs)

#     assert os.path.exists(os.path.join(model_save_root, 'GIN-example.txt'))
#     assert os.path.exists(os.path.join(model_save_root, 'GIN-example', 'model_save_3.pt'))

#     _, graph_preds = spacegm.inference.collect_predict_for_all_nodes(
#         model, dataset, device, inds=np.arange(4), print_progress=False)
#     res = spacegm.inference.full_graph_graph_classification_evaluate_fn(dataset, graph_preds)
#     assert res[0] == 1.0

#     # Test for embedding utilities
#     data_list = get_random_sampled_subgraphs(dataset, inds=np.arange(4), n_samples=128)
#     embs_n, embs_g, (_, g_pred) = get_embedding(model, data_list, device)
#     assert embs_n.shape == (128, model_kwargs['emb_dim'])
#     assert embs_g.shape == (128, model_kwargs['emb_dim'])
#     assert g_pred.shape == (128, 1)

#     _, _, _, tools = dimensionality_reduction_combo(
#         embs_g, n_pca_components=5, cluster_method='kmeans', n_clusters=5, tool_saves=None)
#     all_cluster_labels = collect_cluster_label_for_all_nodes(
#         model, dataset, device, tools, inds=np.arange(4), embedding_from='graph')
#     for i in all_cluster_labels:
#         assert len(all_cluster_labels[i]) == dataset.get_full(i).x.shape[0]

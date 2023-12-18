
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib  import delayed, Parallel

import torch.nn as nn
import spacegm
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

from cluster_comparison import read_run_result_ann_data
from spacegm import build_graph_from_cell_coords, assign_attributes, calcualte_voronoi_from_coords, plot_voronoi_polygons, plot_graph, SubgraphSampler
from helpers import BoundaryDataLoader, SLICE_PIXELS_EDGE_CUTOFF, logger, COLORS_FROM_PLOTLY, NO_JOBS, plotly_spatial_scatter_subgraph, ANNOTATION_DICT
import matplotlib.pyplot as plt

def train_node_classification(dataset_kwargs):


    dataset = spacegm.CellularGraphDataset(root=dataset_kwargs['dataset_root'], **dataset_kwargs)
    logger.info(dataset)

    # idx=328
    # center_ind=250
    # test_fig = dataset.build_subgraph_plotly(idx, center_ind)
    # test_fig.show()

    # fig = plotly_spatial_scatter_subgraph(dataset, color_column, subgraph_edges=None)
    # fig.show()

    # # Initialize sampler
    # dataset.set_subgraph_source('chunk_save')
    # data_iter = SubgraphSampler(dataset)
    # batch = next(data_iter)

    # # i = 0
    # # j = 1234
    # # dataset.plot_subgraph(i, j)



    transformers = [
        # `AddCenterCellType` will add `node_y` attribute to the subgraph for node-level prediction task
        # In this task we will mask the cell type of the center cell (replace it by a placeholder cell type)
        # and use its neighborhood to predict the true cell type
        spacegm.AddCenterCellType(dataset),
        # `AddGraphLabel` will add `graph_y` and `graph_w` attributes to the subgraph for graph-level prediction task
        # spacegm.AddGraphLabel(graph_label_file, tasks=['survival_status']),
        # Transformer `FeatureMask` will zero mask all feature items not included in its argument
        # In this tutorial we perform training/inference using cell types and center cell's size feature
        # spacegm.FeatureMask(dataset),  # , use_center_node_features=dataset.node_feature_names, use_neighbor_node_features=dataset.node_feature_names, use_edge_features=dataset.edge_feature_names
    ]

    # dataset.set_transforms([])  # No transformation
    # d1 = dataset[0]

    dataset.set_transforms(transformers)
    # d2 = dataset[0]


    model_kwargs = {
        'num_layer': dataset.subgraph_size,  # same number of layers as number of hops in the subgraphs
        'num_node_type': len(dataset.cell_type_mapping) + 1,  # number of embeddings for different cell types (plus one placeholder cell type)
        'num_feat': dataset[0].x.shape[1] - 1,  # exclude the cell type column
        'emb_dim': dataset_kwargs['node_embedding_size'], 
        'num_node_tasks': len(dataset.cell_type_mapping),  # A multi-class classification task: predicting center cell type
        'num_graph_tasks': 0,  # a binary classification task
        'node_embedding_output': 'last',
        'drop_ratio': 0.15,
        'graph_pooling': "max",
        'gnn_type': dataset_kwargs['graph_type'],
        'return_node_embedding': False,
    }
    # model_file = "/data/qd452774/spatial_transcriptomics/data/Liver12Slice12_leiden_res_0.7/model_Liver12Slice12_0.7/voronoi_delaunay-gin-emb_size_512/model_save_2998.pt"
    model = spacegm.GNN_pred(**model_kwargs)
    # model.from_pretrained(model_file, strict_bool=False)
    device = 'cpu'

    learning_rate = 0.01
    BATCH_SIZE = 128
    NODE_FEATURES_string = ",".join(dataset_kwargs['node_features'])

    train_kwargs = {
        'experiment_name' : f'{dataset_kwargs["network_type"]}',
        'run_name' : f'{dataset_kwargs["graph_type"]}-emb_dim={model_kwargs["emb_dim"]}-lr={learning_rate}-batch_size={BATCH_SIZE}-features={NODE_FEATURES_string}',
        'batch_size': BATCH_SIZE, # No of subgraphs per batch 
        'lr': learning_rate,
        'graph_loss_weight': 1.0,  # Weight of graph task loss relative to node task loss
        'num_iterations': 50_000,  # In this demo we only train for 50 iterations/batches

        # Loss functions
        'node_task_loss_fn': nn.CrossEntropyLoss(),
        # 'graph_task_loss_fn': spacegm.models.BinaryCrossEntropy(),    

        # Evaluation during training
        'subsample_ratio': 0.1, # Subsample 10% of the data for evaluation
        'evaluate_fn': [spacegm.train.evaluate_by_sampling_subgraphs], #, spacegm.train.evaluate_by_full_graph
        'evaluate_on_train': False,
        'model_save_fn': [spacegm.train.save_model_weight],
        'evaluate_freq': 10,  # Evaluate the model every 10 iterations
    }



    model_save_root = os.path.join(dataset_kwargs['dataset_root'], f'model_{dataset_kwargs["dataset_filter"]}_{dataset_kwargs["leiden_cluster_res"]}')

    evaluate_kwargs = {
        'node_task_evaluate_fn': spacegm.inference.cell_type_prediction_evaluate_fn,
        # 'graph_task_evaluate_fn': spacegm.inference.graph_classification_evaluate_fn,
        # 'full_graph_node_task_evaluate_fn': spacegm.inference.full_graph_cell_type_prediction_evaluate_fn,
        # 'full_graph_graph_task_evaluate_fn': spacegm.inference.full_graph_graph_classification_evaluate_fn,
        'num_eval_iterations': 5,
        'score_file': os.path.join(model_save_root, f'{dataset_kwargs["network_type"]}-{dataset_kwargs["graph_type"]}-emb_size_{dataset_kwargs["node_embedding_size"]}-batch_size={train_kwargs["batch_size"]}-lr={train_kwargs["lr"]}-hops={dataset_kwargs["subgraph_size"]}.txt'),
        'model_folder': os.path.join(model_save_root, f'{dataset_kwargs["network_type"]}-{dataset_kwargs["graph_type"]}-emb_size_{dataset_kwargs["node_embedding_size"]}-batch_size={train_kwargs["batch_size"]}-lr={train_kwargs["lr"]}-hops={dataset_kwargs["subgraph_size"]}'),
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

    train_kwargs['train_inds'] = train_inds
    train_kwargs['valid_inds'] = valid_inds
    # train_kwargs['num_iterations'] = 10_000 -- being set above
    train_kwargs['num_regions_per_segment'] = 0
    train_kwargs['num_iterations_per_segment'] = 10
    train_kwargs['num_workers'] = NO_JOBS

    logger.info("Training model...")
    logger.info(train_kwargs['experiment_name'])
    logger.info(train_kwargs['run_name'])    

    model = spacegm.train.train_subgraph(
        model, 
        dataset,
        device,
        # train_inds=train_inds,
        # valid_inds=valid_inds,
        dataset_kwargs=dataset_kwargs,
        **train_kwargs)


def build_train_kwargs(data_filter_name, resolution, network_type, graph_type):

    # TODO: Add a function to create/read the build biomarkers_list before creating the pytorch graph datasets
    # Change raw_folder_name and processed_folder_name to conduct different experiments

    # dataset_root = "/data/qd452774/spatial_transcriptomics/data/example_dataset"

    dataset_root = f"/data/qd452774/spatial_transcriptomics/data/{data_filter_name}_leiden_res_{resolution}"

    SUBGRAPH_RADIUS_LIMIT = SLICE_PIXELS_EDGE_CUTOFF['Liver1Slice1'] * 4
    SUBGRAPH_SIZE = 3
    NEIGHBORHOOD_SIZE = 15  # SUBGRAPH_SIZE * 5
    NODE_EMBEDDING_SIZE = 128

    biomarkers_list = pd.read_csv(os.path.join(dataset_root, "biomarkers_list.csv"), header=None).values.flatten().tolist()
    # ['1700061G19Rik', 'Abcb4', 'Acaca', 'Acacb', 'Ace', 'Acsbg1', 'Acsbg2', 'Acsf3', 'Acsl1', 'Acsl4', 'Acsl5', 'Acsl6', 'Acss1', 'Acss2', 'Adgre1', 'Adh4', 'Adh7', 'Adpgk', 'Akap14', 'Akr1a1', 'Akr1c18', 'Akr1d1', 'Alas1', 'Alas2', 'Alcam', 'Aldh1b1', 'Aldh3a1', 'Aldh3a2', 'Aldh3b1', 'Aldh3b2', 'Aldh3b3', 'Aldh7a1', 'Aldoart1', 'Aldoart2', 'Aldoc', 'Ammecr1', 'Angpt1', 'Angpt2', 'Apobec3', 'Aqp1', 'Arsb', 'Axin2', 'B4galt6', 'Bank1', 'Bcam', 'Bmp2', 'Bmp5', 'Bmp7', 'Bpgm', 'Calcrl', 'Cald1', 'Cav2', 'Cbr4', 'Ccr1', 'Ccr2', 'Cd177', 'Cd300lg', 'Cd34', 'Cd44', 'Cd48', 'Cd83', 'Cd93', 'Cdh11', 'Cdh5', 'Cebpa', 'Celsr2', 'Chodl', 'Clec14a', 'Cnp', 'Col1a2', 'Col6a1', 'Comt', 'Csf1r', 'Csf3r', 'Csk', 'Csnk1a1', 'Cspg4', 'Ctnnal1', 'Ctsc', 'Cxadr', 'Cxcl12', 'Cxcl14', 'Cxcr2', 'Cybb', 'Cygb', 'Cyp11a1', 'Cyp11b1', 'Cyp17a1', 'Cyp1a1', 'Cyp1a2', 'Cyp21a1', 'Cyp2b19', 'Cyp2b23', 'Cyp2b9', 'Cyp2c23', 'Cyp2c38', 'Cyp2c55', 'Cyp7b1', 'Ddx4', 'Dek', 'Dkk1', 'Dkk2', 'Dkk3', 'Dlat', 'Dld', 'Dll1', 'Dll4', 'Dnase1l3', 'Dpt', 'E2f2', 'Efnb2', 'Egfr', 'Egr1', 'Eif3a', 'Eif3f', 'Elk3', 'Eng', 'Eno2', 'Eno3', 'Eno4', 'Ep300', 'Epas1', 'Epcam', 'Ephb4', 'Errfi1', 'Esam', 'F13a1', 'Fasn', 'Fech', 'Fgf1', 'Fgf2', 'Flt3', 'Flt4', 'Foxq1', 'Fstl1', 'Fyb', 'Fzd1', 'Fzd2', 'Fzd3', 'Fzd4', 'Fzd5', 'Fzd7', 'Fzd8', 'G6pc', 'G6pc3', 'Galm', 'Gapdhs', 'Gata2', 'Gca', 'Gck', 'Gfap', 'Gimap3', 'Gimap4', 'Gimap6', 'Gm2a', 'Gnai2', 'Gnaz', 'Gpd1', 'Gpi1', 'Gpr182', 'Grn', 'Gypa', 'H2afy', 'Hc', 'Hgf', 'Hk1', 'Hk2', 'Hk3', 'Hkdc1', 'Hmgb1', 'Hoxb4', 'Hoxb5', 'Hsd11b2', 'Hsd17b1', 'Hsd17b12', 'Hsd17b2', 'Hsd17b3', 'Hsd17b6', 'Hsd17b7', 'Hsd3b2', 'Hsd3b3', 'Hsd3b6', 'Hvcn1', 'Icam1', 'Igf1', 'Il6', 'Il7r', 'Itga2b', 'Itgal', 'Itgam', 'Itgb2', 'Itgb7', 'Ivl', 'Jag1', 'Jag2', 'Kcnj1', 'Kcnj16', 'Kctd12', 'Kdr', 'Kit', 'Kitl', 'Klrb1c', 'Lad1', 'Lamp3', 'Laptm5', 'Lck', 'Lcp1', 'Ldha', 'Ldhal6b', 'Ldhc', 'Lef1', 'Lepr', 'Lhfp', 'Lmna', 'Lmnb1', 'Lox', 'Lrp2', 'Lsr', 'Ltbp4', 'Lyve1', 'Mal', 'Maml1', 'Mcat', 'Mecom', 'Meis1', 'Meis2', 'Mertk', 'Minpp1', 'Mki67', 'Mkrn1', 'Mmrn1', 'Mpeg1', 'Mpl', 'Mpp1', 'Mrc1', 'Mrvi1', 'Ms4a1', 'Myh10', 'Ncf1', 'Ndn', 'Nes', 'Nid1', 'Nkd2', 'Notch1', 'Notch2', 'Notch3', 'Notch4', 'Npc2', 'Nrp1', 'Obscn', 'Olah', 'Olr1', 'Oxsm', 'Pabpc1', 'Pck1', 'Pck2', 'Pdgfra', 'Pdha1', 'Pdha2', 'Pdhb', 'Pdpn', 'Pecam1', 'Pfkl', 'Pfkm', 'Pfkp', 'Pgam1', 'Pgam2', 'Pgm1', 'Pgm2', 'Pkm', 'Pld4', 'Plvap', 'Podxl', 'Pou2af1', 'Prickle2', 'Procr', 'Proz', 'Psap', 'Psmd3', 'Pth1r', 'Ramp1', 'Rassf4', 'Rbpj', 'Rhoj', 'Rpp14', 'Runx1', 'Ryr2', 'Sardh', 'Satb1', 'Sdc1', 'Sdc3', 'Selp', 'Selplg', 'Serping1', 'Serpinh1', 'Sfrp1', 'Sfrp2', 'Sgms2', 'Shisa5', 'Slamf1', 'Slc12a1', 'Slc25a37', 'Slc34a2', 'Smarca4', 'Smpdl3a', 'Srd5a1', 'Srd5a2', 'Srd5a3', 'Ssh2', 'Stab2', 'Stk17b', 'Sult1e1', 'Sult2b1', 'Tcf3', 'Tcf4', 'Tcf7', 'Tcf7l1', 'Tcf7l2', 'Tek', 'Tent5c', 'Tet1', 'Tet2', 'Tfrc', 'Tgfb2', 'Timp2', 'Timp3', 'Tinagl1', 'Tkt', 'Tmem56', 'Tmod1', 'Tnfrsf13c', 'Tomt', 'Tox', 'Tpi1', 'Trim47', 'Tspan13', 'Tubb6', 'Txlnb', 'Ugt2a3', 'Ugt2b1', 'Unc93b1', 'Vangl2', 'Vav1', 'Vcam1', 'Vwf', 'Wnt2', 'Wnt4']

    NODE_FEATURES = ["cell_type", "center_coord", "biomarker_expression", "SIZE", "neighborhood_composition", "voronoi_polygon"]

    dataset_kwargs = {
        'transform': [],
        'pre_transform': None,
        'dataset_root': dataset_root,
        'dataset_filter': data_filter_name,
        'leiden_cluster_res' : resolution, 
        'raw_folder_name': f'graph_{data_filter_name}_{resolution}/{network_type}',  # os.path.join(dataset_root, "graph") is the folder where we saved nx graphs
        'processed_folder_name': f'graph_{data_filter_name}_{resolution}_processed/{network_type}',  # processed dataset files will be stored here
        'biomarkers': biomarkers_list,  # biomarkers to be used in liver1slice1
        # There are all the cellular features that we want the dataset to compute, cell_type must be the first variable
        'network_type' : network_type,
        'node_features': NODE_FEATURES,
        'edge_features': ["edge_type", "distance"],  # edge_type must be first variable (cell pair) features "edge_type", 
        'graph_type' : graph_type,
        'node_embedding_size': NODE_EMBEDDING_SIZE,
        'subgraph_size': SUBGRAPH_SIZE,  # indicating we want to sample 3-hop subgraphs from these regions (for training/inference), this is a core parameter for SPACE-GM.
        'subgraph_source': 'on-the-fly',
        'subgraph_allow_distant_edge': True,
        'subgraph_radius_limit': SUBGRAPH_RADIUS_LIMIT,
        'sampling_avoid_unassigned' : True,
        'unassigned_cell_type' : 'Unassigned',
    }

    feature_kwargs = {
        # "biomarker_expression_process_method": "linear",
        # "biomarker_expression_lower_bound": 0,
        # "biomarker_expression_upper_bound": 18,
        "neighborhood_size": NEIGHBORHOOD_SIZE,
    }
    dataset_kwargs.update(feature_kwargs)

    return dataset_kwargs    


def run_model_from_config(run_config):
    try:
        dataset_kwargs = build_train_kwargs(*run_config)
        train_node_classification(dataset_kwargs)
    except Exception as e:
        logger.error(f"Error in {run_config}: {str(e)}")

def run_parallel(all_combinations):
    Parallel(n_jobs=NO_JOBS)(delayed(run_model_from_config)(combination) for combination in all_combinations)

def main():
    data_filter_name = "Liver12Slice12"
    resolution = 0.7

    network_types = ["voronoi_delaunay", "given_boundary_delaunay", "given_boundary_r3index"]
    graph_types = ["gin", "gcn", "graphsage", "gat"]

    all_combinations = []
    for network_type in network_types:
        for graph_type in graph_types:
            all_combinations.append((data_filter_name, resolution, network_type, graph_type))

    # dataset_kwargs = build_train_kwargs(data_filter_name, resolution, network_type, graph_type)
    test_run_config = all_combinations[0]
    run_model_from_config(test_run_config)

    # run_parallel(all_combinations)


if __name__ == "__main__":
    main()
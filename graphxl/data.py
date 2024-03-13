"""
Modules to handle pytorch geometric datasets, including data loading, processing, and transformations
from networkx graphs to pytorch geometric data objects, building k-hop subgraphs, Vizgen Cell
Boundary Data Loader, Random Subgraph Sampler, etc.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
import warnings
import h5py
import torch
from torch.utils.data import RandomSampler
import torch_geometric as tg
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
from joblib.externals.loky.backend.context import get_context


from .features import get_feature_names, nx_to_tg_graph
from .graph_build import plot_graph
from .logging_setup import logger
from .plotly_helpers import plotly_spatial_scatter_subgraph
from .utils import (
    get_cell_type_metadata,
    get_biomarker_metadata,
)


class CellularGraphDataset(Dataset):
    """Main dataset structure for cellular graphs
    Inherited from https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Dataset.html
    """

    def __init__(
        self,
        root,
        transform=[],
        pre_transform=None,
        raw_folder_name="nx_files",
        processed_folder_name="nx_files_processed",
        node_features=[
            "cell_type",
            "biomarker_expression",
            "center_coord",
            "volume",
        ],
        edge_features=["edge_type", "distance"],
        edge_types=None,
        cell_type_mapping=None,
        cell_type_freq=None,
        biomarkers=None,
        subgraph_size=0,
        subgraph_node_type="all-types",
        subgraph_source="on-the-fly",
        subgraph_allow_distant_edge=True,
        subgraph_radius_limit=-1,
        **feature_kwargs,
    ):
        """Initialize the dataset

        Args:
            root (str): path to the dataset directory
            transform (list): list of transformations (see `transform.py`),
                applied to each output graph on-the-fly
            pre_transform (list): list of transformations, applied to each graph before saving
            raw_folder_name (str): name of the sub-folder containing raw graphs (gpickle)
            processed_folder_name (str): name of the sub-folder containing processed graphs (pyg data object)
            node_features (list): list of all available node feature items,
                see `features.process_features` for details
            edge_features (list): list of all available edge feature items
            cell_type_mapping (dict): mapping of unique cell types to integer indices,
                see `utils.get_cell_type_metadata`
            cell_type_freq (dict): mapping of unique cell types to their frequencies,
                see `utils.get_cell_type_metadata`
            biomarkers (list): list of biomarkers,
                see `utils.get_expression_biomarker_metadata`
            subgraph_size (int): number of hops for subgraph, 0 means using the full cellular graph
            subgraph_source (str): source of subgraphs, one of 'on-the-fly', 'chunk_save'
            subgraph_allow_distant_edge (bool): whether to consider distant edges
            subgraph_radius_limit (float): radius (distance to center cell in pixel) limit for subgraphs,
                -1 means no limit
            feature_kwargs (dict): optional arguments for processing features
                see `features.process_features` for details
        """
        self.root = root
        self.raw_folder_name = raw_folder_name
        self.processed_folder_name = processed_folder_name
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        # Find all unique cell types in the dataset
        if cell_type_mapping is None or cell_type_freq is None:
            nx_graph_files = [os.path.join(self.raw_dir, f) for f in self.raw_file_names]
            self.cell_type_mapping, self.cell_type_freq, self.cell_type_color = get_cell_type_metadata(nx_graph_files)
        else:
            self.cell_type_mapping = cell_type_mapping
            self.cell_type_freq = cell_type_freq

        self.cell_type_inv_mapping = {v: k for k, v in self.cell_type_mapping.items()}

        # Find all available biomarkers for cells in the dataset
        if biomarkers is None:
            nx_graph_files = [os.path.join(self.raw_dir, f) for f in self.raw_file_names]
            self.biomarkers, _ = get_biomarker_metadata(nx_graph_files)
        else:
            self.biomarkers = biomarkers

        # Node features & edge features
        self.node_features = node_features
        self.edge_features = edge_features
        self.edge_types = edge_types
        self.edge_types_inv = {v: k for k, v in self.edge_types.items()}

        if "cell_type" in self.node_features:
            assert self.node_features.index("cell_type") == 0, "cell_type must be the first node feature"
        if "edge_type" in self.edge_features:
            assert self.edge_features.index("edge_type") == 0, "edge_type must be the first edge feature"

        self.node_feature_names = get_feature_names(
            node_features,
            cell_type_mapping=self.cell_type_mapping,
            biomarkers=self.biomarkers,
        )
        self.edge_feature_names = get_feature_names(
            edge_features,
            cell_type_mapping=self.cell_type_mapping,
            biomarkers=self.biomarkers,
        )

        # Prepare kwargs for node and edge featurization
        self.feature_kwargs = feature_kwargs
        self.feature_kwargs["cell_type_mapping"] = self.cell_type_mapping
        self.feature_kwargs["biomarkers"] = self.biomarkers
        self.feature_kwargs["cell_type_freq"] = self.cell_type_freq
        self.feature_kwargs["edge_types"] = self.edge_types

        # Note this command below calls the `process` function
        super(CellularGraphDataset, self).__init__(root, None, pre_transform)

        # Transformations, e.g. masking features, adding graph-level labels
        self.transform = transform

        # Using n-hop ego graphs (subgraphs) to perform prediction
        assert subgraph_node_type in ["all-types", "cc-types"], "Invalid subgraph node selection choice"
        self.subgraph_node_type = subgraph_node_type
        self.subgraph_size = subgraph_size  # number of hops, 0 = use full graph
        self.subgraph_source = subgraph_source
        self.subgraph_allow_distant_edge = subgraph_allow_distant_edge
        self.subgraph_radius_limit = subgraph_radius_limit

        # sort for consistency between reads
        self.processed_paths.sort()
        assert len(self.processed_paths) == len(
            self.raw_paths
        ), "Processed and raw paths are not consistent, update processed_region_to_raw to handle this case"

        # create mapper for processed to raw paths
        self.processed_region_to_raw = {}
        self.update_processed_region_to_raw()

        # Cache for graphs
        self.cached_data = {}

        self.N = len(self.processed_paths)
        self.region_ids = [self.get_full(i).region_id for i in range(self.N)]
        # inverse cell proportions for each cell type
        self.sampling_freq = {
            self.cell_type_mapping[ct]: 1.0 / (self.cell_type_freq[ct] + 1e-5) for ct in self.cell_type_mapping
        }
        self.sampling_freq = torch.from_numpy(np.array([self.sampling_freq[i] for i in range(len(self.sampling_freq))]))

    def update_processed_region_to_raw(self):
        """This mapper helps to go back and forth between raw and processed paths"""

        for item in self.processed_paths:
            p_key = os.path.basename(item).split(".")[0]
            temp = f"{os.path.basename(item).split('.')[0]}.gpkl"
            temp = temp.replace("-", "_")
            p_val = os.path.join(self.raw_dir, temp)
            self.processed_region_to_raw[p_key] = p_val

    def set_indices(self, inds=None):
        """Limit subgraph sampling to a subset of region indices,
        helpful when splitting dataset into training/validation/test regions
        """
        inds = np.arange(self.N) if inds is None else inds
        idxs = [self._map_to_idx(i, insubset=False) for i in inds]
        self._indices = idxs
        return

    def _map_to_idx(self, i, insubset=False):
        if isinstance(i, (int, np.integer)):
            if not insubset:
                assert 0 <= i < self.N, "Index %d out of range" % i
                idx = i
            else:
                idx = self.indices()[i]
        elif isinstance(i, str) and i in self.region_ids:
            idx = self.region_ids.index(i)
        else:
            raise IndexError("Cannot parse index %s" % str(i))
        return idx

    def set_subgraph_source(self, subgraph_source):
        """Set subgraph source"""
        assert subgraph_source in ["chunk_save", "on-the-fly"]
        self.subgraph_source = subgraph_source

    def set_transforms(self, transform=[]):
        """Set transformation functions"""
        self.transform = transform

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.raw_folder_name)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.processed_folder_name)

    @property
    def raw_file_names(self):
        return sorted([f for f in os.listdir(self.raw_dir) if f.endswith(".gpkl")])

    @property
    def processed_file_names(self):
        # Only files for full graphs
        return sorted([f for f in os.listdir(self.processed_dir) if f.endswith(".gpt") and "hop" not in f])

    def len(self):
        return self.N

    def process(self):
        """Featurize all cellular graphs"""

        from joblib import Parallel, delayed
        from graphxl import NO_JOBS
        from graphxl import NO_JOBS

        def run_parallel_processing(files_names):
            Parallel(n_jobs=NO_JOBS, backend="loky")(delayed(save_file)(f) for f in files_names)

        def save_file(raw_path):
            G = pickle.load(open(raw_path, "rb"))
            region_id = G.region_id
            if os.path.exists(os.path.join(self.processed_dir, "%s.0.gpt" % region_id)):
                return

            # Transform networkx graphs to pyg data objects, and add features for nodes and edges
            data_list = nx_to_tg_graph(
                G,
                node_features=self.node_features,
                edge_features=self.edge_features,
                **self.feature_kwargs,
            )

            for i, d in enumerate(data_list):
                assert d.region_id == region_id  # graph identifier
                assert d.x.shape[1] == len(self.node_feature_names)  # make sure feature dimension matches
                assert d.edge_attr.shape[1] == len(self.edge_feature_names)
                d.component_id = i
                if self.pre_transform is not None:
                    for transform_fn in self.pre_transform:
                        d = transform_fn(d)
                torch.save(
                    d,
                    os.path.join(self.processed_dir, "%s.%d.gpt" % (d.region_id, d.component_id)),
                )

        logger.info("Starting processing of %d graphs" % len(self.raw_file_names))
        all_nx_files = [os.path.join(self.raw_dir, file_item) for file_item in self.raw_file_names]
        run_parallel_processing(all_nx_files)
        logger.info("Done processing of %d graphs" % len(self.raw_file_names))
        return

    def __getitem__(self, i):
        """Sample a graph/subgraph from the dataset and apply transformations"""
        idx = self._map_to_idx(i, insubset=True)
        data = self.get(idx)
        # Apply transformations
        for transform_fn in self.transform:
            data = transform_fn(data)
        return data

    def get(self, idx):
        data = self.get_full(idx)
        if self.subgraph_size > 0:
            center_ind = self.pick_center(data)
            data = self.get_subgraph(idx, center_ind)
        return data

    def get_subgraphs_list(self, idx, size=10):
        """Get a list of subgraphs from the dataset"""
        data = self.get_full(idx)
        subgraph_center_inds = []
        if self.subgraph_size > 0:
            for i in range(size):
                center_ind = self.pick_center(data)
                subgraph_center_inds.append(center_ind)

        subgraphs = [self.get_subgraph(idx, i) for i in subgraph_center_inds]
        return subgraphs

    def get_subgraph(self, idx, center_ind):
        """Get a subgraph from the dataset"""
        # Check cache
        if (idx, center_ind) in self.cached_data:
            return self.cached_data[(idx, center_ind)]
        if self.subgraph_source == "on-the-fly":
            # Construct subgraph on-the-fly
            return self.calculate_subgraph(idx, center_ind)
        elif self.subgraph_source == "chunk_save":
            # Load subgraph from pre-saved chunk file
            return self.get_saved_subgraph_from_chunk(idx, center_ind)

    def get_full(self, idx):
        """Read the full cellular graph of region `idx`"""

        if idx in self.cached_data:
            return self.cached_data[idx]
        else:
            data = torch.load(self.processed_paths[idx])
            self.cached_data[idx] = data
            return data

    def get_full_nx(self, idx):
        """Read the full cellular graph (nx.Graph) of region `idx`"""
        if type(idx) is str:
            return pickle.load(open(self.processed_region_to_raw[idx], "rb"))
        return pickle.load(open(self.raw_paths[idx], "rb"))

    def calculate_subgraph(self, idx, center_ind, return_node_indices=False):
        """Generate the n-hop subgraph around cell `center_ind` from region `idx`"""
        data = self.get_full(idx)

        if data["is_padding"][center_ind]:
            logger.error("center node is part of padding area, invalid")
            return

        if not self.subgraph_allow_distant_edge:
            edge_type_mask = data.edge_attr[:, 0] == self.edge_types["neighbor"]
        else:
            edge_type_mask = None

        if self.subgraph_node_type == "cc-types":
            node_type = int(data.x[int(center_ind), 0])
            node_attributes = data.x
        else:
            node_type = None
            node_attributes = None

        sub_node_inds = k_hop_subgraph(
            int(center_ind),
            self.subgraph_size,
            data.edge_index,
            node_attr=node_attributes,
            node_type=node_type,
            edge_type_mask=edge_type_mask,
            relabel_nodes=False,
            num_nodes=data.x.shape[0],
        )[0]

        if self.subgraph_radius_limit > 0:
            # Restrict to neighboring cells that are within the radius (distance to center cell) limit
            assert "center_coord" in self.node_features
            coord_feature_inds = [i for i, n in enumerate(self.node_feature_names) if n.startswith("center_coord")]
            assert len(coord_feature_inds) == 2
            center_cell_coord = data.x[[center_ind]][:, coord_feature_inds]
            neighbor_cells_coord = data.x[sub_node_inds][:, coord_feature_inds]
            dists = ((neighbor_cells_coord - center_cell_coord) ** 2).sum(1).sqrt()
            sub_node_inds = sub_node_inds[(dists < self.subgraph_radius_limit)]

        # Construct subgraphs as separate pyg data objects
        sub_x = data.x[sub_node_inds]
        sub_cell_ids = [data.cell_id[int(item)] for item in sub_node_inds]
        padding_bool_list = data["is_padding"][sub_node_inds]
        sub_edge_index, sub_edge_attr = subgraph(
            sub_node_inds, data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True
        )

        relabeled_node_ind = list(sub_node_inds.numpy()).index(center_ind)

        sub_data = {
            "center_node_index": relabeled_node_ind,  # center node index in the subgraph
            "original_center_node": center_ind,  # center node index in the original full cellular graph
            "x": sub_x,
            "cell_id": sub_cell_ids,
            "edge_index": sub_edge_index,
            "edge_attr": sub_edge_attr,
            "num_nodes": len(sub_node_inds),
            "is_padding": padding_bool_list,
        }

        # Assign graph-level attributes
        for k in data:
            if not k[0] in sub_data:
                sub_data[k[0]] = k[1]

        sub_data = tg.data.Data.from_dict(sub_data)
        self.cached_data[(idx, center_ind)] = sub_data
        if return_node_indices:
            return sub_data, sub_node_inds
        return sub_data

    def get_saved_subgraph_from_chunk(self, idx, center_ind):
        """Read the n-hop subgraph around cell `center_ind` from region `idx`
        Subgraph will be extracted from a pre-saved chunk file, which is generated by calling
        `save_all_subgraphs_to_chunk`
        """
        full_graph_path = self.processed_paths[idx]
        subgraphs_path = full_graph_path.replace(".gpt", ".%d-hop.gpt" % self.subgraph_size)
        if not os.path.exists(subgraphs_path):
            warnings.warn("Subgraph save %s not found" % subgraphs_path)
            return self.calculate_subgraph(idx, center_ind)

        subgraphs = torch.load(subgraphs_path)
        # Store to cache first
        for j, g in enumerate(subgraphs):
            self.cached_data[(idx, j)] = g
        return self.cached_data[(idx, center_ind)]

    def _get_possible_center_inds(self, idx):
        """Get all possible center indices for subgraph sampling"""
        data = self.get_full(idx)
        return torch.where(data["is_padding"] == False)[0].numpy()  # noqa: E712

    def pick_center(self, data):
        """Randomly pick a center cell from a full cellular graph, cell type balanced"""

        cell_types = data["x"][data["is_padding"] == False][:, 0].long()  # noqa: E712
        freq = self.sampling_freq.gather(0, cell_types)
        freq = freq / freq.sum()
        center_node_ind = np.random.choice(
            torch.where(data["is_padding"] == False)[0].numpy(), p=freq.cpu().data.numpy()  # noqa: E712
        )
        return center_node_ind

    def load_to_cache(self, idx, subgraphs=True):
        """Pre-load full cellular graph of region `idx` and all its n-hop subgraphs to cache"""
        data = torch.load(self.processed_paths[idx])
        self.cached_data[idx] = data
        if subgraphs or self.subgraph_source == "chunk_save":
            subgraphs_path = self.processed_paths[idx].replace(".gpt", ".%d-hop.gpt" % self.subgraph_size)
            if not os.path.exists(subgraphs_path):
                raise FileNotFoundError(
                    "Subgraph save %s not found, please run `save_all_subgraphs_to_chunk`." % subgraphs_path
                )
            neighbor_graphs = torch.load(subgraphs_path)
            for j, ng in enumerate(neighbor_graphs):
                self.cached_data[(idx, j)] = ng

    def save_all_subgraphs_to_chunk(self):
        """Save all n-hop subgraphs for all regions to chunk files (one file per region)"""
        for idx, p in enumerate(self.processed_paths):
            data = self.get_full(idx)
            n_nodes = data.x.shape[0]
            neighbor_graph_path = p.replace(".gpt", ".%d-hop.gpt" % self.subgraph_size)
            if os.path.exists(neighbor_graph_path):
                continue
            subgraphs = []
            for node_i in range(n_nodes):
                subgraphs.append(self.calculate_subgraph(idx, node_i))
            torch.save(subgraphs, neighbor_graph_path)
        return

    def clear_cache(self):
        del self.cached_data
        self.cached_data = {}
        return

    def plot_subgraph(self, idx, center_ind):
        """Plot the n-hop subgraph around cell `center_ind` from region `idx`"""
        xcoord_ind = self.node_feature_names.index("center_coord-x")
        ycoord_ind = self.node_feature_names.index("center_coord-y")

        _subg, sub_node_inds = self.calculate_subgraph(idx, center_ind, return_node_indices=True)
        if torch.all(_subg["is_padding"]):
            logger.error("all cells are from padding area in this graph segment")
            return

        coords = _subg.x.data.numpy()[:, [xcoord_ind, ycoord_ind]].astype(float)
        x_c, y_c = coords[_subg.center_node_index]

        G = self.get_full_nx(idx)
        _G = G.subgraph(np.array(sub_node_inds))

        # G = self.get_full_nx(idx)
        # sub_node_inds = []
        # for n in G.nodes:
        #     c = np.array(G.nodes[n]["center_coord"]).astype(float).reshape((1, -1))
        #     if np.linalg.norm(coords - c, ord=2, axis=1).min() < 1e-2:
        #         sub_node_inds.append(n)
        # assert len(sub_node_inds) == len(coords)
        # _G = G.subgraph(sub_node_inds)

        node_colors = [self.cell_type_mapping[_G.nodes[n]["cell_type"]] for n in _G.nodes]
        node_colors = [matplotlib.cm.tab20(ct) for ct in node_colors]
        plot_graph(_G, node_colors=node_colors)
        xmin, xmax = plt.gca().xaxis.get_data_interval()
        ymin, ymax = plt.gca().yaxis.get_data_interval()

        scale = max(x_c - xmin, xmax - x_c, y_c - ymin, ymax - y_c) * 1.05
        plt.xlim(x_c - scale, x_c + scale)
        plt.ylim(y_c - scale, y_c + scale)
        plt.plot([x_c], [y_c], "x", markersize=5, color="k")

        from io import BytesIO

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)

        return buffer

    def plotly_subgraphs_list(self, idx, filter_self_edges=True, size=10, variation="default"):
        """Plot a list of subgraphs from the dataset"""
        data = self.get_full(idx)
        subgraph_center_inds = []
        if self.subgraph_size > 0:
            for i in range(size):
                center_ind = self.pick_center(data)
                subgraph_center_inds.append(center_ind)

        plots_ = []
        for center_ind in subgraph_center_inds:
            plots_.append(
                self.plotly_subgraph(idx, center_ind, filter_self_edges=filter_self_edges, variation=variation)
            )

        return plots_

    def plotly_subgraph(self, idx, center_ind, filter_self_edges=True, variation="default"):
        xcoord_ind = self.node_feature_names.index("center_coord-x")
        ycoord_ind = self.node_feature_names.index("center_coord-y")
        boundary_feat = "voronoi_polygon" if "voronoi_polygon" in self.node_features else "boundary_polygon"
        edge_type_index = self.edge_feature_names.index("edge_type")

        _subg = self.calculate_subgraph(idx, center_ind)
        if torch.all(_subg["is_padding"]):
            logger.error("all cells are from padding area in this graph segment")
            return

        vectorized_mapping = np.vectorize(self.cell_type_inv_mapping.get)
        cell_types_subgraph = vectorized_mapping(_subg.x[:, 0].numpy())
        _subg["cell_types"] = cell_types_subgraph
        _subg["cell_colors"] = [self.cell_type_color[ct] for ct in cell_types_subgraph]

        G = self.get_full_nx(_subg.region_id)
        node_ids_to_find = _subg.cell_id
        cell_boundaries = [
            data[boundary_feat] for node, data in G.nodes(data=True) if data["cell_id"] in node_ids_to_find
        ]

        edge_types_list = _subg.edge_attr[:, edge_type_index].numpy().astype(int)
        _subg["edge_types"] = [self.edge_types_inv[et] for et in edge_types_list]

        data_df = pd.DataFrame(
            {
                "cell_type": _subg["cell_types"],
                "cell_color": _subg["cell_colors"],
                "x": _subg.x[:, xcoord_ind].numpy(),
                "y": _subg.x[:, ycoord_ind].numpy(),
                "cell_id": _subg.cell_id,
                "boundary": cell_boundaries,
            }
        )

        edges_df = pd.DataFrame(_subg.edge_index.numpy().T, columns=["source", "target"])
        edges_df["edge_type"] = _subg["edge_types"]
        edges_df["source_cell_id"] = edges_df["source"].map(data_df["cell_id"].to_dict())
        edges_df["target_cell_id"] = edges_df["target"].map(data_df["cell_id"].to_dict())
        edges_df["from_loc"] = list(
            zip(
                edges_df["source_cell_id"].map(data_df.set_index("cell_id")["x"]),
                edges_df["source_cell_id"].map(data_df.set_index("cell_id")["y"]),
            )
        )
        edges_df["to_loc"] = list(
            zip(
                edges_df["target_cell_id"].map(data_df.set_index("cell_id")["x"]),
                edges_df["target_cell_id"].map(data_df.set_index("cell_id")["y"]),
            )
        )

        if filter_self_edges:
            edges_df = edges_df[edges_df["edge_type"] != "self"]

        return plotly_spatial_scatter_subgraph(
            data_df, subgraph_edges=edges_df, center_node_index=_subg.center_node_index, variation=variation
        )

    def plot_graph_legend(self):
        """Legend for cell type colors"""
        plt.clf()
        plt.figure(figsize=(2, 2))
        for ct, i in self.cell_type_mapping.items():
            plt.plot([0], [0], ".", label=ct, color=matplotlib.cm.tab20(int(i) % 20))
        plt.legend()
        plt.plot([0], [0], color="w", markersize=10)
        plt.axis("off")


def k_hop_subgraph(
    node_ind,
    subgraph_size,
    edge_index,
    node_attr=None,
    node_type=None,
    edge_type_mask=None,
    relabel_nodes=False,
    num_nodes=None,
):
    """A customized k-hop subgraph fn that filter for edge_type

    Args:
        node_ind (int): center node index
        subgraph_size (int): number of hops for the neighborhood subgraph
        edge_index (torch.Tensor): edge index tensor for the full graph
        edge_type_mask (torch.Tensor): edge type mask
        relabel_nodes (bool): if to relabel node indices to consecutive integers
        num_nodes (int): number of nodes in the full graph

    Returns:
        subset (torch.LongTensor): indices of nodes in the subgraph
        edge_index (torch.LongTensor): edges in the subgraph
        inv (toch.LongTensor): location of the center node in the subgraph
        edge_mask (torch.BoolTensor): edge mask indicating which edges were preserved
    """

    num_nodes = edge_index.max().item() + 1 if num_nodes is None else num_nodes
    col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
    edge_type_mask = torch.ones_like(edge_mask) if edge_type_mask is None else edge_type_mask
    # filtering all edges that are not the same cell type as the center cell if node_type is not None
    node_type_mask = (
        torch.ones_like(edge_mask)
        if node_type is None
        else (
            torch.index_select(node_attr[:, 0] == node_type, 0, row)
            * torch.index_select(node_attr[:, 0] == node_type, 0, col)
        )
    )

    if isinstance(node_ind, (int, list, tuple)):
        node_ind = torch.tensor([node_ind], device=row.device).flatten()
    else:
        node_ind = node_ind.to(row.device)

    subsets = [node_ind]
    next_root = node_ind

    for _ in range(subgraph_size):
        node_mask.fill_(False)
        node_mask[next_root] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask * edge_type_mask * node_type_mask])
        next_root = col[edge_mask * edge_type_mask * node_type_mask]

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[: node_ind.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_ind = row.new_full((num_nodes,), -1)
        node_ind[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_ind[edge_index]

    return subset, edge_index, inv, edge_mask


class InfDataLoader(DataLoader):
    """Overriding __len__ to return a large number"""

    def __len__(self):
        return int(1e10)


class SubgraphSampler(object):
    """Iterator for sampling subgraphs from a CellularGraphDataset

    This iterator should be coupled with a CellularGraphDataset with
    `subgraph_source`='chunk_save' to pipeline data loading. The iterator
    keeps a queue of shuffled regions indices. During each segment,
    `num_graphs_per_segment` regions are selected and all subgraphs from
    these regions will be loaded into `dataset`'s cache, which will provide
    batches of subgraphs in the next `steps_per_segment` steps.
    Then cache is cleared and next segment of regions are selected and loaded.
    """

    def __init__(
        self,
        dataset,
        selected_inds=None,
        batch_size=64,
        num_regions_per_segment=32,
        steps_per_segment=1000,
        num_workers=0,
        seed=None,
    ):
        """Initialize the iterator


        Args:
            dataset (CellularGraphDataset): CellularGraphDataset
            selected_inds (list): list of indices (of regions) to sample from,
                helpful for sampling only training/validation regions
            batch_size (int): batch size
            num_regions_per_segment (int): number of regions to sample from in each segment
            steps_per_segment (int): number of batches in each segment
            num_workers (int): number of workers for multiprocessing
            seed (int): random seed
        """
        self.dataset = dataset
        self.inuse_subset_inds = set()
        self.selected_inds = list(dataset.indices()) if selected_inds is None else list(selected_inds)
        self.set_subset_inds(self.selected_inds)

        self.batch_size = batch_size
        self.num_regions_per_segment = num_regions_per_segment
        self.steps_per_segment = steps_per_segment
        self.num_workers = multiprocessing.cpu_count() if num_workers is None else num_workers

        self.region_inds_queue = []
        self.fill_queue(seed=seed)

        self.step_counter = 0
        self.data_iter = None
        self.get_new_segment()

    def set_subset_inds(self, selected_inds):
        self.dataset.set_indices(selected_inds)
        self.inuse_subset_inds = set(self.dataset.indices())
        return

    def fill_queue(self, seed=None):
        """Fill the queue of region indices randomly"""
        if seed is not None:
            np.random.seed(seed)
        fill_inds = sorted(set(self.selected_inds) - set(self.region_inds_queue))
        np.random.shuffle(fill_inds)
        self.region_inds_queue.extend(fill_inds)

    def get_new_segment(self):
        """Sample regions for the new segment"""
        if self.num_regions_per_segment <= 0:
            self.set_subset_inds(self.selected_inds)
        else:
            graph_inds_in_segment = self.region_inds_queue[: self.num_regions_per_segment]
            self.region_inds_queue = self.region_inds_queue[self.num_regions_per_segment :]  # noqa: E203
            if len(self.region_inds_queue) < self.num_regions_per_segment:
                self.fill_queue()

            self.dataset.clear_cache()
            self.dataset.set_indices(graph_inds_in_segment)
            try:
                for idx in self.dataset.indices():
                    self.dataset.load_to_cache(idx, subgraphs=True)
            except FileNotFoundError as e:
                print(
                    "Cannot find subgraph chunk save files, "
                    + "try running `dataset.save_all_subgraphs_to_chunk()` first"
                )
                raise e
        sampler = RandomSampler(self.dataset, replacement=True, num_samples=int(1e10))
        loader = InfDataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            multiprocessing_context=get_context("loky"),
        )
        self.data_iter = iter(loader)
        self.step_counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.step_counter == self.steps_per_segment:
            # Current segment has sampled enough batches
            logger.info("Using next subsampler - first if there are more regions to sample")
            self.get_new_segment()
        if len(set(self.dataset.indices()) - self.inuse_subset_inds) != 0:
            logger.info("Using next subsampler - second if there are more regions to sample")
            # Change of region subset indices in the dataset or sampler, refresh queue
            self.region_inds_queue = []
            self.get_new_segment()
        batch = next(self.data_iter)
        self.step_counter += 1
        return batch


class BoundaryDataLoader:
    def __init__(
        self,
        slide_name,
        x_range_index,
        y_range_index,
        n_segments=20,
        z_index=0,
        padding=None,
        dir_location=None,
    ) -> None:
        self.slide_name = slide_name
        self.dir_location = dir_location

        self.x_range_index = x_range_index
        self.y_range_index = y_range_index
        self.n_segments = n_segments
        self.padding = padding
        self.centroid_to_fov_x = None
        self.centroid_to_fov_y = None
        self.z_index = z_index

    def generate_centroid_to_fov_dict(self, data, axis):
        mask_z = data.obs["sample"] == self.slide_name
        filtered_data = data[mask_z].obsm["spatial"][:, axis]

        min_val = filtered_data.min()
        max_val = filtered_data.max()
        segment_size = (max_val - min_val) / self.n_segments
        segments = [min_val + i * segment_size for i in range(self.n_segments + 1)]

        centroid_to_fov_dict = {}
        centroid_to_fov_dict_padding = {}

        if self.padding is not None:
            with open(f"{self.dir_location}/images/manifest.json", "r") as json_file:
                boundary_config = json.load(json_file)

            no_pixels = boundary_config["mosaic_width_pixels"] if axis == 0 else boundary_config["mosaic_height_pixels"]
            total_um = (
                abs(boundary_config["bbox_microns"][2] - boundary_config["bbox_microns"][0])
                if axis == 0
                else abs(boundary_config["bbox_microns"][3] - boundary_config["bbox_microns"][1])
            )
            padding_um = self.padding["value"] * total_um / no_pixels

        for i in range(self.n_segments):
            centroid_to_fov_dict[i] = [segments[i], segments[i + 1]]
            if self.padding is not None:
                centroid_to_fov_dict_padding[i] = [
                    segments[i] - padding_um,
                    segments[i + 1] + padding_um,
                ]

        if self.padding is not None:
            self.padding[f"centroid_to_fov_{axis}"] = centroid_to_fov_dict_padding

        if axis == 0:
            self.centroid_to_fov_x = centroid_to_fov_dict
        if axis == 1:
            self.centroid_to_fov_y = centroid_to_fov_dict

    def read_boundaries(self, data):
        if isinstance(self.x_range_index, int):
            self.x_range_index = [self.x_range_index]
        if isinstance(self.y_range_index, int):
            self.y_range_index = [self.y_range_index]

        combined_boundaries = {}

        for x_index in self.x_range_index:
            for y_index in self.y_range_index:
                boundaries_dict = self.get_boundaries_for_indices(data, x_index, y_index)
                combined_boundaries.update(boundaries_dict)

        return combined_boundaries

    def get_boundaries_for_indices(self, data, x_index, y_index):
        if self.centroid_to_fov_x is None:
            self.generate_centroid_to_fov_dict(data, axis=0)
        if self.centroid_to_fov_y is None:
            self.generate_centroid_to_fov_dict(data, axis=1)

        if self.padding is not None:
            x_range = self.padding["centroid_to_fov_0"][x_index]
            y_range = self.padding["centroid_to_fov_1"][y_index]
        else:
            x_range = self.centroid_to_fov_x[x_index]
            y_range = self.centroid_to_fov_y[y_index]

        mask_x = (data.obsm["spatial"][:, 0] >= x_range[0]) & (data.obsm["spatial"][:, 0] < x_range[1])
        mask_y = (data.obsm["spatial"][:, 1] >= y_range[0]) & (data.obsm["spatial"][:, 1] < y_range[1])
        mask_z = data.obs["sample"] == self.slide_name
        mask_combined = mask_x & mask_y & mask_z

        # filtered_ann_data = self.data[mask_combined].obs.index
        fov_values_filtered = list(set(data[mask_combined].obs["fov"]))
        if not fov_values_filtered:
            return {}

        boundaries_filtered_fovs_dict = self.get_boundaries_of_multiple_fov(data, fov_values_filtered)
        filtered_cell_ids = mask_combined[mask_combined == True].index.tolist()  # noqa: E712
        boundaries_filtered_fovs_dict = {
            key: val for key, val in boundaries_filtered_fovs_dict.items() if key in filtered_cell_ids
        }

        if self.padding is not None:
            cell_data = data[list(boundaries_filtered_fovs_dict.keys())]
            mask_padding = (
                (self.centroid_to_fov_x[x_index][0] < cell_data.obsm["spatial"][:, 0])
                & (cell_data.obsm["spatial"][:, 0] < self.centroid_to_fov_x[x_index][1])
                & (self.centroid_to_fov_y[y_index][0] < cell_data.obsm["spatial"][:, 1])
                & (cell_data.obsm["spatial"][:, 1] <= self.centroid_to_fov_y[y_index][1])
            )
            is_padding = dict(
                zip(
                    cell_data[mask_padding].obs.index.tolist(),
                    [False] * len(cell_data[mask_padding].obs.index.tolist()),
                )
            )
            is_padding.update(
                dict(
                    zip(
                        cell_data[~mask_padding].obs.index.tolist(),
                        [True] * len(cell_data[~mask_padding].obs.index.tolist()),
                    )
                )
            )
            self.padding["bool_info"] = is_padding
        return boundaries_filtered_fovs_dict

    def validate_selection(self, chosen_group):
        for name, item in chosen_group.items():
            if isinstance(item, h5py.Dataset):
                return item[()]
            elif isinstance(item, h5py.Group):
                print("Item is a group. Name:", name)
                raise
            else:
                print("Item is of unknown type. Name:", name)
                raise

    def filter_data(self, file, cell_id):
        z_index_group_name = f"zIndex_{self.z_index}"
        z_index_group_sub_name = "p_0"
        final_selection = file["featuredata"][cell_id][z_index_group_name][z_index_group_sub_name]
        cell_boundary_matrix = self.validate_selection(final_selection)
        return cell_boundary_matrix

    def get_boundaries_of_one_fov(self, data, fov_value):
        boundaries = {}

        mask_x = data.obs.fov == fov_value
        mask_y = data.obs["sample"] == self.slide_name
        mask_combined = mask_x & mask_y

        filtered_ann_data = data[mask_combined].obs.index

        # Open the HDF5 file in read mode
        hdf5_file = f"{self.dir_location}/cell_boundaries/feature_data_{fov_value}.hdf5"
        file = h5py.File(hdf5_file, "r")

        for cell_id in filtered_ann_data:
            boundaries[cell_id] = self.filter_data(file, cell_id)

        return boundaries

    def get_boundaries_of_multiple_fov(self, data, fov_values):
        boundaries = {}

        for fov_value in fov_values:
            boundary_dict = self.get_boundaries_of_one_fov(data, fov_value)
            boundaries.update(boundary_dict)

        return boundaries


if __name__ == "__main__":
    dataset_kwargs = {
        "transform": [],
        "pre_transform": None,
        "raw_folder_name": "graph",
        "processed_folder_name": "tg_graph",
        "node_features": [
            "cell_type",
            "volume",
            "biomarker_expression",
            "neighborhood_composition",
            "center_coord",
        ],
        "edge_features": ["edge_type", "distance"],
        "subgraph_size": 3,
        "subgraph_source": "on-the-fly",
        "subgraph_allow_distant_edge": True,
        "subgraph_radius_limit": 200.0,
    }

    feature_kwargs = {
        "biomarker_expression_process_method": "linear",
        "biomarker_expression_lower_bound": 0,
        "biomarker_expression_upper_bound": 18,
        "neighborhood_size": 10,
    }
    dataset_kwargs.update(feature_kwargs)

    # Initialize dataset and save all subgraphs
    dataset = CellularGraphDataset("data/example_dataset", **dataset_kwargs)
    dataset.save_all_subgraphs_to_chunk()

    # Initialize sampler
    dataset.set_subgraph_source("chunk_save")
    data_iter = SubgraphSampler(dataset)
    batch = next(data_iter)

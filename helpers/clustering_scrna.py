import networkx as nx
import pandas as pd
import numpy as np
import scanpy as sc
import scanpy.external as sce
import squidpy as sq
from scipy.spatial import Delaunay
from shapely.geometry import Polygon
import geopandas as gpd
from rtree import index

from .mlflow_client_ import read_run_result_ann_data
from .data_boundary import BoundaryDataLoader
from .graph_build import calcualte_voronoi_from_coords, build_graph_from_cell_coords, assign_attributes, get_edge_type
from .plotly_helpers import plotly_spatial_scatter_categorical, plotly_spatial_scatter_edges

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, facecolor="white")


class spatialPipeline:
    def __init__(self, options):
        # super().__init__()
        self.dir_location = options["dir_loc"]
        self.names_list = options["names_list"]
        if options.get("mlflow_config"):
            y_data_filter_name = options["mlflow_config"]["data_filter_name"]
            y_resolution = options["mlflow_config"]["leiden_resolution"]
            self.data = read_run_result_ann_data(y_data_filter_name, y_resolution)
        else:
            self.data = []
            self.read_data()
        self.segment_instances = {}       

    def read_data(self):
        for name in self.names_list:
            vizgen_dir_ab = f"{self.dir_location}/data/{name}"
            adata_ab = sq.read.vizgen(
                path=vizgen_dir_ab,
                counts_file="cell_by_gene.csv",
                meta_file="cell_metadata.csv",
                transformation_file="micron_to_mosaic_pixel_transform.csv",
            )
            # create a new column with the same value
            adata_ab.obs = adata_ab.obs.assign(sample=f"{name}")
            self.data.append(adata_ab)

        self.data = sc.concat(self.data, axis=0)

    def preprocess_data(self, proprocessor):
        self.data, self.data_raw = proprocessor.extract(self.data)

    def perform_pca(self, pca_options):
        sc.tl.pca(self.data, **pca_options)

    def perform_batch_correction(self, batch_correction_options):
        sce.pp.harmony_integrate(self.data, **batch_correction_options)

    def perform_clustering(self, clustering_options):
        # Preprocess the data
        sc.tl.umap(self.data, **clustering_options["umap"])

        # Louvain clustering
        sc.tl.louvain(self.data, **clustering_options["louvain"])

        # Leiden clustering
        sc.tl.leiden(self.data, **clustering_options["leiden"])

    def compute_neighborhood_graph(self, ng_options):
        sc.pp.neighbors(self.data, **ng_options)  # , n_pcs=50

    def read_boundary_arrays(self, segment_config):
        
        slide_name = segment_config["sample_name"]
        x_range_index = segment_config["region_id"][0]
        y_range_index = segment_config["region_id"][1]
        n_segments=segment_config.get("segments_per_dimension", 20)
        z_index=segment_config.get("z_index", 0)
        padding_info = segment_config.get("padding", None)

        boundary_instance = BoundaryDataLoader(slide_name, x_range_index, y_range_index, n_segments=n_segments, z_index=z_index, padding=padding_info)
        boundary_arrays = boundary_instance.read_boundaries(self.data)
        self.segment_instances[(slide_name, x_range_index, y_range_index, n_segments)] = boundary_instance
        return boundary_arrays

    def get_boundaries_for_indices(self, spatial_loader):
        return spatial_loader.get_boundaries_for_indices(self.data)

    def get_boundaries_of_one_fov(self, spatial_loader, fov_value):
        return spatial_loader.get_boundaries_of_one_fov(self.data, fov_value)

    def get_boundaries_of_multiple_fov(self, spatial_loader, fov_values):
        return spatial_loader.get_boundaries_of_multiple_fov(self.data, fov_values)
    
    def _get_cell_centroids_array(self, cell_ids):
        segment_cells_mask = np.zeros(self.data.obs.shape[0], dtype=bool)
        segment_cells_mask[
            np.where(self.data.obs.index.isin(cell_ids))
        ] = True
        segment_spatial_data = self.data.obsm["spatial"][segment_cells_mask]
        cell_ids_ordered = self.data.obs.index[segment_cells_mask].tolist()
        return cell_ids_ordered, segment_spatial_data

    def _get_cell_centroids_df(self, segment_boundaries, ):

        mask = np.zeros(self.data.obs.shape[0], dtype=bool)
        mask[
            np.where(self.data.obs.index.isin(segment_boundaries.keys()))
        ] = True

        filtered_spatial_data = self.data.obsm["spatial"][mask]

        cell_data = pd.DataFrame(
            filtered_spatial_data,
            columns=["X", "Y"],
            index=self.data.obs.index[mask],
        ).reset_index()
        cell_data = cell_data.rename(columns={"index": "CELL_ID"})
        cell_data["CELL_TYPE"] = self.data.obs['cell_type'][mask].to_list()
        cell_data['volume'] = self.data.obs['volume'][mask].to_list()
        bm_columns = ("BM-" + self.data.var.index).tolist() #[f"BM-{bm_temp}" for bm_temp in .to_list()]
        cell_data[bm_columns] = pd.DataFrame(self.data[cell_data['CELL_ID'], :].X.toarray())

        return cell_data, filtered_spatial_data

    def build_networkx_for_region(self, segment_config, pretransform_networkx_config, network_features_config):

        segment_boundaries_given = self.read_boundary_arrays(segment_config)
        segment_data, segment_centroids = self._get_cell_centroids_df(segment_boundaries_given)
        segment_boundaries = [
            segment_boundaries_given[item][0] for item in segment_data["CELL_ID"].values
        ]
        if pretransform_networkx_config.get("boundary_type") == "voronoi":
            voronoi_polygons_calculated = calcualte_voronoi_from_coords(
                segment_centroids[:, 0], segment_centroids[:, 1]
            )  
            G_segment, node_to_cell_mapping = build_graph_from_cell_coords(
                segment_data, voronoi_polygons_calculated,
                boundary_augments="voronoi_polygon",
                edge_config=pretransform_networkx_config["edge_config"],
            )

            
        else:

            G_segment, node_to_cell_mapping = build_graph_from_cell_coords(
                segment_data,
                segment_boundaries,
                boundary_augments="boundary_polygon",
                edge_config=pretransform_networkx_config["edge_config"]
            )

        padding_info = segment_config.get("padding", None)
        if padding_info is not None:
            padding_dict = self.segment_instances[(segment_config["sample_name"], segment_config["region_id"][0], segment_config["region_id"][1], segment_config.get("segments_per_dimension", 20))].padding["bool_info"]
        else:
            padding_dict = None

        G_segment = assign_attributes(G_segment, segment_data, segment_boundaries_given, node_to_cell_mapping, pretransform_networkx_config["neighbor_edge_cutoff"], padding_dict)

        # TODO: refactor for modular node and edge feature builds
        # cell_ids, cell_centroids = self._get_cell_centroids_array(segment_boundaries.keys())
        # segment_boundaries = calcualte_voronoi_from_coords(
        #     cell_centroids[:, 0], cell_centroids[:, 1]
        # )
        # segment_boundaries = dict(zip(cell_ids, segment_boundaries))
        # G_segment = self._build_network_from_cell_centroids(
        #     segment_boundaries, edge_logic=pretransform_networkx_config["edge_config"]
        # )
        # G_segment = self._assign_network_attributes(
        #     G_segment, network_features_config["node_features"], network_features_config["edge_features"], **pretransform_networkx_config
        # )
        # G_built.region_id = slice_region_id

        return G_segment

    def build_networkx_plots(self, G_segment, pretransform_networkx_config, save_=False, cat_column = "cell_type"):

        plots_ = {}        
        if pretransform_networkx_config["edge_config"]["type"] == "Delaunay":
            given_boundaries = {G_segment.nodes[n]['cell_id'] : G_segment.nodes[n]["boundary_polygon"] for n in G_segment.nodes}
            if pretransform_networkx_config["boundary_type"] == "voronoi":
                voronoi_boundaries = {G_segment.nodes[n]['cell_id'] : G_segment.nodes[n]["voronoi_polygon"] for n in G_segment.nodes}
                vor_bound_fig = plotly_spatial_scatter_categorical(voronoi_boundaries, self.data.obs[cat_column])
                plots_["voronoi_fig.html"] = vor_bound_fig
            given_bound_fig = plotly_spatial_scatter_categorical(given_boundaries, self.data.obs[cat_column])
            plots_["boundary_given_fig.html"] = given_bound_fig
        
        if pretransform_networkx_config["edge_config"]["type"] == "R3Index":
            if pretransform_networkx_config["edge_config"]["bound_type"] == "rectangle":
                rec_boundaries = {G_segment.nodes[n]['cell_id'] : G_segment.nodes[n]["rectangle"] for n in G_segment.nodes}
                boundary_box_fig = plotly_spatial_scatter_categorical(rec_boundaries, self.data.obs[cat_column])            
                plots_["boundary_box_fig.html"] = boundary_box_fig
            if pretransform_networkx_config["edge_config"]["bound_type"] == "rotated_rectangle":
                rot_rec_boundaries = {G_segment.nodes[n]['cell_id'] : G_segment.nodes[n]["rotated_rectangle"] for n in G_segment.nodes}
                boundary_box_rotated_fig = plotly_spatial_scatter_categorical(rot_rec_boundaries, self.data.obs[cat_column])
                plots_["boundary_box_rotated_fig.html"] = boundary_box_rotated_fig

        plots_[pretransform_networkx_config["edge_config"]["type"]] = plotly_spatial_scatter_edges(G_segment, self.data.obs[cat_column], edge_info=pretransform_networkx_config["edge_config"]["type"])

        if save_:
            for key, val in plots_.items():
                val.write_html(f"{self.dir_location}/{key}")


    def _assign_network_attributes(self, G, node_features, edge_features, **kwargs):
        """Assign node and edge attributes to the cellular graph

        Args:
            G (nx.Graph): full cellular graph of the region
            cell_data (pd.DataFrame): dataframe containing cellular data
            node_to_cell_mapping (dict): 1-to-1 mapping between
                node index in `G` and cell id

        Returns:
            nx.Graph: populated cellular graph
        """

        for node, attributes in G.nodes.items():

            assert G.nodes[node]['cell_id'] == attributes['cell_id']
            curr_cell_id = attributes['cell_id']

            build_items = {}
            temp = self.data[curr_cell_id]

            if "biomarker_expression" in node_features:
                gene_panel = list("BM-" + temp.var.index)
                expression_values = temp.X.toarray().flatten()
                build_items["biomarker_expression"] = dict(zip(gene_panel, expression_values))
            
            if "cell_type" in node_features:
                build_items["cell_type"] = temp.obs['cell_type'].iloc[0]

            if "center_coord" in node_features:
                build_items["center_coord"] = tuple(temp.obsm['spatial'].flatten())

            if "volume" in node_features:
                build_items["volume"] = temp.obs['volume'].iloc[0]

            if "boundary_polygon" in node_features:
                # boundary_polygon is already a node attribute
                assert G.nodes[node].get('boundary_polygon') is not None

            G.nodes[node].update(build_items)

        assert "distance" in edge_features
        # Add distance, edge type (by thresholding) to edge feature
        edge_properties = get_edge_type(G, neighbor_edge_cutoff=kwargs.get("neighbor_edge_cutoff", 0))
        nx.set_edge_attributes(G, edge_properties)
        return G
    

    def _build_network_from_cell_centroids(self, cell_boundaries, edge_logic=None):
        """Construct a networkx graph based on cell coordinates

        Args:
            cell_boundaries (list): list of boundaries of cells,
                represented by the coordinates of their exterior vertices
            edge_logic (str, optional): logic to use for edge construction.

        Returns:
            G (nx.Graph): full cellular graph of the region
        """

        cell_ids, cell_centroids = self._get_cell_centroids_array(list(cell_boundaries.keys()))
        # coord_ar = np.column_stack((cell_ids, cell_centroids))

        G = nx.Graph()

        if edge_logic['type'] == 'Delaunay':
            for ind, cell_id in enumerate(cell_ids):
                G.add_node(ind, boundary_polygon=cell_boundaries[cell_id], cell_id=cell_id)
            
            # TODO: fix edge issues
            dln = Delaunay(cell_centroids[np.lexsort((cell_centroids[:, 1], cell_centroids[:, 0]))])
            neighbors = [set()]*len(cell_ids)
            for t in dln.simplices:
                for v in t:
                    neighbors[v].update(t)
            
            for i, ns in enumerate(neighbors):
                for n in ns:
                    G.add_edge(int(i), int(n))

        elif edge_logic['type'] == 'R3Index':

            # Create a GeoDataFrame with the geometry column
            # gdf = gpd.GeoDataFrame(geometry=[Polygon(coords) for coords in cell_boundaries])
            # Create an R-tree spatial index
            spatial_index = index.Index()

            # Populate the spatial index with bounding boxes and cell indices
            for idx, cell_id in enumerate(cell_ids):
                spatial_index.insert(idx, Polygon(cell_boundaries[cell_id]).bounds)

            # Merge the GeoDataFrame with cell_data based on index
            # gdf = gdf.merge(coord_ar, left_index=True, right_index=True)                

            chosen_boundary_type = edge_logic.get("bound_type")
            for i, cell_id in enumerate(cell_ids):
                cell_boundary = Polygon(cell_boundaries[cell_id])
                if chosen_boundary_type == "rectangle":
                    temp_bounds = np.array(list(cell_boundary.bounds))
                    temp_bounds = np.array([[temp_bounds[0], temp_bounds[1]], [temp_bounds[0], temp_bounds[3]], [temp_bounds[2], temp_bounds[3]], [temp_bounds[2], temp_bounds[1]]])
                elif chosen_boundary_type == "rotated_rectangle":
                    temp_bounds = np.array(list(cell_boundary.minimum_rotated_rectangle.exterior.coords))
                else:
                    raise ValueError("bound_type must be either rectangle or rotated_rectangle")

                G.add_node(i, boundary_polygon=cell_boundaries[cell_id], cell_id=cell_id, **{f"{chosen_boundary_type}": temp_bounds})

            # Iterate through cells and find nearby candidates using the spatial index
            threshold_distance = edge_logic.get("threshold_distance", 0)
            for i, cell_id in enumerate(cell_ids):
                # exterior_coords = cell.geometry.exterior.coords
                # intersection = cell.geometry.intersection(candidate_cell.geometry)
                # shared_boundary_length = cell.geometry.boundary.intersection(candidate_cell.geometry.boundary).length
                # shared_area = intersection.area
                cell_boundary = Polygon(cell_boundaries[cell_id])
                candidate_indices = list(spatial_index.intersection(cell_boundary.buffer(threshold_distance).bounds))
                for idx in candidate_indices:
                    if idx != i:
                        G.add_edge(i, idx)

        elif edge_logic['type'] == "MST":
            # logic for minimum spanning tree of a network
            pass


        return G


class spatialPreProcessor:
    def __init__(self, options):
        self.options = options

    def extract(self, data):
        if self.options["var_names_make_unique"] == True:
            data.var_names_make_unique()

        data.obs["is_not_filtered_cells"] = sc.pp.filter_cells(
            data, **self.options["filter_cells"], inplace=False
        )[0]
        data.var["is_not_filtered_genes"] = sc.pp.filter_genes(
            data, **self.options["filter_genes"], inplace=False
        )[0]
        print(f"Shape of AnnData: {data.shape}")
        if self.options["find_mt"] == True:
            data.var["mt"] = data.var_names.str.startswith("mt-")
        if self.options["find_rp"] == True:
            data.var["rp"] = data.var_names.str.contains("^Rp[sl]")

        # if self.options["find_mt"] == True or self.options["find_rp"] == True:
        data_filtered = data[
            data.obs["is_not_filtered_cells"],
            data.var["is_not_filtered_genes"],
        ]
        sc.pp.calculate_qc_metrics(
            data_filtered,
            **self.options["qc_metrics"],
        )
        sc.pp.normalize_total(
            data_filtered,
            **self.options["normalize_totals"],
        )
        sc.pp.log1p(
            data_filtered,
            **self.options["log1p_transformation"],
        )
        sc.pp.scale(
            data_filtered,
            **self.options["scale"],
        )
        print(f"Shape of AnnData filtered: {data_filtered.shape}")
        print(f"Shape of AnnData raw: {data.shape}")
        return data_filtered, data
import os
import h5py
import scanpy as sc
import scanpy.external as sce
import squidpy as sq


sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, facecolor="white")


class spatialPipeline:
    def __init__(self, options):
        # super().__init__()
        self.dir_location = options["dir_loc"]
        self.names_list = options["names_list"]
        self.data = []
        self.read_data()

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

    def filter_by_centroid_coordinates(self, spatial_loader):
        return spatial_loader.filter_by_centroid_coordinates(self.data)

    def get_boundaries_for_indices(self, spatial_loader):
        return spatial_loader.get_boundaries_for_indices(self.data)

    def get_boundaries_of_one_fov(self, spatial_loader, fov_value):
        return spatial_loader.get_boundaries_of_one_fov(self.data, fov_value)

    def get_boundaries_of_multiple_fov(self, spatial_loader, fov_values):
        return spatial_loader.get_boundaries_of_multiple_fov(self.data, fov_values)


class BoundaryDataLoader:
    def __init__(self, slide_name, x_range_index, y_range_index, n_segments=20, z_index=0) -> None:

        self.slide_name = slide_name
        self.dir_location = f"{os.path.realpath(os.path.join(os.getcwd()))}/data/{self.slide_name}"
        
        self.x_range_index = x_range_index
        self.y_range_index = y_range_index
        self.n_segments = n_segments
        self.centroid_to_fov_x = None
        self.centroid_to_fov_y = None
        self.z_index = z_index

    def generate_centroid_to_fov_dict(self, data, axis):

        mask_z = (data.obs["liver_slice"] == self.slide_name)
        filtered_data = data[mask_z].obsm["spatial"][:, axis]

        min_val = filtered_data.min()
        max_val = filtered_data.max()
        segment_size = (max_val - min_val) / self.n_segments
        segments = [min_val + i * segment_size for i in range(self.n_segments + 1)]

        centroid_to_fov_dict = {}

        for i in range(self.n_segments):
            centroid_to_fov_dict[i] = [segments[i], segments[i + 1]]

        return centroid_to_fov_dict

    def filter_by_centroid_coordinates(self, data):
        if isinstance(self.x_range_index, int):
            self.x_range_index = [self.x_range_index]
        if isinstance(self.y_range_index, int):
            self.y_range_index = [self.y_range_index]

        combined_boundaries = {}

        for x_index in self.x_range_index:
            for y_index in self.y_range_index:
                boundaries_dict = self.get_boundaries_for_indices(
                    data, x_index, y_index
                )
                combined_boundaries.update(boundaries_dict)

        return combined_boundaries

    def get_boundaries_for_indices(self, data, x_index, y_index):
        if self.centroid_to_fov_x is None:
            self.centroid_to_fov_x = self.generate_centroid_to_fov_dict(data, axis=0)
        if self.centroid_to_fov_y is None:
            self.centroid_to_fov_y = self.generate_centroid_to_fov_dict(data, axis=1)

        x_range = self.centroid_to_fov_x[x_index]
        y_range = self.centroid_to_fov_y[y_index]

        mask_x = (data.obsm["spatial"][:, 0] >= x_range[0]) & (
            data.obsm["spatial"][:, 0] < x_range[1]
        )
        mask_y = (data.obsm["spatial"][:, 1] >= y_range[0]) & (
            data.obsm["spatial"][:, 1] < y_range[1]
        )
        mask_z = (data.obs["liver_slice"] == self.slide_name)
        mask_combined = mask_x & mask_y & mask_z

        # filtered_ann_data = self.data[mask_combined].obs.index
        fov_values_filtered = list(set(data[mask_combined].obs["fov"]))
        boundaries_filtered_fovs_dict = self.get_boundaries_of_multiple_fov(
            data,
            fov_values_filtered
        )

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
        z_index_group_sub_name = f"p_0"
        final_selection = file["featuredata"][cell_id][z_index_group_name][
            z_index_group_sub_name
        ]
        cell_boundary_matrix = self.validate_selection(final_selection)
        return cell_boundary_matrix

    def get_boundaries_of_one_fov(self, data, fov_value):
        boundaries = {}

        mask_x = (data.obs.fov == fov_value)
        mask_y = (data.obs["liver_slice"] == self.slide_name)
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
import os
import json
import h5py

class BoundaryDataLoader:
    def __init__(self, slide_name, x_range_index, y_range_index, n_segments=20, z_index=0, padding=None) -> None:

        self.slide_name = slide_name
        self.dir_location = f"{os.path.realpath(os.path.join(os.getcwd()))}/data/{self.slide_name}"
        
        self.x_range_index = x_range_index
        self.y_range_index = y_range_index
        self.n_segments = n_segments
        self.padding = padding
        self.centroid_to_fov_x = None
        self.centroid_to_fov_y = None
        self.z_index = z_index

    def generate_centroid_to_fov_dict(self, data, axis):

        mask_z = (data.obs["sample"] == self.slide_name)
        filtered_data = data[mask_z].obsm["spatial"][:, axis]

        min_val = filtered_data.min()
        max_val = filtered_data.max()
        segment_size = (max_val - min_val) / self.n_segments
        segments = [min_val + i * segment_size for i in range(self.n_segments + 1)]

        centroid_to_fov_dict = {}
        centroid_to_fov_dict_padding = {}

        if self.padding is not None:
            with open(f"{self.dir_location}/images/manifest.json", 'r') as json_file:
                boundary_config = json.load(json_file)
            
            no_pixels = boundary_config['mosaic_width_pixels'] if axis == 0 else boundary_config["mosaic_height_pixels"]
            total_um = abs(boundary_config['bbox_microns'][2] - boundary_config['bbox_microns'][0]) if axis ==0 else abs(boundary_config['bbox_microns'][3] - boundary_config['bbox_microns'][1])
            padding_um = self.padding["value"]*total_um/no_pixels

        for i in range(self.n_segments):
            centroid_to_fov_dict[i] = [segments[i], segments[i + 1]]
            if self.padding is not None:
                centroid_to_fov_dict_padding[i] = [segments[i]-padding_um, segments[i + 1]+padding_um]

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
                boundaries_dict = self.get_boundaries_for_indices(
                    data, x_index, y_index
                )
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

        mask_x = (data.obsm["spatial"][:, 0] >= x_range[0]) & (
            data.obsm["spatial"][:, 0] < x_range[1]
        )
        mask_y = (data.obsm["spatial"][:, 1] >= y_range[0]) & (
            data.obsm["spatial"][:, 1] < y_range[1]
        )
        mask_z = (data.obs["sample"] == self.slide_name)
        mask_combined = mask_x & mask_y & mask_z

        # filtered_ann_data = self.data[mask_combined].obs.index
        fov_values_filtered = list(set(data[mask_combined].obs["fov"]))
        if not fov_values_filtered:
            return {}
            
        boundaries_filtered_fovs_dict = self.get_boundaries_of_multiple_fov(
            data,
            fov_values_filtered
        )
        filtered_cell_ids = mask_combined[mask_combined == True].index.tolist()
        boundaries_filtered_fovs_dict = {key: val for key, val in boundaries_filtered_fovs_dict.items() if key in filtered_cell_ids}

        if self.padding is not None:
            cell_data = data[list(boundaries_filtered_fovs_dict.keys())]
            mask_padding = (self.centroid_to_fov_x[x_index][0] < cell_data.obsm["spatial"][:, 0]) \
            & (cell_data.obsm["spatial"][:, 0] < self.centroid_to_fov_x[x_index][1]) \
            & (self.centroid_to_fov_y[y_index][0] < cell_data.obsm["spatial"][:, 1]) \
            & (cell_data.obsm["spatial"][:, 1] <= self.centroid_to_fov_y[y_index][1])
            is_padding = dict(zip(cell_data[mask_padding].obs.index.tolist(), [False]*len(cell_data[mask_padding].obs.index.tolist())))
            is_padding.update(dict(zip(cell_data[~mask_padding].obs.index.tolist(), [True]*len(cell_data[~mask_padding].obs.index.tolist()))))
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
        z_index_group_sub_name = f"p_0"
        final_selection = file["featuredata"][cell_id][z_index_group_name][
            z_index_group_sub_name
        ]
        cell_boundary_matrix = self.validate_selection(final_selection)
        return cell_boundary_matrix

    def get_boundaries_of_one_fov(self, data, fov_value):
        boundaries = {}

        mask_x = (data.obs.fov == fov_value)
        mask_y = (data.obs["sample"] == self.slide_name)
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
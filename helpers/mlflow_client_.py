import os
import mlflow
import scanpy as sc
from .logging_setup import logger

mlflow.set_tracking_uri("/data/qd452774/spatial_transcriptomics/mlruns/")
mlflow_client = mlflow.tracking.MlflowClient()

# Read the SLURM_CPUS_PER_TASK environment variable
# SLURM_CPUS_PER_TASK = os.environ.get("SLURM_CPUS_PER_TASK")
# Set the number of jobs to the value specified in SLURM_CPUS_PER_TASK
# NO_JOBS = int(SLURM_CPUS_PER_TASK) if SLURM_CPUS_PER_TASK is not None else 1
NO_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 3))


def get_run_info(x_experiment, x_resolution):

    xexp_name = f"spatial_clustering_{x_experiment}"
    xrun_name = f"steady-state-using-220-PCs-50-neighbors-leiden-{x_resolution}"

    # Get the experiment ID by name
    experiment = mlflow_client.get_experiment_by_name(xexp_name)
    experiment_id = experiment.experiment_id

    # Search for the run by run name within the specified experiment
    runs = mlflow_client.search_runs(experiment_ids=[experiment_id], filter_string=f"tags.run_name='{xrun_name}'")

    # Check if any runs match the criteria
    if len(runs) > 1:
        logger.error("more runs are there with same name than expected")
        return None
    elif len(runs) == 1:
        # run = runs.iloc[0]  # Assuming there is only one matching run
        # run_id = run.run_id
        return runs[0]
    else:    
        logger.debug(f"No matching run found for run name: {xrun_name} in experiment: {xexp_name}")
        return None


def read_run_result_ann_data(data_filter_name, x_resolution):

    xrun_info = get_run_info(data_filter_name, x_resolution)
    exp_id = xrun_info.info.experiment_id
    run_id = xrun_info.info.run_id
    x_anndata_path = f"/data/qd452774/spatial_transcriptomics/mlruns/{exp_id}/{run_id}/artifacts/data/spatial_clustering_ss_{data_filter_name}.h5ad"
    x_data = sc.read_h5ad(x_anndata_path)
    
    return x_data


import multiprocessing

# BASE_MICRONS = "20um"
# BASE_MICRONS = "10um"
BASE_MICRONS = "5um"
# BASE_MICRONS = "1um"

import json

# Define the directory where your JSON files are located
liver_slices = ["Liver1Slice1", "Liver1Slice2", "Liver2Slice1", "Liver2Slice2"]
base_dir = "/data/qd452774/spatial_transcriptomics/data"

file_names = [f"{base_dir}/{slice_item}/images/manifest.json" for slice_item in liver_slices]

SLICE_PIXELS_EDGE_CUTOFF = {}

for file_name_index in range(len(file_names)):
    # Load the JSON data from the file
    with open(file_names[file_name_index], 'r') as json_file:
        data = json.load(json_file)

    # Extract the microns_per_pixel value
    microns_per_pixel = data["microns_per_pixel"]

    # Calculate the transformed pixels for your NEIGHBOR_EDGE_CUTOFF value
    if BASE_MICRONS == "20um":
        SLICE_PIXELS_EDGE_CUTOFF[liver_slices[file_name_index]] = 20 / microns_per_pixel
    elif BASE_MICRONS == "10um":
        SLICE_PIXELS_EDGE_CUTOFF[liver_slices[file_name_index]] = 10 / microns_per_pixel
    elif BASE_MICRONS == "5um":
        SLICE_PIXELS_EDGE_CUTOFF[liver_slices[file_name_index]] = 5 / microns_per_pixel          
    elif BASE_MICRONS == "1um":
        SLICE_PIXELS_EDGE_CUTOFF[liver_slices[file_name_index]] = 1 / microns_per_pixel        

SLICE_PIXELS_EDGE_CUTOFF = {key: int(value) for key, value in SLICE_PIXELS_EDGE_CUTOFF.items()}


ANNOTATION_DICT = {
    '0': 'Hepatocytes_1',
    '1': 'Hepatocytes_Intermediate',
    '2': 'Hepatocytes_2',
    '3': 'Endothelial',
    '4': 'Hepatocytes_Low',
    '5': 'Kupffer_Monocytes',
    '6': 'Stromal cells',
    '7': 'Neutrophils',
    '8': 'Cholangiocytes',
    '9': 'T_NK_cells',
    '10': 'B_cells',
    '11': 'PDCs',
    '12': 'Pericytes',
    '13': 'Neurons',
    '14': 'Epithelial_Ivl',
    '15': 'Unknown_Hsd3b6',
    '16': 'Unknown_Aldh3b2',
    '17': 'Unknown_Olr1',
    '18': 'Unknown_Aldoart2',
    '19': 'Unknown_Jag2',
    '20': 'Platelets_HSC'
}

NUM_NODE_TYPE, NUM_EDGE_TYPE = 22, 3
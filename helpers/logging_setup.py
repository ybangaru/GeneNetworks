import logging
from logging import getLogger
import os

logger = getLogger(__name__)
logger.setLevel(logging.INFO)

# create console handler and set level to info
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


import multiprocessing

# NO_JOBS = 6

NO_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 2))

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
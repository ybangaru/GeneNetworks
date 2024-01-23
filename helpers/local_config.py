"""
This file contains local configuration variables.
"""
import os

NO_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 2))
PROJECT_DIR = "/data/qd452774/spatial_transcriptomics"
DATA_DIR = f"{PROJECT_DIR}/data"
MLFLOW_TRACKING_URI = f"{PROJECT_DIR}/mlruns/"

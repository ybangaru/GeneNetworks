"""
This file contains local configuration variables.
"""
import os

NO_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 2))
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BASE_DIR = "/data/qd452774/spatial_transcriptomics"
DATA_DIR = f"{BASE_DIR}/data"
MLFLOW_TRACKING_URI = f"{BASE_DIR}/mlruns/"

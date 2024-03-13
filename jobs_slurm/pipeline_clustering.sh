#!/usr/bin/bash

################################################################################
# Description: This script is used for starting the clustering pipeline of scRNA datasets
#              on a Slurm cluster as a batch job.
# **Note: # requires config modifications based on slurm cluster setup
# Usage: To train the model on a Slurm cluster, submit this script as a job using
#        the `sbatch` command.
#        For example, `sbatch jobs_slurm/pipeline_clustering.sh` while in the PROJECT_DIR.
################################################################################

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=40G
#SBATCH --time=14-0:00:00
#SBATCH --job-name=pipeline-clustering
#SBATCH --output=logs/pipeline_clustering-%J.log
#SBATCH --nodelist=compute-node006

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "------------------------------------------------------------"

# Set up environment variables
project_directory="$PROJECT_DIR"

# Validate the environment variable
if [[ -z "$PROJECT_DIR" ]]; then
    echo "Warning: Environment variable PROJECT_DIR is not set. Using the current working directory as the project directory."
    project_directory=$(pwd)
else
    project_directory="$PROJECT_DIR"
fi
echo "The project directory is: $project_directory"

# Change to the project directory
cd "$PROJECT_DIR" || { echo "Error: Unable to change directory to $project_directory"; exit 1; }

# Load environment modules
# Note: You may need to modify this based on your cluster setup
module load python/3.8.18
source .venv/bin/activate || { echo "Error: Failed to activate virtual environment"; }

# export environment variables
export GIT_PYTHON_REFRESH=quiet

# to run the clustering module script
python3 pipeline_clustering_vigzen.py

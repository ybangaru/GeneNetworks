#!/usr/bin/bash

################################################################################
# Description: This script is used for training the node classification model on a Slurm cluster.
# **Note: # requires config modifications based on slurm cluster setup
# Usage: To train the model on a Slurm cluster, submit this script as a job using
#        the `sbatch` command.
#        For example, `sbatch jobs_slurm/pipeline_classification.sh` while in the PROJECT_DIR.
################################################################################

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=80G
#SBATCH --time=0-11:00
#SBATCH --job-name=pipeline-classification
#SBATCH --output=logs/pipeline_classification-%J.log

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
# module load python/3.8.18
# source .venv/bin/activate || { echo "Error: Failed to activate virtual environment"; }
export CONDA_ROOT=/data/qd452774/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
# Now you can activate your configured conda environments
# source .venv/bin/activate
conda activate jupyterlab
# export environment variables
export GIT_PYTHON_REFRESH=quiet
export SLURM_CPUS_PER_TASK=10

# to run the python script
# python3 pipeline_build_nx_graphs.py
python3 pipeline_node_classification.py
# python3 pipeline_node_eval.py
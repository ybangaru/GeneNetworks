#!/usr/bin/bash

################################################################################
# Description: This script deploys the mlflow dashboard to the Slurm cluster.
# **Note: # requires config modifications based on slurm cluster setup
# Usage: To start code-server on a Slurm cluster, run this script as a job using
#        the `sbatch` command.
#        For example, `sbatch jobs_slurm/mlflow_start.sh` while in the PROJECT_DIR.
################################################################################

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=100G
#SBATCH --time=14-0:00:00
#SBATCH --job-name=ml-flow-dashboard
#SBATCH --output=logs/mlflow-%J.log
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
# module load python/3.8.18
# source .venv/bin/activate || { echo "Error: Failed to activate virtual environment";}
export CONDA_ROOT=/data/qd452774/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
# Now you can activate your configured conda environments
# source .venv/bin/activate
conda activate jupyterlab

# setting random ports
TUNNELPORT=`shuf -i 8501-9000 -n 1`
MLFLOWPORT=$(shuf -i 9001-9500 -n 1)

# start reverse tunnel
ssh -R$TUNNELPORT:localhost:$MLFLOWPORT $SLURM_SUBMIT_HOST -N -f

# Print instructions for local tunneling
echo "mlflow dashboard port is $MLFLOWPORT"
echo "On your local machine, run:"
echo ""
echo "ssh -L8888:localhost:$TUNNELPORT $USER@$SLURM_SUBMIT_HOST -N -4"
echo ""
echo "and point your browser to http://localhost:8888/"
echo "Change '8888' to some other value if this port is already in use on your PC,"
echo "for example, you have more than one remote app running."
echo "To stop this app, run 'scancel $SLURM_JOB_ID'"


# Start the server
# export MLFLOW_SERVER_MEMORY=128G
srun -n1 mlflow ui --port=$MLFLOWPORT

#!/usr/bin/bash

################################################################################
# Description: This script deploys a jupyterlab instance interactive shell environment.
#              The environment setup is available under the filename jupyterlab.yml
#              in the project directory which needs to be used to create a conda environment
#              called `jupyterlab`.
# **Note: # requires config modifications based on slurm cluster setup
# Usage: To start jupyterlab on Slurm cluster, run this script as a job using
#        the `sbatch` command.
#        For example, `sbatch jobs_slurm/jupyter_lab.sh` while in the PROJECT_DIR.
################################################################################

# SLURM job parameters
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=256G
#SBATCH --time=0-7:00
#SBATCH --job-name=jlab-instance
#SBATCH --output=logs/jlab-%J.log

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "------------------------------------------------------------"

# Load environment modules
# Note: You may need to modify this based on your cluster setup
export CONDA_ROOT=/data/qd452774/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
# Now you can activate your configured conda environments
# source .venv/bin/activate
conda activate jupyterlab

# set a random port for the notebook, in case multiple notebooks are
# on the same compute node.
NOTEBOOKPORT=`shuf -i 8000-8500 -n 1`

# set a random port for tunneling, in case multiple connections are happening
# on the same login node.
TUNNELPORT=`shuf -i 8501-9000 -n 1`

# set a random access token
TOKEN=`cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 49 | head -n 1`

echo "On your local machine, run:"
echo ""
echo "ssh -L8888:localhost:$TUNNELPORT $USER@$SLURM_SUBMIT_HOST -N -4"
echo ""
echo "and point your browser to http://localhost:8888/?token=$TOKEN"
echo "Change '8888' to some other value if this port is already in use on your PC,"
echo "for example, you have more than one remote notebook running."
echo "To stop this notebook, run 'scancel $SLURM_JOB_ID'"

# Set up a reverse SSH tunnel from the compute node back to the submitting host (login01 or login02)
# This is the machine we will connect to with SSH forward tunneling from our client.
ssh -R$TUNNELPORT\:localhost:$NOTEBOOKPORT $SLURM_SUBMIT_HOST -N -f

# Start the jupyterlab
srun -n1 jupyter lab --no-browser --port=$NOTEBOOKPORT --NotebookApp.token=$TOKEN --log-level WARN
# To stop the notebook, use 'scancel'

# Start the notebook example
# srun -n1 jupyter nbconvert --to python --execute pipeline_visualizations.ipynb

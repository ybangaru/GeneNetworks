#!/usr/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=64G
#SBATCH --time=2-0:00:00
#SBATCH --job-name=qd452774-clustering
#SBATCH --output=logs/pipeline_clustering-%J.log
#SBATCH --nodelist=compute-node004

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "------------------------------------------------------------"

# Load the same python as used for installation
# module load python37
# Insert this AFTER the #SLURM argument section of your job script
export CONDA_ROOT=/data/qd452774/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
# Now you can activate your configured conda environments
# source .venv/bin/activate
conda activate jupyterlab

# Start the notebook
export SLURM_CPUS_PER_TASK=6
# $CONDA_ROOT/envs/jupyterlab/bin/python pipeline_clustering_spatial.py
$CONDA_ROOT/envs/jupyterlab/bin/python pipeline_build_nx_graphs.py
# To stop the notebook, use 'scancel'

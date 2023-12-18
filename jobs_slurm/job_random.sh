#!/usr/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=90G
#SBATCH --time=10-00:00:00
#SBATCH --job-name=10hop
#SBATCH --output=logs/random-%J.log
#SBATCH --nodelist=compute-node003

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

export SLURM_CPUS_PER_TASK=2
# $CONDA_ROOT/envs/jupyterlab/bin/python cluster_comparison.py
$CONDA_ROOT/envs/jupyterlab/bin/python pipeline_node_classification.py

# Start the notebook
# srun -n1 jupyter nbconvert --to python --execute pipeline_visualizations_spatial.ipynb
# To stop the notebook, use 'scancel'

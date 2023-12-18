#!/usr/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=15G
#SBATCH --time=0-11:00
#SBATCH --job-name=qd452774-visualizations
#SBATCH --output=logs/pipeline_visualizations-%J.log

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "------------------------------------------------------------"

# Load the same python as used for installation
# module load python37
# Insert this AFTER the #SLURM argument section of your job script
export CONDA_ROOT=$HOME/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
# Now you can activate your configured conda environments
# source .venv/bin/activate
conda activate jupyterlab

# Start the notebook
srun -n1 jupyter nbconvert --to python --execute pipeline_visualizations.ipynb
# To stop the notebook, use 'scancel'

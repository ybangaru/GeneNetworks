#!/usr/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=128G
#SBATCH --time=2-0:00:00
#SBATCH --account=qd452774
#SBATCH --job-name=qd452774-python-code
#SBATCH --output=logs/python-code-%J.log

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "------------------------------------------------------------"


export CONDA_ROOT=/data/qd452774/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
# Now you can activate your configured conda environments
# source .venv/bin/activate
conda activate jupyterlab

cd /data/qd452774/spatial_transcriptomics

srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --mem-per-cpu=128G --time=0-8:00 --account=qd452774 --job-name=qd452774-code bash -l >> /data/qd452774/spatial_transcriptomics/logs/code-x.log 2>&1
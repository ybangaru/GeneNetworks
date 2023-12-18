#!/usr/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128G
#SBATCH --time=0-4:00
#SBATCH --account=qd452774
#SBATCH --job-name=qd452774-python-code
#SBATCH --output=logs/dash-app-%J.log
#SBATCH --nodelist=compute-node003

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

TUNNELPORT=`shuf -i 8501-9000 -n 1`
DASHPORT=$(shuf -i 9001-9500 -n 1)

ssh -R$TUNNELPORT:localhost:$DASHPORT $SLURM_SUBMIT_HOST -N -f

# Print instructions for local tunneling
echo "dashboard port is $DASHPORT"
echo "On your local machine, run:"
echo ""
echo "ssh -L8888:localhost:$TUNNELPORT $USER@$SLURM_SUBMIT_HOST -N -4"
echo ""
echo "and point your browser to http://localhost:8888/"
echo "Change '8888' to some other value if this port is already in use on your PC,"
echo "for example, you have more than one remote app running."
echo "To stop this app, run 'scancel $SLURM_JOB_ID'"

$CONDA_ROOT/envs/jupyterlab/bin/python dash_app.py --port $DASHPORT
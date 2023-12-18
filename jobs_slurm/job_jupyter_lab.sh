#!/usr/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=256G
#SBATCH --time=0-7:00
#SBATCH --job-name=qd452774-jupyter-lab
#SBATCH --output=spatial_transcriptomics/logs/jupyter-lab-%J.log

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

# Start the notebook
srun -n1 jupyter lab --no-browser --port=$NOTEBOOKPORT --NotebookApp.token=$TOKEN --log-level WARN
# To stop the notebook, use 'scancel'

#!/usr/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --time=0-4:00
#SBATCH --job-name=wandb-dashboard
#SBATCH --output=logs/wandb-%J.log
#SBATCH --nodelist=compute-node001

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "------------------------------------------------------------"

export CONDA_ROOT=/data/qd452774/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

conda activate jupyterlab

cd /data/qd452774/spatial_transcriptomics

# setting random ports
TUNNELPORT=`shuf -i 8501-9000 -n 1`
WANDBPORT=$(shuf -i 9001-9500 -n 1)

# set a random access token
# TOKEN=`cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 49 | head -n 1`

# start reverse tunnel
ssh -R$TUNNELPORT:localhost:$WANDBPORT $SLURM_SUBMIT_HOST -N -f

# Print instructions for local tunneling
echo "WANDB dashboard port is $WANDBPORT"
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
srun -n1 wandb server start --upgrade --port=$WANDBPORT

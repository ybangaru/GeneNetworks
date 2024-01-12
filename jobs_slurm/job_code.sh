#!/usr/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=128G
#SBATCH --time=2-0:00:00
#SBATCH --account=qd452774
#SBATCH --job-name=qd452774-python-code
#SBATCH --output=logs/python-code-%J.log
#SBATCH --nodelist=compute-node001

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "------------------------------------------------------------"

cd /data/qd452774/spatial_transcriptomics
export CONDA_ROOT=/data/qd452774/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

# add code server to root
export CODE_ROOT=/data/qd452774/.local
export PATH="$CODE_ROOT/bin:$PATH"

PASSWORD=`cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 49 | head -n 1` # random access token
CODE_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
# set a random port for tunneling, in case multiple connections are happening
# on the same login node.
TUNNELPORT=`shuf -i 8501-9000 -n 1`

echo "********************************************************************" 
echo "Starting code-server in Slurm"
echo "Environment information:" 
echo "Date:" $(date)
echo "Allocated node:" $(hostname)
echo "Path:" $(pwd)
echo "Password to access VSCode:" $PASSWORD
echo "Listening on code:" $TUNNELPORT
echo "********************************************************************" 

ssh -R$TUNNELPORT\:localhost:$CODE_PORT $SLURM_SUBMIT_HOST -N -f

echo "On your local machine, run:"
echo ""
echo "ssh -L8888:localhost:$TUNNELPORT $USER@$SLURM_SUBMIT_HOST -N -4"
echo ""
echo "and point your browser to http://localhost:8888/" to open the code server
echo "Change '8888' to some other value if this port is already in use on your PC,"
echo "for example, you have more than one remote app running."
echo "To stop this app, run 'scancel $SLURM_JOB_ID'"

srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=3 --mem-per-cpu=128G --time=0-11:00 code-server --bind-addr 0.0.0.0:$CODE_PORT --auth password --disable-telemetry
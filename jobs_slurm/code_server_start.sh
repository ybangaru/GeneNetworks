#!/usr/bin/bash

################################################################################
# Description: This script sets up a code-server instance on a Slurm cluster.
# **Note: # requires code-server to be installed on your local machine along with 
#           hashed password config setup
# **Note: # @ https://coder.com/docs/code-server/latest/install#installsh
# **Note: # requires config modifications based on slurm cluster setup
# Usage: To start code-server on a Slurm cluster, run this script as a job using
#        the `sbatch` command.
#        For example, `sbatch jobs_slurm/code_server_start.sh` while in the PROJECT_DIR.
################################################################################

# SLURM job parameters
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128G
#SBATCH --time=7-0:00:00
#SBATCH --job-name=graphxl
#SBATCH --output=logs/code-server-%J.log
#SBATCH --nodelist=compute-node001

# Print SLURM job information
echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "------------------------------------------------------------"


# add code server to path variable
export CODE_ROOT=/data/qd452774/.local
export PATH="$CODE_ROOT/bin:$PATH"

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
cd "$PROJECT_DIR" || { 
    echo "Error: Unable to change directory to $project_directory"
    exit 1
}

# Load environment modules
# Note: You may need to modify this based on your cluster setup
module load python/3.8.18
source .venv/bin/activate || { 
    echo "Error: Failed to activate virtual environment"
    exit 1
}

# Generate random port for SSH tunnel
CODE_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
# set a random port for tunneling, in case multiple connections are happening
# on the same login node.
TUNNELPORT=`shuf -i 8501-9000 -n 1`

# Print information about the code-server setup
echo "********************************************************************" 
echo "Starting code-server in Slurm"
echo "Environment information:" 
echo "Date:" $(date)
echo "Allocated node:" $(hostname)
echo "Path:" $(pwd)
echo "Listening on code:" $TUNNELPORT
echo "********************************************************************" 

# Set up SSH tunnel for secure access
ssh -R$TUNNELPORT\:localhost:$CODE_PORT $SLURM_SUBMIT_HOST -N -f

# Provide instructions for accessing code-server
echo "On your local machine, run:"
echo ""
echo "ssh -L8888:localhost:$TUNNELPORT $USER@$SLURM_SUBMIT_HOST -N -4"
echo ""
echo "and point your browser to http://localhost:8888/ to open the code server."
echo "Change '8888' to some other value if this port is already in use on your PC,"
echo "for example, you have more than one remote app running."
echo "To stop this app, run 'scancel $SLURM_JOB_ID'"

# Start code-server instance
# Note: You may need to modify this based on your cluster setup
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --mem-per-cpu=128G --time=7-0:00:00 code-server --bind-addr 0.0.0.0:$CODE_PORT --auth password --disable-telemetry

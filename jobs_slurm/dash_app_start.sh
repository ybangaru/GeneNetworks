#!/usr/bin/bash

################################################################################
# Description: This script sets up a Dash app on a Slurm cluster.
# Usage: To start a Dash app on a Slurm cluster, run this script as a job using
#        the `sbatch` command.
#        For example, `sbatch jobs_slurm/dash_app_start.sh` while in the PROJECT_DIR.
################################################################################

# SLURM job parameters
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=0-4:00
#SBATCH --job-name=dash-plotly
#SBATCH --output=logs/dash-plotly-%J.log
#SBATCH --nodelist=compute-node001

# Print SLURM job information
echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "------------------------------------------------------------"

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

cd /data/qd452774/spatial_transcriptomics || { 
    echo "Error: Unable to change directory to /data/qd452774/spatial_transcriptomics"
    exit 1
}

# Generate random port for SSH tunnel and Dash app
TUNNELPORT=$(shuf -i 8501-9000 -n 1)
DASHPORT=$(shuf -i 9001-9500 -n 1)

# Print information about the Dash app setup
echo "********************************************************************"
echo "Starting Dash app in Slurm"
echo "Environment information:"
echo "Date:" $(date)
echo "Allocated node:" $(hostname)
echo "Path:" $(pwd)
echo "Dashboard port:" $TUNNELPORT
echo "********************************************************************"

# Set up SSH tunnel for secure access
ssh -R$TUNNELPORT:localhost:$DASHPORT $SLURM_SUBMIT_HOST -N -f || { echo "Error: Failed to set up SSH tunnel"; exit 1; }

# Provide instructions for accessing the Dash app
echo "On your local machine, run:"
echo ""
echo "ssh -L8888:localhost:$TUNNELPORT $USER@$SLURM_SUBMIT_HOST -N -4"
echo ""
echo "and point your browser to http://localhost:8888/ to view the Dash app."
echo "Change '8888' to some other value if this port is already in use on your PC,"
echo "for example, you have more than one remote app running."
echo "To stop this app, run 'scancel $SLURM_JOB_ID'"

# Start the Dash app
python dash_app.py --port $DASHPORT

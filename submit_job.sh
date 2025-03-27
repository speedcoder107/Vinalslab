#!/bin/bash -l
#SBATCH --job-name=incompatibility_job    # Job name
#SBATCH --partition=agsmall              # Use the standard partition (what else could it be?)
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks-per-node=1               # Number of tasks per node (adjust based on usage)
#SBATCH --cpus-per-task=6                 # Assign 6 CPUs (one for each b value)
#SBATCH --time=28:00:00                   # Maximum runtime (3 days)
#SBATCH --mem=64GB                         # Memory per node
#SBATCH --mail-type=ALL                    # Email notifications (FAIL)
#SBATCH --mail-user=sharm849@umn.edu      # My UMN email

# Load necessary modules
module load python3                # Adjust based on Python version available on MSI

source activate my_XY_env

# Navigate to your code directory 
cd /users/3/sharm849/incompatibility_MSI/		   # (What should this be?)

# Install dependencies from requirements.txt

# Run your Python script
python incompatibility_code.py

# Additional logging (optional)
echo "Job completed at: $(date)" >> job_log.txt
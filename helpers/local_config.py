import os

NO_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 2))


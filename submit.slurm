#!/bin/bash
#SBATCH -n 16               # Number of cores
#SBATCH -t 7-00:00          # Runtime in D-HH:MM
#SBATCH -p gpu       # Partition to submit to: gpu, batch
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:2
#SBATCH -o job_out/log.out       # Write stdout to file named JOBIDNUM.out in current dir
#SBATCH -e job_out/log.err       # Write stderr to file named JOBIDNUM.err in current dir 

FLAGS=""

source activate lwp
python experiments/experiment_train.py ${FLAGS}
conda deactivate

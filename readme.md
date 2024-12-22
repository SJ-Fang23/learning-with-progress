# Dataset
Please get dataset from this link and put in human-demo/can-pick:
https://drive.google.com/drive/folders/1upTXzwI3AsTnKlqR_mz6qz_1J0boQ9I3?usp=sharing

# Training

To run the training, run 
```
experiments/experiment_training.py
```

add 

```
--exp_name=your_name_here
```
to log experiment in different names

# How to change reward shaping
add item to
```
shape_reward = []
```

line 74 in 
```
experiments/experiment_training.py
```

currenly supports: "progress_sign_loss","delta_progress_scale_loss", "value_sign_loss", "advantage_sign_loss" , "progress_value_loss"

# HPC
run `$ sbatch submit.slurm` to submit the script on HPC. By default, this is queued in the `gpu` partition requesting for 2 `A100` gpus and `64G` RAM. The outputs are stored under `job_out/` directory, with `log.err` storing the redirected `stderr` and `log.out` storing the redirected `stdout`.

To change the setting, please edit `submit.slurm` directly. 

Modify `--gres=gpu:${type of gpu}:${number of gpu}` to request other type or number of gpus 

Modify `--mem=${RAM size}` to request RAM 

Modify `FLAGS="${your parameters here}"` to add flags 

Modify `python experiments/experiment_training.py` if you want to run another file 

Run `$ squeue -p ${your partition} | grep ${your name}` to see the queue information and running time 

If you want an interactive session to see immediate outputs or debug, run `$ srun -n ${number of cpus you want} -p preempt -t 7-00:00:00 --mem=${RAM you want} --gres=gpu:${type}:{number} --pty bash`. Do not python any file on the default login terminal! 

I have added `module load anaconda/2021.11` in the `.bashrc`, so by default you are using this anaconda version. If you need another conda version, run `$ module load anaconda/${your version}` in your terminal. I didn't specify cuda. If you need a specific cuda, run `$ module load cuda/${your version}`.

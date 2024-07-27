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

currenly there are: "progress_sign_loss"
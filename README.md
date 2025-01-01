# Manipulation Tasks

We currently support Push-T and Robomimic tasks, with CLIP and DINOv2 as our vision encoders.

Curtsy to the original [diffusion policy paper](https://diffusion-policy.cs.columbia.edu/).

## Environment

```
sudo apt-get install ffmpeg # only for Linux system
conda env create -f environment.yml
conda activate robodiff
conda install -c pytorch3d pytorch3d
```

## Train

Download the datasets from [here](https://diffusion-policy.cs.columbia.edu/data/experiments/image/)

## Train

Change the dataset path in each of the config files that you want to run, and then run
```
python train_baseline.py (--config path_to_config_dir)
```

 which contains the whole training pipeline and saves all the model checkpoints. By default, evaluation will be implemented per 20 epochs. Parameters, loss, eval scores and videos will be uploaded to wandb. 

### Training Modes

There should be separate config files for each supported foundation model and weights for simplicity.

## Evaluate
To evaluate the models, please run

```
python eval_baseline.py (--config path_to_config_dir)
```

which prints out the performance score and saves all the output videos in the output folder. But by default, eval is called every 20 epochs during training.

### Config
Place config files in `./configs`.

Note: if models are evaluated separately from the training file, please make sure pass in the same config.

`num_epochs`: Specifies the number of epochs for training. Default is 200.

`num_diffusion_iters`: Specifies the number of diffusion iterations. Default is 100.

`num_tests`: Specifies the number of trials for evaluation. Default is 20.

`num_demos`: Specifies the number of demos in total per domain. Default is 500. Number of training demos is dependent on ratio (num_demos * ratio)

`num_warmup_steps`: Specifies the number of warmup steps for lr scheduler. Default is 500.

`pred_horizon`: Specifies the prediction horizon. Default is 16.

`obs_horizon`: Specifies the observation horizon. Default is 2.

`action_horizon`: Specifies the action horizon. Default is 8.

`eval_epoch`: Specifies the epoch interval for evaluation. Default is 10.

`lr`: Specifies the lr. Default is 1e-4.

`weight_decay`: Specifies the weight_decay. Default is 1e-6.

`batch_size`: Specifies the batch_size. Default is 64.

`resize_scale`: Specifies the resize_scale. Default is 96 (original image size).

`wandb`: Specifies NOT to use wandb. Default is TO USE.

`verbose`: Enables verbose output during training, including dimension printing and other detailed logs for debugging
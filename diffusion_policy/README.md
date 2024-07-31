# Adaptive Diffusion Policy


The repo includes codes to run diffusion policy on multiple datasets via building invariant and adaptive denoising networks.

Curtsy to the original [diffusion policy paper](https://diffusion-policy.cs.columbia.edu/).

## Environment

```
sudo apt-get install ffmpeg # only for Linux system
conda env create -f environment.yml
conda activate dp
```

## Train

Put multiple datasets into `dataset_path_dir` (remember the .zarr suffix), and within the dp environment, run 

```
python train_baseline.py (--config path_to_config_dir)
```

 which contains the whole training pipeline and saves all the model checkpoints. By default, evaluation will be implemented per 30 epochs. Parameters, loss, eval scores and videos will be uploaded to wandb. Place config files in `./configs`.

### Training Modes

`adapt`: Activates the adaptive learning mode. In this mode, the model will load pre-trained invariant components and focus on optimizing new adaptive layers or mechanisms. Place adapt datasets in `adapt_dataset_path_dir`.

`use_pretrained`: Load and freeze pretrained weights on vision encoder (ResNet18).

`use_mlp`: Insert a MLP between ResNet18 and Unet.

`use_pace`: Use the PACE as optimizer to train diffusion policy on all domains after warmup steps. During warmup we use default AdamW as optimizer.

## Evaluate
To evaluate the models, please run

```
python eval_baseline.py (--config path_to_config_dir)
```

which prints out the performance score and saves all the output videos in the output folder. 

### Config
Place config files in `./configs`.

Note: if models are evaluated separately from the training file, please make sure pass in the same config.

`num_epochs`: Specifies the number of epochs for training. Default is 200.

`num_diffusion_iters`: Specifies the number of diffusion iterations. Default is 100.

`num_tests`: Specifies the number of trials for evaluation. Default is 20.

`num_vis_demos`: Specifies the number of videos saved per case (e.g. evaluate model i on domain j) in output folder. Default is 3.

`num_train_demos`: Specifies the number of demos for training per domain. Default is 500.

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
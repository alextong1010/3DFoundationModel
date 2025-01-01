# Move all training code here, ideally something clean 
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights

from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import argparse
import wandb
import os
import yaml
import shutil

# other files
from utils.common_utils import *
from utils.models import *
from eval_unet import *
from utils.push_t import PushTImageDataset
from utils.robomimic import process_and_cache_robomimic_data, RobomimicImageDataset
from utils.load_vision_encoder import load_vision_encoder

def main():
    parser = argparse.ArgumentParser(description='Training/Eval script for the UNet part of diffusion policy.')
    parser.add_argument('--config', type=str, default='./configs/all_configs/push_t/push_t_CLIP_ViT-B-32_laion2b_s34b_b79k.yml', help='Path to the configuration YAML file.')
    args = parser.parse_args()

    # Parse config file path to determine dataset type
    config_path = args.config
    if '/push_t/' in config_path:
        dataset_type = 'push_t'
    elif any(x in config_path for x in ['/can/', '/lift/', '/square/', '/tool_hang/', '/transport/']):
        dataset_type = 'robomimic'
    else:
        raise ValueError(f"Config path {config_path} does not contain valid dataset type")

    print(f"Using config: {config_path}")
    
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    display_name = config['display_name']
    model_name = config['backbone'] if config['weights'] is None else "{}_{}" .format(config['backbone'], config['weights'])
    models_save_dir = 'outputs/{}_{}_{}' .format(dataset_type, config["vision_encoder"], model_name)

    num_epochs = config['num_epochs']
    num_diffusion_iters = config['num_diffusion_iters']
    num_demos = config['num_demos']
    pred_horizon = config['pred_horizon']
    obs_horizon = config['obs_horizon']
    action_horizon = config['action_horizon']
    eval_epoch = config['eval_epoch']
    lr = config['lr']
    weight_decay = config['weight_decay']
    batch_size = config['batch_size']
    verbose = config['verbose']
    if dataset_type == 'robomimic':
        shape_meta = config['shape_meta']
    else:
        domain_id = config['domain_id']

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if dataset_type == 'robomimic':
        cached_path = process_and_cache_robomimic_data(config['dataset_path'], shape_meta, config['abs_action'], rotation_rep='rotation_6d')
    else:
        cached_path = config['dataset_path']

    # Split dataset into training and testing
    dataset_path, eval_dataset_path = split_dataset(cached_path, config['ratio'])
    
    #|o|o|                             observations: 2
    #| |a|a|a|a|a|a|a|a|               actions executed: 8
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    num_demos = int(np.round(num_demos*config['ratio']))

    # Initialize WandB
    if display_name == "default":
        display_name = None
    if config["wandb"]:
        wandb.init(
            project="foundation_model_manipulation_eval",
            config=config,
            name=display_name
        )
    else:
        print("warning: wandb flag set to False")

    print("Training parameters:")
    print(f"num_epochs: {num_epochs}")
    print(f"num_diffusion_iters: {num_diffusion_iters}")
    print(f"num_demos: {num_demos}")
    print(f"pred_horizon: {pred_horizon}")
    print(f"obs_horizon: {obs_horizon}")
    print(f"action_horizon: {action_horizon}")
    print(f"eval_epoch: {eval_epoch}")
    print("training dataset: {}".format(dataset_path))

    print("\nFreeze foundation model as vision encoder, train the head (Unet) in Diffusion Policy!")

    # Create models_save_dir if it doesn't exist
    if not os.path.exists(models_save_dir):
        os.makedirs(models_save_dir)

    # Initialize networks and noise schedulers
    nets = nn.ModuleDict({})
    noise_schedulers = {}

    # Load vision encoder
    vision_encoder, transform, vision_feature_dim = load_vision_encoder(config["vision_encoder"], config["backbone"], config["weights"])

    # freeze vision encoder weights
    vision_encoder.eval()
    for param in vision_encoder.parameters():
        param.requires_grad = False

    nets['vision_encoder'] = vision_encoder

    if dataset_type == 'robomimic':
        lowdim_shape_sum = sum(
            sum(value['shape'])  # Add up all elements of the shape array
            for key, value in shape_meta['obs'].items()
            if isinstance(value, dict) and value.get('type') != 'rgb'  # Exclude entries with type: rgb
        )
        rgb_count = sum(1 for key, value in shape_meta['obs'].items() if isinstance(value, dict) and value.get('type') == 'rgb')
        action_dim = sum(shape_meta['action']['shape']) # 10

    else: # push_t
        lowdim_shape_sum = 2
        rgb_count = 1
        action_dim = 2
    
    # state is 9 dimensional in total for robomimic: robot0_eef_pos (3), robot0_eef_quat (4), robot0_gripper_qpos (2)
    lowdim_obs_dim = lowdim_shape_sum 
    # for robomimic, observation feature has (rgb_count (2) * vision_feature_dim + lowdim_obs_dim) dims in total per step
    obs_dim = (rgb_count * vision_feature_dim) + lowdim_obs_dim

    unet = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )
    nets['unet'] = unet 
    noise_schedulers["single"] = create_injected_noise(num_diffusion_iters)

    nets = nets.to(device)

    # Exponential Moving Average accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)

    # 2. Dataset & Dataloader
    full_path = os.path.abspath(dataset_path)
    # create dataset from file
    if dataset_type == 'robomimic':
        dataset = RobomimicImageDataset(
            shape_meta=shape_meta,
            cached_dataset_path=full_path,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            num_demos = num_demos,
            transform = transform
        )
    else: # push_t
        dataset = PushTImageDataset(
            dataset_path=full_path,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            id = domain_id,
            num_demos = num_demos,
            transform = transform
        )
    
    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    # TODO: add an option to visualize data in batch with verbose flag

    # 3. Optimizer
    print("Use default AdamW as optimizer.")
    # Standard ADAM optimizer
    # Note that EMA parameters are not optimized directly
    optimizer = torch.optim.AdamW(nets.parameters(), lr=lr, weight_decay=weight_decay)

    # 4. Learning Rate Scheduler
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=config["num_warmup_steps"],
        num_training_steps=len(dataloader) * 3000 # 3000 is the number used in the original diffusion policy paper, could change to num_epochs * len(dataloader)
    )

    # Training head (Unet)
    with tqdm(range(1, num_epochs+1), desc='Epoch', position=0, leave=True) as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            if config['wandb']:
                wandb.log({'epoch': epoch_idx})    
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', position=1, leave=False) as tepoch:
                for nbatch in tepoch:
                    if config["wandb"]:
                        wandb.log({'learning_rate:': lr_scheduler.get_last_lr()[0]})
                    
                    # Prepare images based on dataset type
                    if dataset_type == 'robomimic':
                        images = [
                            nbatch['agentview_image'][:, :obs_horizon].to(device), # shape: (B, obs_horizon, C, H, W)
                            nbatch['robot0_eye_in_hand_image'][:, :obs_horizon].to(device) # shape: (B, obs_horizon, C, H, W)
                        ]
                        agent_states = [
                            nbatch['robot0_eef_pos'][:, :obs_horizon].to(device), # shape: (B, obs_horizon, 3)
                            nbatch['robot0_eef_quat'][:, :obs_horizon].to(device), # shape: (B, obs_horizon, 4)
                            nbatch['robot0_gripper_qpos'][:, :obs_horizon].to(device) # shape: (B, obs_horizon, 2)
                        ]
                    else:  # push_t
                        images = [nbatch['image'][:,:obs_horizon].to(device)] # shape: (B, obs_horizon, C, H, W)
                        agent_states = [nbatch['agent_pos'][:,:obs_horizon].to(device)] # shape: (B, obs_horizon, 2)

                    # Stack and process images
                    nimage = torch.stack(images, dim=2)  # nimage shape: (B, obs_horizon, N=2 (robomimic) or N=1 (push_t), C, H, W)
                    nagent_pos = torch.cat(agent_states, dim=-1) # nagent_pos shape: (B, obs_horizon, 9 (robomimic) or 2 (push_t))
                    
                    # Flatten nimage using reshape
                    B, obs_horizon, N, C, H, W = nimage.shape
                    nimage_flat = nimage.reshape(-1, C, H, W)  # Shape: (B * obs_horizon * N, C, H, W)

                    # Get image features
                    if config["vision_encoder"] == 'clip':
                        image_features = nets["vision_encoder"].encode_image(nimage_flat)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                    else:  # dinov2
                        image_features = nets["vision_encoder"](nimage_flat)
                    # image_features shape: (B * obs_horizon * N, D)

                    # Reshape to include batch and image dimensions
                    # Concatenate features from multiple images (if robomimic)
                    # Concatenate along the feature dimension
                    D = image_features.shape[-1]  # Feature dimension
                    image_features = image_features.reshape(B, obs_horizon, N * D)  # Shape: (B, obs_horizon, N * D)

                    # Get actions and create observation features
                    naction = nbatch['action'].to(device)

                    # Concatenate image features and nagent_pos
                    obs_features = torch.cat([image_features, nagent_pos], dim=-1)  # Shape: (B, obs_horizon, feature_dim)
                    obs_cond = obs_features.flatten(start_dim=1)  # Shape: (B, obs_horizon * feature_dim)

                    # Sample noises and timesteps
                    noise = torch.randn(naction.shape, device=device)
                    timesteps = torch.randint(
                        0, noise_schedulers["single"].config.num_train_timesteps,
                        (B,), device=device).long()
                    
                    # Add noise to actions
                    noisy_actions = noise_schedulers["single"].add_noise(
                        naction, noise, timesteps)
                    
                    # Predict noise residual
                    noise_pred = nets["unet"](noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)
                    
                    # Optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Step lr scheduler every batch
                    lr_scheduler.step()

                    # Update EMA
                    ema.step(nets.parameters())

                    # Logging
                    loss_cpu = loss.item()
                    if config['wandb']:
                        wandb.log({'loss': loss_cpu, 'epoch': epoch_idx})
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))

            # save and eval upon request
            if (epoch_idx % eval_epoch == 0) or (epoch_idx in [1, num_epochs]):
                # remove previous checkpoint
                pre_checkpoint_dir = os.listdir(models_save_dir)
                if len(pre_checkpoint_dir) != 0:
                    pre_checkpoint_path = os.path.join(models_save_dir, pre_checkpoint_dir[0])
                    shutil.rmtree(pre_checkpoint_path)

                # create new checkpoint
                checkpoint_dir = '{}/checkpoint_epoch_{}'.format(models_save_dir, epoch_idx)
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                save(ema, nets, checkpoint_dir, pretrained_VE=True)
                scores = eval_unet(config, checkpoint_dir, dataset_type, eval_dataset_path)
                print(scores)
                scores["epoch"] = epoch_idx

                if config["wandb"]:
                    wandb.log(scores)

if __name__ == "__main__":
    main()
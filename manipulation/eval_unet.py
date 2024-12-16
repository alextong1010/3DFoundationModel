import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights

from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm
import os
import argparse

import yaml

# other files
from utils.common_utils import *
from utils.models import *
from utils.push_t import PushTImageDataset
from utils.robomimic import process_and_cache_robomimic_data, RobomimicImageDataset
from utils.load_vision_encoder import load_vision_encoder


def main():
    parser = argparse.ArgumentParser(description='Training/Eval script for setting various parameters.')
    parser.add_argument('--config', type=str, default='./configs/all_configs/push_t/push_t_CLIP_ViT-B-16_laion2b_s34b_b88k.yml', help='Path to the configuration YAML file.')    
    args = parser.parse_args()

    # Parse config file path to determine dataset type
    config_path = args.config
    if '/push_t/' in config_path:
        dataset_type = 'push_t'
    elif any(x in config_path for x in ['/can/', '/lift/', '/square/', '/tool_hang/', '/transport/']):
        dataset_type = 'robomimic'
    else:
        raise ValueError(f"Invalid config path: {config_path}")

    # Load config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    model_name = config['backbone'] if config['weights'] is None else "{}_{}" .format(config['backbone'], config['weights'])
    models_save_dir = 'outputs/{}_{}_{}' .format(dataset_type, config["vision_encoder"], model_name)
    
    # Eval
    eval_unet(config, models_save_dir, dataset_type)

def eval_unet(config, models_save_dir, dataset_type, eval_dataset_path=None):
    if eval_dataset_path is None:
        _, eval_dataset_path = split_dataset(config['dataset_path'], config['ratio'])
    
    num_diffusion_iters = config['num_diffusion_iters']
    num_demos = config['num_demos']
    pred_horizon = config['pred_horizon']
    obs_horizon = config['obs_horizon']
    action_horizon = config['action_horizon']
    batch_size = config['batch_size']
    if dataset_type == 'robomimic':
        shape_meta = config['shape_meta']
    else:
        domain_id = config['domain_id']

    num_demos = int(np.round(num_demos*config['ratio']))

    print("Evaluation parameters:")
    print(f"num_diffusion_iters: {num_diffusion_iters}")
    print(f"num_demos: {num_demos}")
    print(f"pred_horizon: {pred_horizon}")
    print(f"obs_horizon: {obs_horizon}")
    print(f"action_horizon: {action_horizon}")
    print("eval dataset: {}".format(eval_dataset_path))

    ##################### Instantiating Model and EMA #####################
    nets = nn.ModuleDict({})

    # Load vision encoder
    vision_encoder, transform, vision_feature_dim = load_vision_encoder(config["vision_encoder"], config["backbone"], config["weights"])

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

    # Define UNet
    unet = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )
    nets['unet'] = unet 
    nets = nets.to(device)

    # Define EMA
    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)
    
    ##################### LOADING Model and EMA #####################
    for model_name, model in nets.items():
        if model_name=="vision_encoder":
            continue
        model_path = os.path.join(models_save_dir, f"{model_name}.pth")
        model_state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(model_state_dict)

    ema_nets = nets
    ema_path = os.path.join(models_save_dir, f"ema_nets.pth")
    model_state_dict = torch.load(ema_path, weights_only=True)
    ema.load_state_dict(model_state_dict)

    print("All models have been loaded successfully.")

    ##################### Dataset & Dataloader #####################

    full_path = os.path.abspath(eval_dataset_path)
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
    
    # save training data statistics (min, max) for each dim
    stats = dataset.stats

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

    ##################### Start Inference #####################
    normalized_losses_1st = []
    normalized_losses = []
    noise_scheduler = create_injected_noise(num_diffusion_iters)

    if dataset_type == 'robomimic':
        print("\nEval Diffusion Policy on RoboMimic Dataset.")
    else: # push_t
        print("\nEval Diffusion Policy on PushT Domain #{}:".format(domain_id))

    with torch.no_grad():
        with tqdm(dataloader, desc='Batch', position=1, leave=False) as tepoch:
            for nbatch in tepoch:
                # Prepare images based on dataset type
                if dataset_type == 'robomimic':
                    # Stack the images (as before)
                    nimage = torch.stack([
                        nbatch['agentview_image'][:, :obs_horizon].to(device),             # Shape: (B, obs_horizon, C, H, W)
                        nbatch['robot0_eye_in_hand_image'][:, :obs_horizon].to(device)     # Shape: (B, obs_horizon, C, H, W)
                    ], dim=2)  # nimage shape: (B, obs_horizon, N=2, C, H, W)

                    # Obtain nagent_pos by stacking specified tensors
                    nagent_pos = torch.cat([
                        nbatch['robot0_eef_pos'][:, :obs_horizon].to(device),     # Shape: (B, obs_horizon, pos_dim (3))
                        nbatch['robot0_eef_quat'][:, :obs_horizon].to(device),    # Shape: (B, obs_horizon, quat_dim (4))
                        nbatch['robot0_gripper_qpos'][:, :obs_horizon].to(device) # Shape: (B, obs_horizon, gripper_dim (2))
                    ], dim=-1)  # nagent_pos shape: (B, obs_horizon, total_dim (9))
                    
                    B, obs_horizon, N, C, H, W = nimage.shape

                else:  # push_t
                    nimage = nbatch['image'][:,:obs_horizon].to(device) # shape: (B, obs_horizon, C, H, W)
                    nagent_pos = nbatch['agent_pos'][:,:obs_horizon].to(device) # shape: (B, obs_horizon, 2)
                    
                    B, obs_horizon, C, H, W = nimage.shape
                    N = 1

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

                # Concatenate image features and nagent_pos
                obs_features = torch.cat([image_features, nagent_pos], dim=-1)  # Shape: (B, obs_horizon, feature_dim)
                obs_cond = obs_features.flatten(start_dim=1)  # Shape: (B, obs_horizon * feature_dim)

                # initialize action from Guassian noise
                noisy_action = torch.randn((B, pred_horizon, action_dim), device=device)
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

                for k in noise_scheduler.timesteps:
                    # predict noise 
                    noise_pred = ema_nets["unet"](
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample

                # normalized action
                naction = naction.detach().to('cpu').numpy()    
                # (B, pred_horizon, action_dim)
                normalized_pred_actions = naction 

                # prediction
                # (B, action_dim)
                normalized_pred_1st_action = normalized_pred_actions[:,0,:]
                # (B, action_horizon, action_dim)
                normalized_pred_action = normalized_pred_actions[:,:action_horizon,:]

                # groundtruth
                # (B, action_dim)
                normalized_gt_1st_action = nbatch['action'][:,0,:].numpy()
                # (B, action_horizon, action_dim)
                normalized_gt_action = nbatch['action'][:,:action_horizon,:].numpy()
                
                # (B,)
                normalized_l2_norm_1st = [np.linalg.norm(normalized_pred_1st_action[i] - normalized_gt_1st_action[i]) for i in range(B)]
                normalized_losses_1st += normalized_l2_norm_1st
                
                for i in range(B):
                    # (action_horizon,)
                    normalized_l2_norm_all_steps = [np.linalg.norm(normalized_pred_action[i][j] - normalized_gt_action[i][j]) for j in range(action_horizon)]
                    normalized_l2_norm_avg = sum(normalized_l2_norm_all_steps)/action_horizon
                    normalized_losses.append(normalized_l2_norm_avg)


    # currently we consider mse_loss
    normalized_total_loss_1st = sum(normalized_losses_1st)
    normalized_mse_loss_1st = normalized_total_loss_1st/len(normalized_losses_1st)

    normalized_total_loss = sum(normalized_losses)
    normalized_mse_loss = normalized_total_loss/len(normalized_losses)

    loss_dict = { 'normalized_mse_loss_1st': normalized_mse_loss_1st,
                 'normalized_mse_loss': normalized_mse_loss,}
    
    print("Eval done!")
    return loss_dict


if __name__ == "__main__":
    main()
import numpy as np
import torch
import torch.nn as nn
import collections
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm
import os
import argparse
import json
import yaml

# CLIP
from transformers import CLIPVisionModel
# DinoV2
from transformers import Dinov2Model

# dp defined utils
from utils import *
# from pusht_env import *
from models import *

def main():
    parser = argparse.ArgumentParser(description='Training script for setting various parameters.')
    parser.add_argument('--config', type=str, default='./configs/baseline.yml', help='Path to the configuration YAML file.')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    models_save_dir = config['models_save_dir']

    eval_baseline(config, models_save_dir)

def eval_baseline(config, models_save_dir):
    _, dataset_path = split_dataset(config['dataset_path'], config['eval_mode'], config['ratio'])
    
    # Your training code here
    # For demonstration, we'll just print the values
    num_diffusion_iters = config['num_diffusion_iters']
    num_train_demos = config['num_train_demos']
    pred_horizon = config['pred_horizon']
    obs_horizon = config['obs_horizon']
    action_horizon = config['action_horizon']
    batch_size = config['batch_size']
    verbose = config['verbose']
    resize_scale = 224
    domain_id = config['domain_id']

    if config['eval_mode']==1:
        num_train_demos = num_train_demos-int(np.round(config['num_train_demos']*config['ratio']))


    print("Evaluation parameters:")
    print(f"num_diffusion_iters: {num_diffusion_iters}")
    print(f"num_train_demos: {num_train_demos}")
    print(f"pred_horizon: {pred_horizon}")
    print(f"obs_horizon: {obs_horizon}")
    print(f"action_horizon: {action_horizon}")

    
    # 1. Dataset & Dataloader
    full_path = os.path.abspath(dataset_path)


    # create dataset from file
    dataset = PushTImageDataset(
        dataset_path=full_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        id = 0,
        num_demos = num_train_demos,
        resize_scale = resize_scale,
        pretrained=config["use_pretrained"],
        vision_encoder = config["vision_encoder"]
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


    ##################### Instantiating Model and EMA #####################

    nets = nn.ModuleDict({})

    # TODO: add config argument to support different size of foundation model (small, large, big etc.)
    # add one dp trained on all domains
    if config["vision_encoder"] == "resnet":
        print("Use ResNet18 as vision encoder")
        vision_feature_dim = 512
        if config["use_pretrained"]:
            vision_encoder = get_resnet(weights='IMAGENET1K_V1')
        else:
            vision_encoder = get_resnet()
        vision_encoder = replace_bn_with_gn(vision_encoder)
    elif config["vision_encoder"] == "clip":
        print("Use pretrained CLIP-ViT as vision encoder")
        vision_feature_dim = 768
        vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    elif config["vision_encoder"] == "dinov2":
        print("Use pretrained DinoV2 as vision encoder")
        vision_feature_dim = 384
        vision_encoder = Dinov2Model.from_pretrained("facebook/dinov2-small")
    else:
        raise Exception("vision_encoder is not recognized!")

    nets['vision_encoder'] = vision_encoder

    # agent_pos is 2 dimensional
    lowdim_obs_dim = 2
    # observation feature has (vision_feature_dim + lowdim_obs_dim) dims in total per step
    obs_dim = vision_feature_dim + lowdim_obs_dim
    action_dim = 2

    invariant = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )
    nets['invariant'] = invariant 

    nets = nets.to(device)

    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)

    ##################### LOADING Model and EMA #####################
        
    for model_name, model in nets.items():
        if model_name=="vision_encoder":
            continue
        model_path = os.path.join(models_save_dir, f"{model_name}.pth")
        model_state_dict = torch.load(model_path)
        model.load_state_dict(model_state_dict)

    ema_nets = nets
    ema_path = os.path.join(models_save_dir, f"ema_nets.pth")
    model_state_dict = torch.load(ema_path)
    ema.load_state_dict(model_state_dict)
    ema.copy_to(ema_nets.parameters())

    print("All models have been loaded successfully.")

    ##################### Start Inference #####################
    
    # (num_demos)
    losses = [] 
    normalized_losses = []
    noise_scheduler = create_injected_noise(num_diffusion_iters)

    print("\nEval Diffusion Policy on Domain #{}:".format(domain_id))
    
    with torch.no_grad():
        with tqdm(dataloader, desc='Batch', position=1, leave=False) as tepoch:
            for nbatch in tepoch:
                nimage = nbatch['image'][:,:obs_horizon].to(device)
                nagent_pos = nbatch['agent_pos'][:,:obs_horizon].to(device)
                B = nagent_pos.shape[0]

                # encoder vision features
                if config["vision_encoder"]=='resnet':
                    image_features = nets["vision_encoder"](nimage.flatten(end_dim=1))
                elif config["vision_encoder"]=='clip':
                    outputs = nets["vision_encoder"](pixel_values=nimage.flatten(end_dim=1))
                    image_features = outputs.pooler_output
                elif config["vision_encoder"]=='dinov2':
                    outputs = nets["vision_encoder"](pixel_values=nimage.flatten(end_dim=1))
                    image_features = outputs.pooler_output

                image_features = image_features.reshape(*nimage.shape[:2],-1)
                # (B,obs_horizon,D)

                # concatenate vision feature and low-dim obs
                obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                obs_cond = obs_features.flatten(start_dim=1)
                # (B, obs_horizon * obs_dim)

                # initialize action from Guassian noise
                noisy_action = torch.randn((B, pred_horizon, action_dim), device=device)
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = ema_nets["invariant"](
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

                # unnormalize action
                naction = naction.detach().to('cpu').numpy()
                # (B, pred_horizon, action_dim)
                normalized_pred_actions = naction 
                pred_actions = unnormalize_data(naction, stats=stats['action'])

                # only take the first predicted action
                normalized_pred_action = normalized_pred_actions[:,0,:]
                pred_action = pred_actions[:,0,:]
                # (B, action_dim)

                normalized_gt_action = nbatch['action'][:,0,:].numpy()
                gt_action = unnormalize_data(normalized_gt_action, stats=stats['action'])
                # (B, action_dim)
                l2_norm = np.linalg.norm(pred_action-gt_action)
                normalized_l2_norm = np.linalg.norm(normalized_pred_action-normalized_gt_action)
                losses.append(l2_norm)
                normalized_losses.append(normalized_l2_norm)

    # currently we consider mse_loss
    total_loss = np.sum(losses)
    mse_loss = np.mean(losses)    
    normalized_total_loss = np.sum(normalized_losses)
    normalized_mse_loss = np.mean(normalized_losses)
    loss_dict = {'normalized_total_loss': normalized_total_loss, 
                   'normalized_mse_loss': normalized_mse_loss,
                   'total_loss': total_loss,
                   'mse_loss': mse_loss}
    
    print("Eval done!")
    return loss_dict

if __name__ == "__main__":
    main()

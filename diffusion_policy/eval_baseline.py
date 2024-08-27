import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm
import os
import argparse

import yaml

# OpenCLIP
import open_clip


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
    domain_id = config['domain_id']

    if config['eval_mode']==1:
        num_train_demos = num_train_demos-int(np.round(config['num_train_demos']*config['ratio']))


    print("Evaluation parameters:")
    print(f"num_diffusion_iters: {num_diffusion_iters}")
    print(f"num_train_demos: {num_train_demos}")
    print(f"pred_horizon: {pred_horizon}")
    print(f"obs_horizon: {obs_horizon}")
    print(f"action_horizon: {action_horizon}")
    print("testing dataset: {}".format(dataset_path))

    ##################### Instantiating Model and EMA #####################

    nets = nn.ModuleDict({})

    # TODO: add config argument to support different size of foundation model (small, large, big etc.)
    # add one dp trained on all domains
    if config["vision_encoder"] == "resnet":
        print("Use ResNet18 as vision encoder")
        
        if config["use_pretrained"]:
            vision_encoder = get_resnet(weights='IMAGENET1K_V1')
        else:
            vision_encoder = get_resnet()
        vision_encoder = replace_bn_with_gn(vision_encoder)
        vision_feature_dim = 512
        transform = ResNet18_Weights.IMAGENET1K_V1.transforms(antialias=True)
    elif config["vision_encoder"] == "clip":
        print("Use OpenCLIP as vision encoder")
        
        clip_feature_dim_dict = {'ViT-B-32': 512,
                                'ViT-B-16': 512,
                                'ViT-L-14': 768,
                                'ViT-H-14': 1024,
                                'ViT-g-14': 1024}
        backbone_arch = config["backbone"]

        # Get a dictionary of all available models and their pretrained weights
        open_clip_model_dict = open_clip.list_pretrained()
        open_clip_model_dict = [x for x in open_clip_model_dict if x[0]==backbone_arch]

        flag = False
        for model_pair in open_clip_model_dict:
            if model_pair[1]==config['weights']:
                flag = True
                break
        if not flag:
            raise Exception("{} is an unsupported pretrained CLIP weights!".format(config['weights']))
        
        vision_encoder, _, transform = open_clip.create_model_and_transforms(backbone_arch, pretrained=config['weights']) # This line already loads fine-tuned CLIP weights from local path
        # freeze vision encoder weights
        vision_encoder.eval()
        vision_feature_dim = clip_feature_dim_dict[backbone_arch]
    elif config["vision_encoder"] == "dinov2":
        print("Use pretrained DinoV2 as vision encoder")
        
        dinov2_feature_dim_dict = {'dinov2_vits14': 384,
                                    'dinov2_vitb14': 768,
                                    'dinov2_vitl14': 1024,
                                    'dinov2_vitg14': 1536}

        backbone_arch = config["backbone"]

        # List available models and weights from the dinov2 repository
        dinov2_model_list = torch.hub.list('facebookresearch/dinov2')

        flag = False
        for model_weight in dinov2_model_list:
            if model_weight==backbone_arch:
                flag = True
                break
        if not flag:
            raise Exception("{} is an unsupported pretrained dinov2 weights!".format(config['backbone']))

        vision_encoder = torch.hub.load('facebookresearch/dinov2', backbone_arch) #load the backbone
        # freeze vision encoder weights
        vision_encoder.eval()
        vision_feature_dim = dinov2_feature_dim_dict[backbone_arch]

        # check: https://github.com/facebookresearch/dinov2/tree/main?tab=readme-ov-file#pretrained-heads---image-classification
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
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
        model_state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(model_state_dict)

    ema_nets = nets
    ema_path = os.path.join(models_save_dir, f"ema_nets.pth")
    model_state_dict = torch.load(ema_path, weights_only=True)
    ema.load_state_dict(model_state_dict)
    ema.copy_to(ema_nets.parameters())

    print("All models have been loaded successfully.")


   ##################### Dataset & Dataloader #####################
    full_path = os.path.abspath(dataset_path)

    # create dataset from file
    dataset = PushTImageDataset(
        dataset_path=full_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        id = 0,
        num_demos = num_train_demos,
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
    losses_1st = [] 
    normalized_losses_1st = []
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
                    image_features = nets["vision_encoder"].encode_image(nimage.flatten(end_dim=1))
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                elif config["vision_encoder"]=='dinov2':
                    image_features = nets["vision_encoder"](nimage.flatten(end_dim=1))

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

                # prediction
                # (B, action_dim)
                normalized_pred_1st_action = normalized_pred_actions[:,0,:]
                # (B, action_horizon, action_dim)
                normalized_pred_action = normalized_pred_actions[:,:action_horizon,:]
                # (B, action_dim)
                pred_1st_action = pred_actions[:,0,:]
                # (B, action_horizon, action_dim)
                pred_action = pred_actions[:,:action_horizon,:]

                # groundtruth
                # (B, action_dim)
                normalized_gt_1st_action = nbatch['action'][:,0,:].numpy()
                # (B, action_horizon, action_dim)
                normalized_gt_action = nbatch['action'][:,:action_horizon,:].numpy()
                # (B, action_dim)
                gt_1st_action = unnormalize_data(normalized_gt_1st_action, stats=stats['action'])
                # (B, action_horizon, action_dim)
                gt_action = unnormalize_data(normalized_gt_action, stats=stats['action'])
                
                # (B,)
                l2_norm_1st = [np.linalg.norm(pred_1st_action[i] - gt_1st_action[i]) for i in range(B)]
                losses_1st += l2_norm_1st
                # (B,)
                normalized_l2_norm_1st = [np.linalg.norm(normalized_pred_1st_action[i] - normalized_gt_1st_action[i]) for i in range(B)]
                normalized_losses_1st += normalized_l2_norm_1st
                
                for i in range(B):
                    # (action_horizon,)
                    l2_norm_all_steps = [np.linalg.norm(pred_action[i][j] - gt_action[i][j]) for j in range(action_horizon)]
                    l2_norm_avg = sum(l2_norm_all_steps)/action_horizon
                    losses.append(l2_norm_avg)
                    # (action_horizon,)
                    normalized_l2_norm_all_steps = [np.linalg.norm(normalized_pred_action[i][j] - normalized_gt_action[i][j]) for j in range(action_horizon)]
                    normalized_l2_norm_avg = sum(normalized_l2_norm_all_steps)/action_horizon
                    normalized_losses.append(normalized_l2_norm_avg)


    # currently we consider mse_loss
    total_loss_1st = sum(losses_1st)
    mse_loss_1st = total_loss_1st/len(losses_1st)   
    normalized_total_loss_1st = sum(normalized_losses_1st)
    normalized_mse_loss_1st = normalized_total_loss_1st/len(normalized_losses_1st)

    total_loss = sum(losses)
    mse_loss = total_loss/len(losses)
    normalized_total_loss = sum(normalized_losses)
    normalized_mse_loss = normalized_total_loss/len(normalized_losses)

    loss_dict = {'normalized_total_loss_1st': normalized_total_loss_1st, 
                 'normalized_mse_loss_1st': normalized_mse_loss_1st,
                 'total_loss_1st': total_loss_1st,
                 'mse_loss_1st': mse_loss_1st,
                 'normalized_total_loss': normalized_total_loss, 
                 'normalized_mse_loss': normalized_mse_loss,
                 'total_loss': total_loss,
                 'mse_loss': mse_loss}
    
    print("Eval done!")
    return loss_dict

if __name__ == "__main__":
    main()

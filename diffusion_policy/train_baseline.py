import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import argparse
import wandb
import os
import yaml
import shutil

# CLIP
from transformers import CLIPVisionModel
# DinoV2
from transformers import Dinov2Model

# other files
from utils import *
from pusht_env import *
from models import *
from eval_baseline import eval_baseline

def main():
    parser = argparse.ArgumentParser(description='Training script for setting various parameters.')
    parser.add_argument('--config', type=str, default='./configs/baseline.yml', help='Path to the configuration YAML file.')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    config['dataset_path'], = split_dataset(config['dataset_path'], config['eval_mode'], config['ratio'])
        
    #|o|o|                             observations: 2
    #| |a|a|a|a|a|a|a|a|               actions executed: 8
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
    
    num_epochs = config['num_epochs']
    num_diffusion_iters = config['num_diffusion_iters']
    num_train_demos = config['num_train_demos']
    pred_horizon = config['pred_horizon']
    obs_horizon = config['obs_horizon']
    action_horizon = config['action_horizon']
    eval_epoch = config['eval_epoch']
    lr = config['lr']
    weight_decay = config['weight_decay']
    batch_size = config['batch_size']
    dataset_path = config['dataset_path']
    domain_id = config['domain_id']

    models_save_dir = config['models_save_dir']
    verbose = config['verbose']
    display_name = config['display_name']
    resize_scale = 224

    if display_name == "default":
        display_name = None
    if config["wandb"]:
        # wandb.login(key="c816a85f1488f7f1df913c6f7dae063d173d27b3") 
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
    print(f"num_train_demos: {num_train_demos}")
    print(f"pred_horizon: {pred_horizon}")
    print(f"obs_horizon: {obs_horizon}")
    print(f"action_horizon: {action_horizon}")
    print(f"eval_epoch: {eval_epoch}")

    print("\nFreeze foundation model as vision encoder, train the head (Unet) in Diffusion Policy!")
    
    if not os.path.exists(models_save_dir):
        os.makedirs(models_save_dir)

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
        pretrained = config["use_pretrained"],
        vision_encoder = config["vision_encoder"]
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
    if verbose:
        # visualize data in batch
        batch = next(iter(dataloader))
        print("batch['image'].shape: {}, {}, [{},{}]".format(batch['image'].shape, batch['image'].dtype, torch.min(batch['image']), torch.max(batch['image'])))
        print("batch['agent_pos'].shape: {}, {}, [{},{}]".format(batch['agent_pos'].shape, batch['agent_pos'].dtype, torch.min(batch['agent_pos']), torch.max(batch['agent_pos'])))
        print("batch['action'].shape: {}, {}, [{},{}]".format(batch['action'].shape, batch['action'].dtype, torch.min(batch['action']), torch.max(batch['action'])))
        print("batch['id']: {}, [{},{}]".format(batch['id'].shape, torch.min(batch['id']), torch.max(batch['id'])))


    # 2. Network Instantiation
    nets = nn.ModuleDict({})
    noise_schedulers = {}

    # TODO: add config argument to support different size of foundation model (small, large, big etc.)
    if config["vision_encoder"] == "resnet":
        print("Use ResNet18 as vision encoder")
        vision_feature_dim = 512
        if config["use_pretrained"]:
            vision_encoder = get_resnet(weights='IMAGENET1K_V1')
        else:
            vision_encoder = get_resnet()
        vision_encoder = replace_bn_with_gn(vision_encoder)
        # freeze vision encoder weights
        for param in nets["vision_encoder"].parameters():
            param.requires_grad = False

    elif config["vision_encoder"] == "clip":
        print("Use pretrained CLIP-ViT as vision encoder")
        vision_feature_dim = 768
        vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        # freeze vision encoder weights
        vision_encoder.eval()

    elif config["vision_encoder"] == "dinov2":
        print("Use pretrained DinoV2 as vision encoder")
        vision_feature_dim = 384
        vision_encoder = Dinov2Model.from_pretrained("facebook/dinov2-small")
        # freeze vision encoder weights
        vision_encoder.eval()
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
    noise_schedulers["single"] = create_injected_noise(num_diffusion_iters)        

    nets = nets.to(device)

    # print_model_parameter_sizes(nets)

    # Exponential Moving Average accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)

    # 3. Optimizer & Scheduler
    print("Use default AdamW as optimizer.")
    # Standard ADAM optimizer
    # Note that EMA parameters are not optimized
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=lr, weight_decay=weight_decay)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=config["num_warmup_steps"],
        num_training_steps=len(dataloader) * num_epochs
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
                    
                    # device transfer
                    # data normalized in dataset
                    nimage = nbatch['image'][:,:obs_horizon].to(device)
                    nagent_pos = nbatch['agent_pos'][:,:obs_horizon].to(device)
                    naction = nbatch['action'].to(device)
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

                    # sample noises to add to actions
                    noise= torch.randn(naction.shape, device=device)
                    
                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                            0, noise_schedulers["single"].config.num_train_timesteps,
                            (B,), device=device).long()
                    
                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_schedulers["single"].add_noise(
                        naction, noise, timesteps)
                    
                    # predict the noise residual
                    noise_pred = nets["invariant"](noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)
                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(nets.parameters())

                    # logging
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
                scores = eval_baseline(config, checkpoint_dir)

                if config["wandb"]:
                    wandb.log({"dp_on_domain_{}_avg_eval_score".format(domain_id): np.mean(scores), 'epoch': epoch_idx})
                    
                    for i in range(10):
                        threshold = 0.1*i
                        count = (np.array(scores)>threshold).sum()
                        wandb.log({"num_tests_threshold_{:.1f}".format(threshold): count, 'epoch': epoch_idx})

if __name__ == "__main__":
    main()
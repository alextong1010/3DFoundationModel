#@markdown ### **Imports**
# diffusion policy import
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from PIL import Image
import itertools
from transformers import CLIPImageProcessor, AutoImageProcessor

# env import
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_injected_noise(num_train_timesteps:int, beta_schedule='squaredcos_cap_v2'):
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule=beta_schedule,
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )
    return noise_scheduler

def split_batch_by_id(batch, unique_ids):
    split_batches = []

    for unique_id in unique_ids:
        indices = torch.where(batch['id'] == unique_id)[0]
        mini_batch = {
            'image': batch['image'][indices],
            'agent_pos': batch['agent_pos'][indices],
            'action': batch['action'][indices],
            'id': batch['id'][indices]
        }
        split_batches.append(mini_batch)

    return split_batches

def save(ema, nets, models_save_dir, pretrained_VE=False):
    if not os.path.exists(models_save_dir):
        os.makedirs(models_save_dir)
    torch.save(ema.state_dict(), os.path.join(models_save_dir, "ema_nets.pth"))
    for model_name, model in nets.items():
        if pretrained_VE and model_name=="vision_encoder":
            continue
        model_path = os.path.join(models_save_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"{model_name}.pth saved")

    print("All models have been saved successfully.")


#@markdown ### **Dataset**
#@markdown
#@markdown Defines `PushTImageDataset` and helper functions
#@markdown
#@markdown The dataset class
#@markdown - Load data ((image, agent_pos), action) from a zarr storage
#@markdown - Normalizes each dimension of agent_pos and action to [-1,1]
#@markdown - Returns
#@markdown  - All possible segments with length `pred_horizon`
#@markdown  - Pads the beginning and the end of each episode with repetition
#@markdown  - key `image`: shape (obs_hoirzon, 3, 96, 96)
#@markdown  - key `agent_pos`: shape (obs_hoirzon, 2)
#@markdown  - key `action`: shape (pred_horizon, 2)

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def split_dataset(dataset_path, eval_mode, ratio):
    assert 0 < ratio <= 1, "ratio must be greater than 0 and less than or equal to 1"
    dataset_root = zarr.open(dataset_path, 'r')
    num_max_demos = dataset_root['meta']['episode_ends'][:].shape[0]

    split = ['training', 'testing']
    paths = []
    for idx, s in enumerate(split):
        path = "{}_{}_mode_{}.zarr".format(dataset_path.replace('.zarr', ''), s, eval_mode)
        if os.path.exists(path):
            continue
        paths.append(path)
        dataset = zarr.open(path, mode='w')
        dataset.create_group('data')
        dataset.create_group('meta')

        if (eval_mode == 1): # use the first 90% of demos to train, last 10% to test
            num_train_demos = np.round(ratio * num_max_demos).astype(int)
            num_train_frames = dataset_root['meta']['episode_ends'][num_train_demos-1]
            data_slice = slice(num_train_frames) if idx else slice(num_train_frames, None)
            demo_slice = slice(num_train_demos) if idx else slice(num_train_demos, None)
            
            # create new zarr files
            for key in dataset_root['data']:
                dataset['data'][key] = dataset_root['data'][key][data_slice]
            for key in dataset_root['meta']:
                dataset['meta'][key] = dataset_root['meta'][key][demo_slice]

        elif (eval_mode == 2): # use the first 90% of steps in all demos to train, last 10% of steps in all demos to test
            episode_ends = np.array(dataset_root['meta']['episode_ends'])
            adj_episode_ends = np.copy(episode_ends)
            adj_episode_ends[-1] = 0
            orig_slice_idxs = [int(np.round((episode_ends[i]-adj_episode_ends[i-1])*ratio)) + adj_episode_ends[i-1] for i in range(num_max_demos)]
            if idx: # training split
                slices = [slice(adj_episode_ends[i-1], orig_slice_idxs[i]) for i in range(num_max_demos)]
                new_slice_idxs = [(orig_slice_idxs[i] - adj_episode_ends[i-1]) for i in range(len(episode_ends))]
            else: # testing split
                slices = [slice(orig_slice_idxs[i], episode_ends[i]) for i in range(num_max_demos)]
                new_slice_idxs = [(episode_ends[i] - orig_slice_idxs[i]) for i in range(len(episode_ends))]
            new_slice_idxs = np.array(list(itertools.accumulate(new_slice_idxs)))

            #create new zarr files
            for key in dataset_root['data']:
                dataset['data'][key] = np.concatenate([dataset_root['data'][key][s] for s in slices])
            for key in dataset_root['meta']:
                dataset['meta'][key] = new_slice_idxs
        else:
            raise ValueError("Invalid eval mode. Only eval mode 1 or 2 is allowed.")
    return paths[0], paths[1] # 0 is training, 1 is testing
        
# dataset
class PushTImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 id:int,
                 num_demos: int,
                 resize_scale: int, 
                 pretrained=False, 
                 vision_encoder='resnet',
                 eval_mode=1,
                 ratio=0.9):

        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')

        # limit number of demos
        num_max_demos = dataset_root['meta']['episode_ends'][:].shape[0]
        if num_demos < num_max_demos:
            num_max_demos = num_demos
        num_max_frames = dataset_root['meta']['episode_ends'][num_max_demos-1]

        # float32, [0,255], (N,96,96,3)
        # DO NOT /255 here because PIL Image class needs raw RGB values
        train_image_data = dataset_root['data']['img'][:num_max_frames]
        
        # (N, D)
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            'agent_pos': dataset_root['data']['state'][:num_max_frames,:2],
            'action': dataset_root['data']['action'][:num_max_frames]
        }
        episode_ends = dataset_root['meta']['episode_ends'][:num_max_demos]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        # images are already normalized
        normalized_train_data['image'] = train_image_data

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.dataset_path = dataset_path
        self.id = id
        self.resize_scale = resize_scale
        self.pretrained = pretrained
        self.vision_encoder = vision_encoder

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        images = nsample['image'][:self.obs_horizon,:]

        if self.vision_encoder == 'resnet':
            # "resize to 224 without normalization" reach the best avg score in baseline method
            if self.pretrained:
                transform = v2.Compose([
                    v2.ToImage(),
                    v2.ToDtype(torch.uint8, scale=True),
                    v2.Resize(self.resize_scale),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                transform = v2.Compose([
                    v2.ToImage(),
                    v2.ToDtype(torch.uint8, scale=True),
                    v2.Resize(self.resize_scale),
                    v2.ToDtype(torch.float32, scale=True),
                    # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        
            # PIL Image class will convert the range from [0, 255] to [0, 1] by default (tested)
            images = [transform(image) for image in images]
            
            # float32, (2,3,resize_scale,resize_scale)
            # with v2.Normalize: range[-1.7, 2.7], otherwise: range[0, 1] 
            images = torch.stack(images)
        elif self.vision_encoder == 'clip':
            processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
            images = processor(images=images, return_tensors="pt")["pixel_values"]
        elif self.vision_encoder == 'dinov2':
            image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
            images = image_processor(images=images, return_tensors="pt")["pixel_values"]
        else:
            raise Exception("vision_encoder is not recognized!")

        # discard unused observations
        nsample['image'] = images
        nsample['agent_pos'] = nsample['agent_pos'][:self.obs_horizon,:]
        nsample["id"] = self.id
        
        return nsample


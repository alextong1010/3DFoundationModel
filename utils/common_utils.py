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

def split_dataset(dataset_path, ratio):
    assert 0 < ratio <= 1, "ratio must be greater than 0 and less than or equal to 1"
    print("Zarr Dataset path: {}".format(dataset_path))
    print("Splitting dataset in training and testing with ratio: {}".format(ratio))
    dataset_root = zarr.open(dataset_path, 'r')
    num_max_demos = dataset_root['meta']['episode_ends'][:].shape[0]

    split = ['training', 'testing']
    paths = []
    for idx, s in enumerate(split):
        path = "{}_{}.zarr".format(dataset_path.split(".zarr")[0], s)
        paths.append(path)
        if os.path.exists(path):
            continue
        dataset = zarr.open(path, mode='w')
        dataset.create_group('data')
        dataset.create_group('meta')

        # use the first 90% of demos to train, last 10% to test
        num_train_demos = np.round(ratio * num_max_demos).astype(int)
        num_train_frames = dataset_root['meta']['episode_ends'][num_train_demos-1]
        data_slice = slice(num_train_frames) if not idx else slice(num_train_frames, None)
        demo_slice = slice(num_train_demos) if not idx else slice(num_train_demos, None)
        
        if not idx: # training
            new_idx = dataset_root['meta']['episode_ends'][demo_slice]
        else: # testing
            new_idx = dataset_root['meta']['episode_ends'][demo_slice] - num_train_frames

        # Copy and slice the 'data' group recursively
        copy_and_slice_group(dataset_root['data'], dataset['data'], data_slice)

        # Copy the 'meta' group
        for key in dataset_root['meta']:
            dataset['meta'][key] = new_idx
        
    print("Splitting complete!")
    print("Training path: {}".format(paths[0]))
    print("Testing path: {}".format(paths[1]))
    return paths[0], paths[1]  # 0 is training, 1 is testing
    
# Recursive function to concatenate slices for each group
def concatenate_slices_group(src_group, dest_group, slices):
    for key in src_group:
        item = src_group[key]
        if isinstance(item, zarr.Array):
            dest_group[key] = np.concatenate([item[s] for s in slices])
        elif isinstance(item, zarr.Group):
            new_group = dest_group.create_group(key)
            concatenate_slices_group(item, new_group, slices)

# Recursive function to copy and slice data
def copy_and_slice_group(src_group, dest_group, data_slice):
    for key in src_group:
        item = src_group[key]
        if isinstance(item, zarr.Array):
            dest_group[key] = item[data_slice]
        elif isinstance(item, zarr.Group):
            new_group = dest_group.create_group(key)
            copy_and_slice_group(item, new_group, data_slice)
            
import torch
import torch.utils.data
import numpy as np
from PIL import Image
import zarr
import itertools
import os
from utils.common_utils import normalize_data, get_data_stats, sample_sequence, create_sample_indices

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

class PushTImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 id:int,
                 num_demos: int,
                 transform=None):

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
        self.transform = transform

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

        # PIL Image class will convert the range from [0, 255] to [0, 1] by default (tested)
        images = [self.transform(Image.fromarray(image.astype(np.uint8), 'RGB')) for image in images]
        
        # float32, (2,3,img_h,img_w)
        images = torch.stack(images)

        # discard unused observations
        nsample['image'] = images
        nsample['agent_pos'] = nsample['agent_pos'][:self.obs_horizon,:]
        nsample["id"] = self.id
        
        return nsample

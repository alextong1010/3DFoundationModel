import torch
import torch.utils.data
import numpy as np
import h5py
import os
from PIL import Image
from filelock import FileLock
import zarr
from utils.rotation_transformer import RotationTransformer
from utils.common_utils import normalize_data, get_data_stats, sample_sequence, create_sample_indices
import itertools


class RobomimicImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 shape_meta: dict,
                 cached_dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 num_demos: int,
                 transform=None,
                 ):
        
        self.cached_dataset_path = cached_dataset_path
        self.transform = transform
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.shape_meta = shape_meta

        self._load_from_cache(num_demos)

        # Compute indices using create_sample_indices
        self.indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=self.pred_horizon,
            pad_before=self.obs_horizon - 1,
            pad_after=self.action_horizon - 1
        )

        # Compute stats and normalize data
        self._compute_stats_and_normalize()
        

    def _load_from_cache(self, num_demos):
        print(f"Loading dataset from cache at {self.cached_dataset_path}")
        lock_path = self.cached_dataset_path + '.lock'
        with FileLock(lock_path):
            root = zarr.open(self.cached_dataset_path, mode='r')
            #root = zarr.group(store)
            # Load episode_ends
            self.episode_ends = root['meta']['episode_ends'][:]
            num_max_demos = self.episode_ends.shape[0]
            if num_demos < num_max_demos:
                num_max_demos = num_demos
            num_max_frames = self.episode_ends[num_max_demos-1]

            # Load data
            data_group = root['data']
            # Load images
            img_group = data_group['img']
            self.images = {}
            for key in img_group:
                self.images[key] = img_group[key][:num_max_frames]
            
            # Load low-dimensional observations
            state_group = data_group['state']
            self.lowdim_data = {}
            for key in state_group:
                self.lowdim_data[key] = state_group[key][:num_max_frames]

            # Load actions
            self.actions = data_group['action'][:num_max_frames]

            self.episode_ends = self.episode_ends[:num_max_demos]

        print("Dataset loaded from cache.")

        # Parse shape_meta
        self.rgb_keys = []
        self.lowdim_keys = []
        for key, attr in self.shape_meta['obs'].items():
            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'rgb':
                self.rgb_keys.append(key)
            elif obs_type == 'low_dim':
                self.lowdim_keys.append(key)
            else:
                raise ValueError(f"Unknown observation type {obs_type} for key {key}")

    def _compute_stats_and_normalize(self):
        # Now compute stats and normalize data
        self.stats = {}
        self.normalized_data = {}
        # Normalize lowdim_data
        for key, data in self.lowdim_data.items():
            self.stats[key] = get_data_stats(data)
            self.normalized_data[key] = normalize_data(data, self.stats[key])
        # Normalize actions
        self.stats['action'] = get_data_stats(self.actions)
        self.normalized_data['action'] = normalize_data(self.actions, self.stats['action'])

        # DO NOT /255 here because PIL Image class needs raw RGB values
        for key, data in self.images.items():
            self.normalized_data[key] = data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]
        
        # Get the data using these indices with sample_sequence function
        nsample = sample_sequence(
            train_data=self.normalized_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        for key in self.rgb_keys:
            images = nsample[key][:self.obs_horizon]

            # Ensure images are in uint8 format for PIL
            images = images.astype(np.uint8)
            
            # Convert images to PIL format and apply transforms
            images = [self.transform(Image.fromarray(image, 'RGB')) for image in images]
            
            # Stack images into a tensor
            images = torch.stack(images)  # Shape: (obs_horizon, C, H, W)
            
            # Update nsample with processed images
            nsample[key] = images

        # Process low-dimensional observations
        for key in self.lowdim_keys:
            data = nsample[key][:self.obs_horizon]  # Shape: (obs_horizon, D)
            data = torch.from_numpy(data.astype(np.float32))
            nsample[key] = data

        # Process actions
        actions = nsample['action'][:self.action_horizon]  # Shape: (action_horizon, D)
        actions = torch.from_numpy(actions.astype(np.float32))
        nsample['action'] = actions

        return nsample
    
def process_and_cache_robomimic_data(dataset_path, shape_meta, abs_action=False, rotation_rep='rotation_6d'):
    print("Processing dataset...")
    cache_path = dataset_path + '.zarr.zip'
    if os.path.exists(cache_path) :
        print(f"Cache path exists: {cache_path}. Skipping...")
    else:
        print(f"Cache path does not exist: {cache_path}. Continue processing...")

        # cache_path = dataset_path + '.zarr'
        # Parse shape_meta
        rgb_keys = []
        lowdim_keys = []
        for key, attr in shape_meta['obs'].items():
            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'rgb':
                rgb_keys.append(key)
            elif obs_type == 'low_dim':
                lowdim_keys.append(key)
            else:
                raise ValueError(f"Unknown observation type {obs_type} for key {key}")

        # Read from the hdf5 dataset
        with h5py.File(dataset_path, 'r') as f:
            # Get list of demos
            demos = f['data']
            demo_names = sorted(demos.keys(), key=lambda x: int(x.split('_')[1]))

            # Compute episode ends
            episode_ends = []
            prev_end = 0
            # Collect data arrays
            images_list = {key: [] for key in rgb_keys}
            lowdim_data = {key: [] for key in lowdim_keys}
            actions_list = []
            for demo_name in demo_names:
                demo = demos[demo_name]
                # Get length of demo
                T = demo['actions'].shape[0]
                episode_end = prev_end + T
                episode_ends.append(episode_end)
                prev_end = episode_end

                # Get images
                for key in rgb_keys:
                    images = demo['obs'][key][:]  # (T, H, W, C)
                    images_list[key].append(images)
                # Get low-dimensional observations
                for key in lowdim_keys:
                    data = demo['obs'][key][:]  # (T, D)
                    lowdim_data[key].append(data)
                # Get actions
                actions = demo['actions'][:]  # (T, D)
                actions_list.append(actions)
            
        # Concatenate data
        images = {key: np.concatenate(images_list[key], axis=0) for key in rgb_keys}
        actions = np.concatenate(actions_list, axis=0)  # (N, D)
        for key in lowdim_data:
            lowdim_data[key] = np.concatenate(lowdim_data[key], axis=0)  # (N, D)
        
        # Process actions
        actions = _convert_actions(actions, abs_action, rotation_rep)

        episode_ends = np.array(episode_ends)

        print("Dataset processing complete")

        # Save to cache
        print(f"Saving dataset to cache at {cache_path}")
        lock_path = cache_path + '.lock'
        with FileLock(lock_path):
            # Use zarr to save data efficiently
            store = zarr.ZipStore(cache_path, mode='w')
            # store = zarr.DirectoryStore(cache_path, mode='w')
            root = zarr.group(store)

            # Create 'meta' group
            meta_group = root.create_group('meta')
            # Save episode_ends
            meta_group.create_dataset('episode_ends', data=episode_ends)

            # Create 'data' group
            data_group = root.create_group('data')

            # Save images under 'data/img'
            img_group = data_group.create_group('img')
            for key, imgs in images.items():
                img_group.create_dataset(key, data=imgs, chunks=(1,) + imgs.shape[1:], compressor='default')

            # Save low-dimensional observations under 'data/state'
            state_group = data_group.create_group('state')
            for key, data in lowdim_data.items():
                state_group.create_dataset(key, data=data, chunks=(1,) + data.shape[1:], compressor='default')

            # Save actions under 'data/action'
            data_group.create_dataset('action', data=actions, chunks=(1,) + actions.shape[1:], compressor='default')

            store.close()
        print("Dataset cached successfully.")

        # Ensure lock file is cleaned up
        if os.path.exists(lock_path):
            os.remove(lock_path)
            print("Lock file deleted after successful cache save.")

    return cache_path

def _convert_actions(raw_actions, abs_action=False, rotation_rep='rotation_6d'):
    # Process actions, converting rotations if abs_action is True
    """
    Process actions, converting rotations if abs_action is True.

    Parameters
    ----------
    raw_actions: np.ndarray
        Raw actions, shape (N, D) where N is the sequence length and D is the
        action dimensionality.
    abs_action: bool
        Whether to convert the actions to absolute coordinates.
    rotation_rep: str
        The representation of rotations to use. One of 'axis_angle', 'euler_angles','quaternion', 'rotation_6d', or 'matrix'.

    Returns
    -------
    actions: np.ndarray
        Processed actions, shape (N, D).
    """
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # Dual arm
            raw_actions = raw_actions.reshape(-1, 2, 7)
            is_dual_arm = True

        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]

        # Convert rotation using RotationTransformer
        rot_transformer = RotationTransformer(
            from_rep='axis_angle',
            to_rep=rotation_rep
        )
        rot_converted = rot_transformer.forward(rot)

        raw_actions = np.concatenate([
            pos, rot_converted, gripper
        ], axis=-1).astype(np.float32)

        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1, raw_actions.shape[-2] * raw_actions.shape[-1])
        actions = raw_actions
    return actions
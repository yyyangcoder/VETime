"""
Data loading utilities for VETime anomaly detection.

This module provides dataset classes and collate functions for loading and
preprocessing time series anomaly detection data. It supports:
- Loading preprocessed datasets from pickle files
- Converting time series to image representations on-the-fly
- Padding and batching sequences of variable lengths
- Random masking for self-supervised pretraining
"""
from typing import Tuple, List, Dict, Any, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset
import random

from dataset.pre_image import ts2image_1d


class AnomalyDataset(Dataset):
    """
    PyTorch Dataset for time series anomaly detection.

    This dataset class loads preprocessed time series data from pickle files
    and optionally generates image representations on-the-fly. It supports
    train/test split and filters out very short sequences.

    The dataset expects pickle files containing a list of dictionaries, where
    each dictionary represents a sample with keys:
        - 'time_series': numpy array of shape (L, C)
        - 'normal_time_series': numpy array for normal reference
        - 'labels': numpy array of anomaly labels (0=normal, 1=anomaly)
        - 'attribute': metadata dictionary

    Args:
        dataset_dir: Path to the pickle file containing the dataset.
        patch_size: Size of patches for image generation. Used to determine
                    target image width.
        gen_image: If True, generate image representations for all samples
                   during initialization. Default: True.
        split: Data split to use. 'train' uses all data, 'test' uses the
               last (1 - train_ratio) portion. Default: 'train'.
        train_ratio: Ratio of data to use for training when split='test'.
                     Only used when split='test'. Default: 0.95.
        seed: Random seed for shuffling indices. Default: 42.
        name: Optional name identifier for the dataset. Default: None.

    Attributes:
        data: List of sample dictionaries after filtering and splitting.
        image_type: Type of image representation ('RGB').
        image_h: Height of each channel tile in generated images.

    """

    def __init__(
        self,
        dataset_dir: str,
        patch_size: int,
        gen_image: bool = True,
        split: str = 'train',
        train_ratio: float = 0.95,
        seed: int = 42,
        name: Optional[str] = None
    ):
        """
        Initialize the AnomalyDataset.

        Args:
            dataset_dir: Path to the pickle file containing the dataset.
            patch_size: Size of patches for image generation.
            gen_image: If True, generate image representations during init.
            split: Data split ('train' or 'test').
            train_ratio: Ratio of data for training split.
            seed: Random seed for reproducibility.
            name: Optional dataset name identifier.
        """
        file_path = dataset_dir
        self.image_h = patch_size
        self.gen_image = gen_image
        self.patch_size = patch_size
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
        random.seed(seed)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        num_train = int(len(dataset) * train_ratio)
        if split == 'train':
            selected_indices = indices
        elif split == 'test':
            selected_indices = indices[num_train:]
        else:
            raise ValueError("split must be 'train' or 'test'")

        self.data = [dataset[i] for i in selected_indices]
        self.data = [x for x in self.data if len(x['time_series']) > 100]
        self.data.sort(key=lambda x: len(x['time_series']))

        self.image_type = 'RGB'
        self.name = name
        if self.gen_image:
            self.generate_image(self.data)

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def generate_image(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate image representations for time series samples.

        This function converts each time series in the data list to an image
        using ts2image_1d. The image width is determined by the sequence length
        and patch_size, rounded up to the nearest multiple of patch_size.

        Args:
            data: List of sample dictionaries. Each dictionary should contain
                  at least 'time_series' key. The image, period, and padding
                  value will be added in-place.

        Returns:
            List[Dict[str, Any]]: The same data list with added keys:
                - 'image': Generated image array of shape (3, C*h_size, width)
                - 'period': Detected period (integer)
                - 'padding_value': Padding values for the image

        Note:
            This function modifies the input data list in-place and also
            returns it for convenience.
        """
        for idx, data0 in enumerate(data):
            target_length = ((len(data0['time_series']) + self.patch_size - 1) // self.patch_size) * self.patch_size
            img, period, padding_value = ts2image_1d(data0['time_series'], target_length, self.patch_size)
            data[idx]['image'] = img
            data[idx]['period'] = period
            data[idx]['padding_value'] = padding_value
        return data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict, int, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple containing:
                - time_series: Time series data as float32 tensor (L, C)
                - normal_time_series: Normal reference time series (L, C)
                - image: Image representation as float32 tensor (3, H, W)
                - labels: Anomaly labels as long tensor (L,)
                - attribute: Metadata dictionary
                - period: Detected period (int)
                - padding_value: Padding values as float32 tensor (3, C, 1)
        """
        sample = self.data[idx]
        img_tensor = torch.tensor(sample['image'], dtype=torch.float32)
        time_series = torch.tensor(sample['time_series'], dtype=torch.float32)
        normal_time_series = torch.tensor(sample['normal_time_series'], dtype=torch.float32)
        labels = torch.tensor(sample['labels'], dtype=torch.long)
        attribute = sample['attribute']
        period = sample['period']
        padding_value = torch.tensor(sample['padding_value'], dtype=torch.float32)
        return time_series, normal_time_series, img_tensor, labels, attribute, period, padding_value


def collate_fn(
    batch: List[Tuple],
    patch_size: int
) -> Dict[str, Union[torch.Tensor, List, Tuple]]:
    """
    Collate function for batching anomaly detection samples.

    This function processes a batch of samples from AnomalyDataset and:
    1. Concatenates all time series and computes global mean/std for normalization
    2. Pads all sequences to the same length (multiple of patch_size)
    3. Generates attention masks for valid sequence positions
    4. Applies random masking for self-supervised learning
    5. Pads images to match the target width

    Args:
        batch: List of samples from AnomalyDataset.__getitem__. Each sample is
               a tuple of (time_series, normal_time_series, img_tensor, labels,
               attribute, period, padding_value).
        patch_size: Size of patches for masking and padding alignment.

    Returns:
        A dictionary containing:
            - 'time_series': Padded time series tensor (B, L_max, C)
            - 'normal_time_series': Padded normal reference tensor (B, L_max, C)
            - 'mask_time_series': Time series with random patches masked (B, L_max, C)
            - 'image': Padded image tensor (B, 3, H, W_max)
            - 'mask': Boolean mask indicating masked positions (B, L_max)
            - 'labels': Padded label tensor (B, L_max) with -1 for padding
            - 'attention_mask': Boolean mask for valid positions (B, L_max)
            - 'period': Tuple of periods for each sample in batch
            - 'padding_value': Tensor of padding values (B, 3, C, 1)

    Note:
        - Time series are normalized using batch-wide statistics
        - Labels are padded with -1 (ignored in loss computation)
        - Random masking applies mask_ratio=0.3 to valid sequence regions only
    """
    time_series_list, normal_time_series_list, img_tensor, labels_list, attribute_list, period, padding_value = zip(*batch)
    
    if time_series_list[0].ndim == 1:
        time_series_tensors = [ts.unsqueeze(-1) for ts in time_series_list]
        normal_time_series_tensors = [nts.unsqueeze(-1) for nts in normal_time_series_list]
    else:
        time_series_tensors = [ts for ts in time_series_list]
        normal_time_series_tensors = [nts for nts in normal_time_series_list]

    concatenated = torch.cat(time_series_tensors, dim=0)
    mean = concatenated.mean(dim=0, keepdim=True)
    std = concatenated.std(dim=0, keepdim=True) + 1e-4
    time_series_tensors = [(ts - mean) / std for ts in time_series_tensors]
    normal_time_series_tensors = [(nts - mean) / std for nts in normal_time_series_tensors]

    labels = [label for label in labels_list]
    lengths = [t.size(0) for t in labels]
    max_len = max(lengths)
    max_idx = lengths.index(max_len)
    target_length = ((max_len + patch_size - 1) // patch_size) * patch_size

    def padding_to_target_length(list0, value):
        original_tensor = list0[max_idx]
        pad_shape = [0, 0] * original_tensor.dim()
        pad_shape[-1] = target_length - max_len
        padded_tensor = torch.nn.functional.pad(original_tensor, pad=pad_shape, mode='constant', value=value)
        list0[max_idx] = padded_tensor
        return torch.nn.utils.rnn.pad_sequence(list0, batch_first=True, padding_value=value)

    padded_time_series = padding_to_target_length(time_series_tensors, 0.0)
    normal_time_series_tensors = padding_to_target_length(normal_time_series_tensors, 0.0)
    padded_labels = padding_to_target_length(labels, -1)

    image_inputs = image_right_padding(img_tensor, target_length, padding_value)
    sequence_lengths = [ts.size(0) for ts in time_series_tensors]
    B, max_seq_len, num_features = padded_time_series.shape
    attention_mask = torch.ones(B, max_seq_len, dtype=torch.bool)

    for i, length in enumerate(sequence_lengths):
        attention_mask[i, length:] = False

    mask_time_series, mask = create_random_mask(padded_time_series, attention_mask, patch_size)
    normal_time_series_tensors, mask = create_random_mask(normal_time_series_tensors, attention_mask, patch_size)

    return {
        'time_series': padded_time_series,
        'normal_time_series': normal_time_series_tensors,
        'mask_time_series': mask_time_series,
        'image': image_inputs,
        'mask': mask,
        'labels': padded_labels,
        'attention_mask': attention_mask,
        'period': period,
        'padding_value': padding_value,
    }


def image_right_padding(
    imgs: List[torch.Tensor],
    max_width: int,
    p_values: torch.Tensor
) -> torch.Tensor:
    """
    Pad images on the right side to match target width.

    This function extends images that are shorter than max_width by padding
    on the right side. The padding uses the provided padding values to maintain
    consistency with the time series padding strategy.

    Args:
        imgs: List of image tensors, each of shape (3, H, W_i) where W_i may
              vary across samples.
        max_width: Target width for all images. Images with W < max_width will
                   be padded; images with W >= max_width remain unchanged.
        p_values: Tensor of padding values with shape (B, 3, C, 1) or compatible.
                  Each sample's padding value is used to fill its padded region.

    Returns:
        torch.Tensor: Stacked tensor of padded images with shape (B, 3, H, max_width).
                      All images have the same width after processing.

    Note:
        - Padding is applied only on the right side (width dimension)
        - Padding values are transposed to match the image channel format
    """
    padded_images = []
    for i in range(len(imgs)):
        img = imgs[i]
        C, H_size, W = img.shape
        p_value = p_values[i]
        if max_width > W:
            right_padding = max_width - img.shape[2]
            padding = (0, right_padding, 0, 0)
            padded_img = F.pad(img.unsqueeze(0), padding, mode='constant', value=0).squeeze(0)
            padded_img[:, :, W:] = p_value.T[:, :, None]
        else:
            padded_img = img
        padded_images.append(padded_img)
    return torch.stack(padded_images)


def create_random_mask(
    time_series: torch.Tensor,
    attention_mask: torch.Tensor,
    patch_size: int = 14,
    mask_ratio: float = 0.3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create random mask for time series patches in self-supervised learning.

    This function generates a binary mask that randomly masks patches of the
    time series while respecting the attention mask (only valid sequence
    positions are masked). The masked positions are replaced with small
    Gaussian noise for reconstruction-based pretraining.

    The masking strategy:
    1. Divide the valid sequence into patches of size `patch_size`
    2. Randomly select `mask_ratio` fraction of patches to mask
    3. Apply mask only to valid positions (respecting attention_mask)
    4. Replace masked positions with Gaussian noise (std=0.1)

    Args:
        time_series: Input time series tensor of shape (B, L, C) where B is
                     batch size, L is sequence length, C is number of features.
        attention_mask: Boolean tensor of shape (B, L) indicating valid positions
                        (True = valid, False = padding).
        patch_size: Size of each patch for masking. The sequence is divided
                    into non-overlapping patches of this size. Default: 14.
        mask_ratio: Ratio of patches to mask within valid regions. Must be in
                    [0, 1]. Default: 0.3.

    Returns:
        A tuple containing:
            masked_time_series: Time series with masked positions replaced by
                                small Gaussian noise. Same shape as input (B, L, C).
            mask: Boolean tensor of shape (B, L) indicating masked positions
                  (True = masked, False = visible).
    """
    batch_size, seq_len, num_features = time_series.shape
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    valid_mask = attention_mask
    
    for i in range(batch_size):
        valid_length = valid_mask[i].sum().item()
        num_valid_patches = (valid_length + patch_size - 1) // patch_size
        num_masked = max(1, int(num_valid_patches * mask_ratio)) if mask_ratio > 0 else 0
        num_masked = min(num_masked, num_valid_patches)

        if num_masked > 0:
            masked_patches = torch.randperm(num_valid_patches)[:num_masked]
            for j in masked_patches:
                start_idx = j * patch_size
                end_idx = min((j + 1) * patch_size, valid_length)
                mask[i, start_idx:end_idx] = 1
    
    mask = mask & valid_mask
    masked_time_series = time_series.clone()
    mask_expanded = mask.unsqueeze(-1).expand(-1, -1, num_features)
    masked_time_series[mask_expanded] = torch.randn_like(masked_time_series[mask_expanded]) * 0.1

    return masked_time_series, mask

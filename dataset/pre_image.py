"""
Time series to image conversion utilities for VETime.

This module provides functions to convert 1D time series data into 2D image
representations for vision-based anomaly detection. The conversion leverages
periodicity detection and trend-residual decomposition to create meaningful
visual representations.

Example:
    >>> from dataset.pre_image import ts2image_Test
    >>> ts = np.random.randn(1000, 1)
    >>> img, period, pad_value = ts2image_Test(ts, patch_size=16, img_size=224)
"""
import numpy as np
import torch
import math
from typing import Tuple, Union, List, Optional
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema


def ts2image_Test(
    x: Union[np.ndarray, torch.Tensor],
    patch_size: int,
    T_sqrt: bool = False,
    img_size: int = 224,
    make_RGB: bool = True
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Convert 1D time series to image representation for testing/evaluation.

    This function transforms time series data into a 2D image by:
    1. Detecting periodicity in each channel using autocorrelation
    2. Computing a global period across all channels
    3. Converting to image using ts2image_1d、

    Args:
        x: Input time series data. Shape (L, C) for multivariate or (L,) for 
           univariate time series, where L is sequence length and C is number 
           of channels.
        patch_size: Size of each patch for dividing the time series. Used in 
                    period calculation and image construction.、
        T_sqrt: If True, use sqrt(T) for period height calculation where T is 
                the number of patches. If False, use detected global period.
                Default: False.
        make_RGB: If True, create RGB image using moving average decomposition
                  (original + trend + residual channels). If False, create 
                  grayscale-like 3-channel image. Default: True.

    Returns:
        img: Image array of shape (3, L, img_size), dtype float32.
             Contains the time series visualized as an image.
        period: Detected global period (integer), representing the dominant 
                periodicity in the time series.
        pad_value: Padding values of shape (3, C, 1) used for extending the 
                   time series to fit the image dimensions.
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]
    L, C = x.shape
    
    periods_per_channel = []
    for c in range(C):
        xc = x[:, c].copy()
        mean = xc.mean()
        std = xc.std() + 1e-8
        xc_norm = (xc - mean) / std
        period = find_period(xc_norm)
        periods_per_channel.append(period)
    
    global_period = int(np.max(periods_per_channel)) if periods_per_channel else 1
    global_period = max(1, global_period)
    
    lengths = x.shape[0]
    max_width = ((lengths + patch_size-1) // patch_size) * patch_size

    img, period, pad_values = ts2image_1d(x, max_width, patch_size, h_size=1, make_RGB=make_RGB)

    
    return img, period, pad_values


def ts2image_1d(
    x: Union[np.ndarray, torch.Tensor],
    max_width: int,
    patch_size: int,
    h_size: int = 1,
    make_RGB: bool = True
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Convert time series to RGB image representation with trend/residual decomposition.

    This is the core function that transforms 1D time series into a 2D image by:
    1. Normalizing each channel using z-score normalization
    2. Detecting global periodicity across all channels
    3. Applying moving average decomposition (for RGB mode)
    4. Tiling and stacking channels vertically
    5. Applying gamma correction for visual enhancement

    The output image encodes the time series as a heatmap where:
    - For RGB mode: R=original, G=residual, B=trend components
    - For non-RGB mode: All three channels contain the normalized original signal

    Args:
        x: Input time series data. Shape (L, C) for multivariate or (L,) for 
           univariate time series, where L is sequence length and C is number 
           of channels.
        max_width: Target width for the output image. The time series will be 
                   padded or truncated to this length.
        patch_size: Size of each patch for dividing the time series. Used in 
                    period detection and image construction.
        h_size: Height multiplier for each channel. Each channel will be 
                repeated h_size times vertically. Default: 1.
        make_RGB: If True, create RGB image using moving average decomposition
                  with three components (original, residual, trend). If False,
                  replicate the normalized signal across all three channels.
                  Default: True.

    Returns:
        final_image: Image array of shape (3, C * h_size, max_width), dtype 
                     uint8. Each channel of the time series is converted to a 
                     3-channel image tile and stacked vertically.
        period: Detected global period (integer), representing the maximum 
                periodicity across all channels.
        pad_values: Padding values of shape (3, C, 1), dtype uint8. Contains 
                    the mean values used for padding each channel, scaled to 
                    [0, 255].
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]
    L, C = x.shape
    
    gamma_L = np.linspace(0.5, 1.5, C).tolist()
    np.random.shuffle(gamma_L)
    
    periods_per_channel = []
    for c in range(C):
        xc = x[:, c].copy()
        mean = xc.mean()
        std = xc.std() + 1e-8
        xc_norm = (xc - mean) / std
        period = find_period(xc_norm)
        periods_per_channel.append(period)
    
    global_period = int(np.max(periods_per_channel)) if periods_per_channel else 1
    global_period = max(1, global_period)
    
    pad_value_l = []
    images_per_channel = []
    
    for c in range(C):
        xc = x[:, c].copy()
        
        mean = xc.mean(axis=0, keepdims=True)
        std = xc.std(axis=0, keepdims=True) + 1e-8
        xc_norm = ((xc - mean) / std).reshape(-1, 1)
        
        period = global_period
        
        if make_RGB:
            x_r, x_t = moving_average_decompose(xc_norm, period)
            xc_norm = (xc_norm - xc_norm.min()) / (xc_norm.max() - xc_norm.min() + 1e-5)
            x_r = (x_r - x_r.min()) / (x_r.max() - x_r.min() + 1e-5)
            x_t = (x_t - x_t.min()) / (x_t.max() - x_t.min() + 1e-5)
            img_rgb = np.stack([xc_norm[..., 0], x_r[..., 0], x_t[..., 0]], axis=-1)
        else:
            xc_vis = (xc_norm - xc_norm.min()) / (xc_norm.max() - xc_norm.min() + 1e-5)
            img_rgb = np.repeat(xc_vis[:, np.newaxis], 3, axis=-1)
        
        if img_rgb.shape[0] < max_width:
            img_rgb, pad_value = adaptive_pad_heatmap(img_rgb, max_width=max_width, period=period)
        else:
            img_rgb = img_rgb[:max_width]
            tail_window = max(5, period)
            pad_value = np.mean(img_rgb[-tail_window:], axis=0)
        
        img_tile = np.repeat(img_rgb[np.newaxis, :, :], h_size, axis=0)
        img_tile = np.transpose(img_tile, (2, 0, 1))
        img_tile = np.power(img_tile, gamma_L[c])
        images_per_channel.append(img_tile)
        pad_value_l.append(pad_value)
    
    final_image = np.concatenate(images_per_channel, axis=1)
    final_image = (final_image * 255).clip(0, 255).astype(np.uint8)
    pad_values = (np.stack(pad_value_l) * 255).clip(0, 255).astype(np.uint8)
    
    return final_image, period, pad_values


def adaptive_pad_heatmap(
    img_rgb: np.ndarray,
    max_width: int,
    period: int = 0,
    noise_ratio: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptively pad heatmap to target width while preserving periodic patterns.

    This function extends a time series heatmap to the target width by:
    1. If period is detected: Repeating the last periodic segment to maintain
       the natural pattern of the time series
    2. If no period: Using tail window averaging for constant padding

    This approach avoids introducing artificial patterns that could be 
    misinterpreted as anomalies during model inference.

    Args:
        img_rgb: Input heatmap array. Can be:
            - 2D array of shape (H, W) for single-channel
            - 3D array of shape (H, W, C) for multi-channel RGB
        max_width: Target height after padding. If current height H is already
                   >= max_width, no padding is applied.
        period: Period length detected from the time series. If period > 5,
                periodic padding is used; otherwise, constant padding is applied.
                Default: 0.
        noise_ratio: Ratio of noise to add for simulating normal fluctuations.
                     Currently reserved for future use. Default: 0.05.

    Returns:
        padded_img: Padded heatmap array of shape (max_width, W) for 2D input 
                    or (max_width, W, C) for 3D input. Same dtype as input.
        padding_value: Mean value of the padding content. Shape (W,) for 2D 
                       or (W, C) for 3D, representing the average value added 
                       during padding.
    """
    H = img_rgb.shape[0]
    pad_len = max_width - H
    
    if period > 5 and pad_len > 0:
        start_idx = max(0, H - period - pad_len)
        template = img_rgb[start_idx:H - pad_len]
        repeats = int(np.ceil(pad_len / period))
        extended = np.tile(template, (repeats, 1)) if img_rgb.ndim == 2 else np.tile(template, (repeats, 1, 1))
        padding_content = extended[:pad_len]
    else:
        tail_window = max(5, period)
        tail_mean = np.mean(img_rgb[-tail_window:], axis=0, keepdims=True)
        padding_content = np.tile(tail_mean, (pad_len, 1)) if img_rgb.ndim == 2 else np.tile(tail_mean, (pad_len, 1, 1))
    
    padded_img = np.concatenate([img_rgb, padding_content], axis=0)
    padding_value = np.mean(padding_content, axis=0)
    
    return padded_img.astype(img_rgb.dtype), padding_value


def find_period(
    data: Union[np.ndarray, torch.Tensor],
    top_k: int = 1,
    max_lag_ratio: float = 0.2
) -> int:
    """
    Estimate the period of time series data using autocorrelation function (ACF).

    This function detects periodicity by:
    1. Computing the autocorrelation function (ACF) of the centered time series
    2. Finding local maxima in the ACF values (excluding lag 0)
    3. Selecting the top-k peaks by ACF value strength
    4. Returning the lag corresponding to the k-th strongest peak

    The ACF is computed efficiently using FFT for long sequences.

    Args:
        data: Input time series data. Shape (L,) or (L, 1), where L is the 
              sequence length. If torch.Tensor, it will be converted to numpy.
        top_k: Which local maximum peak to select as the period. 
               top_k=1 returns the strongest peak, top_k=2 the second strongest,
               etc. Higher values may capture sub-harmonics. Default: 1.
        max_lag_ratio: Maximum lag as a ratio of sequence length for ACF 
                       calculation. Controls the maximum detectable period.
                       Default: 0.2 (i.e., max period = 20% of sequence length).

    Returns:
        estimated_period: Detected period as an integer. Returns 1 if:
            - No local maxima found in ACF
            - Strongest ACF peak < 0.5 (weak periodicity)
            - Sequence too short for reliable detection
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    data = data.squeeze()
    
    data = data[:min(20000, len(data))]
    max_lag = int(min(20000, len(data)) * max_lag_ratio)
    if max_lag < 1:
        return 1
    
    auto_corr = acf(data - data.mean(), nlags=max_lag, fft=True)
    acf_vals = auto_corr[1:]
    lags = np.arange(1, max_lag + 1)
    local_max_indices = argrelextrema(acf_vals, np.greater)[0]
    
    if len(local_max_indices) == 0:
        return 1
    
    candidate_lags = lags[local_max_indices]
    candidate_values = acf_vals[local_max_indices]
    sorted_idx = np.argsort(candidate_values)[::-1]
    candidate_lags = candidate_lags[sorted_idx]
    valid_lags = candidate_lags[(candidate_lags >= 1) & (candidate_lags <= max_lag)]
    candidate_values = candidate_values[sorted_idx]
    
    if len(valid_lags) == 0 or candidate_values[0] < 0.5:
        return 1
    
    top_k = min(top_k, len(valid_lags))
    estimated_period = int(valid_lags[top_k - 1])
    
    return estimated_period


def moving_average_decompose(
    X: np.ndarray,
    K: int = 25
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose time series into trend and residual components using moving average.

    This function performs classical time series decomposition:
    1. Apply symmetric padding to handle boundary effects
    2. Compute moving average with a sliding window of size K
    3. Extract trend as the smoothed component
    4. Compute residual as the difference between original and trend

    The decomposition follows: X = trend + residual

    Args:
        X: Input time series data. Shape (T, D) for multivariate or (T,) for 
           univariate time series, where T is sequence length and D is number 
           of channels/features.
        K: Window size for moving average. Larger values produce smoother trends
           but may oversmooth short-term patterns. If K=1, defaults to window 
           size 25. Window size is adjusted to be odd if even. Default: 25.

    Returns:
        trend: Trend component of same shape as input X. Represents the 
               long-term smoothed pattern extracted via moving average.
        residual: Residual component of same shape as input X. Represents 
                  the short-term fluctuations and noise after removing trend.
                  Computed as: residual = X - trend
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    T, D = X.shape
    
    if K == 1:
        kernel_size = 25
    else:
        kernel_size = K
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    pad_len = (kernel_size - 1) // 2
    
    trend = np.zeros_like(X)
    
    if pad_len > 0:
        X_padded = np.pad(X, ((pad_len, pad_len), (0, 0)), mode='reflect')
    else:
        X_padded = X.copy()
    
    for i in range(D):
        col = X_padded[:, i]
        window = np.ones(kernel_size) / kernel_size
        conv_result = np.convolve(col, window, mode='valid')
        trend[:, i] = conv_result
    
    residual = X - trend
    return trend, residual

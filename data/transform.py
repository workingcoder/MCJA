"""MCJA/data/transform.py
   It contains a collection of custom image transformation classes designed for augmenting and preprocessing images.
"""

import math
import numbers
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn


class WeightedGrayscale(nn.Module):
    """
    A module that applies a weighted grayscale transformation to an image, which converts an RGB image to grayscale
    by applying custom weights to each channel before averaging them. This approach allows for more flexibility than
    standard grayscale, potentially emphasizing certain features more than others depending on the chosen weights.

    Args:
    - weights (tuple of floats, optional): The weights to apply to the R, G, and B channels, respectively.
      If not specified, weights are randomly generated for each image.
    - p (float): The probability with which the weighted grayscale transformation is applied.
      A value of 1.0 means the transformation is always applied, whereas a value of 0 means it is never applied.

    Methods:
    - forward(img): Applies the weighted grayscale transformation to the given img with probability p.
      If the transformation is not applied, the original image is returned unchanged.
    """

    def __init__(self, weights=None, p=1.0):
        super().__init__()
        self.weights = weights
        self.p = p

    def forward(self, img):
        if self.p < torch.rand(1):
            return img

        if self.weights is not None:
            w1, w2, w3 = self.weights
        else:
            w1 = random.uniform(0, 1)
            w2 = random.uniform(0, 1)
            w3 = random.uniform(0, 1)
            s = w1 + w2 + w3
            w1, w2, w3 = w1 / s, w2 / s, w3 / s
        img_data = np.asarray(img)
        img_data = w1 * img_data[:, :, 0] + w2 * img_data[:, :, 1] + w3 * img_data[:, :, 2]
        img_data = np.expand_dims(img_data, axis=-1).repeat(3, axis=-1)

        return Image.fromarray(np.uint8(img_data))


class ChannelCutMix(nn.Module):
    """
    A module that implements the ChannelCutMix augmentation, a variant of the CutMix augmentation strategy that operates
     at the channel level. Unlike traditional CutMix, which combines patches from different images, ChannelCutMix
     selectively replaces a region in one channel of an image with the corresponding region from another channel of the
     same image. This process introduces diversity in the training data by blending features from different channels,
     potentially enhancing the robustness of models to variations in input data.

    Args:
    - p (float): The probability with which the ChannelCutMix augmentation is applied.
      A value of 1.0 means the augmentation is always applied, whereas a value of 0 means it is never applied.
    - scale (tuple of floats): The range of scales relative to the original area of the
      image that determines the size of the region to be replaced.
    - ratio (tuple of floats): The range of aspect ratios of the region to be replaced.
      This controls the shape of the region, allowing for both narrow and wide regions to be selected.

    Methods:
    - forward(img): Applies the ChannelCutMix augmentation to the given img with probability p.
      If the augmentation is not applied, the original image is returned unchanged.
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def forward(self, img):
        if self.p < torch.rand(1):
            return img

        img_h, img_w = img.size  # PIL Image Type
        area = img_h * img_w
        log_ratio = torch.log(torch.tensor(self.ratio))
        i, j, h, w = None, None, None, None
        for _ in range(10):
            cutmix_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
            h = int(round(math.sqrt(cutmix_area * aspect_ratio)))
            w = int(round(math.sqrt(cutmix_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue
            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            break

        img_data = np.asarray(img)
        bg_c, fg_c = random.sample(range(3), k=2)
        bg_img_data = img_data[:, :, bg_c]
        fg_img_data = img_data[:, :, fg_c]
        bg_img_data[i:i + h, j:j + w] = fg_img_data[i:i + h, j:j + w]
        img_data = np.expand_dims(bg_img_data, axis=-1).repeat(3, axis=-1)

        return Image.fromarray(np.uint8(img_data))


class SpectrumJitter(nn.Module):
    """
    A module for applying Spectrum Jitter augmentation to an image, which selectively alters the intensity of a randomly
    chosen color channel and blends it with the original image. This augmentation can introduce variations in color
    intensity and distribution across different channels, mimicking conditions of varying spectrum that a model might
    encounter in real-world scenarios. The purpose is to improve the model's robustness to changes in spectrum by
    exposing it to a wider range of color spectrum during training.

    Args:
    - factor (float or tuple of float): Specifies the range of factors to use for blending the altered channel back into
      the original image. If a single float is provided, it's interpreted as the maximum deviation from the default
      intensity of 1.0, creating a range [1-factor, 1+factor]. If a tuple is provided, it directly specifies the range.
      The factor influences how strongly the selected channel's intensity is altered.
    - p (float): The probability with which the Spectrum Jitter augmentation is applied to any given image.
      A value of 1.0 means the augmentation is always applied, while a value of 0 means it is never applied.

    Methods:
    - forward(img): Applies the Spectrum Jitter augmentation to the given img with probability p.
      If the augmentation is not applied, the original image is returned unchanged.
    """

    def __init__(self, factor=0.5, p=0.5):
        super().__init__()
        self.factor = self._check_input(factor, 'spectrum')
        self.p = p

    @torch.jit.unused  # Inspired by the implementation of color jitter in standard libraries
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        if value[0] == value[1] == center:
            value = None
        return value

    def forward(self, img):
        if self.p < torch.rand(1):
            return img

        selected_c = random.randint(0, 2)

        img_data = np.asarray(img)
        img_data = img_data[:, :, selected_c]
        img_data = np.expand_dims(img_data, axis=-1).repeat(3, axis=-1)
        degenerate = Image.fromarray(np.uint8(img_data))

        factor = float(torch.empty(1).uniform_(self.factor[0], self.factor[1]))
        return Image.blend(degenerate, img, factor)


class ChannelAugmentation(nn.Module):
    """
    A module that encapsulates various channel-based augmentation strategies, allowing for flexible and probabilistic
    application of different augmentations to an image. This module supports a range of augmentation modes, including
    selection of individual RGB channels, averaging of channels, mixing channels, and more advanced techniques such
    as weighted grayscale conversion, channel cutmix, and spectrum jitter. The specific augmentation to apply can be
    selected via the `mode` parameter, providing a versatile tool for enhancing the diversity of original data.

    Args:
    - mode (str, optional): Specifies the augmentation technique to apply. Options include:
        - None: No augmentation is applied.
        - 'r', 'g', 'b': Selects a specific RGB channel.
        - 'avg': Averages the RGB channels.
        - 'rg_avg', 'rb_avg', 'gb_avg': Averages specified pairs of RGB channels.
        - 'rand': Randomly selects a channel at each call.
        - 'wg': Applies weighted grayscale augmentation.
        - 'cc': Applies channel cutmix augmentation.
        - 'sj': Applies spectrum jitter augmentation.
      Each mode introduces different types of variations, from simple channel selection to more complex transformations.
    - p (float): The probability with which the selected augmentation is applied.
      A value of 1.0 means the augmentation is always applied, while a value of 0 means it is never applied.

    Methods:
    - forward(img): Applies the configured augmentation to the given img with probability p.
    If the augmentation is not applied (either because p < 1 or mode is None), the original image is returned unchanged.
    """

    def __init__(self, mode=None, p=1.0):
        super().__init__()
        assert mode in [None, 'r', 'g', 'b', 'avg', 'rg_avg', 'rb_avg', 'gb_avg', 'rand', 'wg', 'cc', 'sj']
        self.mode = mode
        self.p = p
        if mode in ['r', 'g', 'b', 'avg', 'rg_avg', 'rb_avg', 'gb_avg', 'rand']:
            self.ca = ChannelSelection(mode=mode, p=p)
        elif mode == 'wg':
            self.ca = WeightedGrayscale(p=p)
        elif mode == 'cc':
            self.ca = ChannelCutMix(p=p)
        elif mode == 'sj':
            self.ca = SpectrumJitter(factor=(0.00, 1.00), p=p)
        else:
            self.ca = NoTransform()

    def forward(self, img):
        return self.ca(img)


class ChannelSelection(nn.Module):
    """
    A module that selectively manipulates the color channels of an image according to a specified mode.
    This augmentation technique can emphasize or de-emphasize certain features in the image based on color,
    which might be beneficial for tasks sensitive to specific color channels. The module supports a variety
    of modes that target different channels or combinations thereof, and it applies these transformations
    with a specified probability, allowing for stochastic data augmentation.

    Args:
    - mode (str, optional): Specifies the channel manipulation mode.
      It can be one of 'r', 'g', 'b', 'avg', 'rg_avg', 'rb_avg', 'gb_avg', or 'rand'. The default is 'rand',
      which randomly selects one of the RGB channels each time the augmentation is applied.
    - p (float): The probability with which the channel selection or modification is applied.
      A value of 1.0 means the transformation is always applied, whereas a value of 0 means it is never applied.
    """

    def __init__(self, mode='rand', p=1.0):
        super().__init__()
        assert mode in ['r', 'g', 'b', 'avg', 'rg_avg', 'rb_avg', 'gb_avg', 'rand']
        self.mode = mode
        self.p = p

    def forward(self, img):
        if self.p < torch.rand(1):
            return img

        img_data = np.asarray(img)
        if 'avg' in self.mode:
            if self.mode == 'avg':
                pass
            elif self.mode == 'rg_avg':
                img_data = np.stack([img_data[:, :, 0], img_data[:, :, 1]])
            elif self.mode == 'rb_avg':
                img_data = np.stack([img_data[:, :, 0], img_data[:, :, 2]])
            elif self.mode == 'gb_avg':
                img_data = np.stack([img_data[:, :, 1], img_data[:, :, 2]])
            img_data = img_data.mean(axis=-1)
        else:
            if self.mode == 'r':
                selected_c = 0
            elif self.mode == 'g':
                selected_c = 1
            elif self.mode == 'b':
                selected_c = 2
            elif self.mode == 'rand':
                selected_c = random.randint(0, 2)
            img_data = img_data[:, :, selected_c]
        img_data = np.expand_dims(img_data, axis=-1).repeat(3, axis=-1)
        return Image.fromarray(np.uint8(img_data))


class NoTransform(nn.Module):
    """
    A module that acts as a placeholder for a transformation step, performing no operation on the input image.
    It is designed to seamlessly integrate into data processing pipelines or augmentation sequences where conditional
    application of transformations is required but, in some cases, no actual transformation should be applied.

    Methods:
    - forward(img): Returns the input image unchanged, serving as a pass-through operation in a transformation pipeline.
    """

    def __init__(self):
        super().__init__()

    def forward(self, img):
        return img

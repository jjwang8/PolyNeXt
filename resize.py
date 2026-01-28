"""
Taken from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/transforms.py
"""

import math
import numbers
import random
import warnings
from typing import List, Sequence, Tuple, Union

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
try:
    from torchvision.transforms.functional import InterpolationMode
    has_interpolation_mode = True
except ImportError:
    has_interpolation_mode = False
from PIL import Image
import numpy as np

__all__ = [
    "ToNumpy", "ToTensor", "str_to_interp_mode", "str_to_pil_interp", "interp_mode_to_str",
    "RandomResizedCropAndInterpolation", "CenterCropOrPad", "center_crop_or_pad", "crop_or_pad",
    "RandomCropOrPad", "RandomPad", "ResizeKeepRatio", "TrimBorder", "MaybeToTensor", "MaybePILToTensor"
]


class ToNumpy:

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img


class ToTensor:
    """ ToTensor with no rescaling of values"""
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, pil_img):
        return F.pil_to_tensor(pil_img).to(dtype=self.dtype)


class MaybeToTensor(transforms.ToTensor):
    """Convert a PIL Image or ndarray to tensor if it's not already one.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pic) -> torch.Tensor:
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return F.to_tensor(pic)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MaybePILToTensor:
    """Convert a PIL Image to a tensor of the same type - this does not scale values.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pic):
        """
        Note: A deep copy of the underlying array is performed.

        Args:
            pic (PIL Image): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return F.pil_to_tensor(pic)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# Pillow is deprecating the top-level resampling attributes (e.g., Image.BILINEAR) in
# favor of the Image.Resampling enum. The top-level resampling attributes will be
# removed in Pillow 10.
if hasattr(Image, "Resampling"):
    _pil_interpolation_to_str = {
        Image.Resampling.NEAREST: 'nearest',
        Image.Resampling.BILINEAR: 'bilinear',
        Image.Resampling.BICUBIC: 'bicubic',
        Image.Resampling.BOX: 'box',
        Image.Resampling.HAMMING: 'hamming',
        Image.Resampling.LANCZOS: 'lanczos',
    }
else:
    _pil_interpolation_to_str = {
        Image.NEAREST: 'nearest',
        Image.BILINEAR: 'bilinear',
        Image.BICUBIC: 'bicubic',
        Image.BOX: 'box',
        Image.HAMMING: 'hamming',
        Image.LANCZOS: 'lanczos',
    }

_str_to_pil_interpolation = {b: a for a, b in _pil_interpolation_to_str.items()}


if has_interpolation_mode:
    _torch_interpolation_to_str = {
        InterpolationMode.NEAREST: 'nearest',
        InterpolationMode.BILINEAR: 'bilinear',
        InterpolationMode.BICUBIC: 'bicubic',
        InterpolationMode.BOX: 'box',
        InterpolationMode.HAMMING: 'hamming',
        InterpolationMode.LANCZOS: 'lanczos',
    }
    _str_to_torch_interpolation = {b: a for a, b in _torch_interpolation_to_str.items()}
else:
    _pil_interpolation_to_torch = {}
    _torch_interpolation_to_str = {}


def str_to_pil_interp(mode_str):
    return _str_to_pil_interpolation[mode_str]


def str_to_interp_mode(mode_str):
    if has_interpolation_mode:
        return _str_to_torch_interpolation[mode_str]
    else:
        return _str_to_pil_interpolation[mode_str]


def interp_mode_to_str(mode):
    if has_interpolation_mode:
        return _torch_interpolation_to_str[mode]
    else:
        return _pil_interpolation_to_str[mode]


_RANDOM_INTERPOLATION = (str_to_interp_mode('bilinear'), str_to_interp_mode('bicubic'))


def _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class RandomResizedCropAndInterpolation:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: random
    """

    def __init__(
            self,
            size,
            scale=(0.08, 1.0),
            ratio=(3. / 4., 4. / 3.),
            interpolation='random',
    ):
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = str_to_interp_mode(interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        img_w, img_h = F.get_image_size(img)
        area = img_w * img_h

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            target_w = int(round(math.sqrt(target_area * aspect_ratio)))
            target_h = int(round(math.sqrt(target_area / aspect_ratio)))
            if target_w <= img_w and target_h <= img_h:
                i = random.randint(0, img_h - target_h)
                j = random.randint(0, img_w - target_w)
                return i, j, target_h, target_w

        # Fallback to central crop
        in_ratio = img_w / img_h
        if in_ratio < min(ratio):
            target_w = img_w
            target_h = int(round(target_w / min(ratio)))
        elif in_ratio > max(ratio):
            target_h = img_h
            target_w = int(round(target_h * max(ratio)))
        else:  # whole image
            target_w = img_w
            target_h = img_h
        i = (img_h - target_h) // 2
        j = (img_w - target_w) // 2
        return i, j, target_h, target_w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return F.resized_crop(img, i, j, h, w, self.size, interpolation)

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = ' '.join([interp_mode_to_str(x) for x in self.interpolation])
        else:
            interpolate_str = interp_mode_to_str(self.interpolation)
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string
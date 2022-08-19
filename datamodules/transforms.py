from typing import Tuple

import numpy as np
import torch
import albumentations as A
import cv2

from PIL import Image
from torchvision.transforms import ToTensor, Compose


class ToNumpy:
  def __call__(self, x: Image) -> np.ndarray:
    """
    Converts a PIL.Image to numpy array.
    
    Args:
        x (Image): a PIL.Image.
        
    Returns:
        np.ndarray: a numpy array.
        
    Author: Adrian
    """
    return np.asarray(x)


class ToGreyScale:
  def __call__(self, x: np.ndarray) -> np.ndarray:
    """
    Converts image to greyscale if an image has 3 channels, assuming RGB.
    
    Args:
        x (np.ndarray): an RGB image.
        
    Returns:
        np.ndarray: a greyscale image.
        
    Author: Adrian
    """
    if len(x.shape) > 2:
      x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    return x


class Normalize:
  def __call__(self, x: torch.Tensor) -> np.ndarray:
    """
    Normalizes an image.
    
    Args:
        x (torch.Tensor): an image to normalize.
        
    Returns:
        np.ndarray: a normalized image.
        
    Author: Adrian
    """
    return x.float() / 255.0


class Albument:
  def __init__(self, augment: A.Compose) -> None:
    """
    Creates albumentations augmentations object.
    
    Args:
        augment (A.Compose): a composition of transforms.
        
    Author: Adrian
    """
    self.augment = augment

  def __call__(self, img: np.ndarray) -> np.ndarray:
    """
    Perfoms albumenations augmentations.
    
    Args:
        img (np.ndarray): an image to transform.
        
    Returns:
        np.ndarray: a transformed image.
        
    Author: Adrian
    """
    return self.augment(image=img)['image']


def train_transforms(target_size: Tuple[int, int], normalize: bool) -> Compose:
    """
    Provides transforms to perform on train images.

    Args:
        target_size (Tuple[int, int]): an image target size for resize augmentation.
        normalize (bool): whether to apply normlization augmentation.
        
    Returns:
        Compose: a composition of augmentations.
        
    Author: Adrian
    """
    augs = A.Compose(
        [
        A.Resize(target_size[0], target_size[1]),
        A.Affine(
            scale=(1, 1.3),
            translate_percent={
                'x': (-0.1, 0.1),
                'y': (-0.1, 0.1)
            },
            rotate=(-20, 20),
            p=0.5
        )
        ]
    )
    albument = Albument(augs)
    transforms_list = [
        ToNumpy(),
        ToGreyScale(),
        albument,
        ToTensor()
    ]
    if normalize:
        transforms_list.append(Normalize())
    return Compose(transforms_list)


def test_val_transforms(target_size: Tuple[int, int], normalize: bool) -> Compose:
    """
    Provides transforms to perform on validation and test images.

    Args:
        target_size (Tuple[int, int]): an image target size for resize augmentation.
        normalize (bool): whether to apply normlization augmentation.

    Returns:
        Compose: a composition of augmentations.
    
    Author: Adrian
    """
    augs = A.Compose(
        [
            A.Resize(target_size[0], target_size[1])
        ]
    )
    albument = Albument(augs)
    transforms_list = [
        ToNumpy(),
        ToGreyScale(),
        albument,
        ToTensor()
    ]
    if normalize:
        transforms_list.append(Normalize())
    return Compose(transforms_list)

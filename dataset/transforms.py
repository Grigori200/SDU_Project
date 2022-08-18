from typing import Tuple

import numpy as np
import torch
import albumentations as A
import cv2

from PIL import Image
from torchvision.transforms import ToTensor, Compose


class ToNumpy:
    def __call__(self, x: Image) -> np.ndarray:
        return np.asarray(x)


class ToGreyScale:
    def __call__(self, x: np.ndarray) -> np.ndarray:
      if len(x.shape) > 2:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
      return x


class Normalize:
  def __call__(self, x: torch.Tensor) -> np.ndarray:
    return x.float() / 255.0


class Albument:
    def __init__(self, augment: A.Compose) -> None:
        self.augment = augment

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.augment(image=img)['image']


def train_transforms(target_size: Tuple[int, int], normalize: bool) -> Compose:
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
            shear=(-10, 10),
            p=0.5
        ),
        A.CLAHE(
            clip_limit=4.0,
            tile_grid_size=(8, 8),
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

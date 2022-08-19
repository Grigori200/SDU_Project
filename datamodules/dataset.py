from typing import Dict, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

import pandas as pd
from torchvision.transforms import Compose


class PneumoniaData(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 transforms: Compose,
                 image_path_name: str = 'filename',
                 label_name: str = 'labels'):
        """
        Provides access to images from a dataframe. Applies transforms to loaded images.
        
        Args:
            df (pd.DataFrame): Dataframe with columns: image_path_name, label_name.
            transforms (Compose): Transforms to apply on loaded images.
            image_path_name (str, optional): A name of a column containing paths to images. Defaults to 'filename'.
            label_name (str, optional): A name of a column containing labels of images. Defaults to 'labels'.
            
        Author: Apichaya
        """
        self.data = df
        self.transforms = transforms
        self.path_name = image_path_name
        self.label_name = label_name

    def __len__(self) -> int:
        """
        Provides the length of the dataset.
        
        Returns:
            int: the length of the dataset.
            
        Author: Apichaya
        """
        return self.data.__len__()

    def _process_image(self, image_path: str) -> torch.Tensor:
        """
        Loads an image and transforms it.
        
        Args:
            image_path (str): a path to the image.
            
        Returns:
            torch.Tensor: a transformed image.
            
        Author: Apichaya
        """
        img = Image.open(image_path)
        return self.transforms(img)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Provides a transformed image and its label.
        
        Args:
            idx (int): an index of an image.
            
        Returns:
            Dict[str, torch.Tensor]: a dictionary with input and target keys representing an image and a label respectively.
            
        Author: Apichaya
        """
        row = self.data.iloc[idx]
        path, label = row[self.path_name], row[self.label_name]
        image = self._process_image(path)
        return {'x': image, 'y': label, 'path': path}

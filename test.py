import os
from PIL import Image
from typing import List
import torch
import numpy as np
import torchvision
from torchvision.transforms import ToTensor, Resize, Compose
from datamodules.transforms import test_val_transforms
import tqdm
import argparse
import pandas as pd


def load_images(filenames: List[str], transform: Compose) -> torch.Tensor:
    """
    Image generator. Yields transformed and normalized images one by one.

    Args:
        :filenames: (List[str]) List of images paths
        :transform: (Compose): Composition of transforms for all images
    
    Yields:
        (torch.Tensor): transformed and normalized image
        
    Author: Adam
    """
    for filename in filenames:
        img = Image.open(filename)
        img = transform(img)
        yield img


def get_paths(dir_path: str) -> List[str]:
    """
    Get image paths from given directory path.

    Args:
        :dir_path: (str): Path to the directory with all the images
    
    Returns:
        (List[str]): a list of image paths from given directory path
        
    Author: Adam
    """
    filenames = []
    if dir_path[-1] != "/" and dir_path[-1] != "\\":
        if "/" in dir_path:
            dir_path += "/"
        else:
            dir_path += "\\"

    for filename in os.listdir(dir_path):
        filenames.append(dir_path + filename)

    return filenames 


def test(model, dir_path: str) -> np.ndarray:
    """
    Test a model on images from provided directory path.

    Args:
        :model: Model for inference
        :dir_path: (str): Path to the directory with all the images
    
    Returns:
        (np.ndarray): an array of class predictions for given images
        
    Author: Adam
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = get_paths(dir_path)
    transform = test_val_transforms((80, 80), normalize=True)
    model = model.to(device)
    model.eval()

    y_hats = []
    for x in tqdm.tqdm(load_images(paths, transform)):
        with torch.no_grad():
            x = x.unsqueeze(0).to(device)
            output = model(x)
        y_hats.append(output)
    y_hats = np.array(y_hats)
    return y_hats.flatten()


def parse_args() -> argparse.Namespace:
    """
    Parse args from input
    
    Returns:
        - argparse.Namespace: Namespace with given attributes
        
    Author: Adam
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    """
    Author: Adam
    """
    parser = parse_args()
    model = torch.load(parser.model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    y_hats = test(model, parser.data_path)
    y_hats = pd.DataFrame(y_hats, columns=["predictions"])
    y_hats.to_csv("predictions.csv")

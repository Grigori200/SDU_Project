import os
from PIL import Image
from typing import List
import torch
import numpy as np
import torchvision
from torchvision.transforms import ToTensor, Resize, Compose
from datamodules.transforms import test_val_transforms
import tqdm


hyperparams = {
    "resolution": 336,
    "model_path": ""
}


def load_images(filenames: List[str], transform: Compose) -> torch.Tensor:
    """
    Image generator. Yields transformed and normalized images one by one.

    Args:
        :filenames: (List[str]) List of images paths
        :transform: (Compose): Composition of transforms for all images
    
    Yields:
        (torch.Tensor): transformed and normalized image
    """
    for filename in filenames:
        img = Image.open(filename)
        img = transform(img)
        img = img.float() / 255.0
        yield img
        

def load_lightning_model(model, ckpt_path: str):
    """
    Load lightning model from given checkpoint path.

    Args:
        :model: a clean model instance
        :ckpt_path: (str): Path to the model checkpoint
    
    Returns:
        loaded model.
    """
    # TODO: Change to correct class
    classifier = Classifier(model=model)
    classifier.load_from_checkpoint(
        checkpoint_path=ckpt_path, model=model
    )
    return classifier.model


def get_paths(dir_path: str) -> List[str]:
    """
    Get image paths from given directory path.

    Args:
        :dir_path: (str): Path to the directory with all the images
    
    Returns:
        (List[str]): a list of image paths from given directory path
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
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = get_paths(dir_path)
    transform = test_val_transforms((hyperparams["resolution"], hyperparams["resolution"]), normalize=True)
    model = model.to(device)
    model.eval()

    y_hats = []
    for x in tqdm.tqdm(load_images(paths, transform)):
        with torch.no_grad():
            x = x.unsqueeze(0).to(device)
            output = model(x)
        y_hat = torch.argmax(output, dim=1).detach().cpu().numpy()
        y_hats.append(y_hat)
    y_hats = np.array(y_hats)
    return y_hats.flatten()


if __name__ == "__main__":
    # Won't work with default resnet, change to our model after it exists
    model = torchvision.models.resnet18(False)
    y_hats = test(model, "D:\\SDU\\project\\data\\")

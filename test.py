import os
from PIL import Image
import torch
import numpy as np
import torchvision
from torchvision.transforms import ToTensor, Resize, Compose


hyperparams = {
    "resolution": 224,
    "model_path": ""
}


def load_images(filenames, transform):
    for filename in filenames:
        img = Image.open(filename)
        img = transform(img)
        img = img.float() / 255.0
        yield img
        

def load_lightning_model(
    model, ckpt_path: str
):
    # TODO: Change to correct class
    classifier = Classifier(model=model)
    classifier.load_from_checkpoint(
        checkpoint_path=ckpt_path, model=model
    )
    return classifier.model


def get_filenames(dir_path):
    filenames = []
    if dir_path[-1] != "/" and dir_path[-1] != "\\":
        if "/" in dir_path:
            dir_path += "/"
        else:
            dir_path += "\\"

    for filename in os.listdir(dir_path):
        filenames.append(dir_path + filename)

    return filenames 


def test(dir_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filenames = get_filenames(dir_path)
    transform = Compose(
        Resize((hyperparams["resolution"], hyperparams["resolution"])),
        ToTensor()
    )

    model = load_lightning_model(Model(), hyperparams["model_path"])
    model = model.to(device)
    model.eval()

    y_hats = []
    for x in load_images(filenames, transform):
        with torch.no_grad():
            x = x.to(device)
            output = model(x)
            y_hat = torch.argmax(output, dim=1).detach().cpu().numpy()
            y_hats.append(y_hat)
    y_hats = np.array(y_hats)
    return y_hats

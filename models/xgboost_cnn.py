from typing import *

from torch import nn
from xgboost import XGBClassifier
import numpy as np
import tqdm


class XGBoostCNN(nn.Module):
    def __init__(
            self,
            width: int,
            height: int,
            in_channels: int = 1,
            num_classes: int = 2
    ) -> None:
        super(XGBoostCNN, self).__init__()
        self.model = nn.Sequential(
            XGBConvBlock(in_channels, 32),
            XGBConvBlock(32, 64),
            XGBConvBlock(64, 128),
            XGBConvBlock(128, 256),
            XGBConvBlock(256, 512),
            XGBConvBlock(512, 1024),
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        return self.model(x)


class XGBConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1
    ) -> None:
        super(XGBConvBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), 1, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.9),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, (3, 3), 1, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.9),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25))

    def forward(self, x):
        return self.model(x)


def train_xgboost(model, train_dataloader, test_dataloader, DEVICE):
    model = model.to(DEVICE)
    model.eval()
    # model.model = nn.Sequential(*[model.model[i] for i in range(7)])
    model.dp = nn.Flatten()
    model.fc = nn.AdaptiveAvgPool1d(1024)
    xgbmodel = XGBClassifier(objective='multi:softprob',
                                num_class=2)
    for x, y in tqdm.tqdm(train_dataloader):
        print('poczatek petli')
        x = x.to(DEVICE)
        y = y.numpy()
        print('srodek petli')
        x_preprocessed = model(x).detach().cpu().numpy()
        print('przed fitem petli')
        xgbmodel.fit(x_preprocessed, y)
        print('koniec petli')
        
    y_trues = []
    y_preds = []
    for x, y_true in tqdm.tqdm(test_dataloader):
        x = x.to(DEVICE)
        x_preprocessed = model(x).detach().cpu().numpy()
        y_true = y_true.numpy()
        y_pred = xgbmodel.predict(x_preprocessed)
        y_preds.extend(y_pred)
        y_trues.extend(y_true)
    
    y_preds = np.array(y_preds)
    y_trues = np.array(y_trues)
    print(f'Test accuracy: {(y_preds == y_trues).sum() / len(y_preds)}')

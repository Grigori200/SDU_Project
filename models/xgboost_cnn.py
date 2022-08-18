from typing import *

from torch import nn
from xgboost import XGBClassifier
import xgboost as xgb


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
            nn.Flatten(),
            nn.Linear(256 * width * height // 2 // 2 // 2 // 2, 256),
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


def train_xgboost(model, x_train, y_train, x_val, y_val):
    model.model = nn.Sequential(*[model.model[i] for i in range(7)])
    x_preprocessed = model(x_train)
    xgbmodel = XGBClassifier(objective='multi:softprob',
                             num_class=2)
    xgb.train()
    xgbmodel.fit(x_preprocessed, y_train)
    print(xgbmodel.score(x_val, y_val))

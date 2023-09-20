import pytorch_lightning as pl
from torch.nn import Module
from torch import Tensor
from ren_util import Errmsg

class CtrlNet(pl.LightningModule,Errmsg):
    def __init__(self):
        pl.LightningModule.__init__(self)
        Errmsg.__init__(self)
        

import torch
import torch.nn as nn
import torch.nn.functional as F
class AlexNet(nn.Module):
    """
    BCHW --> num_classes
    """
    def __init__(self, B, C, H, W, num_classes=1000):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(C, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        
        # Compute the input shape of the classifier
        self.classifier_input_shape = self._compute_classifier_input_shape(B, C, H, W)
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.classifier_input_shape, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _compute_classifier_input_shape(self, B, C, H, W):
        # Create a dummy tensor to compute the output shape of the features module
        dummy_input = torch.zeros((B, C, H, W))
        features_output = self.features(dummy_input)
        _, C, H, W = features_output.shape
        return C * H * W

class Alexnet_Branch(Alexnet):
    def __init__(self,branches_d):
        
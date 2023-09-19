from pathlib import Path

p = Path("/mnt/d/onedrive/OneDrive_cityu/OneDrive - City University of Hong Kong - Student/workspace/Dataset/DenoiseToy/data/test/clean/29026.jpg")
assert p.exists(),p



from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
import torch

print("PSNR(dB): the higher, the cleaner\n 30dB+ acceptable, 40dB+ good")

def load(p,show=True):
    img = Image.open(p)
    imga = np.array(img)
    if show:
        plt.imshow(img)
        plt.show()
    return imga

clean = load(p,show=False)/255


import torch
import torch.nn as nn
import torch.nn.functional as F
class AlexNet(nn.Module):
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
        x = self.features(dummy_input)
        _, C, H, W = x.shape
        return C * H * W



device = torch.device("cuda:0")
f = torch.tensor(clean).permute(2,0,1).unsqueeze(0).float().to(device)
B,C,H,W = f.shape
alexnet = AlexNet(B,C,H,W).to(device)
pred = alexnet(f)

print(pred.shape,pred.min(),pred.max())
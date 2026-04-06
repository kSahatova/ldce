import numpy as np

from typing import List

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50


class SimpleCNNtorch(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        img_size: int = 28,
        num_classes: int = 2,
        in_conv_channels: List[int] = [1, 8, 16, 32, 64],
        out_conv_channels: List[int] = [8, 16, 32, 64, 128],
        conv_kernels: List[int] = [7, 7, 5, 5, 3],
        include_pooling: bool = True,
        softmax_flag=False,
    ):
        super(SimpleCNNtorch, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.input_conv_channels = (
            in_conv_channels
            if in_conv_channels[0] == input_channels
            else [input_channels] + in_conv_channels[1:]
        )
        self.output_conv_channels = out_conv_channels
        self.conv_kernels = conv_kernels
        self.has_pooling = include_pooling
        self.activate_softmax = softmax_flag

        self.features = nn.Sequential
        self.classifier = nn.Sequential

        self.build_conv_layers()
        self.build_classifier()

    def build_conv_block(self, in_ch, out_ch, kernel, stride, pad):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout2d(p=0.1),
        )

    def build_conv_layers(self):
        conv_blocks = [
            self.build_conv_block(
                self.input_conv_channels[i],
                self.output_conv_channels[i],
                kernel=self.conv_kernels[i],
                stride=1,
                pad=1,
            )
            for i in range(len(self.input_conv_channels))
        ]
        self.main = self.features(*conv_blocks, nn.Flatten())

    def calculate_conv_output(self, width, kernel, pad=1, stride=1):
        output_shape = (width - kernel + 2 * pad) / stride + 1
        if self.has_pooling:
            return np.floor(output_shape / 2)
        return output_shape

    def build_classifier(self):
        input_shape_temp = self.img_size
        for kernel in self.conv_kernels:
            output_shape = self.calculate_conv_output(input_shape_temp, kernel)
            input_shape_temp = int(output_shape)

        self.classifier = self.classifier(
            nn.Linear(self.output_conv_channels[-1] * input_shape_temp**2, 128),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, x):
        x = self.main(x)
        logits = self.classifier(x)
        if self.activate_softmax:
            logits = F.log_softmax(logits, dim=1)
        return logits

    def get_params_num(self):
        features_params = self.main.parameters()
        clf_params = self.classifier.parameters()
        total_params = sum(p.numel() for p in features_params if p.requires_grad) + sum(
            p.numel() for p in clf_params if p.requires_grad
        )
        return total_params


def build_resnet50(num_classes, pretrained=True, freeze_backbone=True, unfreeze_last_n=0):
    """Builds an object of the ResNet50 class with the pretrained weights"""
    if not pretrained:
        weights = None
        freeze_backbone = False
    else: 
        weights = "DEFAULT"
    cnn = resnet50(weights=weights)

    if freeze_backbone:
        for param in cnn.parameters():
            param.requires_grad = False
        
        # Optionally unfreeze last N residual blocks
        if unfreeze_last_n > 0:
            layers = [cnn.layer1, cnn.layer2, cnn.layer3, cnn.layer4]
            for layer in layers[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
    cnn.fc = torch.nn.Sequential(torch.nn.Linear(cnn.fc.in_features, 128),
                                 torch.nn.ReLU(True),
                                 torch.nn.Dropout(p=0.1),
                                 torch.nn.Linear(128, num_classes),
)
    return cnn


class CNNtorch(nn.Module):
    def __init__(self, input_channels, num_classes, softmax_flag=False):
        super(CNNtorch, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.activate_softmax = softmax_flag

        self.features = nn.Sequential
        self.classifier = nn.Sequential

        self.build_conv_layers()
        self.build_classifier()

    def build_conv_layers(self):
        # input is Z, going into a convolution
        self.main = self.features(
            nn.Conv2d(self.input_channels, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
            nn.Flatten(),
        )

    def build_classifier(self):
        self.classifier = self.classifier(
            nn.Linear(32 * 3 * 3, 128), nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        x = self.main(x)
        # x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)  # GAP Layer
        logits = self.classifier(x)
        # Added line for C3LT
        if self.activate_softmax:
            logits = F.log_softmax(logits, dim=1)
        return logits

    def get_params_num(self):
        features_params = self.main.parameters()
        clf_params = self.classifier.parameters()
        total_params = sum(p.numel() for p in features_params if p.requires_grad) + sum(
            p.numel() for p in clf_params if p.requires_grad
        )
        return total_params
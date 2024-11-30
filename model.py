import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict

def create_model(architecture='vgg16', hidden_units=2048, dropout_rate=0.3):
    if architecture == 'vgg16':
        base_model = models.vgg16(pretrained=True)
    elif architecture == 'densenet121':
        base_model = models.densenet121(pretrained=True)
    else:
        raise ValueError("Model architecture not supported!")

    # Freeze feature extractor parameters
    for param in base_model.parameters():
        param.requires_grad = False

    input_features = base_model.classifier[0].in_features
    base_model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=dropout_rate)),
        ('fc2', nn.Linear(hidden_units, 256)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=dropout_rate)),
        ('fc3', nn.Linear(256, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    return base_model
el
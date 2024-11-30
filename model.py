import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
from collections import OrderedDict

def Net(backbone='vgg16', hidden_units=2048, dropout=0.3):

  if backbone == 'vgg16':
    model = models.vgg16(pretrained=True)
  elif backbone == 'densenet121':
    model = models.densenet121(pretrained=True)
  else:
    raise ValueError("Unsupported backbone model")

  for param in model.parameters():
    param.requires_grad = False
  model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(model.classifier[0].in_features, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('d_out1', nn.Dropout(p=dropout)),
                          ('fc2', nn.Linear(hidden_units, 256)),
                          ('d_out2', nn.Dropout(p=dropout)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(256, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                        ]))
  return model
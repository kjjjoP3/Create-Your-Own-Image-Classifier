# Image Classifier Project

## Overview

This project allows you to create and train your own image classifier using PyTorch. It includes two main scripts: `train.py` for training a custom image classifier and `predict.py` for making predictions using a pre-trained model. The project also provides a custom model architecture defined in `model.py`.

## Requirements

To run this project, you need:

- Python 3.x
- PyTorch
- torchvision
- NumPy
- PIL (Pillow)
- JSON (for working with JSON files)
- argparse (for command-line argument parsing)

You can install the required dependencies using `pip`. For example:

```bash
pip install torch torchvision numpy pillow jsonlib argparse
```

## `train.py`

### Usage

Train a custom image classifier with the following arguments:

- `data_dir`: Path to the data directory containing training, validation, and test data.
- `--arch`: Model architecture for training (default: 'vgg16' supported, or 'densenet121').
- `--learning_rate`: Learning rate for training (default: 0.001).
- `--hidden_units`: Number of hidden units in the classifier (default: 2048).
- `--dropout`: Dropout rate in the classifier (default: 0.3).
- `--epochs`: Number of training epochs (default: 5).
- `--gpu`: Use GPU for training (optional).

Example usage:

```bash
python train.py /path/to/data --arch vgg16 --learning_rate 0.001 --hidden_units 2048 --dropout 0.3 --epochs 10 --gpu
```

## `predict.py`

### Usage

Make predictions using a pre-trained model with the following arguments:

- `--checkpoint`: Path to the model checkpoint file (default: 'checkpoint_model.pth').
- `--image_path`: Path to the image you want to predict.
- `--json_path`: Path to the JSON file containing class names.
- `--top_k`: Number of top classes to display (default: 5).
- `--gpu`: Use GPU for inference if available (optional).

Example usage:

```bash
python predict.py --checkpoint checkpoint_model.pth --image_path /path/to/image.jpg --json_path class_names.json --top_k 3 --gpu
```

## Model Architecture (`model.py`)

The custom model architecture can be found in `model.py`. It defines a neural network architecture with flexible backbone choices (`vgg16` or `densenet121`) and customizable hidden units and dropout rates. The classifier consists of several fully connected layers.

```python
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
```
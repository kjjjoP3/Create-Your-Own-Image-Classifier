Image Classifier Project
🌟 Introduction
This project enables you to build and train a custom image classifier using PyTorch. It includes the following key components:

train.py: Script to train a custom image classification model.
predict.py: Script to predict image classes using a pre-trained model.
model.py: Defines the custom neural network architecture.
📋 System Requirements
To run this project, ensure you have the following installed:

Python 3.x
PyTorch
torchvision
NumPy
Pillow (PIL)
argparse (built-in Python library)
Install the dependencies using pip:

bash

pip install torch torchvision numpy pillow
🛠️ train.py
📖 Functionality
This script is used to train a custom image classifier.

🔧 Arguments
data_dir: Path to the directory containing training, validation, and test data.
--arch: Model architecture to use (vgg16 by default, supports densenet121).
--learning_rate: Learning rate for training (default: 0.001).
--hidden_units: Number of hidden units in the classifier (default: 2048).
--dropout: Dropout rate in the classifier (default: 0.3).
--epochs: Number of training epochs (default: 5).
--gpu: Enables GPU training if available (optional).
▶️ Example Usage
bash

python train.py /path/to/data --arch vgg16 --learning_rate 0.001 --hidden_units 2048 --dropout 0.3 --epochs 10 --gpu
🔍 predict.py
📖 Functionality
This script allows you to predict the class of an image using a trained model.

🔧 Arguments
--checkpoint: Path to the model checkpoint file (default: checkpoint_model.pth).
--image_path: Path to the image to be classified.
--json_path: Path to the JSON file containing class names.
--top_k: Number of top predicted classes to display (default: 5).
--gpu: Enables GPU inference if available (optional).
▶️ Example Usage
bash

python predict.py --checkpoint checkpoint_model.pth --image_path /path/to/image.jpg --json_path class_names.json --top_k 3 --gpu
🏗️ Custom Model Architecture (model.py)
The custom architecture can be found in the model.py file. It provides flexibility for selecting a backbone model (e.g., vgg16 or densenet121) and allows customization of hidden units and dropout rates. The classifier includes multiple fully connected layers.

python

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
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
from collections import OrderedDict
from model import Net


def make_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('data_dir', type=str, help='Path to data directory')
    parser.add_argument('--arch', type=str, help='Model trainning', default='vgg16')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for trainning')
    parser.add_argument('--hidden_units', type=int, default=2048, help='Number hidden units')
    parser.add_argument('--dropout', type=float, help='Model trainning', default=0.3)
    parser.add_argument('--epochs', type=int, default=5, help='Number of train epochs')
    parser.add_argument('--gpu',action='store_true', default='gpu', help='Use GPU for training')
    
    return parser.parse_args()

def load_data(data_path='flowers'):
    
    data_dir = data_path
    train_dir= data_dir + '/train'
    valid_dir= data_dir + '/valid'
    test_dir= data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64, shuffle = True)

    return train_data, trainloader, testloader, validloader

def main():
    args = make_args()
    if not os.path.exists(args.data_dir):
        print(f"Directory {args.data_dir} does not exist!")
        return
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    train_data, trainloader, validloader, testloader = load_data(args.data_dir)
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    model = Net(backbone=args.arch, hidden_units=args.hidden_units, dropout=args.dropout).to(device)
    # Define the loss function
    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()
    # Define the optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)


    print("--------------------------------------< Model >--------------------------------------------")
    print(model)
    print("--------------------------------------< Start Training >--------------------------------------------")
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        for inputs, labels in trainloader:
          steps += 1

          # Move input and label tensors to the default device
          inputs, labels = inputs.to(device), labels.to(device)

          # Clear the gradients
          optimizer.zero_grad()

          # Forward pass
          logps = model.forward(inputs)
          loss = criterion(logps, labels)

          # Backward pass
          loss.backward()
          optimizer.step()

          running_loss += loss.item()

          if steps % print_every == 0:
            val_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
              for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                val_loss += batch_loss.item()
                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            train_losses.append(running_loss/print_every)
            val_losses.append(val_loss/len(validloader))
            print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(val_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
            running_loss = 0
            model.train()

    print("--------------------------------------< Finished Training >--------------------------------------------")
    # Save the checkpoint
    # Define the file path where you want to save the model
    filepath = 'checkpoint_model.pth'

    # Save the model's state dictionary
    model.class_to_idx = train_data.class_to_idx
    torch.save({'input_size': 25088,
                'output_size': 102,
                'structure': 'vgg16',
                'learning_rate': 0.001,
                'epochs': epochs,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx},filepath)

if __name__ == '__main__':
    main()
import argparse
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from model import create_model

def parse_arguments():
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument('data_dir', type=str, help="Path to the dataset")
    parser.add_argument('--arch', type=str, default='vgg16', help="Model architecture")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--hidden_units', type=int, default=2048, help="Number of hidden units in classifier")
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout rate")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--gpu', action='store_true', help="Use GPU for training")
    return parser.parse_args()

def load_datasets(data_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms['train'])
    valid_data = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=data_transforms['valid'])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)

    return train_data, train_loader, valid_loader

def train_model(model, train_loader, valid_loader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {running_loss/len(train_loader):.3f}")

def main():
    args = parse_arguments()
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    train_data, train_loader, valid_loader = load_datasets(args.data_dir)

    model = create_model(args.arch, args.hidden_units, args.dropout).to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    train_model(model, train_loader, valid_loader, criterion, optimizer, device, args.epochs)

    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': train_data.class_to_idx
    }
    torch.save(checkpoint, 'checkpoint_model.pth')
    print("Training complete. Model saved.")

if __name__ == '__main__':
    main()

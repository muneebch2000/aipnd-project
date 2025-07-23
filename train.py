import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse
import os
from collections import OrderedDict

# ------------------------
# Data loading and transforms
# ------------------------
def get_data_loaders(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # Transforms
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=valid_transforms)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return train_loader, valid_loader, test_loader, train_data.class_to_idx

# ------------------------
# Build model
# ------------------------
def build_model(arch='vgg16', hidden_units=512):
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Custom classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(model.classifier[0].in_features, hidden_units)),
        ('relu', nn.ReLU()),
        ('drop', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    return model

# ------------------------
# Validation helper
# ------------------------
def validation(model, valid_loader, criterion, device):
    model.eval()
    val_loss, accuracy = 0, 0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            val_loss += criterion(output, labels).item()
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            accuracy += (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor).mean().item()

    return val_loss / len(valid_loader), accuracy / len(valid_loader)

# ------------------------
# Test accuracy
# ------------------------
def test_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# ------------------------
# Save checkpoint
# ------------------------
def save_checkpoint(model, optimizer, save_dir, arch, hidden_units, epochs):
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

# ------------------------
# Train script
# ------------------------
def train(args):
    train_loader, valid_loader, test_loader, class_to_idx = get_data_loaders(args.data_dir)

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    model = build_model(args.arch, args.hidden_units)
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss, val_accuracy = validation(model, valid_loader, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs}.. "
              f"Train loss: {running_loss/len(train_loader):.3f}.. "
              f"Validation loss: {val_loss:.3f}.. "
              f"Validation accuracy: {val_accuracy:.3f}")

    # Test
    test_model(model, test_loader, device)

    # Attach mapping and save checkpoint
    model.class_to_idx = class_to_idx
    save_checkpoint(model, optimizer, args.save_dir, args.arch, args.hidden_units, args.epochs)

# ------------------------
# Main CLI
# ------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a new network on a dataset and save the model as a checkpoint.")
    parser.add_argument('data_dir', help='Path to dataset directory')
    parser.add_argument('--save_dir', default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', default='vgg16', choices=['vgg16', 'vgg13'], help='Model architecture: vgg16 or vgg13')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Hidden units in classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    args = parser.parse_args()

    train(args)

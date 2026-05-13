import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms


def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Input variables
batch_size = 256
learning_rate = 0.001
num_epochs = 50
image_size = 224


def get_transforms(target_image_size=image_size):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(target_image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((target_image_size, target_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, test_transform


def load_datasets(data_root="./data", download=False, target_image_size=image_size):
    train_transform, test_transform = get_transforms(target_image_size)
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=download,
        transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=download,
        transform=test_transform
    )
    return train_dataset, test_dataset, test_transform


def create_dataloaders(train_dataset, test_dataset, test_transform, loader_batch_size=32):
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    if hasattr(val_dataset, "dataset") and hasattr(val_dataset.dataset, "transform"):
        val_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=loader_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=loader_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=loader_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, train_dataset


def build_model(num_classes, device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples

    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples

    return epoch_loss, epoch_acc

def run_training():
    device = get_default_device()
    print("Using device:", device)

    train_dataset, test_dataset, test_transform = load_datasets(
        data_root="./data",
        download=False,
        target_image_size=image_size,
    )
    train_loader, val_loader, _, split_train_dataset = create_dataloaders(
        train_dataset, test_dataset, test_transform, loader_batch_size=32
    )

    class_names = split_train_dataset.dataset.classes
    model = build_model(num_classes=len(class_names), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%")
        print("-" * 50)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model, best_val_acc


if __name__ == "__main__":
    run_training()

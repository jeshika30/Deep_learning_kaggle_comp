import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from torchsummary import summary
from sklearn.model_selection import train_test_split
import csv
import json

# Function to unpickle dataset files
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Load CIFAR-10 dataset
cifar_dir = "cifar-10-python/cifar-10-batches-py"  
meta_data = unpickle(os.path.join(cifar_dir, 'batches.meta'))
label_names = [name.decode("utf-8") for name in meta_data[b'label_names']]
print(label_names)

# Load all training data
train_images, train_labels = [], []
for i in range(1, 6):  # Load data_batch_1 to data_batch_5
    batch_dict = unpickle(os.path.join(cifar_dir, f"data_batch_{i}"))
    train_images.append(batch_dict[b'data'])
    train_labels.extend(batch_dict[b'labels'])

train_images = np.vstack(train_images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
train_labels = np.array(train_labels)

# Split into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.1, stratify=train_labels, random_state=42
)

print("Training dataset shape:", train_images.shape)
print("Validation dataset shape:", val_images.shape)

# Data augmentation and normalization
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset class
class CIFARDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        label = self.labels[idx]
        return image, label

# Create datasets and dataloaders
train_dataset = CIFARDataset(train_images, train_labels, transform_train)
val_dataset = CIFARDataset(val_images, val_labels, transform_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SmallResNet, self).__init__()
        self.model = models.resnet18(pretrained=False)

        # ✅ Reduce conv1 output to 32 channels
        self.model.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.bn1 = nn.BatchNorm2d(32)

        # ✅ Modify Layer1 to process 32 channels instead of 64
        self.model.layer1[0].conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[0].bn1 = nn.BatchNorm2d(32)
        self.model.layer1[0].conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[0].bn2 = nn.BatchNorm2d(32)

        self.model.layer1[1].conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[1].bn1 = nn.BatchNorm2d(32)
        self.model.layer1[1].conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[1].bn2 = nn.BatchNorm2d(32)

        # ✅ Modify Layer2: Transition 32 → 64
        self.model.layer2[0].conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.layer2[0].bn1 = nn.BatchNorm2d(64)
        self.model.layer2[0].conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer2[0].bn2 = nn.BatchNorm2d(64)
        self.model.layer2[0].downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
        )

        self.model.layer2[1].conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer2[1].bn1 = nn.BatchNorm2d(64)
        self.model.layer2[1].conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer2[1].bn2 = nn.BatchNorm2d(64)

        # ✅ Modify Layer3: Transition 64 → 128
        self.model.layer3[0].conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.layer3[0].bn1 = nn.BatchNorm2d(128)
        self.model.layer3[0].conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer3[0].bn2 = nn.BatchNorm2d(128)
        self.model.layer3[0].downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
        )

        self.model.layer3[1].conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer3[1].bn1 = nn.BatchNorm2d(128)
        self.model.layer3[1].conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer3[1].bn2 = nn.BatchNorm2d(128)

        # ✅ Modify Layer4: Transition 128 → 256
        self.model.layer4[0].conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.layer4[0].bn1 = nn.BatchNorm2d(256)
        self.model.layer4[0].conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer4[0].bn2 = nn.BatchNorm2d(256)
        self.model.layer4[0].downsample = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
        )

        self.model.layer4[1].conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer4[1].bn1 = nn.BatchNorm2d(256)
        self.model.layer4[1].conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer4[1].bn2 = nn.BatchNorm2d(256)

        # ✅ Modify Fully Connected Layer: 256 → 512 → 10
        self.model.fc = nn.Sequential(
            nn.Linear(256, 512),  # Expand to 512 neurons
            nn.ReLU(),  # Non-linearity
            nn.Dropout(0.3),  # Reduce overfitting
            nn.Linear(512, num_classes)  # Final classification
        )

    def forward(self, x):
        return self.model(x)


# Initialize model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = SmallResNet().to(device)

summary(resnet, input_size=(3, 32, 32))  # Check parameter count

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

# ✅ Use Cosine Annealing Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

# Track results
results = {"epochs": [], "train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}

# Open a CSV file to log results
with open("training_log2.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"])

    best_acc = 0

    for epoch in range(50):
        resnet.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate train accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        scheduler.step()

        # Validation
        resnet.eval()
        val_loss = 0.0
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = resnet(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val

        # ✅ PRINT loss and accuracy
        print(f"Epoch [{epoch+1}/50], Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Accuracy: {val_accuracy:.2f}%")

        writer.writerow([epoch+1, running_loss/len(train_loader), train_accuracy, val_loss/len(val_loader), val_accuracy])

        # Save best model
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(resnet.state_dict(), "best_resnet2.pth")

print(f"✅ Model successfully trained and saved with parameters <5M 🚀")

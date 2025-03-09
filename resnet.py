# import numpy as np 
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import pickle
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset, TensorDataset
# import torchvision.transforms as transforms
# from tqdm.notebook import tqdm
# import time
# import torchvision.models as models


# def unpickle(file):
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict

# cifar_dir = "cifar-10-python/cifar-10-batches-py"  
# meta_data = unpickle(os.path.join(cifar_dir, 'batches.meta'))
# label_names = [name.decode("utf-8") for name in meta_data[b'label_names']]  

# print(label_names)

# # load `data_batch_1`
# batch_dict = unpickle(os.path.join(cifar_dir, "data_batch_1"))
# train_images = batch_dict[b'data']
# train_labels = batch_dict[b'labels']

# train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (10000, 32, 32, 3)

# train_labels = np.array(train_labels)  # (10000,)

# print("training dataset shape:", train_images.shape)  # (10000, 32, 32, 3)
# print("training label shape:", train_labels.shape)  # (10000,)

# # show first 10 images
# plt.figure(figsize=(20, 4))
# for i in range(10):
#     plt.subplot(1, 10, i+1)
#     plt.imshow(train_images[i])
#     plt.title(label_names[train_labels[i]])  # get labels
#     plt.axis('off')

# plt.show()

# print("Labels for first 10 images:", [label_names[label] for label in train_labels[:10]])

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")


# # convert to torch tensors

# train_images_tensor = torch.tensor(train_images, dtype=torch.float32).permute(0, 3, 1, 2)

# train_labels_tensor = torch.tensor(train_labels)  # Labels

# # Perform data normalization, CIFAR-10 mean and standard deviation are usually as follows
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),  # ResNet usually requires 224x224 input images
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # CIFAR-10 normalization
# ])

# # Create dataset and DataLoader
# train_dataset = torch.utils.data.TensorDataset(train_images_tensor, train_labels_tensor)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# # Initialize ResNet-50 model (without pre-trained weights)
# resnet = models.resnet50(pretrained=False)

# # Set model to training mode
# resnet.train()

# # Move the model to GPU (if available)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# resnet.to(device)

# # Define the loss function, assuming a classification task, using CrossEntropyLoss
# criterion = torch.nn.CrossEntropyLoss()

# # Define SGD optimizer
# optimizer = optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

# # Training process
# num_epochs = 30  # Set number of training epochs
# for epoch in range(num_epochs):
#     running_loss = 0.0
#     for inputs, labels in train_loader:
#         # Move data to GPU (if available)
#         inputs, labels = inputs.to(device), labels.to(device)
        
#         # Zero gradients
#         optimizer.zero_grad()
        
#         # Forward propagation
#         outputs = resnet(inputs)
        
#         # Compute loss
#         loss = criterion(outputs, labels)
        
#         # Backpropagation
#         loss.backward()
        
#         # Update weights
#         optimizer.step()
        
#         running_loss += loss.item()

#     # Print loss for each epoch
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

# # After training, save the model
# torch.save(resnet.state_dict(), 'resnet50_sgd.pth')


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
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
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

import torch.nn as nn
import torchvision.models as models

class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SmallResNet, self).__init__()
        self.model = models.resnet18(pretrained=False)

        # Reduce the number of channels in the first conv layer
        self.model.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.bn1 = nn.BatchNorm2d(32)  # Update BatchNorm to match conv1 output
        
        # Modify the number of channels in each residual block
        self.model.layer1[0].conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[0].bn1 = nn.BatchNorm2d(32)

        self.model.layer1[0].conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[0].bn2 = nn.BatchNorm2d(32)

        self.model.layer1[1].conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[1].bn1 = nn.BatchNorm2d(32)

        self.model.layer1[1].conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[1].bn2 = nn.BatchNorm2d(32)

        # Modify layer2 (downsample: 32 -> 64)
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

        # Modify layer3 (downsample: 64 -> 128)
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

        # Modify layer4 (downsample: 128 -> 256)
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

        # Adjust fully connected layer to match last block
        self.model.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.model(x)


# Initialize model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = SmallResNet().to(device)

summary(resnet, input_size=(3, 32, 32))  # Check parameter count

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    resnet.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    scheduler.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Save trained model
torch.save(resnet.state_dict(), 'resnet18_sgd.pth')

# Evaluate on validation dataset
resnet.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = resnet(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Validation Accuracy: {accuracy:.2f}%')
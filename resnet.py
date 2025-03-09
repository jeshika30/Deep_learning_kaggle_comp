import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.transforms as transforms
from tqdm.notebook import tqdm
import time
import torchvision.models as models


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar_dir = "cifar-10-python/cifar-10-batches-py"  
meta_data = unpickle(os.path.join(cifar_dir, 'batches.meta'))
label_names = [name.decode("utf-8") for name in meta_data[b'label_names']]  

print(label_names)

# load `data_batch_1`
batch_dict = unpickle(os.path.join(cifar_dir, "data_batch_1"))
train_images = batch_dict[b'data']
train_labels = batch_dict[b'labels']

train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (10000, 32, 32, 3)

train_labels = np.array(train_labels)  # (10000,)

print("training dataset shape:", train_images.shape)  # (10000, 32, 32, 3)
print("training label shape:", train_labels.shape)  # (10000,)

# show first 10 images
plt.figure(figsize=(20, 4))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(train_images[i])
    plt.title(label_names[train_labels[i]])  # get labels
    plt.axis('off')

plt.show()

print("Labels for first 10 images:", [label_names[label] for label in train_labels[:10]])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# convert to torch tensors

train_images_tensor = torch.tensor(train_images, dtype=torch.float32).permute(0, 3, 1, 2)

train_labels_tensor = torch.tensor(train_labels)  # Labels

# Perform data normalization, CIFAR-10 mean and standard deviation are usually as follows
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # ResNet usually requires 224x224 input images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # CIFAR-10 normalization
])

# Create dataset and DataLoader
train_dataset = torch.utils.data.TensorDataset(train_images_tensor, train_labels_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize ResNet-50 model (without pre-trained weights)
resnet = models.resnet50(pretrained=False)

# Set model to training mode
resnet.train()

# Move the model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)

# Define the loss function, assuming a classification task, using CrossEntropyLoss
criterion = torch.nn.CrossEntropyLoss()

# Define SGD optimizer
optimizer = optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

# Training process
num_epochs = 30  # Set number of training epochs
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Move data to GPU (if available)
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = resnet(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backpropagation
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        running_loss += loss.item()

    # Print loss for each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

# After training, save the model
torch.save(resnet.state_dict(), 'resnet50_sgd.pth')

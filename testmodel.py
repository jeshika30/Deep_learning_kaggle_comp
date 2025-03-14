import numpy as np 
import pandas as pd
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from sklearn.model_selection import train_test_split
import csv


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar_dir = "./cifar-10-python/cifar-10-batches-py"
if not os.path.exists(cifar_dir):
    raise FileNotFoundError(f"Dataset path {cifar_dir} not found. Ensure CIFAR-10 is downloaded.")

meta_data = unpickle(os.path.join(cifar_dir, 'batches.meta'))
label_names = [name.decode("utf-8") for name in meta_data[b'label_names']]
print("Labels:", label_names)

train_images, train_labels = [], []
for i in range(1, 6): 
    batch_dict = unpickle(os.path.join(cifar_dir, f"data_batch_{i}"))
    train_images.append(batch_dict[b'data'])
    train_labels.extend(batch_dict[b'labels'])

train_images = np.vstack(train_images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
train_labels = np.array(train_labels)

train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.1, stratify=train_labels, random_state=42
)

print("Training dataset shape:", train_images.shape)
print("Validation dataset shape:", val_images.shape)

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

train_dataset = CIFARDataset(train_images, train_labels, transform_train)
val_dataset = CIFARDataset(val_images, val_labels, transform_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SmallResNet, self).__init__()
        self.model = models.resnet18(pretrained=False)

        self.model.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.bn1 = nn.BatchNorm2d(32)

        self.model.layer1[0].conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[0].bn1 = nn.BatchNorm2d(32)
        self.model.layer1[0].conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[0].bn2 = nn.BatchNorm2d(32)

        self.model.layer1[1].conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[1].bn1 = nn.BatchNorm2d(32)
        self.model.layer1[1].conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[1].bn2 = nn.BatchNorm2d(32)

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

        self.model.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.model(x)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = SmallResNet().to(device)

total_params = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")
print("Within limit:", total_params <= 5_000_000)


criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)


# test_batch_file = os.path.join(cifar_dir, "test_batch")
test_dict = unpickle('cifar_test_nolabel.pkl')
test_images = test_dict[b'data']
test_ids = test_dict[b'ids']
# test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class TestDataset(Dataset):
    def __init__(self, images, ids, transform):
        self.images = images
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        test_id = self.ids[idx]
        return image, test_id

test_dataset = TestDataset(test_images,test_ids, transform_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

best_model = SmallResNet().to(device)
best_model.load_state_dict(torch.load("best_resnet3.pth"))
best_model.eval()

predictions = []
ids = []
with torch.no_grad():
    for images, idx in test_loader:
        images = images.to(device)
        outputs = best_model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())
        ids.extend(idx.cpu().numpy())

submission_df = pd.DataFrame({"ID": ids, "Labels": predictions})
submission_df.to_csv("submission_2.csv", index=False)

print("saved to submission_2.csv")

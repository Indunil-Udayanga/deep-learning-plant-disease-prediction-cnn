import os
import json
import random
import shutil
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from PIL import Image

# config

SEED = 0
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001
SUBSET_SIZE = 40000

RAW_DATA_DIR = '/content/plantvillage-dataset/plantvillage dataset/color'
SPLIT_DIR    = '/content/split_data'

BEST_MODEL_PATH  = 'best_model.pth'
FINAL_MODEL_PATH = 'plant_disease_cnn.pth'
CLASS_NAMES_PATH = 'class_names.json'

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def download_dataset(raw_dir=RAW_DATA_DIR):
    """Download PlantVillage from Kaggle, but skip if already present."""
    if os.path.isdir(raw_dir):
        print(f'Dataset already present at {raw_dir}, skipping download.')
        return
    import opendatasets as od
    od.download('https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset')

def build_dataset_split(raw_dir=RAW_DATA_DIR, split_dir=SPLIT_DIR, subset_size=SUBSET_SIZE):
    all_images, all_labels = [], []
    for class_name in os.listdir(raw_dir):
        class_path = os.path.join(raw_dir, class_name)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                all_images.append(os.path.join(class_path, img_file))
                all_labels.append(class_name)

    class_buckets = defaultdict(list)
    for img, lbl in zip(all_images, all_labels):
        class_buckets[lbl].append(img)

    imgs_per_class = subset_size // len(class_buckets)
    subset_images, subset_labels = [], []
    for cls, imgs in class_buckets.items():
        sampled = random.sample(imgs, min(imgs_per_class, len(imgs)))
        subset_images.extend(sampled)
        subset_labels.extend([cls] * len(sampled))

    train_imgs, temp_imgs, train_lbls, temp_lbls = train_test_split(
        subset_images, subset_labels, test_size=0.2, random_state=42, stratify=subset_labels
    )
    val_imgs, test_imgs, val_lbls, test_lbls = train_test_split(
        temp_imgs, temp_lbls, test_size=0.5, random_state=42, stratify=temp_lbls
    )

    def copy_images(img_paths, labels, split_name):
        for img_path, label in zip(img_paths, labels):
            dest = os.path.join(split_dir, split_name, label)
            os.makedirs(dest, exist_ok=True)
            shutil.copy(img_path, dest)

    copy_images(train_imgs, train_lbls, 'train')
    copy_images(val_imgs,   val_lbls,   'val')
    copy_images(test_imgs,  test_lbls,  'test')

    return split_dir

#  DATA AUGMENTATION & PREPROCESSING
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, shear=0.2),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

#  DATALOADER

def build_dataloaders(split_dir=SPLIT_DIR):
    train_dataset = datasets.ImageFolder(os.path.join(split_dir, 'train'), transform=train_transform)
    val_dataset   = datasets.ImageFolder(os.path.join(split_dir, 'val'),   transform=test_transform)
    test_dataset  = datasets.ImageFolder(os.path.join(split_dir, 'test'),  transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    class_names = train_dataset.classes
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, class_names

#  MODEL — PlantDiseaseCNN

class PlantDiseaseCNN(nn.Module):
    def __init__(self,num_classes):
        super(PlantDiseaseCNN,self).__init__()

        def conv_block(in_ch,out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1) ,
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2,2)
            )
        
        self.block1 = conv_block(3 , 32)
        self.block2 = conv_block(32, 64)
        self.block3 = conv_block(64, 128)
        self.block4 = conv_block(128, 256)

        self.pool = nn.AdaptiveAvgPool2d((1,1))  # was: nn.AdaptiveAvgPool

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x   # raw logits — this is the "Forward Pass" output
    

# 5–9. TRAINING
#   Forward Pass -> CrossEntropy Loss -> Backpropagation
#   -> Adam Optimizer -> LR Scheduler -> Validation

def train_model(model, train_loader, val_loader, train_dataset, val_dataset):
    criterion = nn.CrossEntropyLoss()                                       # Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)                 # Optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(                 # Scheduler
        optimizer, mode='min', patience=3, factor=0.5
    )

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(EPOCHS):

        # ---- TRAIN ----
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)               # Forward Pass
            loss = criterion(outputs, labels)      # CrossEntropy Loss
            loss.backward()                        # Backpropagation
            optimizer.step()                       # Adam step

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()

        # ---- VALIDATION ----
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()

        train_loss_avg = train_loss / len(train_dataset)
        train_acc_avg  = train_correct / len(train_dataset)
        val_loss_avg   = val_loss / len(val_dataset)
        val_acc_avg    = val_correct / len(val_dataset)

        scheduler.step(val_loss_avg)               # LR Scheduler step

        if val_acc_avg > best_val_acc:
            best_val_acc = val_acc_avg
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        print(f'Epoch [{epoch+1:02d}/{EPOCHS}]  '
              f'Train Loss: {train_loss_avg:.4f}  Acc: {train_acc_avg:.4f}  '
              f'Val Loss: {val_loss_avg:.4f}  Acc: {val_acc_avg:.4f}  '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

    print(f'\nBest Val Acc: {best_val_acc:.4f}  '
          f'(Total time: {(time.time() - start_time) / 60:.2f} min)')

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    return model


#  TESTING

def test_model(model, test_loader, test_dataset):
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            test_correct += (preds == labels).sum().item()

    test_acc = test_correct / len(test_dataset)
    print(f'Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)')
    return test_acc

#  DISEASE PREDICTION — single image inference

def predict_image(image_path, model, class_names):
    img = Image.open(image_path).convert('RGB')
    tensor = test_transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0][pred_idx].item() * 100

    predicted_class = class_names[pred_idx]
    print(f'Predicted: {predicted_class}  ({confidence:.2f}% confidence)')
    return predicted_class, confidence

# MAIN 

if __name__ == "__main__":

    download_dataset()
    split_dir = build_dataset_split()

    (train_loader, val_loader, test_loader,
     train_dataset, val_dataset, test_dataset, class_names) = build_dataloaders(split_dir)

    with open(CLASS_NAMES_PATH, 'w') as f:
        json.dump(class_names, f)

    model = PlantDiseaseCNN(num_classes=len(class_names)).to(device)

    model = train_model(model, train_loader, val_loader, train_dataset, val_dataset)

    test_model(model, test_loader, test_dataset)

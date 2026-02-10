import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import datasets, models
import numpy as np

# =========================
# CONFIG
# =========================
DATASET_DIR = "model2/garbage_classification"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

EPOCHS_HEAD = 12
EPOCHS_FINE = 8

MODEL_OUT = "garbage_mobilenetv2_multiclass.pth"
BEST_OUT = "garbage_mobilenetv2_best.pth"

# =========================
# REPRODUCIBILITY
# =========================
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# =========================
# DATA TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# DATASET & SPLIT
# =========================
full_dataset = datasets.ImageFolder(DATASET_DIR, transform=train_transform)

class_names = full_dataset.classes
num_classes = len(class_names)

print("Clases detectadas:", class_names)
print("Número de clases:", num_classes)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

# Cambiar transform para validación
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# DEVICE
# =========================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("\nVerificando GPU...")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️ No se detectó GPU, se usará CPU")

# =========================
# MODEL
# =========================
def create_model(num_classes):
    model = models.mobilenet_v2(pretrained=True)

    # Congelar base
    for param in model.features.parameters():
        param.requires_grad = False

    # Nuevo clasificador
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, num_classes)
    )

    return model

model = create_model(num_classes).to(device)

# =========================
# TRAIN & VALIDATE FUNCTIONS
# =========================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)
        total += inputs.size(0)

    return running_loss / total, running_corrects.double() / total


def validate_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)
            total += inputs.size(0)

    return running_loss / total, running_corrects.double() / total

# =========================
# PHASE 1: TRAIN HEAD
# =========================
print("\n=== ENTRENANDO HEAD (base congelada) ===")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

best_val_acc = 0.0

for epoch in range(EPOCHS_HEAD):
    print(f"\nEpoch {epoch+1}/{EPOCHS_HEAD}")
    print("-" * 30)

    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), BEST_OUT)
        print("✅ Mejor modelo guardado")

# =========================
# PHASE 2: FINE-TUNING
# =========================
print("\n=== FINE-TUNING (descongelando capas) ===")

for param in model.features[-4:].parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(EPOCHS_FINE):
    print(f"\nFine-tune Epoch {epoch+1}/{EPOCHS_FINE}")
    print("-" * 30)

    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), BEST_OUT)
        print("✅ Mejor modelo guardado")

# =========================
# SAVE FINAL MODEL
# =========================
torch.save(model.state_dict(), MODEL_OUT)

with open("class_names.txt", "w", encoding="utf-8") as f:
    for c in class_names:
        f.write(c + "\n")

print("\n==============================")
print("🔥 ENTRENAMIENTO FINALIZADO")
print(f"✅ Modelo final: {MODEL_OUT}")
print(f"✅ Mejor modelo: {BEST_OUT}")
print(f"✅ Mejor Val Accuracy: {best_val_acc:.4f}")
print("✅ class_names.txt guardado")
print("==============================")

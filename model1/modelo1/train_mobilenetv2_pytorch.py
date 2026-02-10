import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, models
import numpy as np
from PIL import Image

# =========================
# CONFIG
# =========================
DATASET_DIR = "dataset"  # dataset/train, dataset/val, dataset/test
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS_HEAD = 12
EPOCHS_FINE = 8
MODEL_OUT = "waste_mobilenetv2_pytorch.pth"
BEST_OUT = "waste_mobilenetv2_best_pytorch.pth"

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

# Configurar semilla para reproducibilidad
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

# =========================
# DATA TRANSFORMS
# =========================
# Transformaciones para entrenamiento (con augmentation)
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Transformaciones para validación (sin augmentation)
val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =========================
# DATASETS
# =========================
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class_names = train_dataset.classes
num_classes = len(class_names)
print("Clases:", class_names)

if __name__ == "__main__":
    # =========================
    # MODEL: Transfer Learning with MobileNetV2
    # =========================
    def create_model(num_classes, pretrained=True):
        """Crear modelo MobileNetV2 con transfer learning"""
        # Cargar MobileNetV2 preentrenado
        model = models.mobilenet_v2(pretrained=pretrained)
        
        # Congelar las capas base inicialmente
        for param in model.features.parameters():
            param.requires_grad = False
        
        # Modificar el clasificador para nuestro número de clases
        model.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(model.last_channel, num_classes)
        )
        
        return model

    # Verificar si la GPU está disponible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Verificando la disponibilidad de la GPU...")
    if torch.cuda.is_available():
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("No se encontraron GPUs. El entrenamiento se realizará en la CPU.")

    # Crear modelo
    model = create_model(num_classes)
    model = model.to(device)

    # =========================
    # TRAINING FUNCTIONS
    # =========================
    def train_epoch(model, train_loader, criterion, optimizer, device):
        """Entrenar una época"""
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return epoch_loss, epoch_acc

    def validate_epoch(model, val_loader, criterion, device):
        """Validar una época"""
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return epoch_loss, epoch_acc

    # =========================
    # PHASE 1: Train Head (Base Frozen)
    # =========================
    print("\n=== Entrenando 'head' (base congelada) ===")

    # Criterio y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(EPOCHS_HEAD):
        print(f'\nEpoch {epoch+1}/{EPOCHS_HEAD}')
        print('-' * 20)
        
        # Entrenar
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validar
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Guardar métricas
        train_losses.append(train_loss)
        train_accs.append(train_acc.item())
        val_losses.append(val_loss)
        val_accs.append(val_acc.item())
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_OUT)
            print(f'¡Mejor modelo guardado! Val Acc: {val_acc:.4f}')

    # =========================
    # PHASE 2: Fine-tuning (Unfreeze some layers)
    # =========================
    print("\n=== Fine-tuning (descongelando parte de la base) ===")

    # Descongelar las últimas capas de features
    for param in model.features[-3:].parameters():  # Descongelar últimas 3 capas
        param.requires_grad = True

    # Nuevo optimizador con learning rate más bajo
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(EPOCHS_FINE):
        print(f'\nFine-tune Epoch {epoch+1}/{EPOCHS_FINE}')
        print('-' * 25)
        
        # Entrenar
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validar
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Guardar métricas
        train_losses.append(train_loss)
        train_accs.append(train_acc.item())
        val_losses.append(val_loss)
        val_accs.append(val_acc.item())
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_OUT)
            print(f'¡Mejor modelo guardado! Val Acc: {val_acc:.4f}')

    # =========================
    # SAVE FINAL MODEL
    # =========================
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"\n✅ Modelo final guardado en: {MODEL_OUT}")
    print(f"✅ Mejor modelo (checkpoint) en: {BEST_OUT}")
    print(f"✅ Mejor val accuracy: {best_val_acc:.4f}")

    # Guardar nombres de clases
    with open("class_names.txt", "w", encoding="utf-8") as f:
        for c in class_names:
            f.write(c + "\n")
    print("✅ class_names.txt guardado")

    print(f"\n🔥 ¡Entrenamiento completado usando GPU: {torch.cuda.get_device_name(0)}!")
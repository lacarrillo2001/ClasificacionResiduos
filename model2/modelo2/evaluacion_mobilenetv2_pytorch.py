import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# CONFIGURACIÓN
# =========================
DATASET_DIR = "model2/garbage_classification"  # dataset completo
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODELO_PATH = "garbage_mobilenetv2_best.pth"
ARCHIVO_CLASES = "class_names.txt"

# =========================
# DISPOSITIVO
# =========================
dispositivo = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", dispositivo)

# =========================
# CARGAR CLASES
# =========================
with open(ARCHIVO_CLASES, "r", encoding="utf-8") as f:
    nombres_clases = [line.strip() for line in f if line.strip()]

num_clases = len(nombres_clases)
print("Clases:", nombres_clases)

# =========================
# TRANSFORMACIONES (VALIDACIÓN / TEST)
# =========================
transformaciones = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# DATASET Y DATALOADER
# =========================
dataset = datasets.ImageFolder(DATASET_DIR, transform=transformaciones)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# MODELO (MISMA ARQUITECTURA)
# =========================
def crear_modelo(num_clases):
    modelo = models.mobilenet_v2(pretrained=False)
    modelo.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(modelo.last_channel, num_clases)
    )
    return modelo

modelo = crear_modelo(num_clases)
modelo.load_state_dict(torch.load(MODELO_PATH, map_location=dispositivo))
modelo = modelo.to(dispositivo)
modelo.eval()

print("Modelo cargado correctamente")

# =========================
# EVALUACIÓN
# =========================
y_true = []
y_pred = []

with torch.no_grad():
    for imagenes, etiquetas in loader:
        imagenes = imagenes.to(dispositivo)
        etiquetas = etiquetas.to(dispositivo)

        salidas = modelo(imagenes)
        _, predicciones = torch.max(salidas, 1)

        y_true.extend(etiquetas.cpu().numpy())
        y_pred.extend(predicciones.cpu().numpy())

# =========================
# REPORTES
# =========================
print("\n=== Classification Report ===")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=nombres_clases,
        digits=4
    )
)

# =========================
# CONFUSION MATRIX
# =========================
matriz_confusion = confusion_matrix(y_true, y_pred)

print("\n=== Confusion Matrix (valores) ===")
print(matriz_confusion)

# Guardar matriz en CSV
np.savetxt(
    "confusion_matrix.csv",
    matriz_confusion,
    delimiter=",",
    fmt="%d"
)

print("\n✅ confusion_matrix.csv guardado")

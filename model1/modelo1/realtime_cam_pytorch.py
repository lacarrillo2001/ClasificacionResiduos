import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = "waste_mobilenetv2_best_pytorch.pth"
CLASS_NAMES_FILE = "class_names.txt"
IMG_SIZE = (224, 224)

# Cargar clases
with open(CLASS_NAMES_FILE, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

# Crear modelo (misma arquitectura que en el entrenamiento)
def create_model(num_classes):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(model.last_channel, num_classes)
    )
    return model

# Cargar modelo
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = create_model(len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print(f"Modelo cargado en: {device}")
print(f"Clases: {class_names}")

# Transformaciones (mismas que en el entrenamiento)
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_frame(frame_bgr):
    """Procesar frame de OpenCV para PyTorch"""
    # Convertir BGR a RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Convertir a PIL Image
    pil_image = Image.fromarray(frame_rgb)
    
    # Aplicar transformaciones
    tensor = transform(pil_image).unsqueeze(0)  # Añadir dimensión del batch
    
    return tensor.to(device)

cap = cv2.VideoCapture(0)  # cambia a 1 si tienes otra cámara

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

print("Presiona 'q' para salir.")

frame_count = 0
PRED_EVERY_N = 5  # predice cada N frames para ir fluido

label_text = "..."
conf_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % PRED_EVERY_N == 0:
        # Preprocesar frame
        x = preprocess_frame(frame)
        
        # Predicción
        with torch.no_grad():
            outputs = model(x)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Obtener predicción
            conf, idx = torch.max(probs, 0)
            label_text = class_names[idx.item()]
            conf_text = f"{conf.item()*100:.1f}%"

    # Overlay
    cv2.putText(frame, f"Clase: {label_text}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Conf: {conf_text}", (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Mostrar información adicional
    cv2.putText(frame, f"Device: {device}", (15, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Clasificador de Residuos (Tiempo Real) - PyTorch", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
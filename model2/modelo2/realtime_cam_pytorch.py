import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# =========================
# CONFIGURACIÓN
# =========================
RUTA_MODELO = "garbage_mobilenetv2_best.pth"
ARCHIVO_CLASES = "class_names.txt"
TAM_IMAGEN = (224, 224)
PREDECIR_CADA_N = 5  # predice cada N frames

# =========================
# DISPOSITIVO
# =========================
dispositivo = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", dispositivo)

# =========================
# CARGAR NOMBRES DE CLASES (INGLÉS)
# =========================
with open(ARCHIVO_CLASES, "r", encoding="utf-8") as f:
    nombres_clases = [linea.strip() for linea in f if linea.strip()]

num_clases = len(nombres_clases)
print("Clases detectadas:", nombres_clases)

# =========================
# TRADUCCIÓN DE CLASES (EN → ES)
# =========================
TRADUCCION_CLASES = {
    "battery": "batería",
    "biological": "residuo orgánico",
    "brown-glass": "vidrio marrón",
    "cardboard": "cartón",
    "clothes": "ropa",
    "green-glass": "vidrio verde",
    "metal": "metal",
    "paper": "papel",
    "plastic": "plástico",
    "shoes": "calzado",
    "trash": "basura",
    "white-glass": "vidrio blanco"
}

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
modelo.load_state_dict(torch.load(RUTA_MODELO, map_location=dispositivo))
modelo = modelo.to(dispositivo)
modelo.eval()

print("Modelo cargado correctamente")

# =========================
# TRANSFORMACIONES
# =========================
transformaciones = transforms.Compose([
    transforms.Resize(TAM_IMAGEN),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocesar_frame(frame_bgr):
    """Convierte un frame de OpenCV a tensor para PyTorch"""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    imagen_pil = Image.fromarray(frame_rgb)
    tensor = transformaciones(imagen_pil).unsqueeze(0)
    return tensor.to(dispositivo)

# =========================
# CÁMARA
# =========================
camara = cv2.VideoCapture(0)  # cambia a 1 si usas otra cámara

if not camara.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

print("🎥 Cámara iniciada. Presiona 'q' para salir.")

contador_frames = 0
texto_clase = "..."
texto_confianza = ""

# =========================
# BUCLE PRINCIPAL
# =========================
while True:
    ret, frame = camara.read()
    if not ret:
        break

    contador_frames += 1

    if contador_frames % PREDECIR_CADA_N == 0:
        x = preprocesar_frame(frame)

        with torch.no_grad():
            salidas = modelo(x)
            probabilidades = torch.softmax(salidas[0], dim=0)
            confianza, indice = torch.max(probabilidades, 0)

            clase_ingles = nombres_clases[indice.item()]
            texto_clase = TRADUCCION_CLASES.get(clase_ingles, clase_ingles)
            texto_confianza = f"{confianza.item() * 100:.1f}%"

    # =========================
    # TEXTO EN PANTALLA
    # =========================
    cv2.putText(frame, f"Clase: {texto_clase}", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f"Confianza: {texto_confianza}", (15, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f"Dispositivo: {dispositivo}", (15, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Clasificación de Residuos - PyTorch (Tiempo Real)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# LIBERAR RECURSOS
# =========================
camara.release()
cv2.destroyAllWindows()
print("✔️ Inferencia finalizada")

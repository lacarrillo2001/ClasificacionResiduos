import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import csv

# =========================
# CONFIGURACIÓN
# =========================
RUTA_MODELO = "garbage_mobilenetv2_best.pth"
ARCHIVO_CLASES = "class_names.txt"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_IMAGENES = os.path.join(BASE_DIR, "..", "imagenes_prueba")

TAM_IMAGEN = (224, 224)
EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png")

GUARDAR_CSV = True
ARCHIVO_SALIDA = "resultados_reciclaje_4_categorias.csv"

# =========================
# DISPOSITIVO
# =========================
dispositivo = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("🖥️ Usando dispositivo:", dispositivo)

# =========================
# CLASES DEL MODELO (INGLÉS)
# =========================
with open(ARCHIVO_CLASES, "r", encoding="utf-8") as f:
    nombres_clases = [line.strip() for line in f if line.strip()]

num_clases = len(nombres_clases)
print("Clases del modelo:", nombres_clases)

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
# SOLO 4 CATEGORÍAS DE RECICLAJE
# =========================
CATEGORIA_RECICLAJE = {
    # 🟢 Orgánico
    "biological": "🟢 Orgánico",

    # 🔵 Papel / Cartón
    "paper": "🔵 Papel / Cartón",
    "cardboard": "🔵 Papel / Cartón",

    # 🟡 Plástico
    "plastic": "🟡 Plástico"
}

# Todo lo que NO esté arriba → No reciclable

# =========================
# MODELO
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

print("✅ Modelo cargado correctamente")

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

# =========================
# CLASIFICACIÓN
# =========================
resultados = []

print("\n♻️ Clasificando imágenes (4 categorías)...\n")

for nombre_archivo in os.listdir(CARPETA_IMAGENES):
    if not nombre_archivo.lower().endswith(EXTENSIONES_VALIDAS):
        continue

    ruta_imagen = os.path.join(CARPETA_IMAGENES, nombre_archivo)

    try:
        imagen = Image.open(ruta_imagen).convert("RGB")
    except Exception as e:
        print(f"❌ Error al abrir {nombre_archivo}: {e}")
        continue

    tensor = transformaciones(imagen).unsqueeze(0).to(dispositivo)

    with torch.no_grad():
        salidas = modelo(tensor)
        probabilidades = torch.softmax(salidas[0], dim=0)
        confianza, indice = torch.max(probabilidades, 0)

    clase_ingles = nombres_clases[indice.item()]
    tipo_espanol = TRADUCCION_CLASES.get(clase_ingles, clase_ingles)
    categoria = CATEGORIA_RECICLAJE.get(clase_ingles, "⚫ No reciclable")
    confianza_pct = confianza.item() * 100

    resultados.append([
        nombre_archivo,
        tipo_espanol,
        categoria,
        f"{confianza_pct:.2f}%"
    ])

    print(f"📷 {nombre_archivo}")
    print(f"   → Tipo detectado: {tipo_espanol}")
    print(f"   → Categoría de reciclaje: {categoria}")
    print(f"   → Confianza: {confianza_pct:.2f}%\n")

# =========================
# GUARDAR CSV
# =========================
if GUARDAR_CSV and resultados:
    with open(ARCHIVO_SALIDA, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "imagen",
            "tipo_detectado",
            "categoria_reciclaje",
            "confianza"
        ])
        writer.writerows(resultados)

    print(f"📄 Resultados guardados en: {ARCHIVO_SALIDA}")

print("\n✔️ Clasificación finalizada (4 categorías)")

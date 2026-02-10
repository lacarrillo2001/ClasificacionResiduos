import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

DATASET_DIR = "dataset"
TEST_DIR = os.path.join(DATASET_DIR, "test")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = "waste_mobilenetv2_best.keras"  # usa el mejor
CLASS_NAMES_FILE = "class_names.txt"

# Cargar class names
with open(CLASS_NAMES_FILE, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

model = tf.keras.models.load_model(MODEL_PATH)

y_true = []
y_pred = []

for batch_images, batch_labels in test_ds:
    preds = model.predict(batch_images, verbose=0)
    y_true.extend(batch_labels.numpy().tolist())
    y_pred.extend(np.argmax(preds, axis=1).tolist())

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
print("\n=== Confusion Matrix (valores) ===")
print(cm)

# Opcional: guardar matriz en CSV
np.savetxt("confusion_matrix.csv", cm, delimiter=",", fmt="%d")
print("\n✅ confusion_matrix.csv guardado")

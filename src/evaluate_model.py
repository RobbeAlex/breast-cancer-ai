import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("Iniciando evaluación del modelo de Cáncer de Mama...")

try:
    with open(os.path.join(BASE_DIR, 'config', 'mapeo_clases.json'), 'r', encoding='utf-8') as f:
        mapeo_clases = json.load(f)
    print("Mapeo de clases cargado correctamente.")
except FileNotFoundError:
    print("Error: No se encontró 'mapeo_clases.json'.")
    exit()

try:
    model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'models', 'modelo_breast_cancer.keras'))
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

TEST_DIR = os.path.join(BASE_DIR, "Testing Data")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    shuffle=False,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    label_mode='categorical'
)

# Preprocesamiento
AUTOTUNE = tf.data.AUTOTUNE
test_dataset = test_dataset.map(lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

print("\nAnalizando imágenes (recortes) de prueba. Por favor espera...")
predicciones_raw = model.predict(test_dataset)
y_pred = np.argmax(predicciones_raw, axis=1)

y_true = []
for images, labels in test_dataset.unbatch():
    y_true.append(np.argmax(labels.numpy()))
y_true = np.array(y_true)

# Ordenar los nombres de las clases según sus índices
clases_names = [mapeo_clases[str(i)] for i in range(len(mapeo_clases))]

print("\n==================================================")
print("REPORTE DE CLASIFICACIÓN CLÍNICA (PARCHES)")
print("==================================================")
print(classification_report(y_true, y_pred, target_names=clases_names))

# Matriz de Confusión
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(cm, interpolation='nearest', cmap='Reds')
plt.title('Matriz de Confusión: Cáncer de Mama (Parches)')

for i in range(len(clases_names)):
    for j in range(len(clases_names)):
        color = "white" if cm[i, j] > (cm.max() / 2) else "black"
        ax.text(j, i, str(cm[i, j]), va='center', ha='center', color=color)

plt.colorbar(cax)
tick_marks = np.arange(len(clases_names))
plt.xticks(tick_marks, clases_names, rotation=45)
plt.yticks(tick_marks, clases_names)
plt.ylabel('Diagnóstico Real (Lo que es)')
plt.xlabel('Diagnóstico de la IA (Lo que predijo)')
plt.tight_layout()
os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
plt.savefig(os.path.join(BASE_DIR, 'results', 'matriz_confusion.png'))

print("¡Evaluación terminada! La matriz se ha guardado como 'matriz_confusion.png'.")

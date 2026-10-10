import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

print("Configurando el entorno de entrenamiento...")

# Configuración
TRAIN_DIR = "Training Data"
TEST_DIR = "Testing Data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data Augmentation fuerte (las mamografías pueden rotarse o invertirse)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
])

# Carga de datos
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    label_mode='categorical'
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    shuffle=False,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    label_mode='categorical'
)

clases = train_dataset.class_names
print(f"Clases detectadas: {clases}")

# Guardar mapeo de clases
mapeo_clases = {str(i): clase for i, clase in enumerate(clases)}
with open('mapeo_clases.json', 'w') as f:
    json.dump(mapeo_clases, f)

# Calcular pesos de clase para lidiar con el desbalance severo
labels = []
for images, lbls in train_dataset.unbatch():
    labels.append(np.argmax(lbls.numpy()))
labels = np.array(labels)

pesos = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = {i: peso for i, peso in enumerate(pesos)}
print(f"Pesos asignados por clase para equilibrar: {class_weights}")

# Optimización y preprocesamiento de TF Data API
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.map(lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

val_dataset = val_dataset.map(lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

print("Construyendo el modelo con Transfer Learning (MobileNetV2)...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predicciones = Dense(len(clases), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predicciones)

# Métrica Recall para cáncer (Falsos negativos son inaceptables)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
)

callbacks = [
    EarlyStopping(monitor='val_recall', mode='max', patience=4, restore_best_weights=True, verbose=1),
    ModelCheckpoint('modelo_breast_cancer.keras', save_best_only=True, monitor='val_recall', mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
]

print("\n--- INICIANDO FASE 1: Entrenamiento de la cabecera ---")
history_frozen = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)

print("\n--- INICIANDO FASE 2: Fine Tuning ---")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
)

history_fine = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)

print("\n¡Entrenamiento Completado! El modelo ha sido guardado como 'modelo_breast_cancer.keras'.")

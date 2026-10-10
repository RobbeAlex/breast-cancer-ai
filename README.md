# AI in HealthCare: Breast Cancer Detection

![Stack](https://img.shields.io/badge/Stack-Python%20%7C%20TensorFlow/Keras-blue)
![Domain](https://img.shields.io/badge/Domain-Healthcare%20AI-green)

Este proyecto implementa un pipeline completo de Machine Learning / Deep Learning para la detección de cáncer de mama utilizando inteligencia artificial. Abarca desde el preprocesamiento de datos hasta el despliegue del modelo en una aplicación.

## 🌟 Características del Proyecto

*   **Preparación de Datos:** Scripts dedicados para la limpieza y estructuración del dataset médico (`preparar_dataset.py`).
*   **Entrenamiento del Modelo:** Construcción y entrenamiento de una red neuronal utilizando Keras/TensorFlow (`train_model.py`).
*   **Evaluación de Rendimiento:** Generación de métricas y visualizaciones, incluyendo matrices de confusión, para validar la precisión del modelo en diagnósticos médicos (`evaluate_model.py`).
*   **Despliegue (Serving):** Una aplicación lista para consumir el modelo entrenado y realizar inferencias sobre nuevos datos (`app.py`).

## 🛠️ Requisitos Previos

*   Python 3.8+
*   Entorno virtual recomendado (`.venv`)

## 🚀 Instalación y Uso

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/TU_USUARIO/breast-cancer-ai.git
    cd breast-cancer-ai
    ```

2.  **Crear y activar un entorno virtual:**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Flujo de Ejecución:**
    Puedes seguir el flujo lógico del proyecto ejecutando los scripts en este orden:
    *   `python preparar_dataset.py` (Procesamiento de datos)
    *   `python train_model.py` (Entrenamiento)
    *   `python evaluate_model.py` (Validación y métricas)
    *   `python app.py` (Servicio/Inferencia)

## 📂 Estructura Principal
*   `app.py`: Aplicación principal de servicio.
*   `train_model.py` / `evaluate_model.py`: Scripts de entrenamiento y evaluación.
*   `modelo_breast_cancer.keras`: Modelo pre-entrenado y guardado.
*   `matriz_confusion.png`: Resultados visuales de la evaluación del modelo.

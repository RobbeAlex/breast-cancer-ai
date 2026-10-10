import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import cv2

st.set_page_config(
    page_title="IA Cáncer de Mama",
    page_icon="🎗️",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #fff0f5; }
    h1 { color: #d81b60; }
    h3 { color: #ad1457; }
    .stAlert { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🎗️ Detección Temprana de Cáncer de Mama con IA")
st.write("Sube una mamografía. El sistema la dividirá en cuadrículas y buscará **microcalcificaciones** o **masas**.")

@st.cache_data
def cargar_diccionario_clases():
    try:
        with open('mapeo_clases.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_resource
def cargar_modelo():
    try:
        return tf.keras.models.load_model('modelo_breast_cancer.keras')
    except Exception as e:
        return None

CLASES_DICT = cargar_diccionario_clases()
model = cargar_modelo()

PATCH_SIZE = 224

def analizar_imagen(imagen_pil, umbral_confianza):
    # Convertir a arreglo numpy (escala de grises a 3 canales si es necesario)
    img_array = np.array(imagen_pil.convert('RGB'))
    h, w, _ = img_array.shape
    
    img_heatmap = img_array.copy()
    overlay = img_heatmap.copy()
    
    hallazgos = []
    
    progress_bar = st.progress(0)
    
    # Calcular cantidad de parches con superposición (stride = PATCH_SIZE // 2)
    stride = PATCH_SIZE // 2
    y_steps = list(range(0, h - PATCH_SIZE + 1, stride))
    x_steps = list(range(0, w - PATCH_SIZE + 1, stride))
    
    if not y_steps or not x_steps:
        # Imagen muy pequeña
        return imagen_pil, []
        
    total_patches = len(y_steps) * len(x_steps)
    current_patch = 0
    
    # Extraer y analizar parches en cuadrícula
    for y in y_steps:
        for x in x_steps:
            patch = img_array[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            # Convertir a float32 antes de preprocesar
            patch_tensor = np.expand_dims(patch, axis=0).astype(np.float32)
            patch_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(patch_tensor)
            
            prediccion = model.predict(patch_tensor, verbose=0)[0]
            indice_prediccion = np.argmax(prediccion)
            clase = CLASES_DICT[str(indice_prediccion)]
            confianza = prediccion[indice_prediccion]
            
            # Si no es normal y la confianza supera el umbral
            if clase != "Normal" and confianza > umbral_confianza:
                # Evitar agregar cuadros duplicados en la misma área (Non-Maximum Suppression simplificado)
                ya_detectado = any(abs(h_item["x"] - x) < stride and abs(h_item["y"] - y) < stride and h_item["clase"] == clase for h_item in hallazgos)
                if not ya_detectado:
                    hallazgos.append({"clase": clase, "confianza": confianza, "x": x, "y": y})
                    
                    color = (255, 0, 0) if clase == "Masas" else (255, 255, 0) # Rojo para masas, Amarillo para calcificaciones
                    cv2.rectangle(overlay, (x, y), (x+PATCH_SIZE, y+PATCH_SIZE), color, -1)
                
            current_patch += 1
            progress_bar.progress(current_patch / total_patches)
            
    # Mezclar el overlay con la imagen original
    alpha = 0.4
    img_resultado = cv2.addWeighted(overlay, alpha, img_heatmap, 1 - alpha, 0)
    
    # Dibujar bordes fuertes encima
    for h_item in hallazgos:
        color = (255, 0, 0) if h_item["clase"] == "Masas" else (255, 255, 0)
        cv2.rectangle(img_resultado, (h_item["x"], h_item["y"]), (h_item["x"]+PATCH_SIZE, h_item["y"]+PATCH_SIZE), color, 3)
        cv2.putText(img_resultado, f'{h_item["clase"]} ({h_item["confianza"]*100:.0f}%)', 
                    (h_item["x"]+5, h_item["y"]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
    progress_bar.empty()
    return Image.fromarray(img_resultado), hallazgos

if CLASES_DICT is None or model is None:
    st.error("⚠️ Falla de configuración: Faltan archivos del modelo. Entrena la red primero.")
else:
    uploaded_file = st.file_uploader("Sube tu mamografía (PGM, PNG, JPG)...", type=["pgm", "png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Redimensionar para no sobrecargar si es 4K, dejándolo en algo razonable
        image.thumbnail((1024, 1024)) 
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Mamografía Original")
            st.image(image, width=400)
            
        with col2:
            st.subheader("Análisis de Cuadrícula (IA)")
            umbral = st.slider("Sensibilidad (Umbral de confianza)", min_value=0.30, max_value=0.95, value=0.50, step=0.05, help="Disminuye este valor si la IA no detecta anomalías evidentes.")
            
            if st.button("Iniciar Escaneo Clínico"):
                with st.spinner("Dividiendo imagen y analizando tejidos..."):
                    img_procesada, hallazgos = analizar_imagen(image, umbral)
                    st.image(img_procesada, width=400)
                    
                    if len(hallazgos) == 0:
                        st.success("✅ **Diagnóstico:** No se encontraron anomalías significativas.")
                    else:
                        st.error(f"⚠️ **Alerta:** Se detectaron {len(hallazgos)} zona(s) sospechosa(s).")
                        for h in hallazgos:
                            st.write(f"- Detectado **{h['clase']}** con confianza del **{h['confianza']*100:.1f}%**")
                        
                    st.warning("⚠️ *Aviso: Este sistema de IA no sustituye un diagnóstico médico.*")

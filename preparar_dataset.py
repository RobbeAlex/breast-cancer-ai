import os
import tarfile
import requests
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

URL_MIAS = "http://peipa.essex.ac.uk/pix/mias/all-mias.tar.gz"
RAW_DIR = os.path.join("Raw Data", "all-mias")
TRAIN_DIR = "Training Data"
TEST_DIR = "Testing Data"
CLASSES = ["Normal", "Calcificaciones", "Masas"]
PATCH_SIZE = 224

def descargar_y_extraer():
    os.makedirs(RAW_DIR, exist_ok=True)
    info_path = os.path.join(RAW_DIR, "Info.txt")
    
    if not os.path.exists(info_path):
        print(f"ATENCION: No se encontro el archivo Info.txt en la carpeta '{RAW_DIR}'.")
        print("Dado que los servidores oficiales estan inestables, por favor:")
        print("1. Descarga el dataset manualmente desde: https://www.kaggle.com/datasets/kmader/mias-mammography")
        print(f"2. Extrae el archivo ZIP y copia todos los archivos (.pgm y el Info.txt) dentro de la carpeta '{RAW_DIR}'")
        print("3. Vuelve a ejecutar este script.")
        exit(1)
    else:
        print("Archivos base encontrados en Raw Data.")

def preparar_carpetas():
    # Limpiar carpetas anteriores para no mezclar
    for d in [TRAIN_DIR, TEST_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
            
    for base_dir in [TRAIN_DIR, TEST_DIR]:
        for clase in CLASSES:
            os.makedirs(os.path.join(base_dir, clase), exist_ok=True)

def procesar_imagenes():
    info_path = os.path.join(RAW_DIR, "Info.txt")
    
    anotaciones = []
    with open(info_path, 'r') as f:
        for linea in f.readlines():
            partes = linea.strip().split()
            # Valid lines start with mdb and have at least 3 parts
            if len(partes) >= 3 and partes[0].startswith("mdb"):
                anotaciones.append(partes)
                
    # 80% train, 20% test
    train_anotaciones, test_anotaciones = train_test_split(anotaciones, test_size=0.2, random_state=42)
    
    def extraer_parches(lista_anotaciones, base_dir):
        counts = {"Normal": 0, "Calcificaciones": 0, "Masas": 0}
        
        for partes in lista_anotaciones:
            img_name = partes[0] + ".pgm"
            img_path = os.path.join(RAW_DIR, img_name)
            
            if not os.path.exists(img_path):
                continue
                
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            h, w = img.shape
            clase_mias = partes[2]
            
            if clase_mias == 'NORM':
                categoria = "Normal"
                # Extract 2 random background patches (Solo 2 para mantener balance)
                for i in range(2):
                    x_center = np.random.randint(w//4, 3*w//4)
                    y_center = np.random.randint(h//4, 3*h//4)
                    x1 = max(0, x_center - PATCH_SIZE//2)
                    y1 = max(0, y_center - PATCH_SIZE//2)
                    
                    if x1 + PATCH_SIZE > w: x1 = w - PATCH_SIZE
                    if y1 + PATCH_SIZE > h: y1 = h - PATCH_SIZE
                    
                    patch = img[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE]
                    if patch.shape == (PATCH_SIZE, PATCH_SIZE):
                        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
                        cv2.imwrite(os.path.join(base_dir, categoria, f"{partes[0]}_patch_{i}.png"), patch_rgb)
                        counts[categoria] += 1
            else:
                categoria = "Calcificaciones" if clase_mias == 'CALC' else "Masas"
                
                if len(partes) >= 6:
                    try:
                        x_center = int(partes[4])
                        y_center = 1024 - int(partes[5]) 
                        
                        x1 = max(0, x_center - PATCH_SIZE//2)
                        y1 = max(0, y_center - PATCH_SIZE//2)
                        
                        if x1 + PATCH_SIZE > w: x1 = w - PATCH_SIZE
                        if y1 + PATCH_SIZE > h: y1 = h - PATCH_SIZE
                        if x1 < 0: x1 = 0
                        if y1 < 0: y1 = 0
                            
                        patch = img[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE]
                        if patch.shape == (PATCH_SIZE, PATCH_SIZE):
                            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
                            cv2.imwrite(os.path.join(base_dir, categoria, f"{partes[0]}_abnormal_0.png"), patch_rgb)
                            counts[categoria] += 1
                            
                            # === DATA AUGMENTATION OFFLINE (AUMENTO DE DATOS) ===
                            # Aplicamos solo al entrenamiento para no falsear las pruebas
                            if base_dir == TRAIN_DIR:
                                # 1. Espejo Horizontal
                                cv2.imwrite(os.path.join(base_dir, categoria, f"{partes[0]}_abnormal_fh.png"), cv2.flip(patch_rgb, 1))
                                counts[categoria] += 1
                                # 2. Espejo Vertical
                                cv2.imwrite(os.path.join(base_dir, categoria, f"{partes[0]}_abnormal_fv.png"), cv2.flip(patch_rgb, 0))
                                counts[categoria] += 1
                                # 3. Rotaciones
                                for angle in [90, 180, 270]:
                                    M = cv2.getRotationMatrix2D((PATCH_SIZE/2, PATCH_SIZE/2), angle, 1.0)
                                    rotated = cv2.warpAffine(patch_rgb, M, (PATCH_SIZE, PATCH_SIZE))
                                    cv2.imwrite(os.path.join(base_dir, categoria, f"{partes[0]}_abnormal_rot{angle}.png"), rotated)
                                    counts[categoria] += 1
                                # 4. Alteración de brillo para simular distintos rayos X
                                bright = cv2.convertScaleAbs(patch_rgb, alpha=1.2, beta=15)
                                cv2.imwrite(os.path.join(base_dir, categoria, f"{partes[0]}_abnormal_bright.png"), bright)
                                counts[categoria] += 1
                                
                    except ValueError:
                        pass
                        
        return counts

    print("Procesando y extrayendo recortes para Entrenamiento (con Data Augmentation)...")
    train_counts = extraer_parches(train_anotaciones, TRAIN_DIR)
    print("Procesando y extrayendo recortes para Prueba...")
    test_counts = extraer_parches(test_anotaciones, TEST_DIR)
    
    print("\nResumen Train:", train_counts)
    print("Resumen Test:", test_counts)

if __name__ == "__main__":
    print("=== INICIANDO PREPARACIÓN DE DATASET MIAS ===")
    descargar_y_extraer()
    preparar_carpetas()
    procesar_imagenes()
    print("=== PROCESO COMPLETADO ===")

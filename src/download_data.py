import os
import pandas as pd
from sklearn.datasets import fetch_openml

# 1. Asegurar que la carpeta data/raw exista
os.makedirs("data/raw", exist_ok=True)

print("Iniciando descarga de freMTPL2freq desde OpenML...")
# Carga el dataset de Frecuencia (ID: 41214)
freq = fetch_openml(data_id=41214, as_frame=True, parser="auto").frame
freq_path = "data/raw/freMTPL2freq.csv"
freq.to_csv(freq_path, index=False)
print(f"-> Archivo guardado exitosamente en: {freq_path}")

print("Iniciando descarga de freMTPL2sev desde OpenML...")
# Carga el dataset de Severidad (ID: 41215)
sev = fetch_openml(data_id=41215, as_frame=True, parser="auto").frame
sev_path = "data/raw/freMTPL2sev.csv"
sev.to_csv(sev_path, index=False)
print(f"-> Archivo guardado exitosamente en: {sev_path}")

print("\n¡Descarga completada con éxito!")
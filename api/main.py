"""
TPGN - Global Solution 2026/1 - Applied Computer Vision
API de inferencia de cobertura do solo (EuroSAT).

Como rodar (a partir da raiz do projeto, com o modelo ja treinado e salvo):
    uvicorn api.main:app --reload --port 8000
Depois abra http://localhost:8000/

Author: GUILHERME ROCHA BIANCHINI
"""
import io
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).parent
ROOT = BASE.parent

MODEL_PATH = ROOT / "models" / "best_model.keras"
CLASSES_PATH = ROOT / "models" / "classes.json"

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Modelo nao encontrado em {MODEL_PATH}. "
        "Rode o notebook (notebooks/treinamento.ipynb) ate a etapa de salvar o melhor modelo."
    )

model = tf.keras.models.load_model(MODEL_PATH)
CLASSES = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))

app = FastAPI(title="TPGN - EuroSAT Land Cover")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB").resize((64, 64))
    except Exception:
        raise HTTPException(status_code=400, detail="Arquivo de imagem invalido")

    arr = np.expand_dims(np.array(img), 0)  # a normalizacao (Rescaling) esta dentro do modelo
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "classe": CLASSES[idx],
        "confianca": float(probs[idx]),
        "probabilidades": {c: float(p) for c, p in zip(CLASSES, probs)},
    }


# Estaticos montados por ultimo para nao sombrear a rota /predict.
app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")

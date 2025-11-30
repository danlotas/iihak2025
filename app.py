from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

app = FastAPI()

# --------------------------------------
#  STATIC — подключаем фронтенд
# --------------------------------------
FRONTEND_DIR = os.path.join(os.getcwd(), "frontend")

# Все файлы (script.js, css, картинки)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Главная страница
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# --------------------------------------
#  CORS (разрешаем фронтенду обращаться к API)
# --------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # можешь заменить на свой URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------
#  МОДЕЛЬ
# --------------------------------------
MODEL_PATH = "./bert_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# --------------------------------------
# 1) Анализ одного текста
# --------------------------------------
@app.post("/predict_text")
async def predict_text_api(text: str = Form(...)):

    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        logits = model(**tokens).logits

    probs = F.softmax(logits, dim=1).numpy()[0]
    pred = int(np.argmax(probs))

    return {
        "prediction": pred,
        "probabilities": {
            "negative": float(probs[0]),
            "neutral": float(probs[1]),
            "positive": float(probs[2]),
        }
    }

# --------------------------------------
# 2) Анализ CSV
# --------------------------------------
@app.post("/predict_csv")
async def predict_csv_api(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    if "text" not in df.columns:
        return {"error": "CSV must contain 'text' column"}

    preds = []
    negs = []
    neuts = []
    poss = []

    for t in df["text"]:
        tokens = tokenizer(
            str(t),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        with torch.no_grad():
            logits = model(**tokens).logits

        probs = F.softmax(logits, dim=1).numpy()[0]

        preds.append(int(np.argmax(probs)))
        negs.append(float(probs[0]))
        neuts.append(float(probs[1]))
        poss.append(float(probs[2]))

    df["pred"] = preds
    df["prob_neg"] = negs
    df["prob_neu"] = neuts
    df["prob_pos"] = poss

    return df.to_dict(orient="records")

# --------------------------------------
# 3) Оценка качества модели (F1)
# --------------------------------------
@app.post("/evaluate_csv")
async def evaluate_csv_api(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    if "text" not in df.columns or "label" not in df.columns:
        return {"error": "CSV must contain 'text' and 'label'"}

    preds = []
    for t in df["text"]:
        tokens = tokenizer(
            str(t),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        with torch.no_grad():
            logits = model(**tokens).logits

        preds.append(int(np.argmax(logits)))

    df["pred"] = preds

    macro_f1 = f1_score(df["label"], df["pred"], average="macro")

    return {"macro_f1": float(macro_f1)}

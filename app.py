import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =====================================================
# 🔧 ОПТИМИЗАЦИЯ ПОД RENDER FREE
# =====================================================
torch.set_num_threads(1)  # сильно снижает нагрузку CPU
MODEL_PATH = "./bert_model"

# =====================================================
# 🚀 FASTAPI APP
# =====================================================
app = FastAPI()

# Разрешаем фронтенд
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# 📦 ПОДКЛЮЧАЕМ СТАТИКУ И index.html
# =====================================================

# Монтируем директорию frontend/ как статик
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Отдаем главную страницу
@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


# =====================================================
# 📚 ЗАГРУЗКА МОДЕЛИ (оптимизировано)
# =====================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float32
)

model.eval()


# =====================================================
# 1️⃣ API — анализ одного текста
# =====================================================
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


# =====================================================
# 2️⃣ API — пакетный анализ CSV
# =====================================================
@app.post("/predict_csv")
async def predict_csv_api(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    if "text" not in df.columns:
        return {"error": "CSV must contain 'text' column"}

    preds, negs, neuts, poss = [], [], [], []

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


# =====================================================
# 3️⃣ API — валидация модели по CSV
# =====================================================
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

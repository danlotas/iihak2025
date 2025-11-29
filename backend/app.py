from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score
import torch.nn.functional as F
import uvicorn

app = FastAPI()

MODEL_PATH = "./bert_model"

# ---- Load model ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


# ---- 1) Predict single text ----
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
        "text": text,
        "prediction": pred,
        "probabilities": {
            "negative": float(probs[0]),
            "neutral": float(probs[1]),
            "positive": float(probs[2]),
        }
    }


# ---- 2) Predict whole CSV ----
@app.post("/predict_csv")
async def predict_csv_api(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    if "text" not in df.columns:
        return {"error": "CSV must contain 'text' column"}

    preds = []
    prob0 = []
    prob1 = []
    prob2 = []

    for text in df["text"]:
        tokens = tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        with torch.no_grad():
            logits = model(**tokens).logits
        probs = F.softmax(logits, dim=1).numpy()[0]

        preds.append(int(np.argmax(probs)))
        prob0.append(float(probs[0]))
        prob1.append(float(probs[1]))
        prob2.append(float(probs[2]))

    df["pred"] = preds
    df["prob_neg"] = prob0
    df["prob_neu"] = prob1
    df["prob_pos"] = prob2

    out_file = "predicted.csv"
    df.to_csv(out_file, index=False)

    return {"status": "ok", "download": out_file}


# ---- 3) Evaluate CSV with labels ----
@app.post("/evaluate_csv")
async def evaluate_csv_api(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    if "text" not in df.columns or "label" not in df.columns:
        return {"error": "CSV must contain 'text' and 'label'"}

    preds = []
    for text in df["text"]:
        tokens = tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        with torch.no_grad():
            logits = model(**tokens).logits
        preds.append(int(logits.argmax()))

    df["pred"] = preds
    macro_f1 = f1_score(df["label"], df["pred"], average="macro")

    return {"macro_f1": macro_f1}


# ---- Local run ----
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

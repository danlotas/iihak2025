from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd

app = FastAPI()

# Раздаём фронт из папки "frontend" в корне репозитория
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")


def predict_fake(text: str) -> int:
    """Заглушка — всегда нейтральный класс (1). Потом замените на реальную модель."""
    return 1


@app.post("/predict")
async def predict_text(payload: dict):
    text = payload.get("text", "")
    label = predict_fake(text)
    return {"label": label}


@app.post("/upload_csv")
async def predict_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    if "text" not in df.columns:
        return JSONResponse({"error": "CSV must contain 'text' column"}, status_code=400)

    df["label"] = df["text"].apply(predict_fake)

    out_path = "result.csv"
    df.to_csv(out_path, index=False)
    return FileResponse(out_path, filename="result.csv")

import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score


# ---------- Локальная модель ----------
MODEL_PATH = "bert_model"   # <<< ваша модель

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

label_map = {0: "негативный", 1: "нейтральный", 2: "позитивный"}


# ---------- Функция предсказания ----------
def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    return pred, label_map[pred]


# ------------------- UI -------------------
st.title("Анализатор тональности отзывов")

tabs = st.tabs(["Вставить отзыв", "Загрузить CSV", "Валидация"])


# -------------------------------------------------------------------------
# 1. Вставить отзыв
# -------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Введите текст отзыва:")
    text = st.text_area("Текст", height=150)

    if st.button("Проверить тональность"):
        if text.strip():
            cls, cls_text = predict(text)
            st.write(f"**Предсказание модели:** {cls} — {cls_text}")
        else:
            st.warning("Введите текст.")


# -------------------------------------------------------------------------
# 2. Загрузить CSV и разметить
# -------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Загрузите CSV с колонками index, text")

    file = st.file_uploader("CSV-файл", type=["csv"])
    if file:
        df = pd.read_csv(file)

        if "text" not in df.columns:
            st.error("Ошибка: нет столбца 'text'")
        else:
            if st.button("Классифицировать"):
                df["prediction"] = df["text"].apply(lambda x: predict(str(x))[0])
                st.write(df.head())

                st.download_button(
                    label="Скачать результат",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv"
                )


# -------------------------------------------------------------------------
# 3. Валидация модели
# -------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Загрузите CSV с колонками index, text, label")

    val_file = st.file_uploader("CSV-файл", type=["csv"], key="val")
    if val_file:
        df = pd.read_csv(val_file)

        if not {"text", "label"} <= set(df.columns):
            st.error("Ошибка: столбцы 'text' и 'label' должны присутствовать.")
        else:
            if st.button("Проверить качество модели"):
                df["pred"] = df["text"].apply(lambda x: predict(str(x))[0])

                correct = (df["pred"] == df["label"]).sum()
                total = len(df)
                f1 = f1_score(df["label"], df["pred"], average="macro")

                st.write(f"Совпадений: **{correct}/{total}**")
                st.write(f"Macro-F1: **{f1:.4f}**")

                st.write(df.head())

                st.download_button(
                    label="Скачать файл с предсказаниями",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="validation_results.csv",
                    mime="text/csv"
                )

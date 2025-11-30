# ОРП
# Описание проекта
Для запуска проекта: после установки необходимых библеотек ввести в консоль находясь в директори проекта: streamlit run app.py. SORRY не успевали доделать, поэтому так

Проект представляет собой десктоп приложение, в котором реализована работа модели, определяющей тональность текста. В приложении имеется 3 кнопки: оценить 1 отзыв, загрузить csv файл с отзывами и валидация для проверки работы модели. При разработке приложения использовались библеотеки 
import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score

так же в репозитории есть файл - mainFINAL.ipynb - это скрипт который обучал модель.
# Ссылка на видео
https://drive.google.com/file/d/1uwp4B8ABw37qiWvGrECTN65N9R8-Y-uC/view?usp=sharing
# Состав команды
Денисов Артем - фронтенд
Юдина Валерия - бэкенд и аналитика
Левашов Юрий - ML

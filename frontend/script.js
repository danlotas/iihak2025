// ========== URL вашего FastAPI backend ==========
const API_URL = window.location.origin; 
// пример: https://iihak2025.onrender.com


// -----------------------------
// 1) Анализ одного текста
// -----------------------------
async function analyzeTextAPI(text) {
    const formData = new FormData();
    formData.append("text", text);

    const response = await fetch(`${API_URL}/predict_text`, {
        method: "POST",
        body: formData
    });

    return await response.json();
}


// -----------------------------
// 2) Анализ CSV
// -----------------------------
async function analyzeCSVAPI(file) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_URL}/predict_csv`, {
        method: "POST",
        body: formData
    });

    return await response.json();
}


// -----------------------------
// 3) Оценка модели по CSV
// -----------------------------
async function evaluateCSVAPI(file) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_URL}/evaluate_csv`, {
        method: "POST",
        body: formData
    });

    return await response.json();
}


// =====================================
// ======== ЛОГИКА ИНТЕРФЕЙСА ==========
// =====================================

// вкладки
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', function () {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

        this.classList.add('active');
        const tabId = this.getAttribute('data-tab');
        document.getElementById(`${tabId}-tab`).classList.add('active');

        resetResults();
    });
});


// примеры
document.querySelectorAll('.example-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.getElementById('text-input').value = btn.dataset.text;
    });
});


// =============================
// КНОПКА 1 — анализ одиночного текста
// =============================
document.getElementById('analyze-btn').addEventListener('click', async () => {
    const text = document.getElementById('text-input').value.trim();
    if (!text) return alert("Введите текст!");

    const btn = document.getElementById('analyze-btn');
    btn.disabled = true;
    btn.textContent = "Анализируем...";

    const result = await analyzeTextAPI(text);

    btn.disabled = false;
    btn.textContent = "Анализировать тональность";

    displaySingleResult(result);
});


// =============================
// КНОПКА 2 — загрузка CSV
// =============================
document.getElementById('batch-analyze-btn').addEventListener('click', async () => {
    const file = document.getElementById('batch-file').files[0];
    if (!file) return alert("Выберите CSV!");

    const btn = document.getElementById('batch-analyze-btn');
    btn.disabled = true;
    btn.textContent = "Загружаем...";

    const result = await analyzeCSVAPI(file);

    btn.disabled = false;
    btn.textContent = "Запустить пакетный анализ";

    displayBatchResult(result);
});


// =============================
// КНОПКА 3 — валидация CSV
// =============================
document.getElementById('validate-btn').addEventListener('click', async () => {
    const file = document.getElementById('validation-file').files[0];
    if (!file) return alert("Выберите CSV!");

    const btn = document.getElementById('validate-btn');
    btn.disabled = true;
    btn.textContent = "Вычисляем...";

    const result = await evaluateCSVAPI(file);

    btn.disabled = false;
    btn.textContent = "Оценить модель (macro-F1)";

    displayMetrics(result);
});


// ====================================
// ======== ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ ====
// ====================================

// --- 1. одиночный текст ---
function displaySingleResult(res) {
    const sentimentScore = document.getElementById("sentiment-score");
    const sentimentCategory = document.getElementById("sentiment-category");
    const progressFill = document.getElementById("progress-fill");
    const confidenceIndicator = document.getElementById("confidence-indicator");

    const probs = res.probabilities;
    const pred = res.prediction;

    const classNames = ["Негатив", "Нейтрально", "Позитив"];

    sentimentCategory.textContent = classNames[pred];

    // берем максимальную вероятность
    const maxProb = Math.max(...Object.values(probs));

    sentimentScore.textContent = maxProb.toFixed(2);
    progressFill.style.width = `${maxProb * 100}%`;
    confidenceIndicator.textContent = `Уверенность модели: ${(maxProb * 100).toFixed(1)}%`;

    document.getElementById("result-section").style.display = "block";
}


// --- 2. пакетный CSV анализ ---
function displayBatchResult(results) {
    document.getElementById("token-list").innerHTML =
        `<p>Обработано строк: ${results.length}</p>`;

    document.getElementById("download-btn").style.display = "inline-block";
    document.getElementById("result-section").style.display = "block";
}


// --- 3. валидация модели ---
function displayMetrics(result) {
    document.getElementById("f1-score").textContent = result.macro_f1.toFixed(3);

    document.getElementById("metrics-grid").style.display = "grid";
    document.getElementById("result-section").style.display = "block";
}


// --- reset ---
function resetResults() {
    document.getElementById('result-section').style.display = 'none';
    document.getElementById('metrics-grid').style.display = 'none';
    document.getElementById('download-btn').style.display = 'none';
}

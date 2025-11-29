function sendText() {
    const text = document.getElementById("textInput").value;

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
    })
    .then(r => r.json())
    .then(data => {
        document.getElementById("textResult").innerText =
            "Predicted label: " + data.label;
    });
}

function sendCSV() {
    const fileInput = document.getElementById("fileInput");
    const form = new FormData();
    form.append("file", fileInput.files[0]);

    fetch("/upload_csv", {
        method: "POST",
        body: form
    })
    .then(r => r.blob())
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "result.csv";
        document.body.appendChild(a);
        a.click();
        a.remove();
    });
}

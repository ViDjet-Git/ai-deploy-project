from flask import Flask, request, jsonify, Response
import requests
import os

app = Flask(__name__)

AI_API_URL = "http://host.docker.internal:8080"
TRAINER_URL = "http://host.docker.internal:8090"
MONITOR_URL = "http://host.docker.internal:8070"


@app.route("/", methods=["GET"])
def index():
    # максимально простий HTML, без шаблонів
    html = """
    <!DOCTYPE html>
    <html lang="uk">
    <head>
        <meta charset="UTF-8" />
        <title>AI Deploy Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 20px auto; }
            h1 { margin-bottom: 0.2rem; }
            .card { border: 1px solid #ccc; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
            .row { display: flex; gap: 16px; }
            .row > .card { flex: 1; }
            button { padding: 8px 16px; cursor: pointer; }
            #prediction-result, #train-result, #status-result { white-space: pre-wrap; font-family: monospace; }
            input[type="file"] { margin-bottom: 8px; }
        </style>
    </head>
    <body>
        <h1>AI Deploy Dashboard</h1>
        <p>Проста панель для роботи з моделлю: інференс, тренування, моніторинг.</p>

        <div class="row">
            <div class="card">
                <h2>1. Класифікація зображення</h2>
                <form id="predict-form">
                    <input type="file" name="file" id="file-input" accept="image/*" required />
                    <br/>
                    <button type="submit">Надіслати на інференс</button>
                </form>
                <h3>Результат:</h3>
                <div id="prediction-result">—</div>
            </div>

            <div class="card">
                <h2>2. Запуск тренування</h2>
                <form id="train-form">
                    <label>Epochs: <input type="number" name="epochs" id="epochs" value="1" min="1" /></label><br/>
                    <label>Batch size: <input type="number" name="batch_size" id="batch_size" value="64" min="1" /></label><br/>
                    <label>LR: <input type="number" step="0.0001" name="lr" id="lr" value="0.001" /></label><br/><br/>
                    <button type="submit">Запустити тренування</button>
                </form>
                <h3>Статус тренування:</h3>
                <div id="train-result">—</div>
            </div>
        </div>

        <div class="card">
            <h2>3. Статус системи</h2>
            <button id="refresh-status">Оновити статус</button>
            <div id="status-result">—</div>
        </div>

        <script>
            const predictForm = document.getElementById("predict-form");
            const trainForm = document.getElementById("train-form");
            const predictionResult = document.getElementById("prediction-result");
            const trainResult = document.getElementById("train-result");
            const statusResult = document.getElementById("status-result");
            const refreshStatusBtn = document.getElementById("refresh-status");

            predictForm.addEventListener("submit", async (e) => {
                e.preventDefault();
                predictionResult.textContent = "Виконується запит...";
                const fileInput = document.getElementById("file-input");
                if (!fileInput.files.length) {
                    predictionResult.textContent = "Будь ласка, оберіть файл.";
                    return;
                }
                const formData = new FormData();
                formData.append("file", fileInput.files[0]);
                try {
                    const resp = await fetch("/api/predict", {
                        method: "POST",
                        body: formData
                    });
                    const data = await resp.json();
                    predictionResult.textContent = JSON.stringify(data, null, 2);
                } catch (err) {
                    predictionResult.textContent = "Помилка: " + err;
                }
            });

            trainForm.addEventListener("submit", async (e) => {
                e.preventDefault();
                trainResult.textContent = "Відправляю запит на тренування...";
                const payload = {
                    epochs: parseInt(document.getElementById("epochs").value || "1"),
                    batch_size: parseInt(document.getElementById("batch_size").value || "64"),
                    lr: parseFloat(document.getElementById("lr").value || "0.001")
                };
                try {
                    const resp = await fetch("/api/train", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload)
                    });
                    const data = await resp.json();
                    trainResult.textContent = JSON.stringify(data, null, 2);
                } catch (err) {
                    trainResult.textContent = "Помилка: " + err;
                }
            });

            async function refreshStatus() {
                statusResult.textContent = "Оновлення...";
                try {
                    const resp = await fetch("/api/status");
                    const data = await resp.json();
                    statusResult.textContent = JSON.stringify(data, null, 2);
                } catch (err) {
                    statusResult.textContent = "Помилка: " + err;
                }
            }

            refreshStatusBtn.addEventListener("click", refreshStatus);
        </script>
    </body>
    </html>
    """
    return Response(html, mimetype="text/html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    files = {"file": (file.filename, file.stream, file.mimetype)}
    try:
        resp = requests.post(f"{AI_API_URL}/predict", files=files, timeout=10)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def api_train():
    data = request.get_json() or {}
    try:
        resp = requests.post(f"{TRAINER_URL}/train", json=data, timeout=10)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def api_status():
    result = {}
    try:
        r_api = requests.get(f"{AI_API_URL}/health", timeout=5)
        result["ai_api"] = r_api.json()
    except Exception as e:
        result["ai_api"] = {"error": str(e)}

    try:
        r_tr = requests.get(f"{TRAINER_URL}/status", timeout=5)
        result["ai_trainer"] = r_tr.json()
    except Exception as e:
        result["ai_trainer"] = {"error": str(e)}

    try:
        r_mon = requests.get(f"{MONITOR_URL}/metrics_json", timeout=5)
        result["monitor"] = r_mon.json()
    except Exception as e:
        result["monitor"] = {"error": str(e)}

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)

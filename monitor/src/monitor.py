# monitor/src/monitor.py
import time
import json
import os
import requests
import psutil
import threading
from flask import Flask, jsonify

app = Flask(__name__)
os.makedirs("/logs", exist_ok=True)
METRICS_FILE = "/logs/metrics.json"

SERVICES = {
    "ai_api": "http://host.docker.internal:8080/health",
    "ai_trainer": "http://host.docker.internal:8090/metrics"
}

# безпечний локер для роботи з файлами
file_lock = threading.Lock()

def get_service_status(name, url):
    try:
        r = requests.get(url, timeout=3)
        if r.ok:
            try:
                payload = r.json()
            except:
                payload = None
                
            return {"ok": True, "status_code": r.status_code, "data": payload}
        return {"ok": False, "status_code": r.status_code, "error": "Bad response"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def collect_metrics():
    """Збір системних та сервісних метрик."""
    metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_usage": psutil.cpu_percent(),
        "ram_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "services": {}
    }

    for name, url in SERVICES.items():
        metrics["services"][name] = get_service_status(name, url)

    return metrics


def write_metrics(new_metrics):
    """Запис даних у файл з локером."""
    with file_lock:
        data = []
        if os.path.exists(METRICS_FILE):
            try:
                with open(METRICS_FILE, "r") as f:
                    data = json.load(f)
            except:
                data = []

        data.append(new_metrics)
        data = data[-200:]  # зберігаємо останні 200 записів

        with open(METRICS_FILE, "w") as f:
            json.dump(data, f, indent=4)


def background_loop():
    """Цикл періодичного збору метрик."""
    while True:
        m = collect_metrics()
        write_metrics(m)
        time.sleep(15)

@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Повертає останні 20 метрик."""
    if not os.path.exists(METRICS_FILE):
        return jsonify([])

    with file_lock:
        try:
            with open(METRICS_FILE, "r") as f:
                data = json.load(f)
        except:
            return jsonify([])

    return jsonify(data[-20:])


@app.route("/metrics_json", methods=["GET"])
def metrics_json():
    """Формат для Prometheus/Grafana."""
    m = collect_metrics()
    return jsonify(m)


@app.route("/health", methods=["GET"])
def health():
    """Health-check самого монітору."""
    return jsonify({"monitor_ok": True})


if __name__ == "__main__":
    t = threading.Thread(target=background_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=8070)

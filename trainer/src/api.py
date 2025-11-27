# trainer/src/api.py
import threading
import time
import os
import json
import logging
from flask import Flask, jsonify, request
from state import get_state, set_state, reset_state
from train import train_one_run
from registry import list_models

app = Flask(__name__)
os.makedirs("/models", exist_ok=True)
os.makedirs("/trainer_logs", exist_ok=True)
METADATA_PATH = "/trainer_logs/training_metadata.json"

logging.basicConfig(
    filename="/trainer_logs/trainer_api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

_train_thread = None
_thread_lock = threading.Lock()

def load_metadata():
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_metadata(entry):
    data = load_metadata()
    data.append(entry)
    with open(METADATA_PATH, "w") as f:
        json.dump(data, f, indent=4)

def background_train(epochs, batch_size, lr):
    start_time = time.time()
    try:
        logging.info(f"Training started: epochs={epochs}, batch_size={batch_size}, lr={lr}")
        result = train_one_run(epochs=epochs, batch_size=batch_size, lr=lr)
        duration_min = round((time.time() - start_time) / 60, 2)

        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "accuracy": result.get("accuracy"),
            "saved_model": result.get("model_name"),
            "duration_min": duration_min
        }
        save_metadata(entry)
        logging.info(f"Training finished successfully: {entry}")

    except Exception as e:
        logging.exception("Training error")
        set_state(status="failed", last_error=str(e))

@app.route("/train", methods=["POST"])
def start_train():
    global _train_thread
    with _thread_lock:
        st = get_state()
        if st["status"] == "training":
            return jsonify({"status":"busy", "message":"Training already running"}), 409
        # parameters optional in JSON
        data = request.get_json() or {}
        epochs = int(data.get("epochs", 1))
        batch_size = int(data.get("batch_size", 64))
        lr = float(data.get("lr", 1e-3))
        reset_state()
        set_state(status="training", started_at=time.time())
        _train_thread = threading.Thread(target=background_train, args=(epochs, batch_size, lr), daemon=True)
        _train_thread.start()

        logging.info(f"Training launched via API: epochs={epochs}, batch_size={batch_size}, lr={lr}")
        return jsonify({"status":"started", "epochs":epochs}), 202

@app.route("/status", methods=["GET"])
def status():
    return jsonify(get_state())

@app.route("/metrics", methods=["GET"])
def metrics():
    st = get_state()
    return jsonify({
        "accuracy": st.get("accuracy"),
        "loss": st.get("loss"),
        "epoch": st.get("epoch"),
        "progress": st.get("progress"),
        "status": st.get("status")
    })

@app.route("/models", methods=["GET"])
def models():
    return jsonify(list_models())

@app.route("/metadata", methods=["GET"])
def metadata():
    return jsonify(load_metadata())

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "AI Trainer API active",
        "gpu_enabled": os.environ.get("CUDA_VISIBLE_DEVICES", "auto"),
        "endpoints": ["/train", "/status", "/metrics", "/models", "/metadata"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8090, threaded=True)

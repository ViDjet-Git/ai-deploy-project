import torch
import os
import logging
import json
from time import time
from flask import Flask, jsonify, request
from src.model import predict_image_bytes, load_model, MODEL_PATH
import threading

DEPLOY_COLOR = os.getenv("DEPLOY_COLOR", "unknown")
METADATA_PATH = "/models/training_metadata.json"


#Логування
#docker exec -it ai-deploy-project-ai_api-1 tail -f logs/api.log
os.makedirs("/logs", exist_ok=True)
logging.basicConfig(filename="/logs/api.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

app = Flask(__name__)
start_time = time()

# model_loaded flag (reflects if model is loaded in memory)
model_loaded = load_model()

def get_model_version():
    try:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r") as f:
                data = json.load(f)
            # ми ж там зберігали "model_name"
            return data.get("model_name")
    except Exception as e:
        logging.warning(f"Failed to read model metadata: {e}")
    return None

@app.before_request
def log_request_info():
    logging.info(f"Incoming request: {request.method} {request.path} from {request.remote_addr}")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "AI API is running!", "model_path": MODEL_PATH})

@app.route("/status", methods=["GET"])
def status():
    gpu_available = torch.cuda.is_available()
    return jsonify({
    "status": "running",
    "gpu_available": gpu_available,
    "torch_version": torch.__version__
    })

@app.route("/health", methods=["GET"])
def health():
    """
    Простий healthcheck: повертає статус сервера, час роботи, та чи модель завантажена.
    Код відповіді 200 => OK, 500 => problem.
    """
    uptime = int(time() - start_time)
    version = get_model_version()
    status = {"status": "ok" if model_loaded else "degraded",
               "uptime_seconds": uptime,
               "model_loaded": bool(model_loaded),
               "deploy_color": DEPLOY_COLOR,
               "model_version": version}
    return jsonify(status), (200 if model_loaded else 500)

@app.route("/predict", methods=["POST"])
def predict_route():
    global model_loaded
    if "file" not in request.files:
        return jsonify({"error": "No input provide"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    try:
        #Запис логів
        logging.info(f"Received file: {file.filename}, size: {len(file.read())} bytes")
        file.seek(0)

        #Робота ШІ
        img_bytes = file.read()
        result = predict_image_bytes(img_bytes)
        if "error" in result:
            return jsonify(result), 400
        logging.info(f"Prediction result: {result}")
        return jsonify(result)
    except Exception as e:
        logging.exception("Error during prediction")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/reload", methods=["POST"])
def reload_route():
    global model_loaded
    try:
        ok = load_model(force_reload=True)
        model_loaded = bool(ok)
        logging.info(f"/reload called - model_loaded={model_loaded}")
        return jsonify({"reloaded": model_loaded})
    except Exception as e:
        logging.exception("Error loading model")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

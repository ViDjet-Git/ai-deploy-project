# trainer/src/registry.py
import os
from datetime import datetime
import shutil

MODELS_DIR = "/models"  # will be mounted as volume from docker-compose

os.makedirs(MODELS_DIR, exist_ok=True)

def save_model_state(state_dict):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    name = f"model_v{ts}.pth"
    path = os.path.join(MODELS_DIR, name)
    # caller should save torch.save(state_dict, path)
    return path, name

def set_latest(path):
    latest = os.path.join(MODELS_DIR, "model_latest.pth")
    try:
        # atomic replace
        if os.path.exists(latest):
            os.remove(latest)
        shutil.copy2(path, latest)
    except Exception:
        shutil.copy2(path, latest)  # fallback

def list_models():
    return sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")])

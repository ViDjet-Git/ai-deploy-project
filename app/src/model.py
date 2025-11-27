import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
import io

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/models/model_latest.pth"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Преобразування вхідного зображення
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# Lazy-loader model holder
_model = None

def _build_model(num_classes=10):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)

def load_model(force_reload=False):
    """
    Load model_latest.pth into memory. If already loaded and force_reload False, do nothing.
    Returns True if model loaded successfully.
    """
    global _model
    if _model is not None and not force_reload:
        return True
    if not os.path.exists(MODEL_PATH):
        return False
    model = _build_model(num_classes=len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    _model = model
    return True

def predict_image_bytes(image_bytes):
    """
    Returns dict: {"class": <name>, "confidence": <float>} or {"error": ...}
    """
    global _model
    if _model is None:
        ok = load_model()
        if not ok:
            return {"error": f"No model at {MODEL_PATH}"}

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        return {"error": "Invalid image file"}
    
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = _model(x)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        conf, idx = torch.max(probs, 0)
        return {"class": CLASS_NAMES[idx.item()], "confidence": round(conf.item(), 4)}

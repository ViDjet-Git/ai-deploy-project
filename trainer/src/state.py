# trainer/src/state.py
import threading
from time import time

_lock = threading.Lock()

state = {
    "status": "idle",    # idle / training / done / failed / cancelled
    "progress": 0.0,     # 0..100
    "epoch": 0,
    "accuracy": None,
    "loss": None,
    "started_at": None,
    "finished_at": None,
    "last_error": None
}

def set_state(**kwargs):
    with _lock:
        state.update(kwargs)

def get_state():
    with _lock:
        return dict(state)

def reset_state():
    with _lock:
        state.update({
            "status": "idle",
            "progress": 0.0,
            "epoch": 0,
            "accuracy": None,
            "loss": None,
            "started_at": None,
            "finished_at": None,
            "last_error": None
        })

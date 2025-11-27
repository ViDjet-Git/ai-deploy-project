# trainer/src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import time
import os
import json
from registry import save_model_state, set_latest
from state import set_state, get_state

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Використовується пристрій: {DEVICE}")

METADATA_PATH = "/models/training_metadata.json"

def build_model(num_classes=10):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    # replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)

def train_one_run(epochs=1, batch_size=64, lr=1e-3):
    set_state(status="training", progress=0.0, started_at=time.time(), last_error=None)
    start_time = time.time()
    try:
        # transforms for CIFAR10
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

        # download dataset
        dataset = datasets.CIFAR10(root="/data", train=True, download=True, transform=transform)
        val_size = int(0.1 * len(dataset))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        model = build_model(num_classes=10)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        total_steps = epochs * len(train_loader)
        step = 0

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(train_loader, 1):
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                step += 1
                # update progress percent
                progress = min(100.0, (step / total_steps) * 100.0)
                set_state(progress=round(progress, 2), epoch=epoch, loss=round(running_loss / batch_idx, 4))

            # validation pass
            model.eval()
            correct, total = 0, 0
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(DEVICE, non_blocking=True)
                    targets = targets.to(DEVICE, non_blocking=True)

                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            accuracy = correct / total
            # update state with epoch accuracy
            set_state(accuracy=round(accuracy, 4), loss=round(val_loss / len(val_loader), 4), epoch=epoch)
            print(f"[EPOCH {epoch}] loss={val_loss:.4f}, acc={accuracy:.4f}")

        # Save model
        save_path, name = save_model_state(None)  # get path
        # actually save state_dict
        torch.save(model.state_dict(), save_path)
        set_latest(save_path)

        try:
            import request
            ai_api_url = "http://ai_api:8080/reload"
            for _ in range(3):
                try:
                    resp = request.post(ai_api_url, timeout = 5)
                    if resp.ok:
                        print("[INFO] Reload triggered on ai_api")
                        break
                except Exception as e:
                    print(f"[WARN] Reload attempt failed: {e}")
        except Exception:
            pass

        total_time = round(time.time() - start_time, 2)
        metadata = {
            "model_name": name,
            "accuracy": round(accuracy, 4),
            "loss": round(val_loss, 4),
            "epochs": epochs,
            "device": str(DEVICE),
            "train_time_sec": total_time,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=4)
        
        print(f"[INFO] Модель збереження як {name}")
        print(f"[INFO] Метадані записано у training_metadata.json")

        set_state(status="done", progress=100.0, finished_at=time.time(), last_error=None)
        return metadata

    except Exception as e:
        import traceback
        traceback.print_exc()
        set_state(status="failed", last_error=str(e))
        raise

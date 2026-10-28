"""Real MedMNIST training entry point used by all backends.

Trains a small CNN/ResNet on a MedMNIST 2D dataset and writes a JSON result
record: config fingerprint, test accuracy, per-class AUC where applicable,
loss curves, environment info. Importable (run_training) and CLI-runnable.

Novelty note: this is deliberately the *boring* part — a standardized,
auditable training primitive. The novelty of the project lives in the
agents that decide *what* to train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _seed_all(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)


def _get_dataset(name: str, root: str, augment: str, image_size: int):
    import medmnist
    from torchvision import transforms

    name = name.lower()
    info = medmnist.INFO[name]
    n_channels = info["n_channels"]
    base = [transforms.Resize((image_size, image_size)),
            transforms.ToTensor()]
    if n_channels == 1:
        # replicate grayscale -> 3ch for standard backbones
        base.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    base.append(transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3))
    if augment == "basic":
        tfm = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5),
             transforms.RandomRotation(10)] + base
        )
    elif augment == "none":
        tfm = transforms.Compose(base)
    else:
        raise ValueError(f"unknown augment: {augment}")

    cls = getattr(medmnist, info["python_class"])
    train = cls(split="train", root=root, transform=tfm, download=True, size=28)
    test = cls(split="test", root=root, transform=transforms.Compose(base),
               download=True, size=28)
    return train, test, info


def _auc_from_probs(P, Y):
    """Per-column ROC AUC from probability/label arrays.

    Columns whose labels are all one class are skipped (AUC undefined
    there; newer scikit-learn returns NaN with a warning instead of
    raising, so the check is explicit). Returns a list of floats,
    possibly empty. Pure function — unit-tested without any training.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    aucs = []
    for c in range(P.shape[1]):
        col = Y[:, c]
        if np.unique(col).size < 2:
            continue
        try:
            v = float(roc_auc_score(col, P[:, c]))
        except ValueError:
            continue
        if np.isfinite(v):
            aucs.append(v)
    return aucs


def _get_model(arch: str, num_classes: int, in_channels: int = 3):
    import torch
    import torchvision

    if arch == "resnet18":
        m = torchvision.models.resnet18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    elif arch == "tiny-cnn":
        m = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes),
        )
    else:
        raise ValueError(f"unknown arch: {arch}")
    return m


def run_training(config: Dict[str, Any], data_root: str,
                 out_path: str, threads: int = 4) -> Dict[str, Any]:
    import torch

    t_start = time.time()
    torch.set_num_threads(threads)
    seed = int(config["seed"])
    _seed_all(seed)

    dataset = str(config["dataset"])
    arch = str(config["arch"])
    augment = str(config.get("augment", "none"))
    epochs = int(config.get("epochs", 5))
    batch_size = int(config.get("batch_size", 128))
    lr = float(config.get("lr", 1e-3))
    image_size = int(config.get("image_size", 28))

    train_ds, test_ds, info = _get_dataset(dataset, data_root, augment, image_size)
    task = info["task"]
    n_classes = len(info["label"]) if isinstance(info["label"], dict) else int(info["label"])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _get_model(arch, n_classes).to(device)
    if task in ("multi-label, binary-class",):
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    model.train()
    for epoch in range(epochs):
        tot_loss, n_batches = 0.0, 0
        for x, y in train_loader:
            x = x.to(device)
            if task == "multi-label, binary-class":
                y = y.to(device).float()
            else:
                y = y.to(device).squeeze().long()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tot_loss += float(loss.item())
            n_batches += 1
        history.append({"epoch": epoch, "train_loss": tot_loss / max(1, n_batches)})

    # --- evaluation ---
    model.eval()
    correct, total = 0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            if task == "multi-label, binary-class":
                probs = torch.sigmoid(out).cpu()
                preds = (probs > 0.5).long()
                correct += (preds == y).float().mean().item() * y.numel()
                total += y.numel()
                all_probs.append(probs)
                all_labels.append(y)
            else:
                probs = torch.softmax(out, dim=1).cpu()
                preds = probs.argmax(dim=1)
                yv = y.squeeze()
                correct += (preds == yv).sum().item()
                total += yv.numel()
                if task == "binary-class":
                    # keep positive-class probabilities so test AUC (the
                    # standard MedMNIST metric) can be reported as well
                    all_probs.append(probs[:, 1].reshape(-1, 1))
                    all_labels.append(yv.float().reshape(-1, 1))
    metrics: Dict[str, float] = {"test_accuracy": correct / max(1, total)}

    # AUC: per-label for multi-label tasks (mean_auc); positive-class AUC
    # for binary-class tasks (test_auc). Multi-class tasks report accuracy
    # only, as before.
    try:
        import numpy as np

        if all_probs:
            P = torch.cat(all_probs).numpy()
            Y = torch.cat(all_labels).numpy()
            aucs = _auc_from_probs(P, Y)
            if aucs:
                if task in ("multi-label, binary-class",):
                    metrics["mean_auc"] = float(np.mean(aucs))
                else:
                    metrics["test_auc"] = float(aucs[0])
    except ImportError:
        pass

    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    record = {
        "fingerprint": fingerprint,
        "config": config,
        "metrics": metrics,
        "history": history,
        "duration_s": time.time() - t_start,
        "device": str(device),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "env": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }
    Path(out_path).write_text(json.dumps(record, indent=2))
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text())
    record = run_training(config, args.data_root, args.out, threads=args.threads)
    print(json.dumps({"fingerprint": record["fingerprint"],
                      "metrics": record["metrics"]}, indent=2))


if __name__ == "__main__":
    main()

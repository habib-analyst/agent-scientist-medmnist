"""Kaggle GPU pilot: the full 20-run variance campaign in ONE kernel session.

Pushed via the Kaggle API (`python launch.py`) once ~/.kaggle/kaggle.json
exists. Estimated ~3-4 GPU-hours on a T4: 20 runs x ResNet-18 x 5 epochs
on 28x28 MedMNIST 2D data.

This is the Executor's KaggleBackend payload for the pilot campaign —
the same agent loop, just with the GPU backend. Results land in
/kaggle/working/results.jsonl and are pulled back after completion.
"""

import json
import subprocess
import sys
from pathlib import Path

# Campaign spec: 2 datasets x 1 arch x 2 augments x 5 seeds = 20 runs
CONFIGS = [
    {"dataset": ds, "arch": "resnet18", "augment": aug, "seed": seed,
     "epochs": 5, "batch_size": 128, "lr": 1e-3, "image_size": 28}
    for ds in ("pathmnist", "pneumoniamnist")
    for aug in ("none", "basic")
    for seed in (0, 1, 2, 3, 4)
]


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "medmnist==3.0.2", "torch", "torchvision", "scikit-learn"],
        check=True)
    sys.path.insert(0, "/kaggle/working/src")
    from agent_scientist.train_medmnist import run_training

    out_dir = Path("/kaggle/working")
    (out_dir / "src").mkdir(exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done = set()
    if results_path.exists():  # resume-safe: skip finished fingerprints
        for line in results_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["fingerprint"])

    import hashlib
    for i, cfg in enumerate(CONFIGS):
        fp = hashlib.sha256(
            json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
        if fp in done:
            print(f"[{i+1}/20] SKIP {fp} (already done)", flush=True)
            continue
        print(f"[{i+1}/20] TRAIN {cfg['dataset']} aug={cfg['augment']} "
              f"seed={cfg['seed']}", flush=True)
        try:
            rec = run_training(cfg, data_root="/kaggle/working/data",
                               out_path=str(out_dir / "last_result.json"),
                               threads=4)
            rec["run_index"] = i
            with results_path.open("a") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
            print(f"[{i+1}/20] OK acc={rec['metrics'].get('test_accuracy'):.4f} "
                  f"fp={fp}", flush=True)
        except Exception as exc:  # log failures, keep the campaign moving
            with results_path.open("a") as f:
                f.write(json.dumps({
                    "fingerprint": fp, "config": cfg, "status": "failed",
                    "error": str(exc)[:500]}) + "\n")
            print(f"[{i+1}/20] FAILED: {exc}", flush=True)

    print(f"done. results: {results_path}")


if __name__ == "__main__":
    main()

"""Experiment backends: where training jobs actually run.

RunSpec: immutable description of one training run (config + seed + budget).
RunResult: measured outcome (metrics or failure info) — never fabricated.

LocalBackend: runs train_medmnist.py as a subprocess on this machine (CPU).
KaggleBackend: pushes the run to Kaggle kernels via the Kaggle API. It
activates only when ~/.kaggle/kaggle.json exists; otherwise every submit
raises BackendUnavailable with a clear message and the run stays queued.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent


class BackendUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    dataset: str            # e.g. "pathmnist", "pneumoniamnist"
    arch: str               # e.g. "resnet18"
    augment: str            # "none" | "basic"
    seed: int
    epochs: int = 5
    batch_size: int = 128
    lr: float = 1e-3
    image_size: int = 28
    extra: Dict[str, Any] = field(default_factory=dict)

    def config_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        import hashlib

        return hashlib.sha256(
            json.dumps(self.config_dict(), sort_keys=True).encode()
        ).hexdigest()[:16]


@dataclass
class RunResult:
    run_id: str
    fingerprint: str
    status: str             # "ok" | "failed" | "queued"
    metrics: Dict[str, float] = field(default_factory=dict)
    duration_s: float = 0.0
    error: Optional[str] = None
    backend: str = ""
    timestamp_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentBackend:
    name = "base"

    def submit(self, spec: RunSpec, workdir: Path) -> RunResult:
        raise NotImplementedError

    def submit_batch(self, specs: List[RunSpec], workdir: Path) -> List[RunResult]:
        return [self.submit(s, workdir) for s in specs]


class LocalBackend(ExperimentBackend):
    """Run training as a local subprocess (CPU on this VM)."""

    name = "local"

    def __init__(self, python: str = sys.executable, timeout_s: int = 7200,
                 threads: int = 4) -> None:
        self.python = python
        self.timeout_s = timeout_s
        self.threads = threads

    def submit(self, spec: RunSpec, workdir: Path) -> RunResult:
        from datetime import datetime, timezone

        t0 = time.time()
        cfg_path = workdir / f"{spec.run_id}.config.json"
        out_path = workdir / f"{spec.run_id}.result.json"
        cfg_path.write_text(json.dumps(spec.config_dict(), indent=2))
        cmd = [
            self.python, str(HERE / "train_medmnist.py"),
            "--config", str(cfg_path),
            "--out", str(out_path),
            "--threads", str(self.threads),
            "--data-root", str(workdir / "data"),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout_s,
                cwd=str(HERE),
            )
        except subprocess.TimeoutExpired:
            return RunResult(
                run_id=spec.run_id, fingerprint=spec.fingerprint(),
                status="failed", duration_s=time.time() - t0,
                error=f"timeout after {self.timeout_s}s", backend=self.name,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            )
        if proc.returncode != 0 or not out_path.exists():
            err = (proc.stderr or "")[-2000:]
            return RunResult(
                run_id=spec.run_id, fingerprint=spec.fingerprint(),
                status="failed", duration_s=time.time() - t0,
                error=f"exit={proc.returncode}: {err}", backend=self.name,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            )
        payload = json.loads(out_path.read_text())
        return RunResult(
            run_id=spec.run_id, fingerprint=spec.fingerprint(),
            status="ok", metrics=payload.get("metrics", {}),
            duration_s=time.time() - t0, backend=self.name,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )


class KaggleBackend(ExperimentBackend):
    """Submit runs as Kaggle GPU kernels via the Kaggle API.

    Requires ~/.kaggle/kaggle.json (username + key). Without it, submit()
    raises BackendUnavailable — the run stays queued, nothing is faked.
    """

    name = "kaggle"

    def __init__(self, kernel_prefix: str = "agent-scientist",
                 gpu: bool = True) -> None:
        self.kernel_prefix = kernel_prefix
        self.gpu = gpu

    def _client(self):  # lazy import: kaggle package only needed here
        kaggle_json = Path(os.path.expanduser("~/.kaggle/kaggle.json"))
        if not kaggle_json.exists():
            raise BackendUnavailable(
                "KaggleBackend needs ~/.kaggle/kaggle.json (username+key). "
                "Run stays queued; ask the operator to provision credentials."
            )
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as exc:
            raise BackendUnavailable(
                f"kaggle package not installed: {exc}"
            ) from exc
        api = KaggleApi()
        api.authenticate()
        return api

    def submit(self, spec: RunSpec, workdir: Path) -> RunResult:
        from datetime import datetime, timezone

        api = self._client()  # raises BackendUnavailable if not configured
        slug = f"{self.kernel_prefix}-{spec.run_id}".lower().replace("_", "-")
        kdir = workdir / "kaggle_push" / spec.run_id
        kdir.mkdir(parents=True, exist_ok=True)
        # The kernel script is generated from the run config.
        (kdir / "kernel.py").write_text(_kernel_script(spec))
        (kdir / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{_kaggle_user(api)}/{slug}",
            "title": slug,
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": self.gpu,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
        }, indent=2))
        api.kernels_push_cli(str(kdir))
        return RunResult(
            run_id=spec.run_id, fingerprint=spec.fingerprint(),
            status="queued", backend=self.name,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

    def fetch_status(self, spec: RunSpec) -> Optional[Dict[str, Any]]:
        """Best-effort status of a queued kernel; None if unknown."""
        try:
            api = self._client()
            slug = f"{self.kernel_prefix}-{spec.run_id}".lower().replace("_", "-")
            lst = api.kernels_list(search=slug)
            if lst:
                k = lst[0]
                return {"slug": slug, "status": getattr(k, "status", "unknown")}
        except BackendUnavailable:
            pass
        return None


def _kaggle_user(api) -> str:
    try:
        return api.get_config_value("username") or "user"
    except Exception:
        return "user"


def _kernel_script(spec: RunSpec) -> str:
    """Self-contained Kaggle kernel: installs deps, trains, writes results."""
    cfg = json.dumps(spec.config_dict())
    return f'''"""Auto-generated by agent-scientist KaggleBackend. Run: {spec.run_id}"""
import json, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "medmnist==3.0.2", "torch", "torchvision"], check=True)
CONFIG = json.loads({cfg!r})
sys.path.insert(0, "/kaggle/working/agent_scientist_src")
from agent_scientist.train_medmnist import run_training
result = run_training(CONFIG, data_root="/kaggle/working/data", out_path="/kaggle/working/result.json", threads=4)
print(json.dumps(result, indent=2))
'''

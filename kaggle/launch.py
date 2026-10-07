"""Push the pilot campaign to Kaggle as one GPU kernel.

Usage: python launch.py [--dry-run]
Requires ~/.kaggle/kaggle.json. Bundles agent_scientist/ + pilot_kernel.py
into a Kaggle script kernel (private, GPU enabled).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    staging = HERE / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "src" / "agent_scientist").mkdir(parents=True)
    for f in (BUILD / "agent_scientist").glob("*.py"):
        shutil.copy(f, staging / "src" / "agent_scientist" / f.name)
    shutil.copy(HERE / "pilot_kernel.py", staging / "pilot_kernel.py")

    meta = {
        "id": "__USER__/agent-scientist-variance-pilot",
        "title": "agent-scientist-variance-pilot",
        "code_file": "pilot_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (staging / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"staged {len(list(staging.rglob('*')))} files in {staging}")
    if args.dry_run:
        print("dry run: not pushing.")
        return 0

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("ERROR: ~/.kaggle/kaggle.json not found — cannot push. "
              "Ask the operator to provision Kaggle API credentials, "
              "then re-run.", file=sys.stderr)
        return 2
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    user = api.get_config_value("username")
    meta["id"] = f"{user}/agent-scientist-variance-pilot"
    (staging / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    api.kernels_push_cli(str(staging))
    print("pushed. Watch it at: "
          f"https://www.kaggle.com/code/{user}/agent-scientist-variance-pilot")
    return 0


if __name__ == "__main__":
    sys.exit(main())

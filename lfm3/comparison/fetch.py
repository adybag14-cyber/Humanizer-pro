"""Download the pinned comparison artifacts and write a local inference manifest."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


HERE = Path(__file__).resolve().parent
BASE = "LiquidAI/LFM3-1.7B"
BASE_REV = "0dc4268b561499c648c7ba7c9466bf3d8d01517a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="new output JSON file")
    args = parser.parse_args()
    models = json.loads((HERE / "releases.json").read_text(encoding="utf-8"))
    base = snapshot_download(BASE, revision=BASE_REV, allow_patterns=[
        "*.json", "*.py", "*.jinja", "model.safetensors", "LICENSE",
    ])
    candidates = []
    for entry in models["published_adapters"]:
        subdir = entry.get("subfolder", "")
        relative = Path(subdir) / "adapter_model.safetensors"
        folder = Path(snapshot_download(
            entry["repo_id"], revision=entry["revision"],
            allow_patterns=[relative.as_posix(), (Path(subdir) / "adapter_config.json").as_posix()],
        ))
        with (folder / relative).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != entry["adapter_sha256"]:
                raise ValueError(f"Weight hash differs for {entry['candidate']}")
        candidates.append({"id": entry["candidate"], "adapter": str(folder / subdir)})
    candidates.append({"id": "control-base"})
    with args.manifest.open("x", encoding="utf-8") as f:
        json.dump({"base": base, "candidates": candidates}, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()

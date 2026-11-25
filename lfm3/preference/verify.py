"""Check frozen training inputs and a completed adapter without CUDA use."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    args = parser.parse_args()
    record = json.loads((HERE / "training.json").read_text())
    assert record["status"] == "complete" and record["steps"] == 64
    for name, field in (
        ("pairs.json", "pairs_sha256"),
        ("settings.json", "settings_sha256"),
        ("train.py", "script_sha256"),
    ):
        assert sha256(HERE / name) == record[field], name
    weights = args.adapter / "adapter_model.safetensors"
    assert sha256(weights) == record["adapter_sha256"]
    assert weights.stat().st_size == record["adapter_bytes"]
    tensors = load_file(weights, device="cpu")
    assert all(torch.isfinite(value).all() for value in tensors.values())
    elements = sum(value.numel() for value in tensors.values())
    assert elements == record["trainable_parameters"]
    config = json.loads((args.adapter / "adapter_config.json").read_text())
    assert config["base_model_name_or_path"] == record["settings"]["model"]
    assert config["revision"] == record["settings"]["base_revision"]
    print(json.dumps({"verified": True, "tensor_count": len(tensors), "parameters": elements}))


if __name__ == "__main__":
    main()

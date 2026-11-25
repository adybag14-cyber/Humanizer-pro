"""Compare repeated fidelity training artifacts without relying on container-file hashes."""

import hashlib
import itertools
import json
import platform
from pathlib import Path

import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
RUNTIME = Path.home() / "humanizer-pro-runtime"
PATHS = {
    "historical": RUNTIME / "fidelity-20261125/lfm3-fidelity-lora",
    "reproduction": RUNTIME / "fidelity-publication-reproduction/adapter",
    "control": RUNTIME / "first-release-control-57301/adapter",
}
OUTPUTS = {
    "historical": HERE.parent / "evidence/lfm3-fidelity.json",
    "reproduction": RUNTIME / "fidelity-publication-reproduction/lfm3-fidelity.json",
    "control": RUNTIME / "first-release-control-57301/lfm3-fidelity.json",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    torch.set_num_threads(8)
    models = {name: load_file(str(path / "adapter_model.safetensors")) for name, path in PATHS.items()}
    output_text = {name: json.loads(path.read_text()) for name, path in OUTPUTS.items()}
    report = {
        "purpose": "Controlled seed-57301 repeat, excluded from the eight-candidate ranking",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "trainer_sha256": digest(HERE.parent / "train.py"),
        "training_data_sha256": digest(HERE.parent / "training-pairs.json"),
        "artifacts": {},
        "comparisons": [],
    }
    for name, path in PATHS.items():
        tensors = models[name]
        report["artifacts"][name] = {
            "weight_sha256": digest(path / "adapter_model.safetensors"),
            "tensor_count": len(tensors),
            "parameter_count": sum(t.numel() for t in tensors.values()),
            "dtypes": sorted({str(t.dtype) for t in tensors.values()}),
            "output_sha256": digest(OUTPUTS[name]),
        }
    for a, b in itertools.combinations(PATHS, 2):
        ta, tb = models[a], models[b]
        assert ta.keys() == tb.keys()
        changed_tensors = []
        exact_values = 0
        total = 0
        sum_absolute = 0
        sum_squared = 0
        max_absolute = 0
        for key in ta:
            assert ta[key].shape == tb[key].shape
            left, right = ta[key], tb[key]
            equal = left.eq(right)
            exact_values += int(equal.sum())
            total += left.numel()
            if not bool(equal.all()):
                changed_tensors.append(key)
            diff = (left.double() - right.double()).abs()
            sum_absolute += float(diff.sum())
            sum_squared += float(diff.square().sum())
            max_absolute = max(max_absolute, float(diff.max()))
        texts_a, texts_b = output_text[a], output_text[b]
        assert texts_a.keys() == texts_b.keys()
        report["comparisons"].append({
            "a": a, "b": b,
            "equal_tensor_count": len(ta) - len(changed_tensors),
            "changed_tensor_count": len(changed_tensors),
            "exactly_equal_values": exact_values,
            "total_values": total,
            "max_absolute_difference": max_absolute,
            "mean_absolute_difference": sum_absolute / total,
            "rms_difference": (sum_squared / total) ** 0.5,
            "identical_outputs": [k for k in texts_a if texts_a[k] == texts_b[k]],
            "different_outputs": [k for k in texts_a if texts_a[k] != texts_b[k]],
        })
    (HERE / "control-57301-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

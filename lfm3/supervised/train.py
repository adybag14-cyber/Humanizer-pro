"""Train a frozen first-release SFT candidate for factual copy editing."""

import argparse
import datetime
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from runtime import BASE_ID, BASE_REVISION, load, prompt_ids


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", choices=["edit-keep", "minimal"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-dir", type=Path)
    args = parser.parse_args()
    settings_file = HERE / "settings.json"
    settings = json.loads(settings_file.read_text())[args.candidate]
    pairs_file = HERE / (args.candidate + "-pairs.json")
    pairs = json.loads(pairs_file.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "owner.json", {
        "pid": os.getpid(), "ppid": os.getppid(), "executable": sys.executable,
        "argv": sys.argv, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })
    save_json(args.output / "frozen-inputs.json", {
        "settings": settings, "settings_sha256": digest(settings_file),
        "pairs_sha256": digest(pairs_file), "script_sha256": digest(Path(__file__)),
    })
    torch.manual_seed(settings["seed"])
    tokenizer, base = load(args.base_dir or BASE_ID, None if args.base_dir else BASE_REVISION)
    model = get_peft_model(base, LoraConfig(
        r=settings["rank"], lora_alpha=settings["alpha"], lora_dropout=0.0,
        target_modules=settings["target_modules"], task_type="CAUSAL_LM",
    ))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=settings["learning_rate"])
    random.Random(settings["seed"]).shuffle(pairs)
    model.train()
    losses = []
    for step, pair in enumerate(pairs, start=1):
        prompt = prompt_ids(tokenizer, pair["source"])
        target = tokenizer.encode(pair["target"], add_special_tokens=False) + [tokenizer.eos_token_id]
        tokens = torch.tensor([prompt + target], device=model.device)
        labels = tokens.clone()
        labels[:, :len(prompt)] = -100
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=tokens, labels=labels, use_cache=False).loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(loss.item())
        print(f"{args.candidate} {step}/{len(pairs)} loss={loss.item():.5f}", flush=True)
    adapter = args.output / "adapter"
    model.save_pretrained(adapter, save_embedding_layers=False)
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config.update(base_model_name_or_path=BASE_ID, revision=BASE_REVISION)
    save_json(config_path, config)
    record = {
        "candidate": args.candidate, "method": "supervised target-only cross-entropy",
        "base_model": BASE_ID, "base_revision": BASE_REVISION, "settings": settings,
        "pairs_sha256": digest(pairs_file), "settings_sha256": digest(settings_file),
        "script_sha256": digest(Path(__file__)), "examples": len(pairs), "steps": len(losses),
        "first_8_mean_loss": sum(losses[:8]) / len(losses[:8]),
        "last_8_mean_loss": sum(losses[-8:]) / len(losses[-8:]),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "adapter_bytes": (adapter / "adapter_model.safetensors").stat().st_size,
        "adapter_sha256": digest(adapter / "adapter_model.safetensors"),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }
    save_json(args.output / "training.json", record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()

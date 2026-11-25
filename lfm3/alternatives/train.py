"""Train fixed weighted-SFT or supervised-unlikelihood factual editing adapters."""

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from runtime import BASE_ID, BASE_REVISION, load, prompt_ids

SETTINGS = {
    "weighted": {
        "seed": 57303, "method": "critical-span weighted supervised fine-tuning",
        "learning_rate": 5e-5, "rank": 8, "alpha": 16, "dropout": 0.0,
        "targets": ["q_proj", "v_proj", "w1", "w2", "w3"], "critical_weight": 3.0,
        "batch_size": 1, "epochs": 1, "max_grad_norm": 1.0, "weight_decay": 0.01,
    },
    "unlikelihood": {
        "seed": 57304, "method": "supervised fine-tuning plus factual-token unlikelihood",
        "learning_rate": 5e-5, "rank": 8, "alpha": 16, "dropout": 0.0,
        "targets": ["q_proj", "k_proj", "v_proj", "out_proj"], "unlikelihood_weight": 0.5,
        "batch_size": 1, "epochs": 1, "max_grad_norm": 1.0, "weight_decay": 0.01,
    },
}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def freeze_record(candidate):
    return {
        "candidate": candidate, "base_model": BASE_ID, "base_revision": BASE_REVISION,
        "settings": SETTINGS[candidate], "training_examples": 24,
        "files_sha256": {
            path.name: digest(path)
            for path in [ROOT / f"{candidate}-pairs.json", Path(__file__), ROOT.parent / "runtime.py"]
        },
    }


def offsets_mask(tokenizer, text, spans, first_only=False):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    mask = [False] * len(encoded["input_ids"])
    for span in spans:
        start = text.find(span)
        if start < 0:
            raise ValueError(f"Marked text absent from target: {span!r}")
        while start >= 0:
            end = start + len(span)
            for index, (left, right) in enumerate(encoded["offset_mapping"]):
                if left < end and right > start:
                    mask[index] = True
            if first_only:
                break
            start = text.find(span, end)
    if not any(mask):
        raise ValueError("No marked tokens")
    return mask


def token_losses(model, prompt, target):
    tokens = torch.tensor([prompt + target], device=model.device)
    output = model(input_ids=tokens, use_cache=False)
    logits = output.logits[:, len(prompt) - 1:-1, :].float()
    labels = torch.tensor([target], device=model.device)
    return F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", choices=SETTINGS)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = freeze_record(args.candidate)
    frozen_path = ROOT / f"{args.candidate}-frozen.json"
    if args.freeze:
        if frozen_path.exists():
            raise FileExistsError(frozen_path)
        record["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(frozen_path, record)
        print(json.dumps(record), flush=True)
        return
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert {key: frozen[key] for key in record} == record, "Frozen inputs or code changed"
    if args.output is None:
        parser.error("--output is required")
    args.output.mkdir(parents=True, exist_ok=False)
    settings = SETTINGS[args.candidate]
    seed = settings["seed"]
    torch.manual_seed(seed)
    pairs = json.loads((ROOT / f"{args.candidate}-pairs.json").read_text(encoding="utf-8"))
    random.Random(seed).shuffle(pairs)
    assert len(pairs) == 24
    tokenizer, base = load(args.base_dir or BASE_ID, None if args.base_dir else BASE_REVISION)
    model = get_peft_model(base, LoraConfig(
        r=settings["rank"], lora_alpha=settings["alpha"], lora_dropout=settings["dropout"],
        target_modules=settings["targets"], task_type="CAUSAL_LM",
    ))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
    )
    model.train()
    losses = []
    supervised_losses = []
    extra_losses = []
    for step, pair in enumerate(pairs, 1):
        prompt = prompt_ids(tokenizer, pair["source"])
        target = tokenizer.encode(pair["target"], add_special_tokens=False) + [tokenizer.eos_token_id]
        optimizer.zero_grad(set_to_none=True)
        per_token = token_losses(model, prompt, target)
        if args.candidate == "weighted":
            marked = offsets_mask(tokenizer, pair["target"], pair["critical"]) + [False]
            weights = torch.tensor(
                [settings["critical_weight"] if value else 1.0 for value in marked],
                device=model.device,
            )
            loss = (per_token * weights).sum() / weights.sum()
            supervised = per_token.mean().detach().item()
            extra = 0.0
        else:
            supervised = per_token.mean()
            # Backpropagate and release this graph before the rejected completion.
            supervised.backward()
            rejected = tokenizer.encode(pair["rejected"], add_special_tokens=False)
            marked = offsets_mask(
                tokenizer, pair["rejected"], pair["rejected_spans"], first_only=True
            )
            negative_losses = token_losses(model, prompt, rejected)
            selected = negative_losses[torch.tensor(marked, device=model.device)]
            probabilities = (-selected).exp().clamp(max=1.0 - 1e-6)
            extra = -torch.log1p(-probabilities).mean()
            loss = settings["unlikelihood_weight"] * extra
            supervised = supervised.detach().item()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, settings["max_grad_norm"])
        optimizer.step()
        extra_value = float(extra) if isinstance(extra, float) else extra.detach().item()
        total = loss.detach().item()
        if args.candidate == "unlikelihood":
            total += supervised
        losses.append(total)
        supervised_losses.append(supervised)
        extra_losses.append(extra_value)
        print(f"{args.candidate} {step}/{len(pairs)} total={total:.5f}", flush=True)
    adapter = args.output / "adapter"
    model.save_pretrained(adapter, save_embedding_layers=False)
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config.update(base_model_name_or_path=BASE_ID, revision=BASE_REVISION)
    write_json(config_path, config)
    stats = {
        **record, "frozen_at_utc": frozen["frozen_at_utc"], "steps": len(losses),
        "losses": losses, "supervised_losses": supervised_losses, "extra_losses": extra_losses,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "adapter_bytes": (adapter / "adapter_model.safetensors").stat().st_size,
        "adapter_sha256": digest(adapter / "adapter_model.safetensors"),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(args.output / "training.json", stats)
    write_json(ROOT / f"{args.candidate}-training.json", stats)
    print(json.dumps({key: stats[key] for key in [
        "candidate", "steps", "trainable_parameters", "adapter_bytes", "adapter_sha256", "peak_cuda_bytes"
    ]}), flush=True)


if __name__ == "__main__":
    main()

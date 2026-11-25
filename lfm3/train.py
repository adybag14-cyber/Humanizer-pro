"""Reproduce the 48-step LFM3 factual editing experiment on one BF16 CUDA GPU."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from runtime import BASE_ID, BASE_REVISION, load, prompt_ids, rewrite

HERE = Path(__file__).resolve().parent
SEED = 57301


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def evaluate(tokenizer, model, cases):
    model.eval()
    return {case["id"]: rewrite(tokenizer, model, case["source"]) for case in cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for adapter, outputs and run record")
    parser.add_argument("--base-dir", type=Path, help="local copy of the pinned LiquidAI checkpoint")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    tokenizer, base = load(args.base_dir or BASE_ID, None if args.base_dir else BASE_REVISION)
    cases = json.loads((HERE / "evidence/cases.json").read_text(encoding="utf-8"))
    save_json(args.output / "lfm3-base.json", evaluate(tokenizer, base, cases))

    model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "w1", "w2", "w3"],
    ))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=5e-5)
    pairs = json.loads((HERE / "training-pairs.json").read_text(encoding="utf-8"))
    random.Random(args.seed).shuffle(pairs)
    model.train()
    losses = []
    for pair in pairs:
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

    adapter = args.output / "adapter"
    model.save_pretrained(adapter, save_embedding_layers=False)
    outputs = evaluate(tokenizer, model, cases)
    save_json(args.output / "lfm3-fidelity.json", outputs)
    weights = adapter / "adapter_model.safetensors"
    with weights.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    save_json(args.output / "training.json", {
        "model": BASE_ID, "base_revision": BASE_REVISION, "seed": args.seed, "steps": len(losses),
        "first_8_mean_loss": sum(losses[:8]) / 8, "last_8_mean_loss": sum(losses[-8:]) / 8,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "adapter_bytes": weights.stat().st_size, "adapter_sha256": digest,
    })


if __name__ == "__main__":
    main()

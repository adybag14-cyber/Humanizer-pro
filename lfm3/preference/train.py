"""Train the fixed DPO factual-editing pilot with one frozen base reference."""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import BASE_ID, BASE_REVISION, load, prompt_ids

HERE = Path(__file__).resolve().parent


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def make_batch(tokenizer, pair, device, limit):
    prompt = prompt_ids(tokenizer, pair["source"])
    sequences = [
        prompt + tokenizer.encode(pair[name], add_special_tokens=False) + [tokenizer.eos_token_id]
        for name in ("chosen", "rejected")
    ]
    width = max(map(len, sequences))
    if width > limit:
        raise ValueError(f"Pair {pair['id']} needs {width} tokens")
    tokens = torch.full((2, width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(tokens)
    target = torch.zeros_like(tokens, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        tokens[row, :len(sequence)] = torch.tensor(sequence, device=device)
        attention[row, :len(sequence)] = 1
        target[row, len(prompt):len(sequence)] = True
    return tokens, attention, target


def completion_logps(model, batch):
    tokens, attention, target = batch
    logits = model(input_ids=tokens, attention_mask=attention, use_cache=False).logits[:, :-1].float()
    next_token = tokens[:, 1:]
    selected = torch.gather(F.log_softmax(logits, dim=-1), -1, next_token.unsqueeze(-1)).squeeze(-1)
    return (selected * target[:, 1:]).sum(dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new output directory")
    parser.add_argument("--base-dir", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    write_json(args.output / "owner.json", {
        "pid": os.getpid(), "executable": sys.executable,
        "argv": sys.argv, "started_utc": started,
    })
    settings = json.loads((HERE / "settings.json").read_text())
    pairs = json.loads((HERE / "pairs.json").read_text())
    assert len(pairs) == settings["pairs"]
    assert len({p["id"] for p in pairs}) == len(pairs)
    provenance = {
        "settings": settings, "settings_sha256": sha256(HERE / "settings.json"),
        "pairs_sha256": sha256(HERE / "pairs.json"), "script_sha256": sha256(Path(__file__)),
        "started_utc": started, "status": "prepared",
    }
    write_json(args.output / "plan.json", provenance)
    torch.manual_seed(settings["seed"])
    torch.cuda.reset_peak_memory_stats()
    tokenizer, base = load(args.base_dir or BASE_ID, None if args.base_dir else BASE_REVISION)
    model = get_peft_model(base, LoraConfig(
        r=settings["lora_rank"], lora_alpha=settings["lora_alpha"],
        lora_dropout=settings["lora_dropout"], target_modules=settings["target_modules"],
        task_type="CAUSAL_LM",
    ))
    trainable = [p for p in model.parameters() if p.requires_grad]
    batches = [make_batch(tokenizer, pair, model.device, settings["max_sequence_tokens"]) for pair in pairs]
    reference = []
    model.eval()
    with model.disable_adapter(), torch.no_grad():
        for batch in batches:
            reference.append(completion_logps(model, batch).detach())
    optimizer = torch.optim.AdamW(trainable, lr=settings["learning_rate"])
    losses = []
    margins = []
    start = time.monotonic()
    model.train()
    for epoch in range(settings["epochs"]):
        order = list(range(len(pairs)))
        random.Random(settings["seed"] + epoch).shuffle(order)
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            policy = completion_logps(model, batches[index])
            margin = (policy[0] - policy[1]) - (reference[index][0] - reference[index][1])
            loss = -F.logsigmoid(settings["beta"] * margin)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite preference loss")
            if not losses and not math.isclose(loss.item(), math.log(2), abs_tol=0.02):
                raise ValueError("Initial DPO loss differs from the identical-reference expectation")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, settings["gradient_norm"])
            optimizer.step()
            losses.append(loss.item())
            margins.append(margin.item())
            if len(losses) == 1 or len(losses) % 8 == 0:
                print(f"{len(losses)}/{len(pairs) * settings['epochs']} loss={loss.item():.5f}", flush=True)
    adapter = args.output / "adapter"
    model.save_pretrained(adapter, save_embedding_layers=False)
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config.update(base_model_name_or_path=BASE_ID, revision=BASE_REVISION)
    write_json(config_path, config)
    provenance.update(
        status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
        steps=len(losses), first_8_mean_loss=sum(losses[:8]) / 8,
        last_8_mean_loss=sum(losses[-8:]) / 8, mean_reference_adjusted_margin=sum(margins) / len(margins),
        trainable_parameters=sum(p.numel() for p in trainable), elapsed_seconds=time.monotonic() - start,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        adapter_sha256=sha256(adapter / "adapter_model.safetensors"),
        adapter_bytes=(adapter / "adapter_model.safetensors").stat().st_size,
    )
    write_json(args.output / "training.json", provenance)
    write_json(HERE / "training.json", provenance)
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()

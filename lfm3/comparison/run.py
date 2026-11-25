"""Generate every frozen comparison case with one shared base and sequential adapters."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import load, rewrite

import torch
from peft import PeftModel

HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalize(text):
    return " ".join(text.split())


def provenance(candidate):
    key = "adapter" if "adapter" in candidate else "model"
    if key not in candidate:
        return {"kind": "base"}
    folder = Path(candidate[key])
    names = ("adapter_config.json", "adapter_model.safetensors") if key == "adapter" else ("config.json", "model.safetensors")
    return {"kind": key, "location": str(folder), "files": {
        name: digest(folder / name) for name in names if (folder / name).is_file()
    }}


def generate(tokenizer, model, candidate, cases, output):
    results = []
    review = []
    for case in cases:
        try:
            text = rewrite(tokenizer, model.eval(), case["source"])
            generation = {"success": True, "text": text, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
        except Exception as error:
            text = ""
            generation = {"success": False, "text": "", "error": str(error)}
        source_tokens = set(normalize(case["source"]).split())
        result_tokens = set(normalize(text).split())
        results.append({
            "case": case["id"], "source": case["source"], "output": generation,
            "copy_after_whitespace_normalization": normalize(text) == normalize(case["source"]),
            "source_token_recall": len(source_tokens & result_tokens) / len(source_tokens),
        })
        review.append({
            "case": case["id"],
            "claims": [{"claim": claim, "status": None, "evidence": ""} for claim in case["claims"]],
            "unsupported_assertions": None, "clarity": None, "editing_goal_met": None,
            "notes": "", "review_complete": False,
        })
    write_json(output / (candidate["id"] + ".outputs.json"), {
        "candidate": candidate["id"], "provenance": provenance(candidate), "cases": results,
    })
    write_json(output / (candidate["id"] + ".review.json"), {"candidate": candidate["id"], "cases": review})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    candidates = manifest["candidates"]
    identifiers = [candidate["id"] for candidate in candidates]
    if len(set(identifiers)) != len(identifiers) or any(
        not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", identifier) for identifier in identifiers
    ):
        parser.error("Candidate IDs must be unique lowercase filenames")
    if any("model" in candidate and "adapter" in candidate for candidate in candidates):
        parser.error("Each candidate must specify either a merged model or an adapter")
    args.output.mkdir(parents=True, exist_ok=False)
    cases = read_json(HERE / "cases.json")
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "run.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "fixtures_sha256": digest(HERE / "cases.json"), "protocol_sha256": digest(HERE / "protocol.json"),
        "runtime_sha256": digest(HERE.parent / "runtime.py"),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")},
    })
    torch.manual_seed(0)
    shared = [candidate for candidate in candidates if "model" not in candidate]
    if shared:
        tokenizer, base = load(manifest["base"], manifest.get("base_revision"))
        for candidate in shared:
            if "adapter" in candidate:
                model = PeftModel.from_pretrained(base, candidate["adapter"], local_files_only=True)
                generate(tokenizer, model, candidate, cases, args.output)
                base = model.unload()
                del model
            else:
                generate(tokenizer, base, candidate, cases, args.output)
            print(candidate["id"], "complete", flush=True)
        del tokenizer, base
        gc.collect()
        torch.cuda.empty_cache()
    for candidate in candidates:
        if "model" in candidate:
            tokenizer, model = load(candidate["model"], candidate.get("revision"))
            generate(tokenizer, model, candidate, cases, args.output)
            del tokenizer, model
            gc.collect()
            torch.cuda.empty_cache()
            print(candidate["id"], "complete", flush=True)
    record = read_json(args.output / "run.json")
    record["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output / "run.json", record)


if __name__ == "__main__":
    main()

"""Edit one UTF-8 passage with a published adapter or standalone LFM3 checkpoint."""

import argparse
import json
from pathlib import Path

from runtime import load, rewrite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    manifest = Path(__file__).parent / "comparison/releases.json"
    releases = {r["candidate"]: r for r in json.loads(manifest.read_text(encoding="utf-8"))["published_adapters"]}
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model", default="Tdamre/LFM3-1.7B-Humanizer-Pro-Fidelity")
    group.add_argument("--candidate", choices=sorted(releases), help="use the pinned PEFT release")
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.candidate:
        if args.revision:
            parser.error("--candidate uses the release revision recorded in the manifest")
        from peft import PeftModel

        release = releases[args.candidate]
        tokenizer, base = load(release["base_model"], release["base_revision"])
        model = PeftModel.from_pretrained(
            base, release["repo_id"], revision=release["revision"], subfolder=release.get("subfolder", "")
        )
    else:
        tokenizer, model = load(args.model, args.revision)
    print(rewrite(tokenizer, model.eval(), args.source.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()

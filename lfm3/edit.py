"""Edit one UTF-8 passage with the merged LFM3 checkpoint."""

import argparse
from pathlib import Path

from runtime import load, rewrite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--model", default="Tdamre/LFM3-1.7B-Humanizer-Pro-Fidelity")
    parser.add_argument("--revision")
    args = parser.parse_args()
    tokenizer, model = load(args.model, args.revision)
    print(rewrite(tokenizer, model.eval(), args.source.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()

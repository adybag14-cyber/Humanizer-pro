# Humanizer Pro

Humanizer Pro is a local English copy editor built on LiquidAI’s LFM3-1.7B-Base. The current release uses three trained adapters to generate candidate rewrites. A local ranker selects one to return. At the default settings, **34 of 57 rewrites (59.6%) received a 100% human reading from both GPTZero and Pangram**, up from **18 of 57 (31.6%)** for the starting model. [Download the verified release](https://huggingface.co/Tdamre/LFM3-1.7B-Humanizer-Pro/tree/2cde52c70bd4e10d48cd66048828facb92560db5) · [Read the full evaluation](lfm3/joint/RESULTS.md)

## Run the published editor

Use a CUDA-enabled Python 3.12 environment. These commands download the exact release used in the benchmark and run its included example.

```sh
python -m pip install huggingface_hub
hf download Tdamre/LFM3-1.7B-Humanizer-Pro --revision 2cde52c70bd4e10d48cd66048828facb92560db5 --local-dir humanizer-model
cd humanizer-model
python -m pip install -r requirements.txt
python humanize.py example-input.txt --output rewrite.txt
```

For your own edit, replace `example-input.txt` with a UTF-8 English text file of 100–3,500 words and choose a new output filename. The first run downloads the pinned base model and local classifiers. The default settings reproduce the benchmark configuration. To sample another rewrite, add `--seed 28151`. [Model card and inference settings](https://huggingface.co/Tdamre/LFM3-1.7B-Humanizer-Pro/blob/2cde52c70bd4e10d48cd66048828facb92560db5/README.md)

## Benchmark results

A passage passes when GPTZero reports 100% human document probability and Pangram reports 100% human text coverage **on the same returned rewrite**. The first two sets informed model selection. The 16 confirmation passages were held back, and the output hashes were recorded before reviewing their scores.

| Configuration | Historical set | First comparison | Reserved confirmation | Total |
| --- | ---: | ---: | ---: | ---: |
| Starting model | 5/25 | 8/16 | 5/16 | 18/57 |
| Two-adapter release | 9/25 | 13/16 | 10/16 | 32/57 |
| **Published three-adapter release** | **9/25** | **14/16** | **11/16** | **34/57** |

**Repeated runs.** On six fixed passages, the published model passed 3/6 at its default seed, 1/6 at the first alternate seed and 3/6 at a follow-up seed. That gives **7/18 successful runs (38.9%)**. Four of the six passages passed at least once. [Per-passage detector readings](lfm3/joint/evidence/detector-readings.json) · [Per-seed results](lfm3/joint/evidence/seeds-voice-mix.json)

**Factual accuracy.** Across 12 reviewed passages, the published model retained **26 of 60** prespecified details. The starting model retained **31 of 60**. A parcel-desk rewrite earned 100% from both detectors while changing the opening hours and Saturday arrangements. Check operational details, names and numbers against the source before using a rewrite. [Fact checks and scorecard](lfm3/joint/evidence/benchmark-summary.json) · [Two additional inputs and their exact rewrites](lfm3/joint/evidence/post-release-smoke.json)

## Earlier releases

- **Fidelity and Minimal.** The original comparison evaluated eight LFM3 fine-tunes on eight procedural notices. Minimal retained all 64 checked facts and made three of six requested edits. The original Fidelity release includes a standalone BF16 model and its adapter. [Eight-model comparison](lfm3/comparison/RESULTS.md) · [Original training method](lfm3/README.md)
- **V2-R2.** This rank-32 adapter ran two editing passes. On the archived bus notice, Pangram reported 100% human and GPTZero reported 100% AI. [Nine archived cases](lfm3/evidence/prior-v2-evaluation.json) · [V2-R2 model](https://huggingface.co/Tdamre/LFM3-1.7B-Humanizer-Pro-V2-R2/tree/e54c8c4a107115b426490336c51c2396ad0e6188)

The repository code uses Apache 2.0. The published model weights use the LFM Open License v1.0. [Code license](LICENSE) · [Model license](https://huggingface.co/Tdamre/LFM3-1.7B-Humanizer-Pro/blob/2cde52c70bd4e10d48cd66048828facb92560db5/LICENSE)

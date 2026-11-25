# Humanizer Pro

Humanizer Pro compares eight public LFM3-1.7B fine-tunes for English copy editing. The Minimal adapter ranked first on eight procedural notices. It preserved all 64 checked facts and made three of the six requested edits. It copied the other three passages that needed editing. [Comparison, model downloads and reproducibility results](lfm3/comparison/RESULTS.md)

The original Fidelity release includes a standalone BF16 model and its PEFT adapter. Seven additional releases provide PEFT adapters for the pinned LiquidAI base. [Fidelity model](https://huggingface.co/Tdamre/LFM3-1.7B-Humanizer-Pro-Fidelity/tree/79ccfcea39ff768642c7e323e926bad979513a82) · [Minimal adapter](https://huggingface.co/Tdamre/LFM3-1.7B-Humanizer-Pro-Minimal/tree/da35d3d0d3f26df3e5cb507d42ea8f04b469f3a7) · [Original training method](lfm3/README.md)

The earlier **V2-R2** release uses a different rank-32 adapter with two inference passes. On the archived bus notice Pangram reported 100% human and GPTZero reported 100% AI. Both services reported 100% AI on its observatory rewrite. [V2-R2 model and decoder](https://huggingface.co/Tdamre/LFM3-1.7B-Humanizer-Pro-V2-R2/tree/e54c8c4a107115b426490336c51c2396ad0e6188) · [Nine public cases and readings](lfm3/evidence/prior-v2-evaluation.json)

## Run one edit

To use the top-ranked Minimal adapter, run the following commands from the repository root in a Python 3.12 environment with a CUDA-enabled build of PyTorch installed.

```sh
python -m pip install -r lfm3/requirements.txt
python lfm3/edit.py input.txt --candidate minimal
```

For the historical two-pass run, use `python lfm3/edit.py input.txt --candidate lfm-v2-r2`.

On 25 November 2026, all eight releases received a 100% AI document probability from GPTZero and 100% AI text coverage from Pangram on the fixed observatory passage. Three adapters copied the source verbatim. [All eight detector results](lfm3/comparison/RESULTS.md#fixed-live-detector-comparison)

The repository code uses the [Apache 2.0 license](LICENSE). The model files carry the [LFM Open License v1.0](https://huggingface.co/Tdamre/LFM3-1.7B-Humanizer-Pro-Fidelity/blob/79ccfcea39ff768642c7e323e926bad979513a82/LICENSE).

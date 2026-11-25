# Humanizer Pro

Humanizer Pro is an experimental English text editor based on LiquidAI’s LFM3-1.7B. The first public fidelity checkpoint rewrote all three procedural notices in its evaluation. It preserved an inclusive parcel-weight limit and changed the meaning of a refund condition. [Training method and before-and-after results](lfm3/README.md)

Download the standalone BF16 model or the original PEFT adapter from [Hugging Face](https://huggingface.co/Tdamre/LFM3-1.7B-Humanizer-Pro-Fidelity/tree/79ccfcea39ff768642c7e323e926bad979513a82).

## Run one edit

In a Python 3.12 environment with a CUDA-enabled build of PyTorch installed, run the following commands from the repository root.

```sh
python -m pip install -r lfm3/requirements.txt
python lfm3/edit.py input.txt --revision 79ccfcea39ff768642c7e323e926bad979513a82
```

On 25 November 2026 both GPTZero and Pangram classified the merged checkpoint’s fixed observatory edit as AI-generated. The repository includes the full passage and both readings. [Detector evidence](lfm3/evidence/observatory-scans.json)

The repository code uses the [Apache 2.0 license](LICENSE). The model files carry the [LFM Open License v1.0](https://huggingface.co/Tdamre/LFM3-1.7B-Humanizer-Pro-Fidelity/blob/79ccfcea39ff768642c7e323e926bad979513a82/LICENSE).

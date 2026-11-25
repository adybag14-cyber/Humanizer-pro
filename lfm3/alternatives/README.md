# Independent editing pilots

These two first releases start independently from the pinned original LiquidAI LFM3-1.7B checkpoint. Each uses one pass over 24 fixed synthetic examples. Their datasets, code and settings were frozen before training. The training sets were written for this experiment without consulting the evaluation inputs.

| Candidate | Objective | Data | LoRA targets |
| --- | --- | --- | --- |
| weighted | Target cross-entropy with marked factual spans weighted by 3 | Rewrites about access duties, permits, estimates, meeting duties, key charges and attendance uncertainty | q_proj, v_proj, w1, w2, w3 |
| unlikelihood | Preferred-target cross-entropy plus 0.5 times the mean token unlikelihood on a marked factual error | Rewrites paired with corruptions of quantities, timing, responsibility, expiry, decisions and qualification rules | q_proj, k_proj, v_proj, out_proj |

The weighted objective normalises by the sum of token weights. Its end token receives weight one. The unlikelihood objective uses teacher-forced rejected completions and applies `-log(1-p(token))` to the tokens overlapping each marked error. These rejected completions are independently authored training examples.

Both runs use BF16 base weights, rank 8, alpha 16, zero LoRA dropout, a batch of one, AdamW at 0.00005 with weight decay 0.01 and gradient clipping at 1.0. The seeds are 57303 and 57304 respectively. The shared system instruction asks for clear prose while preserving facts, quantities, conditions, uncertainty and responsibilities.

Each dataset has six templates with four parameter variations. These are small pilots with substantial template repetition. The two candidates change both the training objective and the training examples, so a difference in evaluation performance cannot be attributed solely to the loss function.

Use the shared dependency versions from `../requirements.txt`. The following commands reproduce the fixed runs on a CUDA GPU.

```sh
python lfm3/alternatives/train.py weighted --output /models/first-release-weighted
python lfm3/alternatives/train.py unlikelihood --output /models/first-release-unlikelihood
```

An optional `--base-dir` points at a local copy of the pinned base checkpoint. Every output directory must be new. The script checks the frozen data, code and runtime hashes before loading the model, saves a portable PEFT adapter under `adapter/`, and writes a training record with the adapter hash and CUDA allocation peak. Frozen manifests and training records are retained beside the scripts.

The adapter is applied to the original base revision `0dc4268b561499c648c7ba7c9466bf3d8d01517a`. The model weights remain subject to the LFM Open License v1.0.
